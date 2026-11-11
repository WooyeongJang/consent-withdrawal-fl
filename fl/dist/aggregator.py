"""집계 주체 컨테이너: 데이터 없이 데이터센터에 라운드를 요청하고, 알고리즘 2(수락 시점 비소속 확인)로 판정한 뒤 FedAvg한다.

  FL_K=3 python -m fl.dist.aggregator --scenario results/consent/E2-base.json --seed 1 --rounds 50 --out /out \
      --dcs http://dc0:9000,http://dc1:9000,http://dc2:9000 [--delay-ms 0 --rate-mbps 0]
단일 프로세스 모사(fl.run.run_fl)와 같은 순서·같은 함수로 계산한다(기관 순서대로 수락 판정, 같은 순서로 FedAvg).
출력: 라운드별 전역 모델 해시, 커밋(공개 원장 사본, 감사자가 읽음), 사건 기록, 실측 전송량·시간.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from fl import evidence as EV
from fl import train as T
from fl.dist.common import Link, bytes_state, state_bytes, wait_ready
from fl.run import INIT_SEED, INSTS, PURPOSE, Policy, state_cpu


def scenario_policy(path):
    if not path:
        return Policy()
    sc = json.load(open(path))
    return Policy(withdraw={int(p): w for p, w in sc["withdraw"].items()},
                  reconsent={int(p): c for p, c in sc["reconsent"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dcs", required=True)
    ap.add_argument("--ckpt", default="/ckpt", help="전역 모델 체크포인트(집계 주체 저장소)")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="인위적 단방향 지연(요청·응답 각각)")
    ap.add_argument("--rate-mbps", type=float, default=0.0, help="인위적 대역폭(0이면 제한 없음)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device, max_pool = T.configure("gpu_det")
    policy = scenario_policy(a.scenario)
    pol_d = policy.describe()
    urls = a.dcs.split(",")
    assert len(urls) == len(INSTS)
    links = {k: Link(u, a.delay_ms / 1000, a.rate_mbps * 1e6) for k, u in zip(INSTS, urls)}
    for u in urls:
        wait_ready(u)
    pool = ThreadPoolExecutor(len(INSTS))
    call = lambda path, meta, blobs=None: dict(zip(INSTS, pool.map(
        lambda k: links[k].post(path, meta if not callable(meta) else meta(k), (blobs or {}).get(k, b"")), INSTS)))
    events = []
    ev = lambda name, **kw: events.append({"event": name, **kw})
    t_all = time.time()

    # 정보주체 등록(rid 약속)과 표준화 통계(기관별 합만 모음)
    reg = {k: m["subjects"] for k, (m, _) in call("/registry", {}).items()}
    nr = {k: m for k, (m, _) in call("/norm", {"policy": pol_d, "r": 1}).items()}
    tot = sum(nr[k]["size"] for k in INSTS)
    mean = sum(nr[k]["sum"] for k in INSTS) / tot
    std = float(np.sqrt(sum(nr[k]["sumsq"] for k in INSTS) / tot - mean ** 2))
    norm = {"mean": float(mean), "std": std, "scope_roots": {k: nr[k]["scope_root"] for k in INSTS},
            "n": int(sum(nr[k]["n"] for k in INSTS))}
    ev("NormStatsCommitted", **norm)

    g = T.new_model(max_pool, device, INIT_SEED)
    h_prev = T.model_hash(g)[0]
    round_hashes, commits, per_round = {0: h_prev}, [], []
    os.makedirs(a.ckpt, exist_ok=True)
    torch.save(state_cpu(g), os.path.join(a.ckpt, "ckpt_r0.pt"))
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        gb = state_bytes(state_cpu(g))
        meta = {"r": r, "policy": pol_d, "base_seed": a.seed, "rounds": a.rounds, "norm": [norm["mean"], norm["std"]],
                "purpose": PURPOSE}
        res = call("/round", meta, {k: gb for k in INSTS})
        t_train = time.time() - t0
        states, weights, accepted = [], [], []
        for k in INSTS:                                          # 단일 프로세스 모사와 같은 기관 순서
            m, blob = res[k]
            if "error" in m:
                raise RuntimeError(f"dc{k}: {m['error']}")
            if m.get("skipped"):
                ev("InstitutionSkipped", k=k, r=r, reason="empty_input_set")
                continue
            C, U = m["C"], m["U"]
            if U["h_start"] != h_prev or C["sigma"] != policy.version:
                status, reason, invalid = "rejected", "start_or_policy_mismatch", []
            else:
                # 알고리즘 2: 수락 시점에 효력이 있는 이 기관 철회자의 rid 비소속 증명을 받아 확인
                invalid = [q for p, rs in reg[k].items() for q in rs
                           if policy.withdraw.get(int(p), 10 ** 9) <= r and not policy.valid(int(p), r)]
                status, reason = "accepted", ""
                if invalid:
                    pr, _ = links[k].post("/proofs", {"r": r, "rids": invalid})
                    for q in invalid:
                        if not EV.verify_non_membership(C["root_S"], pr["proofs"][q]):
                            status, reason = "rejected", "policy_conflict"
                            break
            ev("RoundInputCommitted", **C); ev("UpdateCommitted", **U)
            ev("UpdateAccepted" if status == "accepted" else "UpdateRejected", k=k, r=r, reason=reason,
               withdrawn_checked=len(invalid))
            commits.append({"C": C, "U": U, "B": m["B"], "status": status})
            if status == "accepted":
                states.append(bytes_state(blob)); weights.append(C["n"]); accepted.append(k)
        g.load_state_dict(T.fedavg(states, weights))
        h_prev = T.model_hash(g)[0]
        round_hashes[r] = h_prev
        torch.save(state_cpu(g), os.path.join(a.ckpt, f"ckpt_r{r}.pt"))
        ev("RoundAggregated", r=r, accepted=accepted, h_out=h_prev)
        per_round.append({"r": r, "wall_s": time.time() - t0, "train_phase_s": t_train})
        if r % 10 == 0:
            print(f"[dist] round {r}: {h_prev[:12]}", flush=True)

    with open(os.path.join(a.out, "events.jsonl"), "w") as f:
        for e in events:
            f.write(EV.canonical(e) + "\n")
    pub = {"policy": pol_d, "norm": norm, "base_seed": a.seed, "rounds": a.rounds, "commits": commits,
           "round_hashes": {str(r): h for r, h in round_hashes.items()}}
    json.dump(pub, open(os.path.join(a.out, "public.json"), "w"), indent=1)
    comm = {k: {"sent": l.sent, "recv": l.recv,
                "by_path": {p: {"calls": sum(c["path"] == p for c in l.calls),
                                "sent": sum(c["sent"] for c in l.calls if c["path"] == p),
                                "recv": sum(c["recv"] for c in l.calls if c["path"] == p)} for p in {c["path"] for c in l.calls}}}
            for k, l in links.items()}
    out = {"final_hash": h_prev, "rounds": a.rounds, "K": len(INSTS), "scenario": a.scenario, "seed": a.seed,
           "delay_ms": a.delay_ms, "rate_mbps": a.rate_mbps, "wall_s": time.time() - t_all, "per_round": per_round,
           "comm": comm, "model_bytes_serialized": len(gb),
           "accepted": sum(c["status"] == "accepted" for c in commits), "rejected": sum(c["status"] != "accepted" for c in commits)}
    json.dump(out, open(os.path.join(a.out, "aggregator.json"), "w"), indent=1)
    print(f"[dist] done final {h_prev[:12]} wall {out['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
