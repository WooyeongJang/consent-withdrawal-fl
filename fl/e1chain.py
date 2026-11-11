"""E1 Kairos 확인 실행(3장 반영 36번 (b), 2026-10-27) — 시나리오마다 1회, acc 규칙(집계 수락 시점 판정, 알고리즘 2).

fl.e1(이산 사건 시뮬레이션)의 판정 논리를 실제 원장 상태와 블록 시각으로 확인한다.
  - 기관은 라운드 시작 때 원장에 기록된 철회(블록 고정 조회)로 입력 집합을 고르고 커밋한다(정책 사본 지연 시나리오는 오래된 사본).
  - 정보주체 철회는 단말 서명 + 플랫폼 대납으로 시나리오가 정한 시점에 제출한다(δ = 0 → 포함 블록에서 효력).
  - 집계 주체는 수락 판정 직전에 원장의 철회 기록을 읽어(checked version = 읽은 철회 수) 커밋 트리에 대한
    비소속 증명을 검사하고 decide를 기록한다. 거절된 기관은 그 라운드에서 빠지고 다음 라운드에 다시 고른다
    (원장은 (기관, 라운드)마다 커밋을 하나만 받는다. 같은 라운드 안 재계산은 시뮬레이션에서만 센다).
  - 사후 판정은 원장 값만 쓴다: 위반 = 수락된 갱신의 커밋에, 수락 블록 시각 τ 이전(≤)에 효력(tEff)이 생긴 철회자의
    기록이 있음. 잘못된 거절 = 거절했는데 그런 기록이 없음. 경합 = 판정 때 읽은 철회 수 < 수락 이벤트의 철회 기록 버전.
  - 로컬 학습은 실제로 한다(resnet1d_wang, 결정적 GPU). 라운드 길이는 축소(R = 6).

  python -m fl.e1chain --cache /data --out /out --runs /runs
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from fl import evidence as EV
from fl import train as T
from fl.ledger_client import Ledger, b32, text32
from fl.run import INIT_SEED, INSTS, PURPOSE, Store, prog_id, state_cpu

SCENARIOS = [("during", {}), ("aftersend", {}), ("aggdelay", {"D": 0.0}), ("aggdelay", {"D": 10.0}),
             ("aggdelay", {"D": 40.0}), ("version", {"sync_q": 0.3}), ("offline", {"m": 2})]
L_S = 20.0     # 축소 라운드 길이(초): 집계 지연 D = 0, 0.5L, 2L


def run_scenario(data, store, scen, kw, seed, R, n_per, device, max_pool, run_dir):
    rng = random.Random(seed)
    L = Ledger()
    dep = L.deploy("ConsentLedger", label=f"E1-chain {scen} {kw}")
    recs = {k: {} for k in INSTS}
    for k in INSTS:
        for i in data.train_idx[k]:
            recs[k].setdefault(int(data.pid[i]), []).append(int(i))
    # 철회 계획: 라운드 2..R−1, 기관마다 n_per명. 시점은 시나리오가 정한다
    plan = {}
    for r in range(2, R):
        for k in INSTS:
            for p in rng.sample(sorted(recs[k]), n_per):
                when = {"during": "after_commit", "aftersend": "after_send", "version": "round_start",
                        "offline": "round_start"}.get(scen) or rng.choice(["round_start", "after_commit", "after_send"])
                plan.setdefault((r, when), []).append((k, p))
    subjects = [kp for v in plan.values() for kp in v]
    sc = {p: EV.H("subject-commit", EV.subject_secret(p)) for _, p in subjects}
    L.batch([("platform", "registerPolicy", (b32(sc[p]), 1, b32(EV.H("policy", p, 1, 1)), 1)) for _, p in subjects])
    # 학습 설정 커밋(v3.1)과 표준화 통계(1라운드 입력 = 전체 학습 집합)
    prog = prog_id("gpu_det")
    scope = np.concatenate([data.train_idx[k] for k in INSTS])
    mean, std = T.norm_stats(data, scope)
    L.tx("platform", "commitTrainingConfig", b32(EV.H("config", seed, EV.SEED_RULE, prog,
                                                       EV.SortedMerkle(store.rids(scope)).root)))
    withdrawn_chain = []                                       # 원장 이벤트: (pid, tEff, reqId)
    req_pid = {}
    t_wall = {}

    def submit_withdrawals(r, when):
        for k, p in plan.get((r, when), []):
            root = EV.SortedMerkle(sorted(store.rid[i] for i in recs[k][p])).root
            t0 = time.time()
            o = L.submit_feepayer(L.sign_subject("requestWithdrawal", b32(sc[p]), 1, b32(root))["raw"], "requestWithdrawal")
            ev = next(e for e in o["events"] if e["event"] == "WithdrawalEffective")
            req_pid[int(ev["args"]["reqId"])] = p                 # 원장 순번 → 정보주체
            withdrawn_chain.append({"pid": p, "k": k, "tEff": int(ev["args"]["tEff"]), "round": r, "when": when,
                                    "inclusion_s": o.get("latency_s"), "block": o.get("block")})
            t_wall[p] = t0

    def chain_withdrawn():
        """원장의 철회 기록(블록 고정 조회): 효력 있는 정보주체 집합과 읽은 철회 수.
        δ = 0이라 요청 순번 0..n−1이 효력 있는 철회이며, 순번은 각 요청 영수증의 reqId로 정보주체에 대응시킨다."""
        n = L.view("effectiveCount")
        return {req_pid[i] for i in range(n)}, n

    g = T.new_model(max_pool, device, INIT_SEED)
    h_prev = T.model_hash(g)[0]
    off = {"k": rng.choice(INSTS), "a": 2} if scen == "offline" else None
    stale_view = {k: (set(), 0) for k in INSTS}
    held = {}                                                  # offline: 오프라인 직전 선택
    units = []
    for r in range(1, R + 1):
        L.tx("platform", "openRound", r, int(time.time()) + 3600, b32(h_prev))
        submit_withdrawals(r, "round_start")
        cur = chain_withdrawn()
        sel = {}
        for k in INSTS:
            if off and off["k"] == k and off["a"] <= r < off["a"] + kw["m"]:
                if r == off["a"]:
                    held[k] = cur[0]                           # 오프라인 직전 선택에 쓴 철회 목록
                continue
            if scen == "version":
                if rng.random() > kw["sync_q"]:
                    stale_view[k] = cur                        # 동기화 성공
                view = stale_view[k]
            elif off and off["k"] == k and r == off["a"] + kw["m"]:
                view = (held[k], None)                         # 재접속: 과거 선택으로 만든 갱신
            else:
                view = cur
            used = np.sort(np.array([i for p, ix in recs[k].items() if p not in view[0] for i in ix], dtype=np.int64))
            tree = EV.SortedMerkle(store.rids(used))
            # 정책 스냅샷 σ = 선택 때 읽은 철회 수(알고리즘 S2). 2026-11-01(코드 검토 C5): 종전에는 1로 고정했다.
            sigma = int(view[1]) if view[1] is not None else 0
            L.tx(f"dc{k}", "commitInput", int(k), r, b32(tree.root), len(used), sigma, text32(PURPOSE))
            sel[k] = {"used": used, "tree": tree, "view_n": view[1]}
        # 철회 제출은 라운드당 한 번. 2026-11-01(C5): 종전 during 조건은 기관 루프 안에서 호출해 같은 철회를 K번
        # 제출했다(최종 재실행 결과의 "72건"은 24명 × 3회). during은 선택 후·학습 전, 나머지는 선택 후 효력.
        submit_withdrawals(r, "after_commit")
        states, weights = [], []
        for k, s in sel.items():
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(g.state_dict())
            m, _ = T.local_train(m, data, k, r, seed, device, R, idx=s["used"], norm=(mean, std))
            s["state"] = state_cpu(m)
            L.tx(f"dc{k}", "commitUpdate", int(k), r, b32(T.model_hash(m)[0]), b32(h_prev))
        submit_withdrawals(r, "after_send")                    # 전송 후, 수락 전 효력
        if kw.get("D"):
            time.sleep(kw["D"])                                # 집계 지연
        for k, s in sel.items():
            wd, n_checked = chain_withdrawn()                  # 수락 판정 직전 원장 조회
            q = [store.rid[i] for p in wd if p in recs[k] for i in recs[k][p]]
            member = any(not EV.verify_non_membership(s["tree"].root, s["tree"].non_membership(x)) for x in q)
            o = L.tx("aggregator", "decide", int(k), r, not member, text32("policy_conflict" if member else "ok"))
            ev = next(e for e in o["events"] if e["event"] in ("UpdateAccepted", "UpdateRejected"))
            tau = int(ev["args"]["tau"])
            ver = int(ev["args"]["withdrawalVersion"]) if ev["event"] == "UpdateAccepted" else None
            units.append({"k": k, "r": r, "accepted": not member, "tau": tau, "checked_version": n_checked,
                          "chain_version": ver, "n_used": int(len(s["used"])),
                          "used_pids": sorted({int(data.pid[i]) for i in s["used"]} & {p for _, p in subjects}),
                          "decide_latency_s": o.get("latency_s")})
            if not member:
                states.append(s["state"]); weights.append(len(s["used"]))
        if states:
            g.load_state_dict(T.fedavg(states, weights))
        h_prev = T.model_hash(g)[0]
        L.tx("aggregator", "aggregate", r, b32(h_prev), b32(h_prev))
    # 사후 판정(원장 값만): 위반·잘못된 거절·경합
    eff = {w["pid"]: w["tEff"] for w in withdrawn_chain}
    viol = wrong = race = correct = 0
    for u in units:
        bad = [p for p in u["used_pids"] if p in eff and eff[p] <= u["tau"]]
        if u["accepted"]:
            viol += len(bad)
            race += int(u["chain_version"] is not None and u["checked_version"] < u["chain_version"])
        else:
            correct += int(bool(bad)); wrong += int(not bad)
    incl = [w["inclusion_s"] for w in withdrawn_chain if w["inclusion_s"] is not None]
    fees = sum((x.get("fee_kei") or 0) for x in L.log) / 1e18
    return {"scenario": scen, "params": kw, "seed": seed, "contract": dep["address"], "rounds": R,
            "withdrawals": len(withdrawn_chain), "updates": len(units), "accepted": sum(u["accepted"] for u in units),
            "violations": viol, "rejections_correct": correct, "rejections_wrong": wrong, "race": race,
            "inclusion_latency_s": {"mean": float(np.mean(incl)), "min": float(np.min(incl)), "max": float(np.max(incl)),
                                    "p95": float(np.percentile(incl, 95))} if incl else None,
            "units": units, "withdrawal_events": withdrawn_chain, "tx_total": len(L.log), "fee_kaia": fees,
            "key_txs": L.key_txs()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--per", type=int, default=2, help="라운드·기관마다 철회자 수")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="", help="시나리오 번호(쉼표) — 스모크용")
    a = ap.parse_args()
    device, max_pool = T.configure("gpu_det")
    data = T.Data(a.cache)
    store = Store(data)
    os.makedirs(a.out, exist_ok=True)
    pick = [int(x) for x in a.only.split(",")] if a.only else range(len(SCENARIOS))
    res = []
    for j in pick:
        scen, kw = SCENARIOS[j]
        t0 = time.time()
        o = run_scenario(data, store, scen, kw, a.seed, a.rounds, a.per, device, max_pool, a.runs)
        o["wall_s"] = time.time() - t0
        res.append(o)
        print(f"[E1-chain] {scen} {kw}: 갱신 {o['updates']} 수락 {o['accepted']} 위반 {o['violations']} "
              f"거절(올바름/잘못됨) {o['rejections_correct']}/{o['rejections_wrong']} 경합 {o['race']} "
              f"포함 지연 {o['inclusion_latency_s']} 거래 {o['tx_total']} 수수료 {o['fee_kaia']:.3f} ({o['wall_s']:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(a.out, "E1_chain.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
