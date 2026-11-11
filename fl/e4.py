"""E4 비용과 확장성(RQ4) (3장 v1.6 8.4 E4).

(1) micro  : 정렬 머클 트리의 생성·비소속 증명 생성·검증 시간과 증명 크기(집합 크기 10²–10⁵), 결속값·모델 해시 시간.
(2) chain  : 기관 수 K(FL_K)마다 Kairos에 R라운드 학습을 기록하고, 정보주체 N명 정책 등록과 그중 일부 철회(정보주체 서명 + 대납),
             첫 사용 라운드 신고를 넣는다. 함수(이벤트)별 가스·대납 수수료·제출~영수증 지연·호출 데이터 크기,
             라운드당 학습 시간과 원장 기록 시간, 재실행 감사(L1) 단위 비용(표본 비율 p별), 라운드당 통신량을 잰다.
테스트넷 수수료는 운영 비용으로 환산하지 않고 가스로 비교한다(8.4). 주체 수·철회 비율에 대한 총량은 함수별 가스가
상수임을 확인한 뒤(분산 보고) 호출 수 × 가스로 외삽한다.

  FL_K=10 python -m fl.e4 --mode chain --cache /cache/ptbxl_k10 --out /out --runs /runs --rounds 10 --subjects 100 --withdraw 20
  python -m fl.e4 --mode micro --cache /data --out /out
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

from fl import audit as AU
from fl import evidence as EV
from fl import train as T
from fl.run import INSTS, Policy, Store, run_fl

# ABI 인코딩 길이(선택자 4바이트 + 정적 인자 32바이트씩). 모든 함수 인자가 정적 타입이다.
N_ARGS = {"registerPolicy": 4, "requestWithdrawal": 3, "declareFirstUse": 2, "openRound": 3, "commitInput": 6,
          "commitUpdate": 4, "decide": 4, "aggregate": 3, "sampleAudit": 1, "createJob": 4, "completeJob": 2,
          "recordVerification": 4, "deploy": 2, "revoke": 2, "approveMethod": 3, "commitTrainingConfig": 1}


def micro(args):
    rng = np.random.default_rng(0)
    out = []
    for n in (100, 1000, 10000, 100000):
        items = sorted({EV.H("rid", int(x)) for x in rng.integers(0, 10 ** 12, n)})
        t = time.perf_counter(); tree = EV.SortedMerkle(items); t_build = time.perf_counter() - t
        qs = [EV.H("q", int(x)) for x in rng.integers(0, 10 ** 12, 200)]
        t = time.perf_counter(); proofs = [tree.non_membership(q) for q in qs]; t_prove = (time.perf_counter() - t) / len(qs)
        t = time.perf_counter(); ok = all(EV.verify_non_membership(tree.root, p) for p in proofs); t_ver = (time.perf_counter() - t) / len(qs)
        size = statistics.mean(len(EV.canonical(p).encode()) for p in proofs)
        out.append({"n": n, "build_s": t_build, "prove_s": t_prove, "verify_s": t_ver, "proof_bytes_json": size, "ok": ok})
        print(out[-1], flush=True)
    device, max_pool = T.configure("gpu_det")
    m = T.new_model(max_pool, device, 0)
    t = time.perf_counter()
    for _ in range(20):
        T.model_hash(m)
    t_hash = (time.perf_counter() - t) / 20
    t = time.perf_counter()
    for _ in range(1000):
        EV.binding_value("0", 1, "a" * 64, "b" * 64, 1, "p", "prog", "c" * 64)
    res = {"merkle": out, "model_hash_s": t_hash, "binding_value_s": (time.perf_counter() - t) / 1000,
           "param_bytes": sum(v.numel() * v.element_size() for v in m.state_dict().values())}
    json.dump(res, open(os.path.join(args.out, "E4_micro.json"), "w"), indent=1)


def chain(args):
    from fl.ledger_client import Ledger, b32, text32
    K, R = len(INSTS), args.rounds
    L = Ledger()
    dep = L.deploy("ConsentLedger", label=f"E4 K{K}")
    device, max_pool = T.configure("gpu_det")
    data = T.Data(args.cache)
    store = Store(data)
    rng = np.random.default_rng(K)
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    subj = sorted(int(p) for p in rng.choice(pids, args.subjects, replace=False))
    wd = sorted(int(p) for p in rng.choice(subj, args.withdraw, replace=False))
    w = max(2, R // 2)
    inst_of = {int(data.pid[i]): k for k in INSTS for i in data.train_idx[k]}
    sc = {p: EV.H("subject-commit", EV.subject_secret(p)) for p in subj}
    policy = Policy(withdraw={p: w for p in wd})
    t0 = time.time()
    L.batch([("platform", "registerPolicy", (b32(sc[p]), 1, b32(EV.H("policy", p, 3, 1)), 3)) for p in subj])
    t_reg = time.time() - t0
    reqs, t_wd = {}, {}

    def on_round_start(r):
        if r != w:
            return
        t = time.time()
        for p in wd:
            root = EV.SortedMerkle(store.subject_rids(p)).root
            o = L.submit_feepayer(L.sign_subject("requestWithdrawal", b32(sc[p]), 1, b32(root))["raw"], "requestWithdrawal")
            reqs[p] = next(int(e["args"]["reqId"]) for e in o["events"] if e["event"] == "WithdrawalRequested")
        t_wd["requests_s"] = time.time() - t
        t = time.time()
        L.batch([(f"dc{inst_of[p]}", "declareFirstUse", (reqs[p], 1)) for p in wd])
        t_wd["declare_s"] = time.time() - t

    round_times = []
    t_run = time.time()
    run = run_fl(data, store, policy, base_seed=1, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                 run_dir=os.path.join(args.runs, f"K{K}"), label=f"E4-K{K}", ledger=L, on_round_start=on_round_start,
                 log_every=1)
    t_run = time.time() - t_run
    # 학습만의 시간(원장 없이 같은 학습) — 원장 기록 시간과 분리
    t = time.time()
    run_fl(data, store, policy, base_seed=1, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
           run_dir=os.path.join(args.runs, f"K{K}_noledger"), label="noledger", log_every=0)
    t_train = time.time() - t
    # 재실행 감사(L1) 단위 비용: 모든 기관·라운드 갱신을 한 번씩 재실행해 단위 시간을 잰다(p별 비용은 p × 단위 수 × 평균)
    l1 = []
    for c in run["commits"]:
        t = time.time()
        ok = AU.reexecute(data, store, policy, os.path.join(args.runs, f"K{K}"), c, 1, device, max_pool, R,
                          run["norm"], run["prog"])["result"] == "통과"
        l1.append({"k": c["C"]["k"], "r": c["C"]["r"], "n": c["C"]["n"], "s": time.time() - t, "ok": ok})
    # 함수별 가스·지연·수수료
    by = {}
    for x in L.log:
        fn = x.get("fn")
        by.setdefault(fn, []).append(x)
    fn_stats = {}
    for fn, xs in by.items():
        g = [x.get("gas_used") for x in xs if x.get("gas_used") is not None]
        lat = [x.get("latency_s") for x in xs if x.get("latency_s") is not None]
        fee = [x.get("fee_kei", 0) or 0 for x in xs]
        fn_stats[fn] = {"count": len(xs), "gas_mean": statistics.mean(g) if g else None,
                        "gas_sd": statistics.pstdev(g) if len(g) > 1 else 0.0, "gas_min": min(g) if g else None,
                        "gas_max": max(g) if g else None, "latency_mean_s": statistics.mean(lat) if lat else None,
                        "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
                        "fee_kaia": sum(fee) / 1e18,
                        "calldata_bytes": 4 + 32 * N_ARGS[fn.split(":")[-1]] if fn and fn.split(":")[-1] in N_ARGS else None}
    pb = sum(v.numel() * v.element_size() for v in run["model"].state_dict().values())
    res = {"K": K, "rounds": R, "subjects": args.subjects, "withdrawals": args.withdraw, "withdraw_round": w,
           "contract": dep["address"], "key_txs": L.key_txs(), "tx_total": len(L.log), "fee_kaia": sum(x.get("fee_kei", 0) or 0 for x in L.log) / 1e18,
           "fn": fn_stats, "time": {"run_with_ledger_s": t_run, "train_only_s": t_train, "register_s": t_reg, **t_wd,
                                    "per_round_ledger_s": (t_run - t_train - t_wd.get("requests_s", 0) - t_wd.get("declare_s", 0)) / R,
                                    "per_round_train_s": t_train / R},
           "tx_per_round": 3 * K + 3, "comm_per_round_bytes": 2 * K * pb, "param_bytes": pb,
           "l1_units": l1, "l1_unit_mean_s": statistics.mean(x["s"] for x in l1), "l1_all_ok": all(x["ok"] for x in l1),
           "l1_cost_by_p": {str(p): p * K * statistics.mean(x["s"] for x in l1) for p in (0.1, 0.3, 0.5, 1.0)},
           "inst_sizes": {k: int(len(data.train_idx[k])) for k in INSTS}}
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"E4_K{K}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("K", "tx_total", "fee_kaia", "time", "l1_unit_mean_s", "l1_all_ok")}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["micro", "chain"], required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="/runs")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--subjects", type=int, default=100)
    ap.add_argument("--withdraw", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (micro if args.mode == "micro" else chain)(args)


if __name__ == "__main__":
    main()
