"""시연 실험: 철회 집단(학습 환자의 5%)에 대한 처리 방법별 L1·L2·L3 (3장 v1.6 4.4·5.3–5.5·8.3절).

철회 집단 X: 학습 환자의 5%를 무작위로 고른다. 라운드 1부터 참여, 라운드 25에 철회 효력.
비교 대상
  P1  학습 참여 차단만(원 학습의 최종 모델, 라운드 1–24에 X가 쓰임)
  U0  처음부터의 재학습(시드 1–5; 시드 1은 처리 작업, 2–5는 θ 산정용)
  U1  체크포인트 재수행(r₀=1, 커밋된 표준화 통계 사용 → 동등 조건 위반)
  U2  FedEraser 조정 재구현(Δt=2, r=0.5)
  U4  경사 상승(임시 구현)
지표: 평가 macro AUC, 멤버십 추론 AUC(X vs fold 9의 같은 크기 비구성원 Z)와 95% CI, L3 판정(θ = U0 AUC 평균 + 2SD),
L1(처리 작업 재실행 해시 비교), L2(입력 커밋 = 기대 집합, X 비소속; U4는 망각 집합 커밋 = X).

  python -m fl.exp_l3 --out /out --runs /runs
"""

import argparse
import json
import os
import statistics

import numpy as np
import torch

from fl import audit as A
from fl import evidence as EV
from fl import mia as MIA
from fl import train as T
from fl import unlearn2 as U
from fl.run import INSTS, Policy, Store, run_fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--withdraw-round", type=int, default=25)
    ap.add_argument("--theta-seeds", type=int, default=5)
    args = ap.parse_args()

    cond = "gpu_det"
    device, max_pool = T.configure(cond)
    data, R = T.Data(args.cache), args.rounds
    store = Store(data)
    common = dict(device=device, max_pool=max_pool, cond=cond, rounds=R)

    # 철회 집단: 학습 환자의 5% 무작위(시드 고정)
    train_pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    rng = np.random.default_rng(20261025)
    cohort = sorted(int(p) for p in rng.choice(train_pids, int(len(train_pids) * args.frac), replace=False))
    cs = set(cohort)
    X_by_k = {k: [int(i) for i in data.train_idx[k] if int(data.pid[i]) in cs] for k in INSTS}
    X_idx = np.sort(np.concatenate([np.array(v, dtype=np.int64) for v in X_by_k.values()]))
    Z_idx = np.sort(np.random.default_rng(1).choice(data.val_idx, min(len(X_idx), len(data.val_idx)), replace=False))
    policy = Policy(withdraw={p: args.withdraw_round for p in cohort})
    pol_wo = policy.without(cohort)
    subj = {k: sorted(store.rids(np.array(X_by_k[k], dtype=np.int64))) for k in INSTS}
    res = {"cohort": {"patients": len(cohort), "records": int(len(X_idx)),
                      "records_by_institution": {k: len(v) for k, v in X_by_k.items()},
                      "withdraw_round": args.withdraw_round, "frac": args.frac,
                      "root_X": EV.SortedMerkle(store.rids(X_idx)).root},
           "Z_records": int(len(Z_idx)), "rounds": R}

    def measure(name, model, norm, extra=None):
        nm = (norm["mean"], norm["std"])
        test = T.macro_auc(data.Y[data.test_idx], T.predict(model, data, data.test_idx, device, norm=nm))[0]
        m = MIA.mia(model, data, X_idx, Z_idx, device, nm)
        res.setdefault("methods", {})[name] = {"test_macro_auc": test, "mia": m, **(extra or {})}
        print(f"{name}: test AUC {test:.4f}, MIA AUC {m['auc']:.4f} {np.round(m['ci'], 4).tolist()}", flush=True)

    # 원 학습(= P1: 학습 참여 차단만)
    orig = run_fl(data, store, policy, base_seed=1, run_dir=os.path.join(args.runs, "orig"), label="orig",
                  retain_every=2, **common)
    measure("P1", orig["model"], orig["norm"], {"final_hash": orig["final_hash"],
                                                 "rejected": sum(c["status"] != "accepted" for c in orig["commits"])})

    # U0: 시드 1(처리 작업) + θ 산정용 시드
    u0_mia = []
    for s in range(1, args.theta_seeds + 1):
        u0 = run_fl(data, store, pol_wo, base_seed=s, run_dir=os.path.join(args.runs, f"U0_s{s}"), label=f"U0_s{s}",
                    log_every=0, **common)
        measure(f"U0_s{s}", u0["model"], u0["norm"], {"final_hash": u0["final_hash"]})
        u0_mia.append(res["methods"][f"U0_s{s}"]["mia"]["auc"])
        if s == 1:
            u0_job = u0
    theta = statistics.mean(u0_mia) + 2 * statistics.stdev(u0_mia)
    res["theta"] = {"value": theta, "rule": "mean + 2·SD of U0 MIA AUC over seeds", "u0_mia_auc": u0_mia}

    # U1: r₀ = 1 → 초기 체크포인트에서, 커밋된 표준화 통계로 재수행(동등 조건 위반)
    u1 = run_fl(data, store, pol_wo, base_seed=1, run_dir=os.path.join(args.runs, "U1"), label="U1",
                start_state=torch.load(os.path.join(args.runs, "orig", "ckpt_r0.pt")), start_round=1,
                norm=orig["norm"], log_every=0, **common)
    measure("U1", u1["model"], orig["norm"], {"final_hash": u1["final_hash"], "equals_U0": u1["final_hash"] == u0_job["final_hash"]})

    # U2: FedEraser
    u2 = U.federaser(data, store, pol_wo, os.path.join(args.runs, "orig"), 1, device, max_pool, R, orig["norm"])
    measure("U2", u2["model"], orig["norm"], {"final_hash": u2["final_hash"], "retained_rounds": u2["retained_rounds"]})

    # U4: 경사 상승(임시)
    final_state = torch.load(os.path.join(args.runs, "orig", f"ckpt_r{R}.pt"))
    u4 = U.gradient_ascent(data, store, pol_wo, X_by_k, final_state, 1, device, max_pool, R, orig["norm"])
    measure("U4_provisional", u4["model"], orig["norm"], {"final_hash": u4["final_hash"]})

    # L1: 처리 작업 재실행(같은 입력·시드로 다시 실행해 결과 해시 비교)
    u0_again = run_fl(data, store, pol_wo, base_seed=1, run_dir=os.path.join(args.runs, "U0_s1_audit"),
                      label="U0_audit", log_every=0, **common)
    u2_again = U.federaser(data, store, pol_wo, os.path.join(args.runs, "orig"), 1, device, max_pool, R, orig["norm"])
    u4_again = U.gradient_ascent(data, store, pol_wo, X_by_k, final_state, 1, device, max_pool, R, orig["norm"])
    l1 = {"U0": u0_again["final_hash"] == u0_job["final_hash"], "U2": u2_again["final_hash"] == u2["final_hash"],
          "U4_provisional": u4_again["final_hash"] == u4["final_hash"]}

    # L2: 입력 커밋 검증
    l2 = {"U0": A.verify_input_set(store, pol_wo, u0_job, subj), "U1": A.verify_input_set(store, pol_wo, u1, subj),
          "U2": A.verify_input_set(store, pol_wo, u2, subj), "U4_provisional": A.verify_input_set(store, pol_wo, u4, subj)}
    l2["U4_provisional"]["forget_set_matches_X"] = all(
        fc["root_X"] == EV.SortedMerkle(subj[fc["k"]]).root for fc in u4["forget_commits"])

    # L3 판정
    for name in ("P1", "U0_s1", "U1", "U2", "U4_provisional"):
        md = res["methods"][name]
        md["L3"] = MIA.decide(md["mia"], theta)
        key = name.replace("_s1", "")
        md["L1"] = {"U0": l1["U0"], "U2": l1["U2"], "U4_provisional": l1["U4_provisional"]}.get(key)
        md["L2"] = l2.get(key, {}).get("result") if key in l2 else None
    res["L2_detail"] = {k: {kk: v[kk] for kk in v if kk != "root_mismatch"} for k, v in l2.items()}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "l3_demo.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("theta", round(theta, 4))
    for name in ("P1", "U0_s1", "U1", "U2", "U4_provisional"):
        md = res["methods"][name]
        print(f"{name:15s} test {md['test_macro_auc']:.4f}  MIA {md['mia']['auc']:.4f}  L1 {md['L1']}  L2 {md['L2']}  L3 {md['L3']}")


if __name__ == "__main__":
    main()
