"""후기 철회(w=45) 조건에서 처리 방법별 L1·L2·L3(LiRA) (3장 v1.6 5.3–5.5·8.3절, L1∧L2 ⇏ L3).

대상: P1(참여 차단만), U0(처음부터 재학습; w와 무관하므로 L3 시연의 U0 시드 1–5 재사용),
     U1(r₀=1 체크포인트 재수행; w와 무관하므로 L3 시연의 U1 재사용), U2 FedEraser, U4 경사 상승(임시)
L3: LiRA(후기 조건 그림자 16개, offline·online), 비교 기준은 U0 시드 1–5의 분포
L1: 처리 작업 재실행 해시 비교, L2: 입력 커밋 검증

  python -m fl.exp_late_methods --out /out --runs /runs --l3runs /l3runs --lirarun /lirarun --l3json results/l3/l3_demo.json
"""

import argparse
import json
import os

import numpy as np
import torch

from fl import audit as A
from fl import evidence as EV
from fl import train as T
from fl import unlearn2 as U
from fl.exp_lira import evaluate, scores, stat
from fl.run import INIT_SEED, INSTS, Policy, Store, run_fl


def main():
    ap = argparse.ArgumentParser()
    for a in ("--out", "--runs", "--l3runs", "--lirarun", "--l3json"):
        ap.add_argument(a, required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--late", type=int, default=45)
    args = ap.parse_args()

    device, max_pool = T.configure("gpu_det")
    data, R = T.Data(args.cache), args.rounds
    store = Store(data)
    l3 = json.load(open(args.l3json))
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    cohort = sorted(int(p) for p in np.random.default_rng(20261025).choice(pids, int(len(pids) * l3["cohort"]["frac"]), replace=False))
    cs = set(cohort)
    X_by_k = {k: [int(i) for i in data.train_idx[k] if int(data.pid[i]) in cs] for k in INSTS}
    X = np.sort(np.array([i for k in INSTS for i in X_by_k[k]]))
    Z = np.sort(np.random.default_rng(1).choice(data.val_idx, min(len(X), len(data.val_idx)), replace=False))
    pool = np.r_[X, np.array([z for z in Z if data.inst[z] in INSTS])]
    nX = len(X)
    policy = Policy(withdraw={p: args.late for p in cohort})
    pol_wo = policy.without(cohort)
    subj = {k: sorted(store.rids(np.array(X_by_k[k], dtype=np.int64))) for k in INSTS}
    z = np.load(os.path.join(args.lirarun, "shadows_late.npz"))
    phis, inmask = z["phis"], z["inmask"]
    res = {"withdraw_round": args.late, "cohort_records": int(nX), "methods": {}}

    def measure(name, model, nm, extra=None):
        test = T.macro_auc(data.Y[data.test_idx], T.predict(model, data, data.test_idx, device, norm=nm))[0]
        on, off = scores(stat(model, data, pool, device, nm), phis, inmask)
        res["methods"][name] = {"test_macro_auc": test, "lira_offline": evaluate(off, nX), "lira_online": evaluate(on, nX),
                                "final_hash": T.model_hash(model)[0], **(extra or {})}
        print(f"{name}: test {test:.4f}  LiRA offline {res['methods'][name]['lira_offline']['auc']:.4f} "
              f"{np.round(res['methods'][name]['lira_offline']['ci'], 4).tolist()}  online {res['methods'][name]['lira_online']['auc']:.4f}", flush=True)

    def load(path, nm):
        m = T.new_model(max_pool, device, INIT_SEED); m.load_state_dict(torch.load(path)); return m, nm

    def norm_of(run_dir):
        e = next(json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl")) if "NormStatsCommitted" in l)
        return (e["mean"], e["std"]), e

    # 후기 원 학습을 업데이트 보존과 함께 다시 실행(결정적 재현이면 LiRA 실험의 orig_late와 해시가 같아야 함)
    orig = run_fl(data, store, policy, base_seed=1, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                  run_dir=os.path.join(args.runs, "orig_late"), label="orig_late", log_every=0, retain_every=2)
    prev_hash = T.model_hash(load(os.path.join(args.lirarun, "orig_late", f"ckpt_r{R}.pt"), None)[0])[0]
    res["orig_late_rerun_identical"] = orig["final_hash"] == prev_hash
    nm = (orig["norm"]["mean"], orig["norm"]["std"])
    measure("P1", orig["model"], nm)

    for s in range(1, 6):
        n_s, _ = norm_of(os.path.join(args.l3runs, f"U0_s{s}"))
        measure(f"U0_s{s}", load(os.path.join(args.l3runs, f"U0_s{s}", f"ckpt_r{R}.pt"), n_s)[0], n_s)
    n_l3, e_l3 = norm_of(os.path.join(args.l3runs, "orig"))
    res["U1_reuse_valid"] = n_l3 == nm   # U1은 같은 표준화 통계와 같은 정책으로 r₀=1부터 재수행하므로 w와 무관
    measure("U1", load(os.path.join(args.l3runs, "U1", f"ckpt_r{R}.pt"), nm)[0], nm)

    u2 = U.federaser(data, store, pol_wo, os.path.join(args.runs, "orig_late"), 1, device, max_pool, R, orig["norm"])
    u2b = U.federaser(data, store, pol_wo, os.path.join(args.runs, "orig_late"), 1, device, max_pool, R, orig["norm"])
    measure("U2", u2["model"], nm, {"L1": u2["final_hash"] == u2b["final_hash"],
                                     "L2": A.verify_input_set(store, pol_wo, u2, subj)["result"]})

    final_state = torch.load(os.path.join(args.runs, "orig_late", f"ckpt_r{R}.pt"))
    u4 = U.gradient_ascent(data, store, pol_wo, X_by_k, final_state, 1, device, max_pool, R, orig["norm"])
    u4b = U.gradient_ascent(data, store, pol_wo, X_by_k, final_state, 1, device, max_pool, R, orig["norm"])
    l2 = A.verify_input_set(store, pol_wo, u4, subj)
    measure("U4_provisional", u4["model"], nm, {"L1": u4["final_hash"] == u4b["final_hash"], "L2": l2["result"],
            "forget_set_matches_X": all(fc["root_X"] == EV.SortedMerkle(subj[fc["k"]]).root for fc in u4["forget_commits"])})

    u0_off = [res["methods"][f"U0_s{s}"]["lira_offline"]["auc"] for s in range(1, 6)]
    res["U0_offline_range"] = [min(u0_off), max(u0_off)]
    with open(os.path.join(args.out, "late_methods.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("orig_late rerun identical:", res["orig_late_rerun_identical"], "| U1 reuse valid:", res["U1_reuse_valid"])
    print("U0 offline range:", np.round(res["U0_offline_range"], 4).tolist())


if __name__ == "__main__":
    main()
