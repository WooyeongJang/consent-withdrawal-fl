"""후기 철회(w=45) 조건에서 U4 FedOSD의 L1·L2·L3(LiRA)와 성능 (임시 경사 상승 구현을 대체).

  python -m fl.exp_fedosd --out /out --late /late --lirarun /lirarun --l3json results/l3/l3_demo.json --latejson results/late/late_methods.json
"""

import argparse
import json
import os

import numpy as np
import torch

from fl import audit as A
from fl import evidence as EV
from fl import fedosd as F
from fl import train as T
from fl.exp_lira import evaluate, scores, stat
from fl.run import INSTS, Policy, Store


def main():
    ap = argparse.ArgumentParser()
    for a in ("--out", "--late", "--lirarun", "--l3json", "--latejson"):
        ap.add_argument(a, required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--w", type=int, default=45)
    ap.add_argument("--settings", default="5:10,20:10", help="T_u:post 목록")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--u0run", default=None)
    ap.add_argument("--tag", default="")
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
    pol_wo = Policy(withdraw={p: args.w for p in cohort}).without(cohort)
    subj = {k: sorted(store.rids(np.array(X_by_k[k], dtype=np.int64))) for k in INSTS}
    zf = np.load(os.path.join(args.lirarun, "shadows_late.npz"))
    phis, inmask = zf["phis"], zf["inmask"]
    ev = next(json.loads(l) for l in open(os.path.join(args.late, "orig_late", "events.jsonl")) if "NormStatsCommitted" in l)
    norm = {"mean": ev["mean"], "std": ev["std"]}
    nm = (norm["mean"], norm["std"])
    start = torch.load(os.path.join(args.late, "orig_late", f"ckpt_r{R}.pt"))
    ref = json.load(open(args.latejson))
    res = {"reference": {k: {"test": v["test_macro_auc"], "lira_offline": v["lira_offline"]["auc"]}
                         for k, v in ref["methods"].items()}, "U0_offline_range": ref["U0_offline_range"],
           "lr": args.lr, "runs": {}}

    for st in args.settings.split(","):
        tu, post = (int(x) for x in st.split(":"))
        out = F.fedosd(data, store, pol_wo, X, start, 1, device, max_pool, R, norm, tu, post, args.lr)
        again = F.fedosd(data, store, pol_wo, X, start, 1, device, max_pool, R, norm, tu, post, args.lr)
        m = out["model"]
        test = T.macro_auc(data.Y[data.test_idx], T.predict(m, data, data.test_idx, device, norm=nm))[0]
        # 망각 집합에 대한 성능(참고): X의 macro AUC
        x_auc = T.macro_auc(data.Y[X], T.predict(m, data, X, device, norm=nm))[0]
        on, off = scores(stat(m, data, pool, device, nm), phis, inmask)
        l2 = A.verify_input_set(store, pol_wo, out, subj)
        name = f"FedOSD_Tu{tu}_post{post}"
        res["runs"][name] = {"test_macro_auc": test, "X_macro_auc": x_auc, "lira_offline": evaluate(off, len(X)),
                             "lira_online": evaluate(on, len(X)), "L1": out["final_hash"] == again["final_hash"],
                             "L2": l2["result"], "forget_set_matches_X": out["forget_commit"]["root_X"] == EV.SortedMerkle(store.rids(X)).root,
                             "unlearn_conflicts": [r["conflicts"] for r in out["log"] if r["stage"] == "unlearn"],
                             "dist_to_w0": [round(r["dist_to_w0"], 4) for r in out["log"]], "final_hash": out["final_hash"]}
        r = res["runs"][name]
        print(f"{name}: test {test:.4f}  X-AUC {x_auc:.4f}  LiRA offline {r['lira_offline']['auc']:.4f} "
              f"{np.round(r['lira_offline']['ci'], 4).tolist()}  L1 {r['L1']}  L2 {r['L2']}", flush=True)
    # 참고: P1과 U0(X를 한 번도 보지 않은 모델)의 X macro AUC — "충분히 잊었는가"의 기준선
    p1 = T.new_model(max_pool, device, 20261025); p1.load_state_dict(start)
    res["P1_X_macro_auc"] = T.macro_auc(data.Y[X], T.predict(p1, data, X, device, norm=nm))[0]
    if args.u0run:
        e0 = next(json.loads(l) for l in open(os.path.join(args.u0run, "events.jsonl")) if "NormStatsCommitted" in l)
        u0 = T.new_model(max_pool, device, 20261025); u0.load_state_dict(torch.load(os.path.join(args.u0run, f"ckpt_r{R}.pt")))
        res["U0_X_macro_auc"] = T.macro_auc(data.Y[X], T.predict(u0, data, X, device, norm=(e0["mean"], e0["std"])))[0]
    print("P1 X-AUC", round(res["P1_X_macro_auc"], 4), "U0 X-AUC", round(res.get("U0_X_macro_auc", float("nan")), 4))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"fedosd{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
