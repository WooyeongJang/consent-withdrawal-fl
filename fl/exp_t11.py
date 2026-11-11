"""T11 오프라인 예비 실험(원장 미사용). E3-B 시드 1의 캐시 체크포인트(원 모델·U0)를 쓴다.

  eps   : 비결정 GPU 모드로 U0를 두 번 학습해 허용 오차 ε(상대 거리)를 보정한다.
  pilot : T11a·T11b 사례를 만들어 L3(TOST)·결정적 U0와의 상대 거리·평가 AUC를 잰다.

  python -m fl.exp_t11 --mode eps   --out /out --runs /runs --e3b /e3b
  python -m fl.exp_t11 --mode pilot --out /out --runs /runs --e3b /e3b --lirarun /lirarun --latejson results/late/late_methods.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from fl import t11 as F
from fl import train as T
from fl.exp_lira import evaluate, scores, stat
from fl.l3 import tost
from fl.run import INSTS, Policy, Store, run_fl


def norm_of(data, store, policy):
    scope = np.concatenate([store.input_set(policy, k, 1) for k in INSTS])
    m, s = T.norm_stats(data, scope)
    return {"mean": m, "std": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eps", "eps_step", "pilot", "pilot_a2", "pilot_bk"], required=True)
    for a in ("--out", "--runs", "--e3b"):
        ap.add_argument(a, required=True)
    ap.add_argument("--lirarun")
    ap.add_argument("--latejson")
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--w", type=int, default=45)
    ap.add_argument("--frac", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.R
    data = T.Data(args.cache)
    store = Store(data)
    pids, cohort, X, pool = F.late_setup(data, store, args.frac)
    policy = Policy(withdraw={p: args.w for p in cohort})
    pol_wo = policy.without(cohort)
    u0_det = torch.load(os.path.join(args.e3b, "B_s1", "U0", f"ckpt_r{R}.pt"))

    if args.mode == "eps_step":
        # 단계(라운드) 단위 재현 검증의 허용 오차: 같은 ckpt_r{R-1}에서 마지막 라운드 하나를 비결정 모드로 두 번
        device, max_pool = T.configure("gpu_nondet")
        norm = norm_of(data, store, pol_wo)
        start = torch.load(os.path.join(args.e3b, "B_s1", "U0", f"ckpt_r{R - 1}.pt"))
        finals = []
        for i in range(3):
            o = run_fl(data, store, pol_wo, base_seed=1, device=device, max_pool=max_pool, cond="gpu_nondet", rounds=R,
                       start_state=start, start_round=R, norm=norm, run_dir=os.path.join(args.runs, f"step_nd{i}"),
                       label=f"snd{i}", log_every=0)
            finals.append({k: v.detach().cpu() for k, v in o["model"].state_dict().items()})
        pairs = [F.rel_dist(finals[a], finals[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
        out = {"rel_dist_nondet_step_pairs": pairs, "rel_dist_nondet_step_vs_det": [F.rel_dist(f, u0_det) for f in finals],
               "eps_step": max(pairs),
               "note": "ε_step = 같은 시작 상태(U0 ckpt_r49)에서 라운드 50 하나를 비결정 모드로 3회 실행한 결과 사이의 최대 상대 거리"}
        json.dump(out, open(os.path.join(args.out, "t11_eps_step.json"), "w"), indent=1)
        print(out)
        return

    if args.mode == "eps":
        device, max_pool = T.configure("gpu_nondet")
        norm = norm_of(data, store, pol_wo)
        finals = []
        for i in range(2):
            o = run_fl(data, store, pol_wo, base_seed=1, device=device, max_pool=max_pool, cond="gpu_nondet",
                       rounds=R, norm=norm, run_dir=os.path.join(args.runs, f"U0_nondet{i}"), label=f"nd{i}", log_every=0)
            finals.append({k: v.detach().cpu() for k, v in o["model"].state_dict().items()})
        out = {"rel_dist_nondet_pair": F.rel_dist(finals[0], finals[1]),
               "rel_dist_nondet_vs_det": [F.rel_dist(f, u0_det) for f in finals],
               "note": "ε = 정직한 비결정 재실행 두 번 사이의 상대 거리(같은 입력·시드·이미지, cudnn.benchmark, 결정적 알고리즘 미사용)"}
        out["eps"] = out["rel_dist_nondet_pair"]
        json.dump(out, open(os.path.join(args.out, "t11_eps.json"), "w"), indent=1)
        print(out)
        return

    device, max_pool = T.configure("gpu_det")
    late = json.load(open(args.latejson))
    zf = np.load(os.path.join(args.lirarun, "shadows_late.npz"))
    phis, inmask = zf["phis"], zf["inmask"]
    u0_aucs = [late["methods"][f"U0_s{s}"]["lira_offline"]["auc"] for s in range(1, 6)]
    norm_orig, norm_u0 = norm_of(data, store, policy), norm_of(data, store, pol_wo)
    nm_orig = (norm_orig["mean"], norm_orig["std"])

    def l3_check(model, nm=nm_orig):
        _, off = scores(stat(model, data, pool, device, nm), phis, inmask)
        e = evaluate(off, len(X))
        return {**tost(e["auc"], e["ci"], u0_aucs), "auc": e["auc"]}

    g = T.new_model(max_pool, device, 0)
    g.load_state_dict(u0_det)
    base = {"U0_test_auc": F.test_auc(g, data, device, (norm_u0["mean"], norm_u0["std"])),
            "U0_L3": l3_check(g, (norm_u0["mean"], norm_u0["std"]))}
    orig = torch.load(os.path.join(args.e3b, "B_s1", "orig", f"ckpt_r{R}.pt"))
    g.load_state_dict(orig)
    base["P1_rel_dist_to_U0"] = F.rel_dist(orig, u0_det)
    base["P1_L3"] = l3_check(g)
    print(base, flush=True)
    rng = random.Random(11)
    rows = []
    if args.mode == "pilot_bk":
        # T11b 주입 규모별 단계 편차: 마지막 라운드에 철회자 n명만 다시 넣는다
        nm_u0 = (norm_u0["mean"], norm_u0["std"])
        for nk in (1, 1, 5, 25, 100, len(cohort)):
            inject = rng.sample(cohort, nk)
            o = F.forge_inject(data, store, cohort, inject, 1, os.path.join(args.e3b, "B_s1", "U0"), norm_u0, R,
                               device, max_pool, os.path.join(args.runs, "T11b"))
            st = {k: v.detach().cpu() for k, v in o["model"].state_dict().items()}
            l3 = l3_check(o["model"], nm_u0)
            rows.append({"variant": "b", "n_inject": nk, "rel_dist_to_U0": F.rel_dist(st, u0_det), "L3": l3["result"],
                         "auc": l3["auc"], "bit_identical_to_U0": all(torch.equal(st[k], u0_det[k]) for k in st)})
            print(rows[-1], flush=True)
        json.dump({"rows": rows}, open(os.path.join(args.out, "t11_pilot_bk.json"), "w"), indent=1, default=float)
        return
    if args.mode == "pilot_a2":
        retain = np.setdiff1d(np.concatenate([data.train_idx[k] for k in INSTS]), X)
        for i in range(args.n):
            m, steps, l3 = F.forge_l3match(data, X, pool, retain, orig, nm_orig, device, max_pool, phis, inmask, l3_check, seed=200 + i)
            st = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            rows.append({"variant": "a2", "steps": steps, "L3": l3["result"], "auc": l3["auc"], "diff": l3["diff"],
                         "ci90": l3["ci90"], "rel_dist_to_U0": F.rel_dist(st, u0_det),
                         "rel_dist_to_P1": F.rel_dist(st, orig), "test_auc": F.test_auc(m, data, device, nm_orig)})
            print(rows[-1], flush=True)
        json.dump({"base": base, "rows": rows}, open(os.path.join(args.out, "t11_pilot_a2.json"), "w"), indent=1, default=float)
        return
    for i in range(args.n):
        o, m, l3 = F.forge_finetune(data, store, pol_wo, orig, norm_orig, R, device, max_pool,
                                    os.path.join(args.runs, "T11a"), l3_check, base_seed=100 + i)
        st = {k: v.detach().cpu() for k, v in o["model"].state_dict().items()}
        rows.append({"variant": "a", "extra_rounds": m, "L3": l3["result"], "auc": l3["auc"], "diff": l3["diff"],
                     "ci90": l3["ci90"], "rel_dist_to_U0": F.rel_dist(st, u0_det),
                     "test_auc": F.test_auc(o["model"], data, device, nm_orig)})
        print(rows[-1], flush=True)
    nm_u0 = (norm_u0["mean"], norm_u0["std"])
    for i in range(args.n):
        m_inj, f = [(1, 1.0), (1, 0.5), (2, 1.0), (1, 0.25)][i % 4]
        inject = rng.sample(cohort, int(len(cohort) * f))
        o = F.forge_inject(data, store, cohort, inject, m_inj, os.path.join(args.e3b, "B_s1", "U0"), norm_u0, R,
                           device, max_pool, os.path.join(args.runs, "T11b"))
        st = {k: v.detach().cpu() for k, v in o["model"].state_dict().items()}
        l3 = l3_check(o["model"], nm_u0)
        rows.append({"variant": "b", "m_inj": m_inj, "inject_frac": f, "L3": l3["result"], "auc": l3["auc"],
                     "diff": l3["diff"], "ci90": l3["ci90"], "rel_dist_to_U0": F.rel_dist(st, u0_det),
                     "test_auc": F.test_auc(o["model"], data, device, nm_u0)})
        print(rows[-1], flush=True)
    json.dump({"base": base, "rows": rows}, open(os.path.join(args.out, "t11_pilot.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
