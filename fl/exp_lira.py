"""강한 멤버십 추론(LiRA, Carlini et al., 2022)과 철회 시점 조건 (3장 v1.6 5.5·8.1·8.4 E2).

후보 집합 = 철회 집단 X(838건) ∪ 비구성원 Z(fold 9, 838건). 그림자 모델 N개는 각각 후보의 무작위 절반을 구성원으로
넣고, 대상과 같은 과정(구성원은 철회 라운드 w 전까지만 사용)으로 연합학습한다. 기록별 통계량
φ = 클래스 평균 logit(정답 확신도)를 IN/OUT 모델에서 모아 정규분포로 근사하고, 대상 모델의 φ에 대해
online 점수 log N(φ; μ_in, σ_in) − log N(φ; μ_out, σ_out), offline 점수 (φ − μ_out)/σ_out를 계산한다.
σ는 표본이 적으므로 Carlini et al.의 권고대로 기록 공통(global) 분산을 쓴다.
대상: 중기 철회(w=25) P1, 후기 철회(w=45) P1, U0(시드 1–5, X를 전혀 쓰지 않음)
지표: X vs Z AUC(부트스트랩 95% CI), 낮은 위양성률에서의 참양성률(TPR@FPR 1%, 5%)

  python -m fl.exp_lira --out /out --runs /runs --l3runs /l3runs --l3json results/l3/l3_demo.json
"""

import argparse
import json
import os

import numpy as np
import torch

from fl import train as T
from fl.run import INIT_SEED, INSTS, Policy, Store, run_fl


def stat(model, data, idx, device, norm):
    P = np.clip(T.predict(model, data, idx, device, norm=norm), 1e-6, 1 - 1e-6)
    Y = data.Y[idx]
    pt = np.where(Y == 1, P, 1 - P)
    return np.log(pt / (1 - pt)).mean(1)


def shadow_fl(data, base_by_k, members_by_k, w, seed, device, max_pool, rounds, norm):
    g = T.new_model(max_pool, device, T.derive_seed(seed, "init", 0))
    weights = None
    for r in range(1, rounds + 1):
        states, ws = [], []
        for k in INSTS:
            S = base_by_k[k] if r >= w else np.sort(np.concatenate([base_by_k[k], members_by_k[k]]))
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(g.state_dict())
            m, _ = T.local_train(m, data, k, r, seed, device, rounds, idx=S, norm=norm)
            states.append({n: v.detach().clone() for n, v in m.state_dict().items()}); ws.append(len(S))
        g.load_state_dict(T.fedavg(states, ws))
    return g


def scores(phi_t, phis, inmask):
    """phis: (N, P) 그림자 통계, inmask: (N, P) bool."""
    # 2026-11-01(코드 검토 C4): 한 레코드가 모든 그림자에서 IN(또는 OUT)이면 평균이 NaN이 되어 AUC로 전파된다.
    # 그 경우 전체 평균으로 대체한다(최종 재실행에서는 발생하지 않았다).
    g_in, g_out = phis[inmask].mean(), phis[~inmask].mean()
    mu_in = np.array([phis[inmask[:, j], j].mean() if inmask[:, j].any() else g_in for j in range(phis.shape[1])])
    mu_out = np.array([phis[~inmask[:, j], j].mean() if (~inmask[:, j]).any() else g_out for j in range(phis.shape[1])])
    sd_in = np.std(phis[inmask] - mu_in[np.where(inmask)[1]])
    sd_out = np.std(phis[~inmask] - mu_out[np.where(~inmask)[1]])
    online = -0.5 * ((phi_t - mu_in) / sd_in) ** 2 + 0.5 * ((phi_t - mu_out) / sd_out) ** 2 - np.log(sd_in / sd_out)
    offline = (phi_t - mu_out) / sd_out
    return online, offline


def evaluate(score, nX, seed=0, n_boot=1000):
    y = np.r_[np.ones(nX), np.zeros(len(score) - nX)]
    a = T.auc(y, score)
    rng = np.random.default_rng(seed)
    xs, zs = score[:nX], score[nX:]
    boots = [T.auc(y, np.r_[xs[rng.integers(0, nX, nX)], zs[rng.integers(0, len(zs), len(zs))]]) for _ in range(n_boot)]
    out = {"auc": a, "ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
    for fpr in (0.01, 0.05):
        thr = np.quantile(zs, 1 - fpr)
        out[f"tpr@fpr{int(fpr * 100)}%"] = float((xs > thr).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--l3runs", required=True)
    ap.add_argument("--l3json", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--shadows", type=int, default=16)
    ap.add_argument("--late", type=int, default=45)
    ap.add_argument("--mid", type=int, default=25)
    args = ap.parse_args()

    device, max_pool = T.configure("gpu_det")
    data, R = T.Data(args.cache), args.rounds
    store = Store(data)
    l3 = json.load(open(args.l3json))
    # exp_l3와 같은 철회 집단·비구성원
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    cohort = sorted(int(p) for p in np.random.default_rng(20261025).choice(pids, int(len(pids) * l3["cohort"]["frac"]), replace=False))
    cs = set(cohort)
    X = np.sort(np.array([i for k in INSTS for i in data.train_idx[k] if int(data.pid[i]) in cs]))
    Z = np.sort(np.random.default_rng(1).choice(data.val_idx, min(len(X), len(data.val_idx)), replace=False))
    assert len(X) == l3["cohort"]["records"]
    Zk = np.array([z for z in Z if data.inst[z] in INSTS])          # 기관이 없는 비구성원은 그림자 학습에서 제외
    pool = np.r_[X, Zk]
    nX = len(X)
    base_by_k = {k: np.array([i for i in data.train_idx[k] if int(data.pid[i]) not in cs], dtype=np.int64) for k in INSTS}
    norm_line = next(json.loads(l) for l in open(os.path.join(args.l3runs, "orig", "events.jsonl")) if "NormStatsCommitted" in l)
    norm = (norm_line["mean"], norm_line["std"])
    res = {"pool": {"X": int(nX), "Z": int(len(Zk))}, "shadows": args.shadows, "rounds": R}

    # 후기 철회 대상 모델(P1_late)
    late = run_fl(data, store, Policy(withdraw={p: args.late for p in cohort}), base_seed=1, device=device,
                  max_pool=max_pool, cond="gpu_det", rounds=R, run_dir=os.path.join(args.runs, "orig_late"),
                  label="orig_late", log_every=0)
    targets = {"P1_mid": (os.path.join(args.l3runs, "orig", f"ckpt_r{R}.pt"), norm),
               "P1_late": (os.path.join(args.runs, "orig_late", f"ckpt_r{R}.pt"), (late["norm"]["mean"], late["norm"]["std"]))}
    for s in range(1, 6):
        ev = next(json.loads(l) for l in open(os.path.join(args.l3runs, f"U0_s{s}", "events.jsonl")) if "NormStatsCommitted" in l)
        targets[f"U0_s{s}"] = (os.path.join(args.l3runs, f"U0_s{s}", f"ckpt_r{R}.pt"), (ev["mean"], ev["std"]))
    phi_t = {}
    for name, (path, nm) in targets.items():
        m = T.new_model(max_pool, device, INIT_SEED); m.load_state_dict(torch.load(path))
        phi_t[name] = stat(m, data, pool, device, nm)
        if name == "P1_late":
            res["P1_late_test_macro_auc"] = T.macro_auc(data.Y[data.test_idx], T.predict(m, data, data.test_idx, device, norm=nm))[0]

    # 그림자 모델: 조건(w)마다 N개
    res["conditions"] = {}
    for cond_name, w, tgt in (("mid", args.mid, "P1_mid"), ("late", args.late, "P1_late")):
        f = os.path.join(args.runs, f"shadows_{cond_name}.npz")
        if os.path.exists(f):
            z = np.load(f); phis, inmask = z["phis"], z["inmask"]
        else:
            phis, inmask = np.zeros((args.shadows, len(pool))), np.zeros((args.shadows, len(pool)), bool)
            for i in range(args.shadows):
                mask = np.zeros(len(pool), bool)
                mask[np.random.default_rng(1000 + i).choice(len(pool), len(pool) // 2, replace=False)] = True
                mem = pool[mask]
                members_by_k = {k: np.array([j for j in mem if data.inst[j] == k], dtype=np.int64) for k in INSTS}
                g = shadow_fl(data, base_by_k, members_by_k, w, 1000 + i, device, max_pool, R, norm)
                phis[i], inmask[i] = stat(g, data, pool, device, norm), mask
                print(f"[{cond_name}] shadow {i + 1}/{args.shadows}", flush=True)
            np.savez(f, phis=phis, inmask=inmask)
        out = {}
        for name in [tgt] + [f"U0_s{s}" for s in range(1, 6)]:
            on, off = scores(phi_t[name], phis, inmask)
            out[name] = {"online": evaluate(on, nX), "offline": evaluate(off, nX)}
            print(f"[{cond_name}] {name}: online AUC {out[name]['online']['auc']:.4f} {np.round(out[name]['online']['ci'], 4).tolist()} "
                  f"TPR@1% {out[name]['online']['tpr@fpr1%']:.3f} | offline {out[name]['offline']['auc']:.4f}", flush=True)
        # 그림자 모델 자체에 대한 공격 성능(검정력 확인): 각 그림자를 대상으로, 나머지로 점수
        sanity = []
        for i in range(args.shadows):
            keep = np.arange(args.shadows) != i
            on, _ = scores(phis[i], phis[keep], inmask[keep])
            sanity.append(T.auc(inmask[i].astype(float), on))
        out["shadow_leave_one_out_auc"] = {"mean": float(np.mean(sanity)), "min": float(np.min(sanity)), "max": float(np.max(sanity))}
        print(f"[{cond_name}] shadow LOO AUC mean {np.mean(sanity):.4f}", flush=True)
        res["conditions"][cond_name] = {"withdraw_round": w, **out}
    with open(os.path.join(args.out, "lira.json"), "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
