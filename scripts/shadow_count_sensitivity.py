"""그림자 모델 수 민감도 (모의심사 M3 대응; 저장된 그림자 출력만 읽는다 — 학습·GPU·거래 없음).

무엇을 재는가
  E2의 각 시나리오에는 그림자 모델 16개의 기록별 통계(phis)와 구성원 표시(inmask)가 shadows.npz로 남아 있다.
  그림자 하나를 대상 모델로 두고(구성원 = 그 그림자의 IN 절반), 나머지 15개 중 n개만 써서 offline LiRA 점수를
  만든 뒤 AUC와 1% FPR에서의 TPR을 구한다. n = 4, 8, 15에서 값이 어떻게 달라지는지가 결과다.
  n = 15는 fl/e2.py가 이미 기록하는 shadow_loo_auc(offline)와 같은 계산이다(자체 점검에 쓴다).

무엇을 재지 않는가
  논문의 대상 모델(P1, U0–U4s)을 그림자 수를 줄여 다시 채점하지는 않는다. 그 모델들의 기록별 통계는 저장되지
  않았고 체크포인트는 실행 뒤 지워졌다(--keep 없이 실행). 16개보다 많은 그림자도 새 학습이 필요해 다루지 않는다.
  따라서 결과는 "공격의 판별력이 그림자 수 4–15 범위에서 얼마나 변하는가"이지 L3 판정의 재계산이 아니다.

실행 (저장소 루트에서)
  python3 scripts/shadow_count_sensitivity.py --runs <E2 실행 폴더(시나리오별 하위 폴더에 shadows.npz)> \
      --out results/final/shadow_count_sensitivity.json
  --selftest : 합성 자료로 계산식만 점검(자료 불필요)
"""
import argparse, glob, json, os, statistics as st
import numpy as np

SIZES = (4, 8, 15)
N_SUBSETS = 20          # n < 15일 때 대상마다 뽑는 부분집합 수
EARLY = {"E2-early", "E2-rate01-early", "E2-rate10-early", "E2-rate20-early"}
MID = {"E2-mid", "E2-rate01-mid", "E2-rate10-mid", "E2-rate20-mid"}
LATE = {"E2-base", "E2-rate01", "E2-rate10", "E2-rate20", "E2-diagMI", "E2-F65", "E2-inst2", "E2-seq3", "E2-exit2", "E2-latejoin"}


def auc(y, s):
    """fl/train.py auc와 같은 순위 기반 AUROC(동점은 평균 순위)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def offline_score(phi_t, phis, inmask):
    """fl/exp_lira.py scores()의 offline 점수와 같은 식: (phi_t - mu_out) / sd_out."""
    g_out = phis[~inmask].mean()
    mu_out = np.array([phis[~inmask[:, j], j].mean() if (~inmask[:, j]).any() else g_out for j in range(phis.shape[1])])
    sd_out = np.std(phis[~inmask] - mu_out[np.where(~inmask)[1]])
    return (phi_t - mu_out) / sd_out


def tpr_at_fpr(y, s, fpr=0.01):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def one_scenario(phis, inmask, seed=0):
    N = len(phis)
    rng = np.random.default_rng(seed)
    out = {n: {"auc": [], "tpr1": []} for n in SIZES if n <= N - 1}
    for i in range(N):
        others = np.array([j for j in range(N) if j != i])
        y = inmask[i].astype(int)
        for n in out:
            subsets = [others] if n == len(others) else [rng.choice(others, n, replace=False) for _ in range(N_SUBSETS)]
            a, t = [], []
            for sub in subsets:
                s = offline_score(phis[i], phis[sub], inmask[sub])
                a.append(auc(y, s)); t.append(tpr_at_fpr(y, s))
            out[n]["auc"].append(float(np.mean(a))); out[n]["tpr1"].append(float(np.mean(t)))
    return {str(n): {"auc_mean": float(np.mean(v["auc"])), "auc_sd_targets": float(np.std(v["auc"], ddof=1)),
                     "tpr1_mean": float(np.mean(v["tpr1"]))} for n, v in out.items()}


def selftest():
    rng = np.random.default_rng(1)
    N, P = 16, 1200
    inmask = np.zeros((N, P), bool)
    for i in range(N):
        inmask[i, rng.choice(P, P // 2, replace=False)] = True
    base = rng.normal(0, 1.0, P)                                   # 기록별 난이도
    phis = base + rng.normal(0, 0.5, (N, P)) + 0.35 * inmask       # 구성원이면 통계가 조금 높다
    r = one_scenario(phis, inmask)
    print("selftest:", json.dumps(r))
    assert 0.5 < r["4"]["auc_mean"] <= r["8"]["auc_mean"] + 0.01 <= r["15"]["auc_mean"] + 0.02, "AUC가 그림자 수에 따라 줄면 안 된다"
    null = one_scenario(base + rng.normal(0, 0.5, (N, P)), inmask)  # 구성원 효과가 없으면 0.5 근처
    assert abs(null["15"]["auc_mean"] - 0.5) < 0.02, null
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--logged", default="results/final/e2", help="shadow_loo_auc 대조용 실행 로그 폴더")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    files = sorted(glob.glob(os.path.join(a.runs, "*", "shadows.npz")))
    res, missing = {}, []
    for f in files:
        name = os.path.basename(os.path.dirname(f))
        z = np.load(f)
        res[name] = one_scenario(z["phis"], z["inmask"])
        res[name]["n_shadows"], res[name]["pool"] = int(z["phis"].shape[0]), int(z["phis"].shape[1])
        logs = glob.glob(os.path.join(a.logged, f"{name}_s*.json"))      # 자체 점검: n=15 값 = 기록된 shadow_loo_auc
        if logs:
            logged = json.load(open(logs[0]))["shadow_loo_auc"]["offline"]["mean"]
            res[name]["logged_loo_offline"] = logged
            res[name]["matches_log"] = bool(abs(logged - res[name]["15"]["auc_mean"]) < 1e-6) if "15" in res[name] else None
    main19 = EARLY | MID | LATE | {"E2-reconsent"}
    missing = sorted(main19 - set(res))
    groups = {}
    for gname, g in (("late", LATE), ("mid", MID), ("early", EARLY)):
        have = [s for s in g if s in res]
        groups[gname] = {"n_scenarios": len(have),
                         **{n: {"auc_mean": st.mean(res[s][n]["auc_mean"] for s in have),
                                "tpr1_mean": st.mean(res[s][n]["tpr1_mean"] for s in have)}
                            for n in ("4", "8", "15") if have and all(n in res[s] for s in have)}}
    out = {"sizes": SIZES, "subsets_per_target": N_SUBSETS, "scenarios": res, "groups": groups, "missing_main_scenarios": missing,
           "note": "shadow-as-target leave-one-out; offline LiRA; not a re-scoring of P1/U0-U4s (their per-record statistics were not stored)"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"scenarios found: {len(res)}; missing of the 19 main: {missing}")
    bad = [s for s, v in res.items() if v.get("matches_log") is False]
    print("n=15 equals logged shadow_loo_auc:", "all" if not bad else f"MISMATCH in {bad}")
    for gname, g in groups.items():
        print(gname, {n: (round(v["auc_mean"], 4), round(v["tpr1_mean"], 4)) for n, v in g.items() if n != "n_scenarios"})


if __name__ == "__main__":
    main()
