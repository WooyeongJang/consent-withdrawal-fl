"""A2 분석: 참조 U0 5개의 기록별 점수로 (1) 저장값 재현, (2) 대응 기록 부트스트랩, (3) 그림자 모델 민감도를 계산한다.

입력: ~/data/runs/validation/A2/<시나리오>.npz, reeval_summary.json, ~/data/runs/final_e2/<시나리오>/shadows.npz, results/final/e2/*.json
출력: validation/A_l3_power/results/a2_summary.json
AUC와 점수 계산은 여기서 새로 구현한다(순위 기반 AUC, offline 점수 = (φ − μ_out)/σ_out, σ는 기록 공통).
"""
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from a1_common import DELTA, ROOT, Z90, Z975, load_runs

A2 = os.path.expanduser("~/data/runs/validation/A2")
RUNS = os.path.expanduser("~/data/runs/final_e2")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 20261109
B = 2000


def auc_xz(xs, zs):
    """순위 기반 AUROC(동점 평균 순위)."""
    s = np.r_[xs, zs]
    r = stats.rankdata(s)
    n1, n0 = len(xs), len(zs)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_many(S, nX):
    """S: (m, P) 점수. 각 행의 AUC."""
    r = stats.rankdata(S, axis=1)
    n1, n0 = nX, S.shape[1] - nX
    return (r[:, :nX].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def offline(phi_t, phis, inmask):
    """(φ − μ_out)/σ_out. μ_out은 기록별(그 기록이 OUT인 그림자들의 평균), σ_out은 기록 공통."""
    out = ~inmask
    cnt = out.sum(0)
    g_out = phis[out].mean()
    mu = np.where(cnt > 0, (phis * out).sum(0) / np.maximum(cnt, 1), g_out)
    sd = np.std((phis - mu[None, :])[out])
    return (phi_t - mu) / sd


def offline_runtime(phi_t, phis, inmask):
    """실행 당시(tag final-rerun-v1) 계산: NaN 방지가 없어, 모든 그림자에서 IN인 기록은 μ_out = NaN → 점수 NaN."""
    out = ~inmask
    cnt = out.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (phis * out).sum(0) / cnt
    sd = np.std((phis - mu[None, :])[out])
    return (phi_t - mu) / sd


def auc_runtime(score, nX):
    """fl/train.py auc의 정의(안정 정렬 순위, 동점 평균). NaN은 정렬에서 맨 뒤 = 가장 높은 순위가 된다."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / cnt)[inv]
    n1, n0 = nX, len(score) - nX
    return float((ranks[:nX].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    meta = json.load(open(os.path.join(A2, "reeval_summary.json")))
    runs = load_runs()
    stored = {}
    for r in runs:
        stored.setdefault(r["scenario"], {"refs": r["refs"], "ci": {}})
        u0 = r["methods"]["U0"]
        if u0["auc"] in r["refs"]:
            stored[r["scenario"]]["ci"][r["refs"].index(u0["auc"])] = u0["ci"]
    rng = np.random.default_rng(SEED)
    res = {"meta": {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "B": B, "reeval": {k: meta[k] for k in ("started", "finished", "torch", "image_digest", "script_sha256")}},
           "scenarios": {}, "reproduction": {}}
    max_auc_diff = max_ci_diff = max_own_diff = max_rt_diff = 0.0
    n_auc = n_ci = n_exact_current = 0
    nan_records = {}
    for sc in sorted(meta["scenarios"]):
        z = np.load(os.path.join(A2, f"{sc}.npz"))
        phi, off_c, nX = z["phi"], z["offline"], int(z["nX"])
        sh = np.load(os.path.join(RUNS, sc, "shadows.npz"))
        phis, inmask = sh["phis"], sh["inmask"]
        P = phi.shape[1]
        # (1) 재현: 컨테이너에서 원래 함수로 계산한 AUC·CI와 저장값, 그리고 여기서 새로 구현한 점수·AUC
        ev = meta["scenarios"][sc]["eval"]
        for j in range(5):
            d = abs(ev[j]["auc"] - stored[sc]["refs"][j]); max_auc_diff = max(max_auc_diff, d); n_auc += 1
            if j in stored[sc]["ci"]:
                dc = max(abs(a - b) for a, b in zip(ev[j]["ci"], stored[sc]["ci"][j])); max_ci_diff = max(max_ci_diff, dc); n_ci += 1
        off = np.array([offline(phi[j], phis, inmask) for j in range(5)])
        own = auc_many(off, nX)
        n_exact_current += int(sum(abs(ev[j]["auc"] - stored[sc]["refs"][j]) <= 1e-9 for j in range(5)))
        rt = [auc_runtime(offline_runtime(phi[j], phis, inmask), nX) for j in range(5)]
        max_rt_diff = max(max_rt_diff, max(abs(a - b) for a, b in zip(rt, stored[sc]["refs"])))
        all_in, all_out = np.where(inmask.all(0))[0], np.where((~inmask).all(0))[0]
        if len(all_in) or len(all_out):
            nan_records[sc] = {"all_IN_pool_index": all_in.tolist(), "all_OUT_pool_index": all_out.tolist(), "in_X": [bool(i < nX) for i in all_in],
                               "auc_runtime_code": rt, "auc_current_code": [e["auc"] for e in ev], "stored": stored[sc]["refs"]}
        max_own_diff = max(max_own_diff, float(np.abs(own - np.array(stored[sc]["refs"])).max()))
        # (2) 대응 기록 부트스트랩: X·Z를 층화해 같은 재표본에서 5개 모델의 AUC를 함께 계산
        aucs = np.empty((B, 5))
        for b in range(B):
            ix = np.r_[rng.integers(0, nX, nX), nX + rng.integers(0, P - nX, P - nX)]
            aucs[b] = auc_many(off[:, ix], nX)
        sig_rec = aucs.std(0, ddof=1)
        d_i = np.array([aucs[:, i] - np.delete(aucs, i, axis=1).mean(1) for i in range(5)]).T      # (B, 5)
        sig_rec_d = d_i.std(0, ddof=1)
        corr = np.corrcoef(aucs.T)
        iu = np.triu_indices(5, 1)
        # 규칙이 쓰는 se: 대상의 se_boot(95% 백분위 CI 폭/3.92)와 참조 4개 평균의 SE 합성
        se_boot_new = (np.percentile(aucs, 97.5, axis=0) - np.percentile(aucs, 2.5, axis=0)) / (2 * Z975)
        ref_auc = np.array(stored[sc]["refs"])
        se_rule = np.array([math.sqrt(se_boot_new[i] ** 2 + np.delete(ref_auc, i).var(ddof=1) / 4) for i in range(5)])
        # (3) 그림자 모델 민감도
        jk = np.array([auc_many(np.array([offline(phi[j], np.delete(phis, i, 0), np.delete(inmask, i, 0)) for j in range(5)]), nX)
                       for i in range(16)])                                            # (16, 5)
        jk_se = np.sqrt(15 / 16 * ((jk - jk.mean(0)) ** 2).sum(0))                      # 잭나이프 SE(그림자 표집)
        jk_d = np.array([jk[:, i] - np.delete(jk, i, axis=1).mean(1) for i in range(5)]).T
        jk_se_d = np.sqrt(15 / 16 * ((jk_d - jk_d.mean(0)) ** 2).sum(0))
        sub = []
        for _ in range(200):
            keep = np.sort(rng.choice(16, 8, replace=False))
            ok = (~inmask[keep]).any(0).all()
            sub.append(auc_many(np.array([offline(phi[j], phis[keep], inmask[keep]) for j in range(5)]), nX))
        sub = np.array(sub)
        sub_d = np.array([sub[:, i] - np.delete(sub, i, axis=1).mean(1) for i in range(5)]).T
        s_ref5 = float(ref_auc.std(ddof=1))
        res["scenarios"][sc] = {
            "nX": nX, "nZ": P - nX, "auc": own.tolist(),
            "sigma_rec": sig_rec.tolist(), "sigma_rec_mean": float(sig_rec.mean()), "se_boot_recomputed": se_boot_new.tolist(),
            "sigma_rec_d": sig_rec_d.tolist(), "sigma_rec_d_mean": float(sig_rec_d.mean()),
            "corr_between_models_mean": float(corr[iu].mean()), "corr_between_models_min": float(corr[iu].min()),
            "se_rule_mean": float(se_rule.mean()),
            "s_ref_5": s_ref5,
            # 실제 변동(참조 5개 중 하나를 나머지 4개와 비교할 때): 시드 변동 sqrt(σ²(1+1/4)) + 대응 기록 표집
            "sd_actual_seed_only": float(s_ref5 * math.sqrt(1.25)),
            "sd_actual_seed_plus_records": float(math.sqrt(1.25 * s_ref5 ** 2 + sig_rec_d.mean() ** 2)),
            "shadow_jackknife_se_auc_mean": float(jk_se.mean()), "shadow_jackknife_se_d_mean": float(jk_se_d.mean()),
            "shadow_half8_sd_auc_mean": float(sub.std(0, ddof=1).mean()), "shadow_half8_sd_d_mean": float(sub_d.std(0, ddof=1).mean()),
            "shadow_half8_mean_shift_auc": float((sub.mean(0) - own).mean())}
        print(f"[{sc}] σ_rec {sig_rec.mean():.4f} σ_rec,d {sig_rec_d.mean():.4f} corr {corr[iu].mean():.3f} se_rule {se_rule.mean():.4f} "
              f"s_ref5 {s_ref5:.4f} jkSE(auc) {jk_se.mean():.4f} jkSE(d) {jk_se_d.mean():.4f}", flush=True)
    res["reproduction"] = {"n_auc": n_auc, "max_abs_diff_auc_container_vs_stored": max_auc_diff, "n_ci": n_ci,
                           "max_abs_diff_ci_container_vs_stored": max_ci_diff, "max_abs_diff_auc_reimplemented_vs_stored": max_own_diff,
                           "criterion": 1e-9, "n_auc_within_criterion_current_code": n_exact_current,
                           "pass_current_code": bool(max_auc_diff <= 1e-9 and max_ci_diff <= 1e-9),
                           "max_abs_diff_auc_runtime_code_vs_stored": max_rt_diff, "pass_runtime_code": bool(max_rt_diff <= 1e-9),
                           "records_IN_or_OUT_in_all_shadows": nan_records}
    json.dump(res, open(os.path.join(OUT, "a2_summary.json"), "w"), indent=1)
    print(json.dumps(res["reproduction"], indent=1))


if __name__ == "__main__":
    print(f"# a2_analysis start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} script_sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
    main()
