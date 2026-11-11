"""A1: 기존 실행 로그만으로 하는 L3 검정력·판정 신뢰도 분석(PREREG.md 실험 A, Q1–Q7).

관측(로그에서 다시 계산)과 모의(정규 모형)를 결과 파일에서 구분한다: results/a1_observed.json, results/a1_simulated.json.
부트스트랩 반복값은 독립 시드로 쓰지 않는다. se_boot는 규칙의 CI 폭 성분으로만 들어간다.

  .venv/bin/python validation/A_l3_power/code/a1_analysis.py
"""
import collections
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from a1_common import DELTA, METHODS, ROOT, Z90, Z975, classify, load_runs, refs_for, tost

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 20261109
REPS = 100_000
GRID = np.round(np.arange(-0.100, 0.1501, 0.001), 3)          # 계산 격자(0.001), 보고는 0.005 간격
NREFS = (2, 3, 4, 5, 10, 20)
MAIN4 = ("E2-rate01", "E2-base", "E2-rate10", "E2-rate20")


def se_from_ci(ci):
    return (ci[1] - ci[0]) / (2 * Z975)


# ---------------------------------------------------------------- 관측
def observed(runs):
    by = collections.defaultdict(list)
    for r in runs:
        by[r["scenario"]].append(r)
    out = {}
    pooled_std = []
    for sc, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["seed"])
        refs = rs[0]["refs"]
        assert all(r["refs"] == refs for r in rs), sc
        u0_all = list(refs) + [r["methods"]["U0"]["auc"] for r in rs if r["methods"]["U0"]["auc"] not in refs]
        assert len(u0_all) == 10 and len(set(u0_all)) == 10, (sc, len(u0_all))
        s_ref = statistics.stdev(u0_all)
        pooled_std += [(a - statistics.mean(u0_all)) / s_ref for a in u0_all]
        chi_lo, chi_hi = stats.chi2.ppf(0.95, 9), stats.chi2.ppf(0.05, 9)
        o = {"nX": rs[0]["nX"], "nZ": rs[0]["nZ"], "u0_aucs_10": u0_all, "u0_mean_10": statistics.mean(u0_all),
             "s_ref_10": s_ref, "s_ref_10_ci90": [s_ref * math.sqrt(9 / chi_lo), s_ref * math.sqrt(9 / chi_hi)],
             "s_ref_5": statistics.stdev(refs), "methods": {}}
        for m in METHODS:
            aucs = [r["methods"][m]["auc"] for r in rs]
            ses = [se_from_ci(r["methods"][m]["ci"]) for r in rs]
            ts = [tost(r["methods"][m]["auc"], r["methods"][m]["ci"], refs_for(r, m)) for r in rs]
            hw = [Z90 * t["se"] for t in ts]
            o["methods"][m] = {"auc_mean": statistics.mean(aucs), "s_seed": statistics.stdev(aucs), "se_boot_mean": statistics.mean(ses),
                               "se_boot_min": min(ses), "se_boot_max": max(ses),
                               "diff_mean": statistics.mean(t["d"] for t in ts), "diff_sd": statistics.stdev([t["d"] for t in ts]),
                               "class": dict(collections.Counter(t["class"] for t in ts)),
                               "halfwidth_min": min(hw), "halfwidth_max": max(hw),
                               "pass_arith_impossible": sum(h >= DELTA for h in hw),
                               # 참조 수가 무한이어도(참조 평균의 SE = 0) 반폭이 δ 이상인 실행 수
                               "pass_arith_impossible_inf_refs": sum(Z90 * s >= DELTA for s in ses)}
        out[sc] = o
    sw = stats.shapiro(pooled_std)
    return out, {"shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue), "n": len(pooled_std),
                 "skew": float(stats.skew(pooled_std)), "excess_kurtosis": float(stats.kurtosis(pooled_std))}


def subset_enumeration(runs, obs):
    """Q4(관측): U0 모델 10개에서 크기 n 부분집합을 참조로 써서 각 실행을 다시 분류한다(전수 열거)."""
    by = collections.defaultdict(list)
    for r in runs:
        by[r["scenario"]].append(r)
    res = {}
    agg = collections.defaultdict(lambda: collections.Counter())          # (method, n) -> counts
    for sc, rs in sorted(by.items()):
        pool = np.array(obs[sc]["u0_aucs_10"])
        res[sc] = {}
        for m in METHODS:
            res[sc][m] = {}
            for n in range(2, 10):
                same = tot = 0
                cls_cnt = collections.Counter()
                for r in rs:
                    x = r["methods"][m]
                    p1 = r["methods"]["P1"]
                    cand = [i for i in range(10) if not (m == "U0" and pool[i] == x["auc"])]
                    idx = np.array(list(itertools.combinations(cand, n)))
                    R = pool[idx]
                    mu, sd = R.mean(1), R.std(1, ddof=1)

                    def cls(auc, ci):
                        se = np.sqrt(se_from_ci(ci) ** 2 + sd ** 2 / n)
                        d = auc - mu
                        lo, hi = d - Z90 * se, d + Z90 * se
                        ok = (lo > -DELTA) & (hi < DELTA)
                        return np.where(ok, 0, np.where(lo > 0, 1, 2))     # 0 pass, 1 exceeds, 2 inconclusive
                    c = cls(x["auc"], x["ci"])
                    if x["kind"] == "approx":                               # 원고 규칙의 양성 대조(같은 부분집합으로)
                        c1 = cls(p1["auc"], p1["ci"])
                        c = np.where((c == 0) & (c1 != 1), 2, c)
                    # 보고된 분류(원고 규칙, 원래 참조)
                    t = tost(x["auc"], x["ci"], refs_for(r, m))["class"]
                    if x["kind"] == "approx" and t == "pass" and tost(p1["auc"], p1["ci"], refs_for(r, "P1"))["class"] != "exceeds":
                        t = "inconclusive"
                    code = {"pass": 0, "exceeds": 1, "inconclusive": 2}[t]
                    same += int((c == code).sum()); tot += len(c)
                    for k, v in zip(*np.unique(c, return_counts=True)):
                        cls_cnt[int(k)] += int(v)
                res[sc][m][n] = {"subsets_per_run": int(tot / len(rs)), "agree_with_reported": same / tot,
                                 "pass": cls_cnt[0] / tot, "exceeds": cls_cnt[1] / tot, "inconclusive": cls_cnt[2] / tot}
                agg[(m, n)]["same"] += same; agg[(m, n)]["tot"] += tot
                for k in (0, 1, 2):
                    agg[(m, n)][k] += cls_cnt[k]
    overall = {m: {n: {"agree_with_reported": agg[(m, n)]["same"] / agg[(m, n)]["tot"],
                       "pass": agg[(m, n)][0] / agg[(m, n)]["tot"], "exceeds": agg[(m, n)][1] / agg[(m, n)]["tot"],
                       "inconclusive": agg[(m, n)][2] / agg[(m, n)]["tot"]} for n in range(2, 10)} for m in METHODS}
    # 실행 단위 안정성: n=5 부분집합 중 보고된 분류가 유지되는 비율이 0.90 이상인 실행의 수
    return res, overall


def run_level_stability(runs, obs, n=5):
    by = collections.defaultdict(list)
    for r in runs:
        by[r["scenario"]].append(r)
    out = {m: {"stable_runs": 0, "runs": 0, "by_reported": collections.defaultdict(lambda: [0, 0])} for m in METHODS}
    for sc, rs in by.items():
        pool = np.array(obs[sc]["u0_aucs_10"])
        for r in rs:
            for m in METHODS:
                x, p1 = r["methods"][m], r["methods"]["P1"]
                cand = [i for i in range(10) if not (m == "U0" and pool[i] == x["auc"])]
                idx = np.array(list(itertools.combinations(cand, n)))
                R = pool[idx]; mu, sd = R.mean(1), R.std(1, ddof=1)

                def cls(auc, ci):
                    se = np.sqrt(se_from_ci(ci) ** 2 + sd ** 2 / n); d = auc - mu
                    lo, hi = d - Z90 * se, d + Z90 * se
                    return np.where((lo > -DELTA) & (hi < DELTA), 0, np.where(lo > 0, 1, 2))
                c = cls(x["auc"], x["ci"])
                if x["kind"] == "approx":
                    c = np.where((c == 0) & (cls(p1["auc"], p1["ci"]) != 1), 2, c)
                t = tost(x["auc"], x["ci"], refs_for(r, m))["class"]
                if x["kind"] == "approx" and t == "pass" and tost(p1["auc"], p1["ci"], refs_for(r, "P1"))["class"] != "exceeds":
                    t = "inconclusive"
                frac = float((c == {"pass": 0, "exceeds": 1, "inconclusive": 2}[t]).mean())
                out[m]["runs"] += 1; out[m]["stable_runs"] += frac >= 0.90
                out[m]["by_reported"][t][0] += frac >= 0.90; out[m]["by_reported"][t][1] += 1
    return {m: {"runs": v["runs"], "stable_runs": int(v["stable_runs"]),
                "by_reported": {k: {"stable": int(a), "n": b} for k, (a, b) in v["by_reported"].items()}} for m, v in out.items()}


# ---------------------------------------------------------------- 모의
def sim_curves(rng, sigma_ref, sigma_t, se_boot, n, reps=REPS, sigma_rec=0.0):
    """Δ 격자 전체의 P(pass), P(exceeds), P(inconclusive). 공통 난수를 쓴다."""
    R = rng.standard_normal((reps, n)) * sigma_ref
    rbar, s = R.mean(1), R.std(1, ddof=1)
    e = rng.standard_normal(reps) * sigma_t + (rng.standard_normal(reps) * sigma_rec if sigma_rec else 0.0)
    base = (e - rbar)[:, None]                       # d = Δ + base
    hw = (Z90 * np.sqrt(se_boot ** 2 + s ** 2 / n))[:, None]
    d = GRID[None, :] + base
    lo, hi = d - hw, d + hw
    p = ((lo > -DELTA) & (hi < DELTA))
    ex = (~p) & (lo > 0)
    return p.mean(0), ex.mean(0)


def first_cross(curve, level):
    """Δ ≥ 0 구간에서 curve가 level 이상이 되는 가장 작은 Δ(없으면 None)."""
    pos = GRID >= 0
    idx = np.where(pos & (curve >= level))[0]
    return float(GRID[idx[0]]) if len(idx) else None


def at(curve, x):
    return float(curve[int(np.argmin(np.abs(GRID - x)))])


def simulate(obs, sigma_rec_by_sc=None):
    rng = np.random.default_rng(SEED)
    sim = {}
    report_grid = [float(x) for x in np.round(np.arange(-0.10, 0.1501, 0.005), 3)]
    for sc, o in obs.items():
        s_ref, (s_lo, s_hi) = o["s_ref_10"], o["s_ref_10_ci90"]
        seb_u0, seb_p1 = o["methods"]["U0"]["se_boot_mean"], o["methods"]["P1"]["se_boot_mean"]
        sim[sc] = {"params": {"sigma_ref": s_ref, "sigma_ref_ci90": [s_lo, s_hi], "se_boot_U0": seb_u0, "se_boot_P1": seb_p1,
                              "sigma_T_P1": o["methods"]["P1"]["s_seed"]}, "models": {}}
        variants = {"M1": (s_ref, 0.0, 0.0), "M2": (s_ref, s_ref, 0.0), "M2_sT0.5": (s_ref, 0.5 * s_ref, 0.0),
                    "M2_sT1.5": (s_ref, 1.5 * s_ref, 0.0), "M2_sref_lo": (s_lo, s_lo, 0.0), "M2_sref_hi": (s_hi, s_hi, 0.0)}
        if sigma_rec_by_sc and sc in sigma_rec_by_sc:
            variants["M3"] = (s_ref, s_ref, sigma_rec_by_sc[sc])
        for name, (sr, st, srec) in variants.items():
            sim[sc]["models"][name] = {}
            for n in NREFS:
                p, ex = sim_curves(rng, sr, st, seb_u0, n, sigma_rec=srec)
                inc = 1 - p - ex
                agree = p ** 2 + ex ** 2 + inc ** 2
                sim[sc]["models"][name][n] = {
                    "power_at_0": at(p, 0.0), "power_at_0.025": at(p, 0.025), "power_at_-0.025": at(p, -0.025),
                    "false_equiv_at_0.05": at(p, 0.05), "false_equiv_at_-0.05": at(p, -0.05),
                    "false_equiv_at_0.06": at(p, 0.06), "false_equiv_at_0.075": at(p, 0.075),
                    "exceeds_at_0": at(ex, 0.0), "replicate_agreement_at_0": at(agree, 0.0),
                    "replicate_agreement_min": float(agree.min()), "replicate_agreement_min_at": float(GRID[int(agree.argmin())]),
                    "curve": {"delta": report_grid, "pass": [at(p, x) for x in report_grid], "exceeds": [at(ex, x) for x in report_grid]}}
        # Q5 양성 대조: P1형 대상
        sim[sc]["positive_control"] = {}
        for n in NREFS:
            p, ex = sim_curves(rng, s_ref, o["methods"]["P1"]["s_seed"], seb_p1, n)
            _, ex1 = sim_curves(rng, s_ref, 0.0, seb_p1, n)             # 대상 모델 고정(M1)
            sim[sc]["positive_control"][n] = {
                "MDR50": first_cross(ex, 0.5), "MDR80": first_cross(ex, 0.8), "MDR90": first_cross(ex, 0.9),
                "MDR80_M1": first_cross(ex1, 0.8),
                "detect_at_observed_P1_diff": at(ex, round(o["methods"]["P1"]["diff_mean"], 3)),
                "P1_pass_at_observed_P1_diff": at(p, round(o["methods"]["P1"]["diff_mean"], 3)),
                "curve": {"delta": report_grid, "exceeds": [at(ex, x) for x in report_grid], "pass": [at(p, x) for x in report_grid]}}
    return sim


def hierarchical(obs, G=20_000):
    """Q3·Q6: 참조 집합을 공유하는 시드 10개의 통과 수 분포. 실제 구조(시드 1–5는 참조 중 하나, n=4; 시드 6–10은 n=5)를 따른다."""
    rng = np.random.default_rng(SEED + 1)
    out = {}
    tot_plug = np.zeros(G); tot_post = np.zeros(G)
    for sc, o in obs.items():
        seb = o["methods"]["U0"]["se_boot_mean"]
        res = {}
        for label in ("plug_in", "posterior"):
            if label == "plug_in":
                sig = np.full(G, o["s_ref_10"])
            else:                                      # σ² | 자료 ~ 9 s² / χ²₉ (무정보 사전분포)
                sig = o["s_ref_10"] * np.sqrt(9 / rng.chisquare(9, G))
            U = rng.standard_normal((G, 10)) * sig[:, None]
            passes = np.zeros(G, dtype=int)
            for i in range(5):                          # 시드 1–5: 대상 = 참조 i, 참조 = 나머지 4개
                others = np.delete(U[:, :5], i, axis=1)
                d = U[:, i] - others.mean(1)
                hw = Z90 * np.sqrt(seb ** 2 + others.var(1, ddof=1) / 4)
                passes += ((d - hw > -DELTA) & (d + hw < DELTA))
            mu5, v5 = U[:, :5].mean(1), U[:, :5].var(1, ddof=1)
            hw5 = Z90 * np.sqrt(seb ** 2 + v5 / 5)
            for i in range(5, 10):                      # 시드 6–10: 참조 5개
                d = U[:, i] - mu5
                passes += ((d - hw5 > -DELTA) & (d + hw5 < DELTA))
            res[label] = {"mean": float(passes.mean()), "pi95": [int(np.percentile(passes, 2.5)), int(np.percentile(passes, 97.5))],
                          "dist": [float((passes == k).mean()) for k in range(11)]}
            if label == "plug_in":
                tot_plug += passes
            else:
                tot_post += passes
        res["observed_pass"] = o["methods"]["U0"]["class"].get("pass", 0)
        res["observed_in_pi95_plug_in"] = res["plug_in"]["pi95"][0] <= res["observed_pass"] <= res["plug_in"]["pi95"][1]
        res["observed_in_pi95_posterior"] = res["posterior"]["pi95"][0] <= res["observed_pass"] <= res["posterior"]["pi95"][1]
        out[sc] = res
    tot = {"plug_in": {"mean": float(tot_plug.mean()), "pi95": [float(np.percentile(tot_plug, 2.5)), float(np.percentile(tot_plug, 97.5))]},
           "posterior": {"mean": float(tot_post.mean()), "pi95": [float(np.percentile(tot_post, 2.5)), float(np.percentile(tot_post, 97.5))]},
           "observed_pass_total": sum(o["methods"]["U0"]["class"].get("pass", 0) for o in obs.values())}
    return out, tot


def shared_ref_icc(obs, G=20_000, n=5):
    """Q3(모의): 같은 참조 집합(n=5, 독립 참조)을 공유하는 독립 시드 10개의 통과 수 분포와 급내 상관(M2, Δ=0·0.025·0.05)."""
    rng = np.random.default_rng(SEED + 2)
    out = {}
    for sc, o in obs.items():
        s_ref, seb = o["s_ref_10"], o["methods"]["U0"]["se_boot_mean"]
        out[sc] = {}
        R = rng.standard_normal((G, n)) * s_ref
        rbar, hw = R.mean(1), Z90 * np.sqrt(seb ** 2 + R.var(1, ddof=1) / n)
        E = rng.standard_normal((G, 10)) * s_ref
        for dl in (0.0, 0.025, 0.05):
            d = dl + E - rbar[:, None]
            ok = (d - hw[:, None] > -DELTA) & (d + hw[:, None] < DELTA)
            k = ok.sum(1)
            p = ok.mean()
            # 급내 상관: Var(조건부 통과 확률) / (p(1−p)). 조건부 확률은 참조 집합마다의 통과 비율로 근사(10개 시드) → ANOVA 추정
            msb = 10 * np.var(ok.mean(1), ddof=1)
            msw = (ok.var(1, ddof=1)).mean()
            icc = float((msb - msw) / (msb + 9 * msw)) if (msb + 9 * msw) > 0 else None
            out[sc][str(dl)] = {"p_pass": float(p), "P_0_of_10": float((k == 0).mean()), "P_10_of_10": float((k == 10).mean()),
                                "sd_count": float(k.std()), "binomial_sd_count": float(math.sqrt(10 * p * (1 - p))), "icc": icc}
    return out


def main():
    t0 = time.time()
    runs = load_runs()
    obs, normality = observed(runs)
    enum, enum_overall = subset_enumeration(runs, obs)
    stab = run_level_stability(runs, obs)
    # Hanley–McNeil 근사와 관측 se_boot의 대조, 통과가 산술적으로 가능한 최소 |X|(외삽)
    hm = {}
    for sc, o in obs.items():
        n1, n0 = o["nX"], o["nZ"]
        hm[sc] = {"hanley_mcneil_se_at_0.5": math.sqrt((n1 + n0 + 1) / (12 * n1 * n0)), "se_boot_U0_mean": o["methods"]["U0"]["se_boot_mean"]}
    se_max = DELTA / Z90
    extrap = {"se_total_max_for_possible_pass": se_max,
              "min_n_equal_groups_no_ref_error": math.ceil(1 / (6 * se_max ** 2)),
              "note": "외삽: AUC≈0.5, |X|=|Z|=n, 참조 평균의 오차 0을 가정한 Hanley–McNeil 근사 se≈sqrt(1/(6n))"}
    observed_out = {"meta": {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": "results/final/e2/*.json (190)",
                             "kind": "observed (recomputed from stored statistics)"},
                    "scenarios": obs, "u0_normality_pooled": normality, "subset_enumeration": enum,
                    "subset_enumeration_overall": enum_overall, "run_level_stability_n5": stab,
                    "hanley_mcneil_check": hm, "extrapolation_min_n": extrap}
    json.dump(observed_out, open(os.path.join(OUT, "a1_observed.json"), "w"), indent=1, ensure_ascii=False)

    sigma_rec = None
    a2 = os.path.join(OUT, "a2_summary.json")
    if os.path.exists(a2):
        sigma_rec = {sc: v["sigma_rec_d_mean"] for sc, v in json.load(open(a2))["scenarios"].items()}
    sim = simulate(obs, sigma_rec)
    hier, hier_tot = hierarchical(obs)
    icc = shared_ref_icc(obs)
    mc_se_max = 0.5 / math.sqrt(REPS)
    simulated_out = {"meta": {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": "simulated (normal model, parameters = observed estimates)",
                              "seed": SEED, "reps": REPS, "mc_se_max": mc_se_max, "nrefs": NREFS, "M3_included": bool(sigma_rec)},
                     "scenarios": sim, "calibration_U0_shared_refs": hier, "calibration_U0_total": hier_tot, "shared_ref_counts": icc}
    json.dump(simulated_out, open(os.path.join(OUT, "a1_simulated.json"), "w"), indent=1, ensure_ascii=False)
    print(f"done in {time.time() - t0:.0f}s; M3={'yes' if sigma_rec else 'no'}")


if __name__ == "__main__":
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"# a1_analysis start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} git {head} script_sha256 "
          f"{hashlib.sha256(open(__file__, 'rb').read()).hexdigest()} numpy {np.__version__} seed {SEED}")
    main()
