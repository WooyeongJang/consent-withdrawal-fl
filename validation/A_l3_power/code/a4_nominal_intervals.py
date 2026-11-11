"""A4(사후 추가, 사전 등록 밖): 명목 수준을 지키는 구간을 기존 실행 로그에 적용한다.

2026-11-09 A1 결과(방법 수준 경계 통과 확률 최대 0.11)를 본 뒤 추가했다. 주 규칙·마진은 바꾸지 않는다(DEVIATIONS.md A절).
실행 단위(190회, 양성 대조는 같은 규칙을 P1에 적용한 '원고 규칙'):
  z     se^2 = se_boot^2 + s^2/n,        z 분위수          (주 규칙; Table S10 첫 행을 재현해야 한다)
  t     같은 se,                          t_{n-1} 분위수    (Table S10 셋째 행을 재현해야 한다)
  Pz    se^2 = se_boot^2 + s^2(1+1/n),   z 분위수          (기존 scripts/posthoc_checks.py의 예측구간 변형, U0 123)
  Pt    같은 se,                          t_{n-1} 분위수
시나리오 단위(19개): 시드 10개의 대응 차이 d_i = AUC_m,i − AUC_U0,i 의 평균에 t_9 90% 구간(같은 마진 ±0.05, 같은 분류 규칙).
경계 통과 확률: A1의 모의 모형(M2: σ_T = σ_ref, 그리고 σ_T = 1.5σ_ref)에서 Δ = 0.05일 때 통과할 확률, 100,000회, 시드 20261109.
"""
import collections
import json
import math
import os
import statistics as st
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import a1_common as c

RES = os.path.join(os.path.dirname(__file__), "..", "results")
APP = c.APPROX_DEFAULT
LATE = ["E2-base", "E2-rate01", "E2-rate10", "E2-rate20", "E2-diagMI", "E2-F65", "E2-inst2", "E2-seq3", "E2-exit2", "E2-latejoin"]
EARLY_MID = [f"E2-{p}{t}" for p in ("", "rate01-", "rate10-", "rate20-") for t in ("early", "mid")]


def cls(d, se, crit, delta=c.DELTA):
    lo, hi = d - crit * se, d + crit * se
    if lo > -delta and hi < delta:
        return "pass"
    if lo > 0:
        return "exceeds"
    return "inconclusive"


def run_class(r, m, mode):
    refs = c.refs_for(r, m)
    n = len(refs)
    x = r["methods"][m]
    se_t = (x["ci"][1] - x["ci"][0]) / (2 * c.Z975)
    s = st.stdev(refs)
    d = x["auc"] - st.mean(refs)
    var = se_t ** 2 + s * s * ((1 + 1 / n) if mode[0] == "P" else 1 / n)
    crit = stats.t.ppf(0.95, n - 1) if mode.endswith("t") else c.Z90
    return cls(d, math.sqrt(var), crit)


def main():
    runs = c.load_runs()
    by = collections.defaultdict(list)
    for r in runs:
        by[r["scenario"]].append(r)
    assert len(runs) == 190 and len(by) == 19 and set(LATE) | set(EARLY_MID) | {"E2-reconsent"} == set(by)
    out = {"meta": {"kind": "post hoc (not pre-registered); observed counts recomputed from stored statistics, boundary probabilities simulated",
                    "source": "results/final/e2/*.json (190)", "seed": 20261109, "reps": 100000}}

    # ---- 실행 단위
    run_level = {}
    for mode in ("z", "t", "Pz", "Pt"):
        row = {"pass_with_control": collections.Counter(), "pass_no_control": collections.Counter(),
               "pass_with_control_late": collections.Counter(), "pass_with_control_early_mid": collections.Counter(), "P1": collections.Counter()}
        for r in runs:
            p1 = run_class(r, "P1", mode)
            row["P1"][p1] += 1
            for m in ("U0",) + APP:
                k = run_class(r, m, mode)
                row["pass_no_control"][m] += k == "pass"
                ok = k == "pass" and (m == "U0" or p1 == "exceeds")
                row["pass_with_control"][m] += ok
                if r["scenario"] in LATE:
                    row["pass_with_control_late"][m] += ok
                if r["scenario"] in EARLY_MID:
                    row["pass_with_control_early_mid"][m] += ok
        run_level[mode] = {k: dict(v) for k, v in row.items()}
    assert [run_level["z"]["pass_with_control"][m] for m in ("U0",) + APP] == [145, 77, 66, 68, 3, 11], "Table S10 row 1"
    assert [run_level["t"]["pass_with_control"][m] for m in ("U0",) + APP] == [122, 61, 51, 54, 2, 8], "Table S10 row 3"
    assert run_level["Pz"]["pass_with_control"]["U0"] == 123, "extended results: prediction-interval variant"
    out["run_level"] = run_level

    # ---- 시나리오 단위
    t9 = stats.t.ppf(0.95, 9)
    scen = {}
    tot = collections.defaultdict(collections.Counter)
    credited = collections.Counter()
    for sc, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        assert len(rs) == 10
        row = {}
        for m in ("P1",) + APP:
            d = np.array([r["methods"][m]["auc"] - r["methods"]["U0"]["auc"] for r in rs])
            mu, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
            row[m] = {"mean": mu, "sd": float(d.std(ddof=1)), "lo": mu - t9 * se, "hi": mu + t9 * se, "class": cls(mu, se, t9)}
        scen[sc] = row
        for m in ("P1",) + APP:
            tot[m][row[m]["class"]] += 1
        for m in APP:
            credited[m] += row[m]["class"] == "pass" and row["P1"]["class"] == "exceeds"
    out["scenario_level"] = {"scenarios": scen, "class_counts": {m: dict(v) for m, v in tot.items()}, "pass_with_control": dict(credited),
                             "P1_exceeds": sorted(sc for sc in scen if scen[sc]["P1"]["class"] == "exceeds"),
                             "halfwidth_range": [min((v["hi"] - v["lo"]) / 2 for r in scen.values() for k, v in r.items() if v["sd"] > 0),
                                                 max((v["hi"] - v["lo"]) / 2 for r in scen.values() for v in r.values())]}

    # ---- 경계 통과 확률(모의)
    sim = json.load(open(os.path.join(RES, "a1_simulated.json")))["scenarios"]
    rng = np.random.default_rng(20261109)
    N, n = 100000, 5
    tq = stats.t.ppf(0.95, n - 1)
    bnd = {}
    for sc, v in sorted(sim.items()):
        sr, sb = v["params"]["sigma_ref"], v["params"]["se_boot_U0"]
        bnd[sc] = {}
        for mult in (1.0, 1.5):
            refs = rng.normal(0, sr, (N, n))
            d = c.DELTA + rng.normal(0, mult * sr, N) - refs.mean(1)
            s = refs.std(1, ddof=1)
            for name, var, crit in (("z", sb ** 2 + s * s / n, c.Z90), ("t", sb ** 2 + s * s / n, tq),
                                    ("Pz", sb ** 2 + s * s * (1 + 1 / n), c.Z90), ("Pt", sb ** 2 + s * s * (1 + 1 / n), tq)):
                se = np.sqrt(var)
                bnd[sc][f"{name}_sT{mult}"] = float(np.mean((d - crit * se > -c.DELTA) & (d + crit * se < c.DELTA)))
    thr = 0.05 + 3 * math.sqrt(0.05 * 0.95 / N)
    summ = {}
    for key in next(iter(bnd.values())):
        vals = [bnd[sc][key] for sc in bnd]
        summ[key] = {"max": max(vals), "n_above_nominal": sum(v > thr for v in vals)}
    out["boundary"] = {"threshold": thr, "per_scenario": bnd, "summary": summ}

    json.dump(out, open(os.path.join(RES, "a4_nominal_intervals.json"), "w"), indent=1, ensure_ascii=False)
    for mode, row in run_level.items():
        print(mode, row["pass_with_control"], "| late", row["pass_with_control_late"], "| early/mid", row["pass_with_control_early_mid"], "| P1", row["P1"])
    print("scenario level:", out["scenario_level"]["class_counts"], "credited", dict(credited), "hw", out["scenario_level"]["halfwidth_range"])
    print("P1 exceeds:", out["scenario_level"]["P1_exceeds"])
    print("boundary:", {k: (round(v["max"], 3), v["n_above_nominal"]) for k, v in summ.items()})


if __name__ == "__main__":
    main()
