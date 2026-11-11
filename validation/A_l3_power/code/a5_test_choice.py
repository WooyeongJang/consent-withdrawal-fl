"""A5(사후, 2026-11-10 통계 감사): 저장된 E2 결과(읽기 전용)에서 (1) 대응 검정 선택의 민감도, (2) L3 판정 190회 × 6방법의
재계산, (3) LiRA 부트스트랩 CI의 대칭성과 Hanley–McNeil 표준오차 대비 비율을 확인한다. 새 학습·거래 없음.

(1) 시나리오 안 대응 비교(방법 − U0, AUROC·LiRA AUC)는 fl/stats.py paired가 Shapiro–Wilk(p > 0.05)로 대응 t와 Wilcoxon을 고른다.
    같은 자료에 대응 t만 / Wilcoxon만 적용하고 시나리오 안 Holm(α = 0.05)으로 판정이 바뀌는 비교를 센다.
(2) lira_offline.auc·ci와 u0_ref_lira_auc만으로 fl/l3.py tost를 적용(U0 자기 참조 제외, 양성 대조는 원고 규칙)해 Table S10 첫 행을 재현한다.
(3) 모델 수준 95% 백분위 CI의 비대칭 |(상한−AUC)−(AUC−하한)|과, 폭/(2·1.96)으로 둔 se_boot를 Hanley–McNeil 근사 표준오차로 나눈 비율.

저장소 루트에서:  .venv/bin/python validation/A_l3_power/code/a5_test_choice.py [results/final] [validation/A_l3_power/results/a5_test_choice.json]
"""
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fl.l3 import tost  # noqa: E402
from fl.stats import holm  # noqa: E402
from summarize_e2 import exclude_self_ref  # noqa: E402

LATE = {"E2-base", "E2-rate01", "E2-rate10", "E2-rate20", "E2-diagMI", "E2-F65", "E2-inst2", "E2-exit2", "E2-latejoin", "E2-seq3"}


def reported_claim(sc, key):
    """본문 V절이 p값(Holm)으로 보고한 비교: 후기 10개 시나리오의 P1 LiRA 초과("significant in each"),
    모든 시나리오의 U2·U3 AUROC 손실("significant within every main scenario")."""
    return (key == "P1:lira" and sc in LATE) or key in ("U2:test", "U3:test")


def main(final="results/final", out_path="validation/A_l3_power/results/a5_test_choice.json"):
    agg = json.load(open(os.path.join(final, "E2_e2.json")))
    runs = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(final, "e2", "*_s*.json"))):
        r = exclude_self_ref(json.load(open(f)))
        runs[r["scenario"]][r["seed"]] = r
    out = {}
    # (1) 검정 선택 민감도
    tests = collections.Counter(c.get("test") for r in agg.values() for c in r["compare_vs_U0"].values())
    flips = {"all_paired_t": [], "all_wilcoxon": []}
    n_cmp = 0
    for sc, by in runs.items():
        seeds = sorted(by)
        keys, p_t, p_w, rep = [], [], [], []
        for k, c in agg[sc]["compare_vs_U0"].items():
            m, met = k.split(":")
            ref = "U0-seq" if m.endswith("-seq") else "U0"
            f = (lambda x: x["test_macro_auc"]) if met == "test" else (lambda x: x["lira_offline"]["auc"])
            a = [f(by[s]["methods"][m]) for s in seeds if "test_macro_auc" in by[s]["methods"].get(m, {})]
            b = [f(by[s]["methods"][ref]) for s in seeds if "test_macro_auc" in by[s]["methods"].get(m, {})]
            d = np.array(a) - np.array(b)
            if len(d) < 3 or np.allclose(d, d[0]):
                p_t.append(None); p_w.append(None)
            else:
                p_t.append(float(st.ttest_rel(a, b).pvalue)); p_w.append(float(st.wilcoxon(a, b).pvalue))
            keys.append(k); rep.append(c.get("p_holm"))
        for k, r0, t, w in zip(keys, rep, holm(p_t), holm(p_w)):
            if r0 is None:
                continue
            n_cmp += 1
            for name, p in (("all_paired_t", t), ("all_wilcoxon", w)):
                if (r0 < 0.05) != (p < 0.05):
                    flips[name].append({"scenario": sc, "comparison": k, "p_holm_reported": r0, "p_holm_alt": p,
                                        "reported_claim": reported_claim(sc, k)})
    out["test_choice"] = {"comparisons": n_cmp, "tests_used": dict(tests), "flips": flips,
                          "n_flips": {k: len(v) for k, v in flips.items()},
                          "flips_among_reported_claims": {k: sum(x["reported_claim"] for x in v) for k, v in flips.items()}}
    # (2) L3 재계산
    cnt, p1_exceeds = collections.Counter(), 0
    for sc, by in runs.items():
        for s, r in by.items():
            refs = r["u0_ref_lira_auc"]
            p1 = r["methods"]["P1"]["lira_offline"]
            c0 = tost(p1["auc"], p1["ci"], refs)["class"]
            p1_exceeds += c0 == "기준 초과"
            for m in ["U0", "U1", "U2", "U3", "U4", "U4s"]:
                x = r["methods"].get(m)
                if not x:
                    continue
                rr = list(refs)
                if m == "U0" and x["lira_offline"]["auc"] in rr:
                    rr.remove(x["lira_offline"]["auc"])
                c = tost(x["lira_offline"]["auc"], x["lira_offline"]["ci"], rr)["class"]
                exact = m == "U0" or (m == "U1" and "latejoin" in sc)
                cnt[m] += int(c == "통과" and (exact or c0 == "기준 초과"))
    out["l3_recomputed"] = {"passes_rule_as_written": dict(cnt), "P1_exceeds": p1_exceeds,
                            "table_s10_row1": [145, 77, 66, 68, 3, 11]}
    assert [cnt[m] for m in ["U0", "U1", "U2", "U3", "U4", "U4s"]] == out["l3_recomputed"]["table_s10_row1"]
    # (3) 부트스트랩 CI
    asym, ratio = [], []
    for sc, by in runs.items():
        for s, r in by.items():
            nX, nZ = r["X_records"], r["Z_records"]
            for m, x in r["methods"].items():
                if not x or "lira_offline" not in x:
                    continue
                a = x["lira_offline"]["auc"]; lo, hi = x["lira_offline"]["ci"]
                asym.append(abs((hi - a) - (a - lo)))
                se = (hi - lo) / (2 * 1.959963984540054)
                q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
                hm = np.sqrt((a * (1 - a) + (nX - 1) * (q1 - a * a) + (nZ - 1) * (q2 - a * a)) / (nX * nZ))
                ratio.append(se / hm)
    out["bootstrap_ci"] = {"n_intervals": len(asym), "asymmetry_median": float(np.median(asym)), "asymmetry_max": float(max(asym)),
                           "se_over_hanley_mcneil": {"median": float(np.median(ratio)), "min": float(min(ratio)), "max": float(max(ratio))}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
    tc = out["test_choice"]
    print("comparisons", tc["comparisons"], "tests", tc["tests_used"], "flips", tc["n_flips"], "among reported claims", tc["flips_among_reported_claims"])
    for x in flips["all_wilcoxon"]:
        print("  wilcoxon-only flip:", x["scenario"], x["comparison"], round(x["p_holm_reported"], 4), "->", round(x["p_holm_alt"], 4))
    print("L3 recomputed", out["l3_recomputed"]["passes_rule_as_written"], "P1 exceeds", p1_exceeds)
    print("bootstrap CI", out["bootstrap_ci"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
