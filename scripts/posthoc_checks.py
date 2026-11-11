"""원고의 사후 계산 수치를 저장된 결과에서 다시 계산한다(새 학습·거래 없음, 2026-11-01).

E2(results/final/e2, E2_e2.json)
- U0의 L3 분류(통과·정밀도 부족·과잉 제거·초과, 1% 철회 내역), 다른 참조로 시험한 U0의 |z| > 1.645 비율
- 예측구간 변형(대상 모델의 학습 분산을 더함)에서 U0 통과 수
- 시나리오 간 Holm(전체 대응 비교): 후기 P1 LiRA 최대 p, 중기 20% P1, U2 AUROC
- 양성 대조 효과(철회율 × 시점 격자의 초기·중기 80회): 대조 전후 근사 방법 통과 수, 남은 통과가 나온 실행과 P1 하한
- 초기·중기 철회에서 P1 자신의 통과 수, 후기 P1 잔존의 효과크기 d_z 범위
- U5(데이터센터 제외)와 U2 과잉 제거 크기(기관 집중·이탈)
E3(results/final/e3)
- T1: 원장에 저장된 철회 수가 누락하지 않은 철회 수와 같은지(누락된 요청이 원장에 없음)
- 정직한 학습 단위 재실행 수: 유형 A 표본에서 공격 단위를 뺀 수, 유형 B 정규 라운드 표본 수

.venv/bin/python scripts/posthoc_checks.py [results/final] [results/final/posthoc_checks.json]
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from fl.consent import SPECS  # noqa: E402
from fl.l3 import Z90  # noqa: E402
from fl.stats import holm  # noqa: E402
from summarize_e2 import exclude_self_ref, power_ok  # noqa: E402

APPROX = ["U1", "U2", "U3", "U4", "U4s"]
GRID_EARLY_MID = ["E2-early", "E2-mid", "E2-rate01-early", "E2-rate01-mid", "E2-rate10-early", "E2-rate10-mid",
                  "E2-rate20-early", "E2-rate20-mid"]


def late(sc):
    s = SPECS.get(sc, {})
    return s.get("timing") == "late" or "waves" in s


def l3_class(lo, hi, delta=0.05):
    if lo > -delta and hi < delta:
        return "pass"
    return "exceeds" if lo > 0 else "over-removal" if hi < 0 else "insufficient precision"


def e2(root):
    runs = defaultdict(dict)
    for f in sorted(glob.glob(f"{root}/e2/*_s*.json")):
        r = exclude_self_ref(json.load(open(f)))
        runs[r["scenario"]][r["seed"]] = r
    out = {}
    # U0 분류, |z|, 예측구간 변형
    cls, cls1, z_out, pi_pass, n = Counter(), Counter(), 0, 0, 0
    for sc, bs in runs.items():
        for r in bs.values():
            x = r["methods"]["U0"]
            lo, hi = x["L3"]["ci90"]
            c = l3_class(lo, hi)
            cls[c] += 1
            if "rate01" in sc:
                cls1[c] += 1
            se = (hi - lo) / (2 * Z90)
            z_out += abs(x["L3"]["diff"] / se) > Z90
            refs = list(r["u0_ref_lira_auc"])
            auc, ci = x["lira_offline"]["auc"], x["lira_offline"]["ci"]
            if x["L3"].get("self_ref_excluded"):
                refs.remove(auc)
            se_t = (ci[1] - ci[0]) / (2 * 1.959963984540054)
            sep = math.sqrt(se_t ** 2 + statistics.stdev(refs) ** 2 * (1 + 1 / len(refs)))
            d = auc - statistics.mean(refs)
            pi_pass += (d - Z90 * sep > -0.05 and d + Z90 * sep < 0.05)
            n += 1
    out["U0_L3"] = {"runs": n, "classes": dict(cls), "classes_at_1pct": dict(cls1),
                    "outside_1.645se": z_out, "prediction_interval_passes": pi_pass}
    # 시나리오 간 Holm
    agg = json.load(open(f"{root}/E2_e2.json"))
    keys, ps = [], []
    for sc, res in agg.items():
        for k, c in res["compare_vs_U0"].items():
            if c.get("p") is not None:
                keys.append((sc, k)); ps.append(c["p"])
    g = dict(zip(keys, holm(ps)))
    late_sc = sorted(sc for sc in agg if late(sc))
    out["holm_across_scenarios"] = {
        "comparisons": len(ps),
        "late_P1_lira_max_p": max(g[(sc, "P1:lira")] for sc in late_sc), "late_scenarios": late_sc,
        "rate20_mid_P1_lira_p": g[("E2-rate20-mid", "P1:lira")],
        "U2_test_p": {sc: g[(sc, "U2:test")] for sc in agg if (sc, "U2:test") in g and g[(sc, "U2:test")] >= 0.05}}
    out["late_P1_lira_dz"] = [min(agg[sc]["compare_vs_U0"]["P1:lira"]["effect"].get("d_z", float("nan")) for sc in late_sc),
                              max(agg[sc]["compare_vs_U0"]["P1:lira"]["effect"].get("d_z", float("nan")) for sc in late_sc)]
    # 양성 대조 효과
    before, after, left = Counter(), Counter(), []
    p1_pass_em = Counter()
    for sc in GRID_EARLY_MID:
        for s, r in sorted(runs[sc].items()):
            p1 = r["methods"]["P1"]["L3"]
            p1_pass_em[sc] += p1["result"] == "통과"
            for m in APPROX:
                x = r["methods"].get(m, {}).get("L3")
                if x and x["result"] == "통과":
                    before[m] += 1
                    if power_ok(p1):
                        after[m] += 1
                        left.append({"scenario": sc, "seed": s, "method": m, "P1_ci90_lo": p1["ci90"][0]})
    out["positive_control_grid"] = {"runs": sum(len(runs[sc]) for sc in GRID_EARLY_MID),
                                    "passes_before": dict(before), "passes_after": dict(after), "remaining": left,
                                    "P1_passes_early_mid": dict(p1_pass_em)}
    # U5, U2 과잉 제거
    out["U5"] = {sc: {"L3": agg[sc]["methods"]["U5"]["L3_class"],
                      "test_diff_vs_U0": agg[sc]["compare_vs_U0"]["U5:test"]["diff"]["mean"]}
                 for sc in agg if "U5" in agg[sc]["methods"]}
    out["U2_lira_diff_vs_U0"] = {sc: agg[sc]["compare_vs_U0"]["U2:lira"]["diff"]["mean"] for sc in ("E2-inst2", "E2-exit2")}
    return out


def e3(root):
    t1_ok, n_a = 0, 0
    sampled, attacked = 0, 0
    for f in sorted(glob.glob(f"{root}/e3/A_seed*_p0.3.json")):
        d = json.load(open(f))
        honest = {x["pid"] for x in d["instances"] if x["attack"] != "T1"}
        t1 = [x for x in d["instances"] if x["attack"] == "T1"]
        t1_ok += len(t1) == 1 and d["withdrawal_count_on_chain"] == len(honest)
        n_a += 1
        sampled += d["l1_sampled_units"]
        attacked += len({(x["k"], x["r"]) for x in d["instances"] if x["attack"] != "T1" and x.get("unit_sampled")})
    b = [json.load(open(f))["training_L1"] for f in sorted(glob.glob(f"{root}/e3/B_seed*.json"))]
    return {"T1_onchain_count_excludes_dropped_request": f"{t1_ok}/{n_a}",
            "typeA_sampled_units": sampled, "typeA_sampled_attacked_units": attacked,
            "typeA_honest_reexecutions": sampled - attacked,
            "typeB_regular_round_reexecutions": sum(x["sampled"] for x in b),
            "typeB_failures": sum(x["fail"] for x in b)}


def main(root="results/final", out_path="results/final/posthoc_checks.json"):
    out = {"E2": e2(root), "E3": e3(root)}
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float)[:4000])


if __name__ == "__main__":
    main(*sys.argv[1:3])
