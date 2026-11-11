"""L3 민감도(3장 반영 56번 ②, 2026-10-29): 저장된 E2 결과로 판정 규칙만 바꿔 다시 판정한다(새 학습 없음).

규칙: 주 규칙(δ=0.05, 정규 분위수) / δ=0.025 / t 분위수(자유도 = 참조 수 − 1, 보수적) / 둘 다.
각 규칙 안에서 양성 대조도 같은 규칙으로 다시 적용한다(P1이 그 규칙에서 기준 초과일 때만 근사 방법 통과 인정).
U0(정확 방법)는 배포와 무관하지만, 같은 규칙에서 U0 자신이 통과하는 비율을 함께 보고한다.
저FPR 지표: LiRA TPR@1%FPR의 방법별 평균(후기 철회 시나리오).

.venv/bin/python scripts/l3_sensitivity.py results/final/e2 results/final/L3_sensitivity.json
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fl.consent import SPECS  # noqa: E402

APPROX = ["U1", "U2", "U3", "U4", "U4s"]
RULES = {"main": (0.05, "z"), "delta0.025": (0.025, "z"), "t": (0.05, "t"), "delta0.025_t": (0.025, "t")}


def classify(auc, ci95, refs, delta, q):
    """fl/l3.py tost와 같은 계산에 분위수만 바꾼다. 반환: pass / exceeds / other."""
    se_t = (ci95[1] - ci95[0]) / (2 * 1.959963984540054)
    mu, sd = statistics.mean(refs), statistics.stdev(refs)
    se = math.sqrt(se_t ** 2 + sd ** 2 / len(refs))
    k = stats.norm.ppf(0.95) if q == "z" else stats.t.ppf(0.95, len(refs) - 1)
    d = auc - mu
    lo, hi = d - k * se, d + k * se
    if lo > -delta and hi < delta:
        return "pass"
    return "exceeds" if lo > 0 else "other"


def late(sc):
    s = SPECS.get(sc, {})
    return s.get("timing") == "late" or "waves" in s


def main(root="results/e2", out_path="results/L3_sensitivity.json"):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/*_s*.json"))]
    res = {r: {"all": defaultdict(lambda: [0, 0]), "late": defaultdict(lambda: [0, 0])} for r in RULES}
    tpr = defaultdict(list)
    for run in runs:
        refs = run.get("u0_ref_lira_auc") or []
        ms = run["methods"]
        if len(refs) < 3 or "P1" not in ms:
            continue
        grp = "late" if late(run["scenario"]) else "early_mid"
        for rule, (delta, q) in RULES.items():
            p1 = ms["P1"]["lira_offline"]
            powered = classify(p1["auc"], p1["ci"], refs, delta, q) == "exceeds"
            for m in APPROX + ["U0"]:
                x = ms.get(m)
                if not (x and "lira_offline" in x):
                    continue
                lo = x["lira_offline"]
                rr = refs
                if m == "U0" and lo["auc"] in refs:            # 자기참조 제외(49번)
                    j = refs.index(lo["auc"])
                    rr = refs[:j] + refs[j + 1:]
                exact = m == "U0" or (m == "U1" and "latejoin" in run["scenario"])   # summarize_e2.py와 같은 정확 방법 판정
                ok = classify(lo["auc"], lo["ci"], rr, delta, q) == "pass" and (exact or powered)
                for g in ("all",) + (("late",) if grp == "late" else ()):
                    res[rule][g][m][0] += ok
                    res[rule][g][m][1] += 1
        if grp == "late":
            for m in ["P1", "U0"] + APPROX:
                if m in ms and "lira_offline" in ms[m]:
                    tpr[m].append(ms[m]["lira_offline"]["tpr@fpr1%"])
    out = {"runs": len(runs), "rules": {r: {g: {m: {"k": v[0], "n": v[1]} for m, v in d.items()} for g, d in gs.items()}
                                        for r, gs in res.items()},
           "tpr_at_fpr1_late": {m: {"mean": statistics.mean(v), "n": len(v)} for m, v in tpr.items()}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    for r in RULES:
        print(r, {g: {m: f"{v['k']}/{v['n']}" for m, v in d.items()} for g, d in out["rules"][r].items()})
    print("TPR@1%FPR (late):", {m: round(v["mean"], 4) for m, v in out["tpr_at_fpr1_late"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
