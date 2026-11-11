"""A1 공통: 실행 로그 읽기와 L3 규칙의 독립 재구현(원고 서술 기준, fl/l3.py를 import하지 않는다)."""
import glob
import json
import math
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
E2 = os.path.join(ROOT, "results", "final", "e2")
DELTA = 0.05
Z90 = 1.6448536269514722
Z975 = 1.959963984540054
METHODS = ("P1", "U0", "U1", "U2", "U3", "U4", "U4s")
APPROX_DEFAULT = ("U1", "U2", "U3", "U4", "U4s")


def classify(d, se, delta=DELTA, z=Z90):
    """통과: CI ⊂ (−δ, δ). 기준 초과: 통과가 아니고 하한 > 0. 그 외 판정 불가."""
    lo, hi = d - z * se, d + z * se
    if lo > -delta and hi < delta:
        return "pass"
    if lo > 0:
        return "exceeds"
    return "inconclusive"


def tost(auc, ci95, refs, delta=DELTA):
    se_t = (ci95[1] - ci95[0]) / (2 * Z975)
    mu, sd = statistics.mean(refs), statistics.stdev(refs)
    se = math.sqrt(se_t ** 2 + sd ** 2 / len(refs))
    d = auc - mu
    return {"d": d, "se": se, "se_t": se_t, "lo": d - Z90 * se, "hi": d + Z90 * se, "class": classify(d, se, delta)}


KO = {"통과": "pass", "기준 초과": "exceeds", "판정 불가": "inconclusive"}


def load_runs():
    """[{scenario, seed, nX, nZ, refs[5], methods{m: {auc, ci, kind, stored_class, stored_reason, self_ref, deploy}}}]"""
    runs = []
    for f in sorted(glob.glob(os.path.join(E2, "*.json"))):
        j = json.load(open(f))
        ms = {}
        for m in METHODS:
            x = j["methods"].get(m)
            if not x or "lira_offline" not in x:
                continue
            ms[m] = {"auc": x["lira_offline"]["auc"], "ci": x["lira_offline"]["ci"], "kind": x.get("kind"),
                     "stored_class": KO.get((x.get("L3") or {}).get("class")), "stored_reason": (x.get("L3") or {}).get("reason"),
                     "stored_power_ok": (x.get("L3") or {}).get("power_ok"),
                     "self_ref": x.get("self_ref"), "deploy": x.get("deploy")}
        runs.append({"scenario": j["scenario"], "seed": j["seed"], "nX": j["X_records"], "nZ": j["Z_records"],
                     "refs": j["u0_ref_lira_auc"], "methods": ms, "file": os.path.basename(f)})
    return runs


def refs_for(run, m, exclude_self=True):
    """원고 규칙: 대상이 참조 중 하나(U0 시드 1–5)이면 자기 자신을 참조에서 뺀다.
    실행 로그에는 self_ref 필드가 없어(최종 실행 당시 코드) AUC 값이 같은 참조를 자기 자신으로 본다.
    exclude_self=False는 실행 로그에 저장된 L3 필드의 계산 방식(자기 자신 포함)이다."""
    refs = run["refs"]
    a = run["methods"][m]["auc"]
    if exclude_self and m == "U0" and a in refs:
        j = refs.index(a)
        return refs[:j] + refs[j + 1:]
    return list(refs)
