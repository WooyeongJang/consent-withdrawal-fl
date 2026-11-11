"""A0: 기존 L3 결과 재현. 저장된 통계량(AUC, 95% CI, 참조 AUC)만으로 분류를 다시 계산해 저장된 분류, Table S10 첫 행과 대조한다."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from a1_common import APPROX_DEFAULT, METHODS, load_runs, refs_for, tost

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
runs = load_runs()
assert len(runs) == 190, len(runs)
res = {"n_runs": len(runs), "mismatch": [], "counts": {}}
raw = collections.Counter(); asw = collections.Counter(); runtime = collections.Counter(); stored = collections.Counter()
n_cmp = 0
for r in runs:
    p1 = tost(r["methods"]["P1"]["auc"], r["methods"]["P1"]["ci"], refs_for(r, "P1"))
    for m in METHODS:
        x = r["methods"][m]
        t = tost(x["auc"], x["ci"], refs_for(r, m))
        cls = t["class"]
        cls_log = tost(x["auc"], x["ci"], refs_for(r, m, exclude_self=False))["class"]   # 실행 로그의 계산 방식
        raw[(m, cls)] += 1
        # 원고 규칙: 근사 방법의 통과는 P1이 기준 초과일 때만 인정
        kind = x["kind"]
        final = cls
        if kind == "approx" and cls == "pass" and p1["class"] != "exceeds":
            final = "inconclusive"
        asw[(m, final)] += 1
        rt = cls_log
        if kind == "approx" and cls_log == "pass" and not (p1["lo"] > 0):
            rt = "inconclusive"
        runtime[(m, rt)] += 1
        # 저장된 분류: 실행 당시 규칙으로 기록됨(로그 불변). 비교는 실행 당시 규칙과 한다.
        if x["stored_class"] is not None:
            n_cmp += 1
            stored[(m, x["stored_class"])] += 1
            if x["stored_class"] != rt:
                res["mismatch"].append({"file": r["file"], "method": m, "stored": x["stored_class"], "recomputed_runtime_rule": rt,
                                        "recomputed_as_written": final})
for name, c in (("raw_tost", raw), ("as_written", asw), ("runtime_rule", runtime), ("stored_in_logs", stored)):
    res["counts"][name] = {m: {k: c[(m, k)] for k in ("pass", "exceeds", "inconclusive")} for m in METHODS}
res["n_compared_with_stored"] = n_cmp
res["table_S10_row1_paper"] = {"U0": 145, "U1": 77, "U2": 66, "U3": 68, "U4": 3, "U4s": 11}
res["table_S10_row1_recomputed_raw"] = {m: raw[(m, "pass")] for m in ("U0", "U1", "U2", "U3", "U4", "U4s")}
res["table_S10_row1_recomputed_as_written"] = {m: asw[(m, "pass")] for m in ("U0", "U1", "U2", "U3", "U4", "U4s")}
res["u0_self_reference"] = {"seeds_1_5_auc_equals_a_reference": sum(1 for r in runs if r["methods"]["U0"]["auc"] in r["refs"]), "stored_self_ref_field_present": sum(1 for r in runs if r["methods"]["U0"]["self_ref"] is not None)}
res["kinds"] = {m: dict(collections.Counter(r["methods"][m]["kind"] for r in runs)) for m in METHODS}
json.dump(res, open(os.path.join(OUT, "a0_reproduction.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ("n_compared_with_stored", "table_S10_row1_paper", "table_S10_row1_recomputed_raw",
                                      "table_S10_row1_recomputed_as_written", "u0_self_reference", "kinds")}, indent=1))
print("mismatch vs stored (runtime rule):", len(res["mismatch"]))
for k, v in res["counts"].items():
    print(k, v)
