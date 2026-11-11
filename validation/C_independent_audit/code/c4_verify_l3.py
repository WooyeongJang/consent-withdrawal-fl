"""C-4(L3)·C-5·C-6·C-7: 양성 대조 규칙 변경, 배포 필드 재분류, 보충 S7절·Table S8·S9의 수치를 실행 로그의 통계량에서 다시 계산한다.

규칙은 보충 Algorithm S5의 서술에서 다시 구현했다(fl/l3.py, scripts/*를 import하지 않음).
  d = a − mean(refs), se = sqrt(se_a² + s²/n), [l, u] = d ± 1.6449·se
  pass: −δ < l 이고 u < δ / exceeds: pass가 아니고 l > 0 / 그 외 inconclusive
  양성 대조(원고 규칙): 근사 방법의 pass는 같은 시험에서 P1이 exceeds일 때만 인정
  양성 대조(실행 당시 규칙): P1의 l > 0이면 인정
se_a는 로그에 저장된 95% 백분위 부트스트랩 CI의 폭 / (2·1.959964)로 둔다(로그에는 CI만 있다).
자기 참조 제외: 대상의 AUC와 정확히 같은 참조값을 뺀다(U0 시드 1–5). 로그의 self_ref 필드와 일치하는지 따로 확인한다.
실행 로그의 L3.class·deploy·power_ok 필드는 비교 대상으로만 읽는다.

  python3 c4_verify_l3.py
"""
import collections
import json
import math
import statistics
import subprocess
import time

import c_lib as L

OUT = L.EXP / "results"
DELTA, Z90, Z975 = 0.05, 1.6448536269514722, 1.959963984540054
APPROX_LABELS = ("U1", "U2", "U3", "U4", "U4s")
KO = {"통과": "pass", "기준 초과": "exceeds", "판정 불가": "inconclusive"}
EARLY_MID = ["E2-early", "E2-mid", "E2-rate01-early", "E2-rate01-mid", "E2-rate10-early", "E2-rate10-mid", "E2-rate20-early", "E2-rate20-mid"]
NAMES = {"E2-rate20-mid": "20%, mid", "E2-rate10-mid": "10%, mid", "E2-exit2": "institution exit", "E2-seq3": "three waves"}


def classify(auc, ci95, refs, delta=DELTA):
    se_a = (ci95[1] - ci95[0]) / (2 * Z975)
    mu, s, n = statistics.mean(refs), statistics.stdev(refs), len(refs)
    d = auc - mu
    se = math.sqrt(se_a ** 2 + s ** 2 / n)
    l, u = d - Z90 * se, d + Z90 * se
    if -delta < l and u < delta:
        c = "pass"
    elif l > 0:
        c = "exceeds"
    else:
        c = "inconclusive"
    return {"class": c, "d": d, "l": l, "u": u, "n_ref": n}


def analyse(path):
    j = json.load(open(path))
    refs_all = j["u0_ref_lira_auc"]
    out = {"file": path.name, "scenario": j["scenario"], "seed": j["seed"], "methods": {}}
    m = j["methods"]
    p1 = classify(m["P1"]["lira_offline"]["auc"], m["P1"]["lira_offline"]["ci"], refs_all)
    out["P1"] = p1
    for label, v in m.items():
        if not isinstance(v, dict) or not v.get("L3") or label in ("P1",):
            continue
        a = v["lira_offline"]["auc"]
        refs = [r for r in refs_all if r != a]
        self_idx = [i for i, r in enumerate(refs_all) if r == a]
        raw = classify(a, v["lira_offline"]["ci"], refs)
        kind = v.get("kind")
        base = label.replace("-seq", "")
        no_ctrl = raw["class"]
        runtime = "inconclusive" if (kind == "approx" and raw["class"] == "pass" and not p1["l"] > 0) else raw["class"]
        written = "inconclusive" if (kind == "approx" and raw["class"] == "pass" and p1["class"] != "exceeds") else raw["class"]
        out["methods"][label] = {"base": base, "kind": kind, "raw": raw, "no_control": no_ctrl, "runtime": runtime, "written": written,
                                 "self_ref_by_equality": self_idx, "json_self_ref": v.get("self_ref"),
                                 "json_class": KO.get(v["L3"]["class"], v["L3"]["class"]), "json_reason": v["L3"].get("reason"),
                                 "json_deploy": v.get("deploy"), "json_power_ok": v["L3"].get("power_ok"),
                                 "json_ci90": v["L3"]["ci90"], "json_diff": v["L3"]["diff"]}
    return out


def deploy_L3(x):
    return x["json_deploy"] == "배포 허용(L3 통과)"


def main():
    hdr = L.run_header(__file__)
    res = {"header": hdr}
    main_logs = [analyse(p) for p in sorted((L.REPO / "results/final/e2").glob("*.json"))]
    res["n_main_logs"] = len(main_logs)
    rows = [(r, lab, x) for r in main_logs for lab, x in r["methods"].items()]
    # ---- C-4(L3): 저장된 통계량에서 다시 분류 ----
    num_ok = sum(abs(x["raw"]["l"] - x["json_ci90"][0]) < 1e-12 and abs(x["raw"]["u"] - x["json_ci90"][1]) < 1e-12 and
                 abs(x["raw"]["d"] - x["json_diff"]) < 1e-12 for _, _, x in rows)
    res["L3"] = {"fields": len(rows), "ci90_and_diff_reproduced": num_ok,
                 "class_equals_log_under_runtime_rule": sum(x["runtime"] == x["json_class"] for _, _, x in rows),
                 "class_equals_log_under_written_rule": sum(x["written"] == x["json_class"] for _, _, x in rows),
                 "self_ref_consistent": sum((x["self_ref_by_equality"] == ([] if x["json_self_ref"] is None else [x["json_self_ref"]])) for _, _, x in rows),
                 "power_ok_field_equals_runtime_rule": sum(x["json_power_ok"] == (r["P1"]["l"] > 0) for r, _, x in rows if x["json_power_ok"] is not None),
                 "power_ok_fields": sum(x["json_power_ok"] is not None for _, _, x in rows),
                 "passes_190_written": {lab: sum(x["written"] == "pass" for r in main_logs for l2, x in r["methods"].items() if l2 == lab)
                                        for lab in ("U0", "U1", "U2", "U3", "U4", "U4s")},
                 "kinds": dict(collections.Counter(f"{lab}:{x['kind']}" for _, lab, x in rows))}
    # ---- C-5: 세 규칙의 통과 수(초기·중기 80개 실행) ----
    em = [r for r in main_logs if r["scenario"] in EARLY_MID]
    res["rules_80"] = {"n_runs": len(em), **{rule: {lab: sum(r["methods"][lab][rule] == "pass" for r in em) for lab in APPROX_LABELS}
                                             for rule in ("no_control", "runtime", "written")}}
    diff_scen = {}
    for sc in sorted({r["scenario"] for r in main_logs}):
        rs = [r for r in main_logs if r["scenario"] == sc]
        rt = {lab: sum(r["methods"][lab]["runtime"] == "pass" and r["methods"][lab]["kind"] == "approx" for r in rs) for lab in APPROX_LABELS}
        wr = {lab: sum(r["methods"][lab]["written"] == "pass" and r["methods"][lab]["kind"] == "approx" for r in rs) for lab in APPROX_LABELS}
        if rt != wr:
            diff_scen[sc] = {"runtime": rt, "written": wr,
                             "P1_pass_with_l_gt_0_runs": sum(r["P1"]["class"] == "pass" and r["P1"]["l"] > 0 for r in rs)}
    res["scenarios_that_differ"] = diff_scen
    res["P1_pass_with_l_gt_0_by_scenario"] = {sc: n for sc, n in collections.Counter(
        r["scenario"] for r in main_logs if r["P1"]["class"] == "pass" and r["P1"]["l"] > 0).items()}
    # ---- C-6: 배포 필드 ----
    dep = [(r, lab, x) for r, lab, x in rows if deploy_L3(x)]
    changed = [(r, lab, x) for r, lab, x in dep if x["written"] != "pass"]
    res["deploy_fields"] = {"recorded_deployable_on_L3_pass": len(dep), "changed_under_written_rule": len(changed),
                            "changed_by_method": dict(collections.Counter(lab for _, lab, _ in changed)),
                            "changed_by_scenario": dict(collections.Counter(r["scenario"] for r, _, _ in changed)),
                            "changed_to": dict(collections.Counter(x["written"] for _, _, x in changed)),
                            "base_scenario_changed": sum(r["scenario"] == "E2-base" for r, _, _ in changed),
                            "log_deploy_consistent_with_runtime_rule": sum((x["runtime"] == "pass") == deploy_L3(x) for _, _, x in rows if x["kind"] == "approx"),
                            "approx_fields": sum(x["kind"] == "approx" for _, _, x in rows),
                            "written_pass_but_not_recorded_deployable": sum(x["written"] == "pass" and x["kind"] == "approx" and not deploy_L3(x) for _, _, x in rows)}
    # 보조 로그
    aux = {}
    for name in ("e2_perwave", "e2_k5", "e2_leadI", "e2_c2021"):
        logs = [analyse(p) for p in sorted((L.REPO / "results/final" / name).glob("*.json"))]
        ar = [(r, lab, x) for r in logs for lab, x in r["methods"].items()]
        d = [(r, lab, x) for r, lab, x in ar if deploy_L3(x)]
        ch = [(r, lab, x) for r, lab, x in d if x["written"] != "pass"]
        aux[name] = {"logs": len(logs), "recorded_deployable_on_L3_pass": len(d), "changed": len(ch),
                     "changed_by_method": dict(collections.Counter(lab for _, lab, _ in ch)),
                     "class_equals_log_under_runtime_rule": sum(x["runtime"] == x["json_class"] for _, _, x in ar), "fields": len(ar)}
    res["aux"] = aux
    # ---- C-7: 본문에 인용된 예 ----
    def p1ci(sc, seed):
        r = next(r for r in main_logs if r["scenario"] == sc and r["seed"] == seed)
        return [round(r["P1"]["l"], 3), round(r["P1"]["u"], 3), r["P1"]["class"]]
    res["examples"] = {"rate20-mid_s1_P1_ci": p1ci("E2-rate20-mid", 1), "base_s1_P1_ci": p1ci("E2-base", 1)}
    res["base_scenario_rules_identical"] = all(x["runtime"] == x["written"] for r in main_logs if r["scenario"] == "E2-base" for x in r["methods"].values())
    res["U0_rules_identical"] = all(x["runtime"] == x["written"] == x["no_control"] for r, lab, x in rows if x["kind"] == "exact")
    res["max_passes_80"] = {"runtime": max(res["rules_80"]["runtime"].values()), "written": max(res["rules_80"]["written"].values())}
    # ---- 코드 출처(git) ----
    def show(ref):
        src = subprocess.run(["git", "-C", str(L.REPO), "show", f"{ref}:fl/l3.py"], capture_output=True, text=True).stdout
        i = src.index("def power_ok")
        body = src[i:src.index("\ndef ", i + 5)]
        return [ln.strip() for ln in body.splitlines() if ln.strip().startswith("return")]
    git = lambda *a: subprocess.run(["git", "-C", str(L.REPO), *a], capture_output=True, text=True).stdout.strip()
    res["code_provenance"] = {"power_ok_return_at_final-rerun-v1": show("final-rerun-v1"), "power_ok_return_at_HEAD": show("HEAD"),
                              "tag_final-rerun-v1": git("log", "-1", "--format=%H %cI", "final-rerun-v1"),
                              "commit_that_changed_rule": git("log", "-1", "--format=%H %cI %s", "-S", 'p1_l3["class"] == "기준 초과"', "--", "fl/l3.py"),
                              "results_final_e2_commit": git("log", "-1", "--format=%H %cI", "--diff-filter=A", "--", "results/final/e2/E2-base_s1.json"),
                              "results_final_e2_last_change": git("log", "-1", "--format=%H %cI", "--", "results/final/e2")}
    json.dump(res, open(OUT / "l3_rules.json", "w"), indent=1, ensure_ascii=False)
    with open(L.EXP / "logs" / f"c4_verify_l3_{time.strftime('%Y%m%dT%H%M%S')}.log", "w") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "header"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
