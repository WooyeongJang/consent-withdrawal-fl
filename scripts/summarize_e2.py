"""E2 집계: results/e2/<시나리오>_s<시드>.json → results/E2.json과 표 출력 (8.5 통계 규칙).

방법별(시나리오마다): 평가 매크로 AUROC·매크로 F1·LiRA offline AUC·출력 차이(X−Z)의 평균과 95% CI,
L3 통과 비율(윌슨 CI), 처리 시간·전송량. 조건 간 비교: 방법 − U0(같은 시드 대응), Holm 보정(시나리오 안의 비교 전체).
컨테이너 안에서 실행(scipy 필요):  python scripts/summarize_e2.py [results/e2] [results/E2.json]
"""

import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fl.l3 import tost  # noqa: E402
from fl.stats import holm, mean_ci, paired, wilson  # noqa: E402

METHODS = ["P1", "P2", "U0", "U1", "U2", "U3-orig", "U3", "U4", "U4s", "U5",
           "U0-seq", "U1-seq", "U2-seq", "U3-seq", "U4-seq", "U4s-seq",                     # -seq: 차례별 처리(39번)
           "U4-pa", "U4-up", "U4-off", "U4s-off"]                                           # FedOSD 공식 코드 차이 점검(46번)
REF = {"U0": "U0", "U0-seq": "U0-seq"}                                                        # 대응 비교 기준
METRICS = {"test": lambda m: m["test_macro_auc"], "f1": lambda m: m["test_macro_f1"],
           "lira": lambda m: m["lira_offline"]["auc"], "gapXZ": lambda m: m["gap_X"] - m["gap_Z"]}


def exclude_self_ref(r):
    """U0 자기참조 제외(49번, 2026-10-28): 시드 s ≤ nref인 U0는 참조 U0 집합의 s번째 모델과 같은 모델이므로,
    그 U0의 L3를 자기 자신을 뺀 나머지 참조로 다시 계산한다. 같은 모델인지는 LiRA AUC가 같은지로 확인한다."""
    refs = r.get("u0_ref_lira_auc") or []
    for lab in ("U0", "U0-seq", "U5"):                      # U5: 기관 탈퇴 시나리오에서 U0와 같은 모델(2026-11-01, C2)
        x = r["methods"].get(lab)
        if not (x and x.get("L3") and "lira_offline" in x):
            continue
        auc = x["lira_offline"]["auc"]
        if x.get("self_ref") is not None:                    # 실행 때 기록한 인덱스가 있으면 그것을 쓴다(2026-11-01, C2)
            idx = [x["self_ref"]]
        else:                                                # 없으면(최종 재실행 결과) AUC 동일성으로 판단
            idx = [j for j, a in enumerate(refs) if a == auc]
        if idx and len(refs) - 1 >= 2:
            others = [a for j, a in enumerate(refs) if j != idx[0]]
            x["L3"] = {**tost(auc, x["lira_offline"]["ci"], others), "self_ref_excluded": True, "n_ref": len(others)}
    return r


def power_ok(p1_l3):
    """양성 대조(33번 (a)): 원 모델 P1이 같은 시험에서 *기준 초과*로 검출되어야 검정력이 있다.
    기준 초과 = TOST 불통과이면서 90% CI 하한 > 0. 실행 중 기록(fl/l3.py)은 하한 > 0만 보아 P1이 통과여도
    검정력을 인정했으므로(설계 이탈, 2026-10-29 심사관 검토) 집계에서 설계대로 다시 판정한다."""
    return bool(p1_l3) and p1_l3["result"] != "통과" and p1_l3["ci90"][0] > 0


def main(root="results/e2", out_path="results/E2.json"):
    runs = defaultdict(dict)
    for f in sorted(glob.glob(f"{root}/*_s*.json")):
        r = exclude_self_ref(json.load(open(f)))
        runs[r["scenario"]][r["seed"]] = r
    out = {}
    for sc, by_seed in runs.items():
        seeds = sorted(by_seed)
        res = {"seeds": seeds, "methods": {}, "compare_vs_U0": {}}
        for m in METHODS:
            rows = [by_seed[s]["methods"].get(m) for s in seeds]
            rows = [x for x in rows if x and "test_macro_auc" in x]
            if not rows:
                continue
            d = {k: mean_ci([f(x) for x in rows]) for k, f in METRICS.items() if not (m == "P2" and k == "gapXZ")}
            l3 = [x["L3"]["result"] == "통과" for x in rows if x.get("L3")]
            d["L3_pass"] = {"k": sum(l3), "n": len(l3), "wilson95": wilson(sum(l3), len(l3))} if l3 else None
            # 알고리즘 5의 세 결과(44번 (a)): 통과 / 기준 초과(잔존, 90% CI 하한 > 0) /
            # 판정 불가 — 과잉 제거(CI 상한 < 0), 정밀도 부족(CI가 0을 포함), 검정력 부족(33번 (a): 근사 방법의 통과인데
            # 같은 실행의 P1이 기준 초과로 검출되지 않음). 옛 결과(kind 없음)도 같은 규칙으로 다시 분류한다.
            prow = [by_seed[s]["methods"].get("P1", {}).get("L3") for s in seeds]
            prow = [p for p, x in zip(prow, [by_seed[s]["methods"].get(m) for s in seeds]) if x and "test_macro_auc" in x]
            kind = lambda x: x.get("kind") or ("exact" if m in ("U0", "U0-seq", "U5") or (m in ("U1", "U1-seq") and "latejoin" in sc)
                                               else "approx")
            def cls_of(x, p1):
                if x["L3"]["result"] == "통과":
                    return "pass" if m in ("P1", "U3-orig") or kind(x) == "exact" or power_ok(p1) else "lowpower"
                return "residual" if x["L3"]["ci90"][0] > 0 else "overremoval" if x["L3"]["ci90"][1] < 0 else "inconclusive"
            cls = [cls_of(x, p1) for x, p1 in zip(rows, prow) if x.get("L3")]
            d["L3_class"] = {c: cls.count(c) for c in ("pass", "residual", "inconclusive", "overremoval", "lowpower")} if cls else None
            # 배포 전이(44번 (a))도 위 판정으로 다시 계산한다(실행 중 기록의 deploy는 옛 양성 대조 규칙을 썼다)
            dep = ["배포 허용(L1·L2)" if kind(x) == "exact" else
                   "배포 허용(L3 통과)" if x.get("L3") and cls_of(x, p1) == "pass" else "미완료 → 정확 방법 전환"
                   for x, p1 in zip(rows, prow) if x.get("deploy")]
            d["deploy"] = {v: dep.count(v) for v in sorted(set(dep))} if dep else None
            d["wall_s"] = mean_ci([x["cost"].get("wall_s", 0) for x in rows])
            d["bytes"] = mean_ci([x["cost"].get("bytes", 0) for x in rows])
            if "subgroup" in rows[0]:
                d["subgroup"] = {g: mean_ci([x["subgroup"][g]["macro_auc"] for x in rows]) for g in rows[0]["subgroup"]}
            res["methods"][m] = d
        # 대응 비교: 방법 − U0(차례별 처리 결과는 − U0-seq)
        comps, keys = [], []
        for m in METHODS:
            ref = "U0-seq" if m.endswith("-seq") else "U0"
            if m in REF or m not in res["methods"] or ref not in res["methods"]:
                continue
            for k in ("test", "lira"):
                a = [METRICS[k](by_seed[s]["methods"][m]) for s in seeds if "test_macro_auc" in by_seed[s]["methods"].get(m, {})]
                b = [METRICS[k](by_seed[s]["methods"][ref]) for s in seeds if "test_macro_auc" in by_seed[s]["methods"].get(m, {})]
                comps.append(paired(a, b)); keys.append(f"{m}:{k}")
        adj = holm([c.get("p") for c in comps])
        for key, c, p in zip(keys, comps, adj):
            c["p_holm"] = p
            res["compare_vs_U0"][key] = c
        out[sc] = res
        print(f"\n## {sc} (n={len(seeds)})")
        print("| 방법 | 평가 AUROC | 매크로 F1 | LiRA AUC | L3 통과/잔존/정밀도부족/과잉제거/검정력부족 | 처리 시간(초) |")
        print("|---|---|---|---|---|---|")
        for m, d in res["methods"].items():
            f = lambda x: f"{x['mean']:.4f} [{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]" if x["ci"][0] is not None else f"{x['mean']:.4f}"
            c = d["L3_class"]
            l3 = f"{c['pass']}/{c['residual']}/{c['inconclusive']}/{c['overremoval']}/{c['lowpower']}" if c else "—"
            print(f"| {m} | {f(d['test'])} | {f(d['f1'])} | {f(d['lira'])} | {l3} | {d['wall_s']['mean']:.0f} |")
        sig = [f"{k} Δ={c['diff']['mean']:+.4f} p_holm={c['p_holm']:.3g} ({c['test']})" for k, c in res["compare_vs_U0"].items()
               if c.get("p_holm") is not None and c["p_holm"] < 0.05]
        print("유의한 차이(방법 − U0 또는 U0-seq, Holm):", "; ".join(sig) if sig else "없음")
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:3])
