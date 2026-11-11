"""E3·E3-A 집계: results/e3/A_seed*_p*.json, results/e3/B_seed*.json → results/E3.json과 표 출력.
  python scripts/summarize_e3.py [results/e3] [results/E3.json]

E3-A 구성요소(3장 8.4): (a) 정보주체 측 원장 직접 등록 (b) 집계 수락 시점 비소속 검사 (c) 재실행 감사
(d) 입력 집합 검사 (e) 잔존 영향 시험 (f) 단말 배포 모델 대조.
추가로 구분하는 증거: (s) 정보주체 자기 비소속 검증(5.4), (g) 원장 계보 기록(출력 해시의 계보 중복),
(L) 원장 자체(상태 전이 규칙·교차 대조·불변성). (s)는 L2의 정보주체 측 검증이며 (d)와 따로 센다.
"""

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

STRUCTS = ["B0", "B1", "B2", "B3", "B4", "B5"]
COMPONENTS = ["a", "b", "c", "d", "e", "f", "s", "g", "L"]
COMP_NAME = {"a": "정보주체 원장 직접 등록", "b": "수락 시점 비소속 검사", "c": "재실행 감사(L1)",
             "d": "입력 집합 검사(L2, 감사자)", "e": "잔존 영향 시험(L3)", "f": "단말 대조(L4)",
             "s": "정보주체 자기 비소속 검증", "g": "원장 계보 기록", "L": "원장 상태 전이·교차 대조·불변성"}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def b6_detects(attack, ev, type_a):
    """B6 Fdhila형(57번, 2026-10-29): 커밋 대비 학습 단위 표본 재실행 + 변경 불가 원장만 있는 감사 구조.
    정보주체 등록·수락 비소속·L2·L3·L4·자기검증·계보 검사·처리 작업 재실행·방법 승인이 없다.
    → 유형 A에서는 T2b의 (c)만 인정한다. T2a·T3·T10의 (c)는 감사자가 동의 정책으로 입력 집합을 재구성해서
    생기는 불일치(fl/audit.py reexecute)라 동의 정보가 없는 감사자에게는 일어나지 않는다. T2b는 커밋과 실제 계산이
    달라 커밋 대비 재실행만으로 드러난다. T9(기록 누락·이중 제시·사후 수정)에서는 (L)만 인정한다."""
    if type_a:
        return attack == "T2b" and "c" in ev
    return attack == "T9" and "L" in ev


def evidence_A(x, p):
    e = x["evidence"]
    return {c for c, v in (("a", e["a"]), ("b", e["b"]), ("s", e["s"]), ("c", e["c"].get(str(p), False))) if v}


def evidence_B(x):
    a = x["attack"]
    if a in ("T4", "T5", "T6", "T7", "T11"):
        ev = set()
        if not x["L1"]:
            ev.add("c")
        if not x["L2"]:
            ev.add("d")
        if x.get("L3") and x["L3"]["result"] != "통과":
            ev.add("e")
        if x.get("lineage_duplicate"):
            ev.add("g")
        if x.get("ledger_refused"):                 # T11f1: 원장이 미승인 (방법, prog) 작업 생성을 거부
            ev.add("L")
        return ev
    if a == "T8":
        return {"f"} if not x["device_accepts"] else set()
    if a == "T9":
        return {"L"} if x["detected"]["B4"] else set()
    return set()


def main(root="results/e3", out_path="results/E3.json", p_main=0.3):
    A = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/A_seed*_p{p_main}.json"))]
    B = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/B_seed*.json"))]
    det = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_seed = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # 유형 B: 공격 → 학습 시드 → B4 [탐지, n](28번 (c), 시드 단위 묶음)
    evid = defaultdict(list)
    pgrid = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cf = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # T11 비교 판정(L3 단독, 허용 오차 재현 검증)
    dists, eps_v = defaultdict(list), None
    rdev, grind = defaultdict(list), defaultdict(list)     # T11c·d 라운드 단위 최대 편차, T11e 시드 고르기 이득
    fp = {"A_units_rejected_honest": 0, "A_units_l1_fail_honest": 0, "A_subject_check_fail_honest": 0,
          "A_l1_sampled_honest_units": 0, "B_honest_job_blocked": 0, "B_honest_device_rejected": 0, "B_training_l1_fail": 0}
    for r in A:
        for x in r["instances"]:
            for s in STRUCTS:
                det[x["attack"]][s][0] += bool(x["detected"].get(s))
                det[x["attack"]][s][1] += 1
            evid[x["attack"]].append(evidence_A(x, p_main))
            det[x["attack"]]["B6"][0] += b6_detects(x["attack"], evid[x["attack"]][-1], True)
            det[x["attack"]]["B6"][1] += 1
            for pp, v in x["evidence"]["c"].items():
                if x["attack"] in ("T2b", "T10"):
                    pgrid[x["attack"]][pp][0] += bool(v)
                    pgrid[x["attack"]][pp][1] += 1
        f = r["false_positives"]
        fp["A_units_rejected_honest"] += len(f["rejected_honest"])
        fp["A_units_l1_fail_honest"] += len(f["l1_fail_honest"])
        fp["A_subject_check_fail_honest"] += len(f["subject_check_fail_honest"])
        fp["A_l1_sampled_honest_units"] += r["l1_sampled_units"]
    for r in B:
        for x in r["instances"]:
            if x["attack"] == "honest":
                continue
            key = f"{x['attack']}{x['variant']}" if x["attack"] in ("T9", "T11") else x["attack"]
            for s in STRUCTS:
                det[key][s][0] += bool(x["detected"].get(s))
                det[key][s][1] += 1
            by_seed[key][r["seed"]][0] += bool(x["detected"].get("B4"))
            by_seed[key][r["seed"]][1] += 1
            evid[key].append(evidence_B(x))
            det[key]["B6"][0] += b6_detects(x["attack"], evid[key][-1], False)
            det[key]["B6"][1] += 1
            if x["attack"] == "T11":
                for c, v in x["counterfactual"].items():
                    cf[key][c][0] += bool(v)
                    cf[key][c][1] += 1
                cf[key]["B4"][0] += bool(x["detected"]["B4"])
                cf[key]["B4"][1] += 1
                dists[key].append(x["rel_dist_to_U0"])
                eps_v = x["eps_end"]
                if x.get("round_dev"):
                    rdev[key].append({vk: max(d_.values()) for vk, d_ in x["round_dev"].items()})
                if x["variant"] == "e":
                    grind[key].append({"dist_to_P1": x["dist_to_P1"], "u0_dist_to_P1": x["u0_dist_to_P1"],
                                       "lira": (x.get("L3") or {}).get("auc")})
        f = r["false_positives"]
        fp["B_honest_job_blocked"] += int(not f["honest_job_deployable"])
        fp["B_honest_device_rejected"] += int(not f["honest_device_accepts"])
        fp["B_training_l1_fail"] += f["training_L1_fail"]

    order = ["T1", "T2a", "T2b", "T3", "T4", "T5", "T6", "T7", "T8", "T9a", "T9b", "T9c", "T10",
             "T11a", "T11b", "T11c", "T11d", "T11e", "T11f"]
    table = {}
    print("| 공격 | n | " + " | ".join(STRUCTS + ["B6"]) + " |")
    print("|---|---:|" + "---|" * (len(STRUCTS) + 1))
    for a in [o for o in order if o in det]:
        row = {s: {"k": det[a][s][0], "n": det[a][s][1], "rate": det[a][s][0] / det[a][s][1],
                   "wilson95": wilson(det[a][s][0], det[a][s][1])} for s in STRUCTS + ["B6"]}
        if a in by_seed and len(by_seed[a]) > 1:
            # 시드 단위 묶음: 시드별 탐지율과 그 평균·범위(독립 단위 = 학습 시드). 합동 윌슨 CI는 시드 내 주입을 독립으로 본 값이다
            rates = [v[0] / v[1] for _, v in sorted(by_seed[a].items())]
            row["B4"]["by_seed"] = {str(k): v for k, v in sorted(by_seed[a].items())}
            row["B4"]["seed_rate"] = {"mean": sum(rates) / len(rates), "min": min(rates), "max": max(rates),
                                      "n_seeds": len(rates)}
        table[a] = row
        print(f"| {a} | {row['B4']['n']} | " + " | ".join(f"{row[s]['k']}/{row[s]['n']}" for s in STRUCTS + ["B6"]) + " |")

    # E3-A: 구성요소 하나를 뺐을 때 B4의 탐지율
    ablation = {}
    for a, evs in evid.items():
        base = sum(bool(e) for e in evs)
        ablation[a] = {"B4": base / len(evs), "n": len(evs),
                       **{c: sum(bool(e - {c}) for e in evs) / len(evs) for c in COMPONENTS}}
    # 쌍 제거(30번 (a)): 서로 대체하는 증거 (b) 수락 시점 비소속 검사 + (s) 정보주체 자기 검증, (c) L1 + (g) 계보
    PAIRS = {"b+s": {"b", "s"}, "c+g": {"c", "g"}}
    for a, evs in evid.items():
        ablation[a]["pairs"] = {k: sum(bool(e - v) for e in evs) / len(evs) for k, v in PAIRS.items()}
    print("\nE3-A 쌍 제거:", {a: ablation[a]["pairs"] for a in [o for o in order if o in ablation]})
    print("\nE3-A(구성요소 제거 시 B4 탐지율):")
    print("| 공격 | B4 | " + " | ".join(f"−{c}" for c in COMPONENTS) + " |")
    for a in [o for o in order if o in ablation]:
        r = ablation[a]
        print(f"| {a} | {r['B4']:.2f} | " + " | ".join(f"{r[c]:.2f}" for c in COMPONENTS) + " |")

    pg = {a: {pp: {"k": v[0], "n": v[1], "rate": v[0] / v[1], "theory_m1": float(pp), "wilson95": wilson(v[0], v[1])}
              for pp, v in sorted(d.items(), key=lambda kv: float(kv[0]))} for a, d in pgrid.items()}
    print("\n표본 비율 p별 재실행 탐지율(공격 단위 1개, 이론값 1−(1−p)^1 = p):")
    for a, d in pg.items():
        print(a, {pp: f"{v['k']}/{v['n']} (이론 {v['theory_m1']:.1f})" for pp, v in d.items()})
    t11 = {a: {c: {"k": v[0], "n": v[1], "rate": v[0] / v[1], "wilson95": wilson(v[0], v[1])} for c, v in d.items()}
           | {"rel_dist_to_U0": {"min": min(dists[a]), "max": max(dists[a])}}
           | ({"round_dev_max": {vk: {"min": min(r_[vk] for r_ in rdev[a]), "max": max(r_[vk] for r_ in rdev[a])}
                                 for vk in rdev[a][0]}} if rdev.get(a) else {})
           | ({"grinding": grind[a]} if grind.get(a) else {}) for a, d in cf.items()}
    if t11:
        print(f"\nT11 비교 판정(ε = {eps_v:.4g}):")
        for a, d in t11.items():
            print(a, {c: f"{v['k']}/{v['n']}" for c, v in d.items() if "k" in v}, "거리", d["rel_dist_to_U0"],
                  *(["라운드 편차", d["round_dev_max"]] if "round_dev_max" in d else []),
                  *(["시드 고르기", len(d["grinding"]), "건"] if "grinding" in d else []))
    print("\n오탐:", fp)
    out = {"runs": {"A": len(A), "B": len(B)}, "p_main": p_main, "detection": table, "ablation": ablation,
           "component_names": COMP_NAME, "p_grid": pg, "t11": t11, "t11_eps": eps_v, "false_positives": fp,
           "fees_kaia": {"A": sum(r["fee_kaia"] for r in A), "B": sum(r["fee_kaia"] for r in B)},
           "tx": {"A": sum(r["tx_total"] for r in A), "B": sum(r["tx_total"] for r in B)},
           "contracts": {"A": [r["contract"] for r in A], "B": [r["contract"] for r in B]}}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
