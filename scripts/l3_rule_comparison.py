#!/usr/bin/env python3
"""보충 S7·Table S9 재현: E2 실행 로그에서 근사 방법(U1–U4s)의 L3 통과 수를 세 규칙으로 센다.

- no control   : 양성 대조 없이 TOST 통과(실행 로그의 class '통과' 또는 '판정 불가/검정력 부족')
- run-time rule: 실행 당시 코드(fl/l3.py, 2026-10-27~30)의 대조 — P1의 90% CI 하한 > 0 이면 인정
- rule as written: Algorithm S5 — P1이 '기준 초과'(TOST 불통과 ∧ 하한 > 0)여야 인정(scripts/summarize_e2.py와 동일)

실행 로그의 deploy 필드 중 "배포 허용(L3 통과)" 가운데 규칙 정렬로 판정 불가가 되는 수(53/269)도 함께 출력한다
(scripts/recount_l3_deploy.py와 같은 계산). 새 학습·거래 없음. 사용: python3 -I scripts/l3_rule_comparison.py results/final
"""
import glob
import json
import sys
from collections import Counter, defaultdict

APPROX = ["U1", "U2", "U3", "U4", "U4s"]
EARLY_MID = {"E2-early", "E2-mid", "E2-rate10-early", "E2-rate10-mid", "E2-rate20-early", "E2-rate20-mid"}
NAMES = {"E2-rate20-mid": "20% mid", "E2-rate10-mid": "10% mid", "E2-exit2": "institution exit", "E2-seq3": "three waves"}


def main(root: str) -> None:
    per = defaultdict(lambda: defaultdict(Counter))
    changed = Counter()
    deployable = 0
    for f in sorted(glob.glob(f"{root}/e2/*_s*.json")):
        r = json.load(open(f))
        sc = r.get("scenario") or r.get("name")
        p1 = r["methods"]["P1"]["L3"]
        runtime_ok = p1["ci90"][0] > 0
        written_ok = p1["class"] == "기준 초과"
        for m in APPROX:
            x = r["methods"].get(m)
            if not x:
                continue
            l3 = x["L3"]
            raw_pass = l3["class"] == "통과" or l3.get("reason") == "검정력 부족"
            if raw_pass:
                per[sc][m]["none"] += 1
                per[sc][m]["runtime"] += int(runtime_ok)
                per[sc][m]["written"] += int(written_ok)
            if x.get("deploy") == "배포 허용(L3 통과)":
                deployable += 1
                if not written_ok:
                    changed[m] += 1
    print("early/mid grid (80 runs): method  none  run-time  as-written")
    for m in APPROX:
        t = Counter()
        for sc in EARLY_MID:
            t.update(per[sc][m])
        print(f"  {m:4s} {t['none']:4d} {t['runtime']:6d} {t['written']:8d}")
    print("scenarios whose counts differ between the two control rules (run-time -> as-written):")
    for sc in sorted(per):
        diff = {m: (per[sc][m]["runtime"], per[sc][m]["written"]) for m in APPROX if per[sc][m]["runtime"] != per[sc][m]["written"]}
        if diff:
            print(f"  {NAMES.get(sc, sc):18s}", "  ".join(f"{m} {a}->{b}" for m, (a, b) in diff.items()))
    print(f"run-log fields deployable on an L3 pass: {deployable}; changed to inconclusive by the rule as written: {sum(changed.values())} {dict(changed)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/final")
