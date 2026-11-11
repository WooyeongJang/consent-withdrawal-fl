"""보충 S2 수치 재현: E2 실행 로그의 배포 필드 중 "배포 허용(L3 통과)" 건수와, 양성 대조 규칙을 설계대로
(P1이 *기준 초과*) 다시 적용했을 때 판정 불가로 바뀌는 건수. 사용: python3 scripts/recount_l3_deploy.py results/final
(2026-11-01 반영 검토 권고에 따라 추가; 새 학습·거래 없음.)"""
import glob, json, sys
from collections import Counter
root = sys.argv[1]
APPROX = ["U1", "U2", "U3", "U4", "U4s"]
before, after = Counter(), Counter()
runs = 0
for f in sorted(glob.glob(f"{root}/e2/*_s*.json")):
    r = json.load(open(f)); runs += 1
    p1 = r["methods"]["P1"]["L3"]
    for m in APPROX:
        x = r["methods"].get(m)
        if not x or x.get("deploy") != "배포 허용(L3 통과)":
            continue
        before[m] += 1
        assert p1["ci90"][0] > 0, (f, m)          # run-time rule: P1 lower bound > 0
        if p1["class"] != "기준 초과":              # rule as written: P1 must be 'exceeds'
            after[m] += 1
print("runs", runs)
print("deployable-on-L3-pass (deploy field):", sum(before.values()), dict(before))
print("changed to inconclusive after re-applying rule:", sum(after.values()), dict(after))
