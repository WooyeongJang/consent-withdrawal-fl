"""E2 기준 시나리오 단독 재측정 확인: 방법별 모델 해시가 최종 E2와 같은지 보고, 처리 시간을 비교한다.

python3 scripts/check_e2_timing.py [results/final]
"""
import glob
import json
import os
import statistics as st
import sys


def main(root="results/final"):
    a = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(f"{root}/e2/E2-base_s*.json"))}
    b = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(f"{root}/e2_timing/E2-base_s*.json"))}
    assert b, "e2_timing 결과가 없다"
    mismatch, wall = [], {}
    for name, rb in b.items():
        ra = a[name]
        for m, xb in rb["methods"].items():
            xa = ra["methods"].get(m)
            if xa is None or xa.get("hash") != xb.get("hash"):
                mismatch.append((name, m))
            if xb.get("cost") and "wall_s" in xb["cost"]:
                wall.setdefault(m, ([], []))
                wall[m][0].append(xa["cost"]["wall_s"])
                wall[m][1].append(xb["cost"]["wall_s"])
    print(f"seeds {len(b)}/10, hash mismatch {len(mismatch)}", mismatch[:5])
    print("| 방법 | 동시 실행(초) | 단독(초) |\n|---|---|---|")
    for m, (x, y) in wall.items():
        print(f"| {m} | {st.mean(x):.0f} | {st.mean(y):.0f} |")
    out = {m: {"concurrent": x, "alone": y} for m, (x, y) in wall.items()}
    json.dump({"hash_mismatch": mismatch, "wall_s": out}, open(f"{root}/E2_timing.json", "w"), indent=1)
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
