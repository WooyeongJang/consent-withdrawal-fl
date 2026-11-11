"""E0 결과 집계: results/e0/*.json → results/E0.json (시연 화면의 '실험 결과' 탭이 읽는다)과 표 출력."""

import glob
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/e0")
SUMMARY = Path(sys.argv[2] if len(sys.argv) > 2 else "results/E0.json")


def main():
    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(OUT / "*.json")))]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["mode"], r["cond"] + r.get("tag", ""))].append(r)

    summary = {"groups": [], "pilot": None}
    print("| 모드 | 조건 | 실행 수 | 서로 다른 해시 수 | 최빈 해시 일치율 | 평균 시간(초) |")
    print("|---|---|---:|---:|---:|---:|")
    for (mode, cond), rs in sorted(groups.items()):
        if mode == "pilot":
            continue
        hs = Counter(r["hash"] for r in rs)
        top, n = hs.most_common(1)[0]
        secs = [r.get("seconds") or sum(r.get("round_seconds", [])) for r in rs]
        g = {"mode": mode, "cond": cond, "runs": len(rs), "distinct_hashes": len(hs),
             "agreement": n / len(rs), "modal_hash": top, "mean_seconds": statistics.mean(secs),
             "torch": rs[0]["env"]["torch"]}
        if mode == "chain":
            # 라운드별로 실행 간 해시가 같은지
            g["rounds_identical"] = [len({r["round_hashes"][i] for r in rs}) == 1 for i in range(len(rs[0]["round_hashes"]))]
        if len(hs) > 1 and mode == "unit":
            # 처음 갈라지는 텐서(state_dict 순서)
            a, b = rs[0]["per_tensor"], next(r for r in rs if r["hash"] != rs[0]["hash"])["per_tensor"]
            g["differing_tensors"] = [k for k in a if a[k] != b[k]]
        summary["groups"].append(g)
        print(f"| {mode} | {cond} | {len(rs)} | {len(hs)} | {n / len(rs):.0%} | {g['mean_seconds']:.2f} |")

    # 조건 간 비교(같은 시드, 같은 초기값)
    modal = {(g["mode"], g["cond"]): g["modal_hash"] for g in summary["groups"]}
    pairs = [("gpu_det", "cpu"), ("gpu_det", "gpu_det_pipvenv"), ("cpu", "cpu_t4")]
    summary["cross"] = {f"{a} vs {b}": modal.get(("unit", a)) == modal.get(("unit", b))
                        for a, b in pairs if ("unit", a) in modal and ("unit", b) in modal}
    print("\n조건 간 해시 일치:", summary["cross"])

    summary["pilot"] = {}
    for tag, label in (("gpu_det", "초기값 고정"), ("gpu_det_vinit", "초기값도 시드별")):
        pil = groups.get(("pilot", tag), [])
        if not pil:
            continue
        aucs = [r["test_macro_auc"] for r in pil]
        m, sd = statistics.mean(aucs), statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(aucs), 2.0)
        half = t * sd / math.sqrt(len(aucs))
        per = [statistics.mean(r["test_auc_per_class"][j] for r in pil) for j in range(5)]
        summary["pilot"][tag] = {
            "label": label, "seeds": [r["base_seed"] for r in pil], "test_macro_auc": aucs, "mean": m, "sd": sd,
            "ci95": [m - half, m + half], "per_class_mean": dict(zip(["NORM", "MI", "STTC", "CD", "HYP"], per)),
            "val_curve": {r["base_seed"]: r["val_auc"] for r in pil},
            "seconds_per_run": statistics.mean(sum(r["round_seconds"]) for r in pil),
            "benchmark_reference": 0.930}
        s = summary["pilot"][tag]
        print(f"\n예비 측정[{label}](R=50, 시드 {len(aucs)}개) 평가 macro AUC: 평균 {m:.4f}, SD {sd:.4f}, "
              f"95% CI [{m - half:.4f}, {m + half:.4f}], 1회 {s['seconds_per_run']:.0f}초")
        print("범주별 평균:", {k: round(v, 4) for k, v in s["per_class_mean"].items()})

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
