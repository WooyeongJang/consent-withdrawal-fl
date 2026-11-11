"""PTB-XL 100Hz 신호와 메타데이터를 학습용 캐시로 만든다(호스트에서 한 번 실행, wfdb 필요).

출력(--out):
  ptbxl_100hz.npy  float32 (N, 12, 1000), ecg_id 오름차순
  meta.csv         ecg_id, patient_id, strat_fold, institution(--k, 기본 3), NORM..HYP, has_label, excluded
제외 규칙(3장 v1.6 8.1, 2026-10-25 결정):
  - 환자의 어느 기록에도 site가 없는 학습 기록(site_check/excluded_records.csv)
  - 상위 범주 레이블이 없는 기록(벤치마크 방식) — 학습·검증·평가 모두
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_ptbxl_sites import SUPERCLASSES, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--site-check", type=Path, default=Path("site_check"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3, help="기관 수(site_check/patient_map_K{k}.csv)")
    ap.add_argument("--signals", type=Path, help="이미 만든 ptbxl_100hz.npy(같은 기록 순서)를 링크로 재사용")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    db = load(args.root).sort_index()
    pm = pd.read_csv(args.site_check / f"patient_map_K{args.k}.csv")
    excl = set(pd.read_csv(args.site_check / "excluded_records.csv").ecg_id)
    meta = pd.DataFrame({
        "ecg_id": db.index,
        "patient_id": db.patient_id.values,
        "strat_fold": db.strat_fold.values,
        "institution": db.patient_id.map(dict(zip(pm.patient_id, pm.institution.astype(str)))).fillna("none").values,
        **{c: db[c].values for c in SUPERCLASSES},
        "has_label": (db.NONE == 0).values,
        "excluded": db.index.isin(excl),
    })
    if args.signals:
        dst = args.out / "ptbxl_100hz.npy"
        if not dst.exists():
            dst.symlink_to(os.path.relpath(args.signals.resolve(), args.out.resolve()))   # 상대 링크(컨테이너에서 캐시 상위 디렉터리를 연결해 사용)
    else:
        X = np.empty((len(db), 12, 1000), dtype=np.float32)
        for i, f in enumerate(db.filename_lr):
            sig, _ = wfdb.rdsamp(str(args.root / f))
            X[i] = sig.T
        np.save(args.out / "ptbxl_100hz.npy", X)
    meta.to_csv(args.out / "meta.csv", index=False)
    for name in ("ptbxl_100hz.npy", "meta.csv"):
        h = hashlib.sha256((args.out / name).read_bytes()).hexdigest()
        print(f"{h}  {name}")
    print("records", len(meta), "train usable", int(((meta.strat_fold <= 8) & meta.has_label & ~meta.excluded).sum()))


if __name__ == "__main__":
    main()
