"""Challenge 2021 보조 데이터 캐시 (3장 8.1 보조 데이터, 코호트 = 기관).

- 코호트: ningbo, georgia, chapman_shaoxing, cpsc_2018 + cpsc_2018_extra(한 기관으로 병합, 3장 반영 13번) → 기관 4곳(0–3)
- 신호: 500Hz → 100Hz(폴리페이즈 리샘플링), 앞 10초(1,000표본). 10초보다 짧은 기록은 제외하고 목록을 남긴다.
- 레이블: c2021/mapping_superclass_v1.csv(초안 B, 형태 문장 포함 — 3장 반영 41번 결정 2026-10-27).
  비교용으로 초안 A(v0, 보수적) 레이블을 meta_A.csv로 함께 쓴다(학습에는 쓰지 않음).
  NORM = NSR(426783006) 외 코드가 없는 기록. 상위 범주가 없는 기록은 has_label=False.
- 나이·성별은 헤더의 Age·Sex를 meta.csv에 넣는다(결측은 빈 값). 동의 사건 생성기와 E2가 PTB-XL 표 대신 이 열을 쓴다.
- 철회 단위는 기록이다(환자 식별자 없음). patient_id 열에는 기록 번호를 넣어 PTB-XL 파이프라인을 그대로 쓴다.
- 체크섬 제외: c2021/excluded_checksum.csv의 기록(배포 SHA256과 다른 서버 파일)은 읽지 않는다(2026-10-27 사용자 결정).
- 무효 표본 제외: 앞 10초 안에 WFDB 무효 값(16비트 형식 −32768, wfdb가 NaN으로 읽음)이 하나라도 있는 기록은 제외하고
  excluded_invalid.csv에 남긴다(2026-10-27 사용자 결정, 커밋본 c2021/excluded_invalid_samples.csv).
- 분할: 기록 이름의 SHA-256으로 fold 1–10(1–8 학습, 9 검증, 10 평가)을 정한다(PTB-XL strat_fold와 같은 의미, 층화는 안 함).

  .venv/bin/python fl/prepare_c2021.py --root ~/data/physionet.org/files/challenge-2021/1.0.3/training --out ~/data/cache/c2021
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

COHORTS = [("ningbo", "0"), ("georgia", "1"), ("chapman_shaoxing", "2"), ("cpsc_2018", "3"), ("cpsc_2018_extra", "3")]
CATS = ["NORM", "MI", "STTC", "CD", "HYP"]
NSR = "426783006"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def labels(codes, m):
    y = {c: 0 for c in CATS}
    for c in codes:
        if m.get(c):
            y[m[c]] = 1
    if set(codes) == {NSR}:
        y["NORM"] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, default=Path("c2021/mapping_superclass_v1.csv"))
    ap.add_argument("--mapping-a", type=Path, default=Path("c2021/mapping_superclass_v0.csv"))
    ap.add_argument("--exclude", type=Path, default=Path("c2021/excluded_checksum.csv"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    load = lambda p: {r["snomed"]: r["superclass"] for r in csv.DictReader(open(p))}
    mapping, mapping_a = load(a.mapping), load(a.mapping_a)
    skip = {(r["cohort"], r["record"]) for r in csv.DictReader(open(a.exclude))} if a.exclude.exists() else set()
    heads = sorted((coh, inst, h) for coh, inst in COHORTS for h in (a.root / coh).glob("*/*.hea") if (coh, h.stem) not in skip)
    print("excluded(checksum)", len(skip))
    X = np.lib.format.open_memmap(a.out / "c2021_100hz.npy", mode="w+", dtype=np.float32, shape=(len(heads), 12, 1000))
    meta, metaA, short, invalid, n = [], [], [], [], 0
    for coh, inst, h in heads:
        rec = wfdb.rdheader(str(h.with_suffix("")))
        if rec.fs != 500 or rec.sig_len < 10 * rec.fs:
            short.append((coh, h.stem, rec.fs, rec.sig_len))
            continue
        codes = next((ln.split(":", 1)[1].strip().split(",") for ln in rec.comments if ln.startswith("Dx")), [])
        sig, _ = wfdb.rdsamp(str(h.with_suffix("")), sampto=10 * rec.fs)
        if not np.isfinite(sig).all():                     # 원 기록의 무효 표본(결측)
            bad = ~np.isfinite(sig)
            invalid.append((coh, h.stem, int(bad.sum()), ",".join(rec.sig_name[j] for j in np.where(bad.any(0))[0])))
            continue
        order = [rec.sig_name.index(l) for l in LEADS]
        X[n] = resample_poly(sig[:, order], 1, 5, axis=0).T.astype(np.float32)
        fold = int(hashlib.sha256(f"c2021|{coh}|{h.stem}".encode()).hexdigest()[:8], 16) % 10 + 1
        y, ya = labels(codes, mapping), labels(codes, mapping_a)
        hdr = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip() for ln in rec.comments if ":" in ln}
        age = hdr.get("Age", "")
        age = age if age.replace(".", "", 1).isdigit() else ""                          # NaN·빈 값은 결측
        sex = {"Male": 0, "Female": 1}.get(hdr.get("Sex", ""), "")                      # PTB-XL과 같은 부호(0=남, 1=여)
        base = {"ecg_id": n, "patient_id": n, "strat_fold": fold, "institution": inst, "cohort": coh, "record": h.stem,
                "age": age, "sex": sex}
        meta.append({**base, **y, "has_label": any(y.values()), "excluded": False, "dx": ",".join(codes)})
        metaA.append({**base, **ya, "has_label": any(ya.values()), "excluded": False})
        n += 1
    X.flush()
    if n < len(heads):                                     # 제외한 만큼 잘라 다시 저장
        Y = np.array(X[:n]); del X
        np.save(a.out / "c2021_100hz.npy", Y)
    for name, rows in (("meta.csv", meta), ("meta_A.csv", metaA)):
        with open(a.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(a.out / "excluded_short.csv", "w", newline="") as f:
        csv.writer(f).writerows([("cohort", "record", "fs", "sig_len")] + short)
    with open(a.out / "excluded_invalid.csv", "w", newline="") as f:
        csv.writer(f).writerows([("cohort", "record", "n_invalid_samples", "leads")] + invalid)
    print("records", n, "excluded(short/fs)", len(short), "excluded(invalid samples)", len(invalid))
    for name in ("c2021_100hz.npy", "meta.csv", "meta_A.csv", "excluded_short.csv", "excluded_invalid.csv"):
        print(hashlib.sha256((a.out / name).read_bytes()).hexdigest(), name)


if __name__ == "__main__":
    main()
