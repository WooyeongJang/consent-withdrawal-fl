"""데이터센터별 학습 기록 조각 만들기(플랫폼 준비 단계, 학습 개시 전).

전체 캐시(prepare_data / prepare_c2021 결과)에서 기관 k의 학습 집합(fold 1–8, 사용 가능 기록)만 떼어
<out>/dc{k}/X.npy, meta.csv로 쓴다. 기록 순서는 전체 캐시의 위치 오름차순이다(단일 프로세스 모사와 같은 순서).
데이터센터 컨테이너는 자기 조각만 읽기 전용으로 연결한다.

  FL_K=3 python -m fl.dist.split --cache /cache/ptbxl_k3 --out /split
"""

import argparse
import csv
import hashlib
import os

import numpy as np

from fl import train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["FL_LEADS"] = ""                    # 조각에는 12유도 전체를 둔다(유도 선택은 데이터센터가 학습 때 한다)
    data = T.Data(a.cache)
    for k in T.inst_keys():
        idx = data.train_idx[k]
        d = os.path.join(a.out, f"dc{k}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "X.npy"), np.asarray(data.X[idx], dtype=np.float32))
        with open(os.path.join(d, "meta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ecg_id", "patient_id"] + T.CATS)
            for i in idx:
                w.writerow([int(data.ecg[i]), int(data.pid[i])] + [int(v) for v in data.Y[i]])
        hs = {n: hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() for n in ("X.npy", "meta.csv")}
        print(f"dc{k}: records {len(idx)}", " ".join(f"{n}={h[:12]}" for n, h in hs.items()))


if __name__ == "__main__":
    main()
