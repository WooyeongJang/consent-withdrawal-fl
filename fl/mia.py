"""L3 잔존 영향 시험의 멤버십 추론 (3장 v1.6 알고리즘 5).

구성원 후보 X(철회 기록)와 같은 분포의 비구성원 Z를 기록별 손실로 구분하는 AUC를 잰다(손실이 낮을수록
구성원으로 판단). 신뢰구간은 X·Z를 각각 복원 추출한 부트스트랩으로 구한다.
알고리즘 5의 "공격 a를 n회 독립 학습·적용"은 학습이 필요 없는 손실 기반 공격 + 부트스트랩으로 단순화했다.
그림자 모델을 쓰는 더 강한 공격은 E2에서 검토한다.
"""

import numpy as np
import torch

from fl import train as T


@torch.no_grad()
def record_loss(model, data, idx, device, norm):
    P = np.clip(T.predict(model, data, idx, device, norm=norm), 1e-7, 1 - 1e-7)
    Y = data.Y[idx]
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean(1)


def mia(model, data, X_idx, Z_idx, device, norm, n_boot=1000, seed=0):
    lx, lz = record_loss(model, data, X_idx, device, norm), record_loss(model, data, Z_idx, device, norm)
    s = np.concatenate([-lx, -lz]); y = np.concatenate([np.ones(len(lx)), np.zeros(len(lz))])
    a = T.auc(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bx, bz = rng.integers(0, len(lx), len(lx)), rng.integers(0, len(lz), len(lz))
        boots.append(T.auc(y, np.concatenate([-lx[bx], -lz[bz]])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": a, "ci": [float(lo), float(hi)], "loss_X": float(lx.mean()), "loss_Z": float(lz.mean())}


def decide(m, theta):
    """알고리즘 5의 4단계: 상한 ≤ θ 통과, 하한 > θ 기준 초과, 그 외 판정 불가(검정력 부족)."""
    lo, hi = m["ci"]
    return "통과" if hi <= theta else ("기준 초과" if lo > theta else "판정 불가")
