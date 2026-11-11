"""U3: 기관 내부 분할 학습에 의한 부분 재학습(SISA, Bourtoule et al. 2021의 연합 적용, 재구현).

- 각 기관의 학습 환자를 해시로 S개 조각에 나눈다(환자 단위, 같은 환자의 기록은 같은 조각).
- 조각 s마다 구성 모델 하나를 기관 간 연합학습(FedAvg, 같은 라운드 수)으로 학습한다(S개의 독립 연합).
- 예측은 구성 모델 출력 확률의 평균이다(원 논문의 다수결 대신, AUROC 계산을 위해 확률 평균).
- 철회가 생기면 철회자가 속한 조각의 구성 모델만 처음부터 다시 학습한다. 조각 내부 슬라이스(체크포인트)는 쓰지 않는다.
원 논문의 공개 구현은 단일 서버 이미지 분류용이므로, 연합 구조에 맞게 재구현했다(3장 8.3 "재구현 명시").
"""

import hashlib
import os

import torch
import torch.nn as nn

from fl import train as T
from fl.run import Policy, run_fl, state_cpu


def shard_of(pid, S):
    return int(hashlib.sha256(f"sisa|{pid}".encode()).hexdigest()[:8], 16) % S


class ShardPolicy(Policy):
    def __init__(self, base, s, S):
        super().__init__(base.enroll, base.withdraw, base.version, base.reconsent)
        self.s, self.S = s, S

    def valid(self, pid, r):
        return shard_of(pid, self.S) == self.s and super().valid(pid, r)


class Ensemble(nn.Module):
    """forward는 구성 모델 확률 평균의 logit을 돌려준다(train.predict·exp_lira.stat과 호환)."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        p = torch.stack([torch.sigmoid(m(x)) for m in self.models]).mean(0).clamp(1e-7, 1 - 1e-7)
        return torch.log(p / (1 - p))


def train(data, store, policy, S, base_seed, device, max_pool, rounds, norm, run_dir, only=None, prev=None):
    """구성 모델 S개를 학습한다. only가 주어지면 그 조각만 다시 학습하고 나머지는 prev(상태 목록)를 재사용한다."""
    states, runs = [], {}
    for s in range(S):
        if only is not None and s not in only:
            states.append(prev[s])
            continue
        o = run_fl(data, store, ShardPolicy(policy, s, S), base_seed=base_seed * 100 + s, device=device,
                   max_pool=max_pool, cond="gpu_det", rounds=rounds, norm=norm,
                   run_dir=os.path.join(run_dir, f"shard{s}"), label=f"sisa{s}", log_every=0)
        states.append(state_cpu(o["model"]))
        runs[s] = o
    models = []
    for st in states:
        m = T.new_model(max_pool, device, 0)
        m.load_state_dict(st)
        models.append(m)
    return Ensemble(models).to(device), states, runs
