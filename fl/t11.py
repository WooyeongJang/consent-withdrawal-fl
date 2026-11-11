"""T11 적응형 위조 공격(2026-10-26 사용자 승인, 3장 반영 31번).

운영자가 프로토콜(L1·L2·L3와 감사자의 그림자 모델)을 알고, 선언한 처리(U0 재학습)를 하지 않은 채
L2(정직한 입력 커밋 제시)와 L3(잔존 영향 동등성)를 통과하도록 만든 모델을 처리 결과로 제출한다.

  T11a  L3 표적 미세조정: 원 모델(P1)을 철회자를 뺀 자료로 한 라운드씩 더 학습하며, 감사자와 같은
        그림자 모델로 스스로 L3를 시험해 통과하는 즉시 제출(Thudi et al. 2022의 "모델만으로는 증명 불가",
        Zhang et al. 2024의 검증 회피에 대응하는 적응형 공격자).
  T11b  근사 재현 위조: U0를 정직하게 재학습하다가 마지막 m 라운드에만 철회자 일부를 다시 넣는다.
        결과는 정직한 U0와 매우 가까워 허용 오차(ε>0) 재현 검증은 통과할 수 있으나 비트 단위 재실행(ε=0)은 통과하지 못한다
        (Zhang et al. 2024의 재현 검증 위조가 ε>0에서만 성립한다는 점의 실측).

허용 오차 ε는 같은 U0를 비결정 GPU 모드로 두 번 실행한 모델 사이의 상대 거리로 보정한다(정직한 비결정 재실행을 받아들이는 데 필요한 최소 허용치).
"""

import os

import numpy as np
import torch

from fl import train as T
from fl.run import INSTS, Policy, run_fl, state_cpu


def flat(state):
    return torch.cat([v.detach().double().flatten().cpu() for k, v in sorted(state.items())
                      if torch.is_floating_point(v)])


def rel_dist(a, b):
    """‖θa − θb‖ / ‖θb‖ (부동소수점 매개변수·버퍼 전체)."""
    fa, fb = flat(a), flat(b)
    return float(torch.linalg.norm(fa - fb) / torch.linalg.norm(fb))


def test_auc(model, data, device, norm):
    P = T.predict(model, data, data.test_idx, device, norm=norm)
    return T.macro_auc(data.Y[data.test_idx], P)[0]


def late_setup(data, store, frac, seed=20261025):
    """fl.e3b와 같은 후기 철회 집단(학습 환자의 frac, 같은 난수)."""
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    cohort = sorted(int(p) for p in np.random.default_rng(seed).choice(pids, int(len(pids) * frac), replace=False))
    cs = set(cohort)
    X = np.sort(np.array([int(i) for k in INSTS for i in data.train_idx[k] if int(data.pid[i]) in cs]))
    Z = np.sort(np.random.default_rng(1).choice(data.val_idx, min(len(X), len(data.val_idx)), replace=False))
    pool = np.r_[X, np.array([z for z in Z if data.inst[z] in INSTS])]
    return pids, cohort, X, pool


def forge_finetune(data, store, pol_wo, start_state, norm, R, device, max_pool, run_dir, l3_check, base_seed,
                   max_extra=10):
    """T11a. 반환: (마지막 run_fl 결과, 추가 라운드 수, 마지막 L3 판정)."""
    state, o, l3 = start_state, None, None
    for m in range(1, max_extra + 1):
        o = run_fl(data, store, pol_wo, base_seed=base_seed, device=device, max_pool=max_pool, cond="gpu_det",
                   rounds=R + m, start_state=state, start_round=R + m, norm=norm, run_dir=run_dir,
                   label="T11a", log_every=0)
        l3 = l3_check(o["model"])
        if l3["result"] == "통과":
            break
        state = state_cpu(o["model"])
    return o, m, l3


def forge_inject(data, store, cohort, inject, m_inj, u0_dir, u0_norm, R, device, max_pool, run_dir, base_seed=1):
    """T11b. U0 궤적(ckpt_r{R−m})에서 시작해 마지막 m 라운드에 inject 환자를 다시 넣는다."""
    keep_out = [p for p in cohort if p not in set(inject)]
    pol = Policy(enroll={p: 10 ** 9 for p in keep_out}, version=99)
    start = torch.load(os.path.join(u0_dir, f"ckpt_r{R - m_inj}.pt"))
    return run_fl(data, store, pol, base_seed=base_seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                  start_state=start, start_round=R - m_inj + 1, norm=u0_norm, run_dir=run_dir, label="T11b",
                  log_every=0)


def forge_l3match(data, X, pool, retain_idx, start_state, norm, device, max_pool, phis, inmask, l3_check, seed,
                  lam=1.0, lr=1e-4, bs=128, check_every=50, max_steps=6000):
    """T11a(L3 표적). 감사자와 같은 그림자 모델을 가진 공격자가, 잔류 자료의 학습 손실을 유지하면서
    철회 기록의 LiRA offline 점수가 같은 모델에서 비구성원(Z) 점수의 평균과 같아지도록 원 모델(P1)을 조정한다.
    목표 φ_j = μ_out,j + s̄_Z·σ_out(s̄_Z는 확인할 때마다 다시 계산). φ를 μ_out에 그대로 맞추면 X 점수가 0에 몰려
    비구성원 분포와 달라지므로(예비 실험에서 AUC 상승) 비구성원 분포의 위치에 맞춘다.
    φ는 fl.exp_lira.stat과 같은 정의(평가 모드, 2.5초 구간 최댓값, 클래스별 logit 평균). 반환: (모델, 단계 수, L3)."""
    import torch.nn as nn
    mean, std = norm
    model = T.new_model(max_pool, device, 0)
    model.load_state_dict(start_state)
    l3 = l3_check(model)
    if l3["result"] == "통과" or phis.shape[1] != len(pool):   # 이미 통과(스모크: 그림자 불일치로 L3 생략)
        return model, 0, l3
    mu_out = np.array([phis[~inmask[:, j], j].mean() for j in range(len(X))], dtype=np.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    starts = list(range(0, 1000 - T.WIN + 1, T.WIN // 2))
    from fl.exp_lira import scores, stat
    sd_out = float(np.std(phis[~inmask] - np.array([phis[~inmask[:, j], j].mean() for j in range(phis.shape[1])])[np.where(~inmask)[1]]))

    def target():
        off = scores(stat(model, data, pool, device, norm), phis, inmask)[1]
        return torch.from_numpy(mu_out + np.float32(off[len(X):].mean() * sd_out)).to(device)

    tgt = target()
    step = 0
    while l3["result"] != "통과" and step < max_steps:
        for _ in range(check_every):
            model.train()
            rec = retain_idx[torch.randint(0, len(retain_idx), (bs,), generator=g).numpy()]
            off = torch.randint(0, 1000 - T.WIN + 1, (bs,), generator=g).numpy()
            x = torch.from_numpy((np.stack([data.X[i, :, o:o + T.WIN] for i, o in zip(rec, off)]) - mean) / std).float().to(device)
            loss = bce(model(x), torch.from_numpy(data.Y[rec]).to(device))
            model.eval()
            j = torch.randint(0, len(X), (bs,), generator=g).numpy()
            xr = (np.asarray(data.X[X[j]], dtype=np.float32) - mean) / std
            z = torch.stack([model(torch.from_numpy(np.ascontiguousarray(xr[:, :, s:s + T.WIN])).to(device)) for s in starts]).max(0).values
            y = torch.from_numpy(data.Y[X[j]]).to(device)
            p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
            pt = torch.where(y == 1, p, 1 - p)
            phi = torch.log(pt / (1 - pt)).mean(1)
            loss = loss + lam * ((phi - tgt[torch.from_numpy(j).to(device)]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        l3 = l3_check(model)
        tgt = target()
    return model, step, l3


# ---- 35번 (b) 추가 변형(2026-10-27). 모두 정직한 U0 입력 커밋(S_wo)을 제시하고 결과를 "U0"라고 주장한다 ----

def perturb_trajectory(orig_dir, R, gamma, seed, out_dir):
    """T11c(Zhang et al. 2024 Adv-F형 기록 위조를 라운드 단위로 옮김). 철회자 자료로 학습한 원 궤적(P1)의 라운드별
    전역 모델 M_r에 상대 크기 γ의 섭동을 더해 U0 궤적이라고 제시하고(재계산 없음), 잊을 기록 자리는 이웃으로 바꾼 배치를
    제출 배치로 주장한다. 제출 배치 검증자에게 라운드 편차는 '이웃 치환 한 라운드의 효과 + γ' 크기다. ckpt_r{0..R}을 out_dir에 쓴다."""
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for r in range(0, R + 1):
        st = torch.load(os.path.join(orig_dir, f"ckpt_r{r}.pt"))
        if r > 0:
            st = {k: (v + gamma * v.abs() * torch.randn(v.shape, generator=g, dtype=v.dtype)) if v.is_floating_point() else v
                  for k, v in st.items()}
        torch.save(st, os.path.join(out_dir, f"ckpt_r{r}.pt"))
    return torch.load(os.path.join(out_dir, f"ckpt_r{R}.pt"))


def round_deviation(data, store, pol, claimed_dir, rounds, norm, base_seed, device, max_pool, R, tmp_dir, tamper=None):
    """라운드 단위 허용 오차 재현 검증(비교 판정): 제시된 M'_{r−1}에서 한 라운드를 다시 실행해 제시된 M'_r과의 상대 거리를 잰다.
    두 검증자 모형:
      시드 규칙 검증자  — 정직한 입력 집합 S_wo와 시드 규칙의 배치로 다시 계산(본 연구의 배치 결정성, 전제 (i))
      제출 배치 검증자  — 공격자가 제출한 배치(이웃 치환 포함)로 다시 계산하고 배치에 잊을 기록이 없는지만 본다
                          (Zhang et al. 2024의 Reproducible·Removable 정의). data·tamper로 제출 배치를 재현한다.
    반환: {r: 거리}."""
    out = {}
    for r in rounds:
        o = run_fl(data, store, pol, base_seed=base_seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                   start_state=torch.load(os.path.join(claimed_dir, f"ckpt_r{r - 1}.pt")), start_round=r, stop_round=r,
                   norm=norm, run_dir=tmp_dir, label="T11-roundcheck", log_every=0, tamper=tamper)
        out[r] = rel_dist(state_cpu(o["model"]), torch.load(os.path.join(claimed_dir, f"ckpt_r{r}.pt")))
    return out


def neighbor_map(data, X_idx, retain_idx, device, chunk=256):
    """T11d(Adv-R): 철회 기록마다 같은 기관 잔류 자료에서 레이블 벡터가 같은(없으면 해밍 거리가 가장 작은) 기록 가운데
    신호 L2 거리가 가장 가까운 기록. 반환: {철회 기록 위치: 이웃 위치}."""
    out = {}
    for k in INSTS:
        xk = np.array([i for i in X_idx if data.inst[i] == k]); rk = np.array([i for i in retain_idx if data.inst[i] == k])
        if len(xk) == 0:
            continue
        R_ = torch.from_numpy(np.asarray(data.X[rk], dtype=np.float32).reshape(len(rk), -1)).to(device)
        Yr = torch.from_numpy(data.Y[rk]).to(device)
        for b in range(0, len(xk), chunk):
            q = xk[b:b + chunk]
            Q = torch.from_numpy(np.asarray(data.X[q], dtype=np.float32).reshape(len(q), -1)).to(device)
            ham = torch.cdist(torch.from_numpy(data.Y[q]).to(device), Yr, p=1)       # 레이블 해밍 거리
            dist = torch.cdist(Q, R_)
            score = torch.where(ham == ham.min(1, keepdim=True).values, dist, torch.full_like(dist, float("inf")))
            for i, j in zip(q, score.argmin(1).cpu().numpy()):
                out[int(i)] = int(rk[j])
    return out


class _RemapX:
    def __init__(self, X, remap):
        self.X, self.remap, self.shape, self.dtype = X, remap, X.shape, X.dtype

    def __getitem__(self, key):
        head, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if np.isscalar(head):
            return self.X[(self.remap.get(int(head), int(head)),) + rest]
        idx = np.array([self.remap.get(int(i), int(i)) for i in np.asarray(head)])   # 표준화 통계 등 묶음 읽기
        return self.X[(idx,) + rest]


class RemapData:
    """신호만 이웃 기록으로 바꿔 읽는 Data 보기(레이블은 같음). 학습 경로(fl.train.local_train)에서만 쓴다."""
    def __init__(self, data, remap):
        self._d, self.X = data, _RemapX(data.X, remap)

    def __getattr__(self, name):
        return getattr(self._d, name)


def substitute_setup(data, cohort, nbr, w):
    """제출 배치 재현: 원 학습과 같은 기록 위치 집합(S_wo ∪ 철회 기록, w 이전 라운드)을 쓰되 철회 기록의 신호는 이웃으로 바꾼다.
    기록 위치 집합이 원 학습과 같으므로 시드 규칙의 순열·배치 자리도 원 학습과 같고, 잊을 기록 자리에만 이웃이 들어간다."""
    pol = Policy(withdraw={p: 1 for p in cohort}, version=98)              # 커밋 = S_wo, 철회 기록은 '숨겨 사용'
    tamper = {(k, r): {"mode": "hidden", "pids": set(cohort)} for k in INSTS for r in range(1, w)}
    return RemapData(data, nbr), pol, tamper


def forge_substitute(data, store, cohort, nbr, R, w, device, max_pool, base_seed, run_dir):
    """T11d(Zhang et al. 2024 Adv-R형). 처음부터 다시 학습하되, 원 학습 배치에서 철회 기록 자리를 같은 레이블 최근접 이웃의
    신호로 바꿔 그대로 계산한다. 입력 커밋은 정직한 S_wo를 제시한다(L2 통과 목표). 제출 배치 검증자에게는 계산이 정확히
    재현되고 배치에 잊을 기록도 없으므로 ε = 0에서도 통과한다. 막는 것은 배치 결정성(전제 (i))이다."""
    dv, pol, tamper = substitute_setup(data, cohort, nbr, w)
    return run_fl(dv, store, pol, base_seed=base_seed, device=device, max_pool=max_pool,
                  cond="gpu_det", rounds=R, run_dir=run_dir, label="T11d", log_every=0, tamper=tamper)
