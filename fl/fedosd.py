"""U4: FedOSD (Pan et al., 2025, AAAI, "Federated Unlearning with Gradient Descent and Conflict Mitigation")
재구현. 원문 알고리즘 1을 따른다(공개 코드 github.com/zibinpan/FedOSD는 참고하지 않고 논문 서술로 작성).

원문 구성
  1. 망각 대상 u는 UCE 손실 L = −Σ_c y_c·log(1 − p_c/2)를 최소화하는 로컬 학습을 한다(식 3).
  2. 서버는 g_i = (ω − ω_i)/η를 받아, 남은 클라이언트 경사 행렬 G에 직교(Gd = 0)하면서 −g_u에 가장 가깝고
     ‖d‖ = ‖g_u‖인 방향 d를 구해 ω ← ω + η·d로 갱신한다(식 5–6). 이는 −g_u를 G의 영공간에 사영한 것과 같다.
  3. T_u 라운드 후 사후 학습: 남은 클라이언트의 g_i가 g_a = ω − ω⁰와 g_i·g_a > 0이면 g_a의 법평면에 사영하고
     원래 노름으로 맞춘 뒤 평균해 ω ← ω − η·ḡ로 갱신한다(식 7).
  원문 설정: 로컬 SGD, E = 1. 일부 기록 삭제는 망각 기록만 가진 "가상 클라이언트"로 바꿔 처리한다(원문 2.2절).
본 연구의 조정(8.3 "재구현하며 그 사실을 명시")
  - 다중 레이블(sigmoid): UCE를 클래스별 정답 확신도 p_t(y=1이면 p, y=0이면 1−p)에 적용해 평균한다.
  - 가상 클라이언트 u = 철회 기록 X 전체(여러 기관에 걸침). 남은 클라이언트 = 세 기관의 유효 집합.
  - BN 이동 통계(경사 없는 버퍼)는 남은 기관 로컬 모델의 가중 평균으로 둔다.
공식 코드와의 교차 검증(3장 반영 46번 (a), 2026-10-27, 커밋 41cc106, tests/test_fedosd_official.py):
  - 직교 사영(식 5–6)은 공식 get_nearest_oth_d와 같은 방향이다(부호는 갱신식에 흡수).
  - 사후 학습: 공식 코드는 모든 g_i를 g_a의 법평면에 사영한다(조건 없음). 본 구현은 논문 서술대로 g_i·g_a > 0일 때만 사영한다.
    → post_proj="all"로 공식 코드 방식을 고를 수 있다.
  - UCE: 공식 코드는 단일 레이블 softmax에서 정답 클래스 항만 쓴다. 본 구현은 다중 레이블 sigmoid의 모든 클래스(정답 확신도)를
    쓴다. → uce="pos"로 양성 클래스 항만 쓰는 변형을 고를 수 있다.
  - 공식 실행 예는 사후 학습 학습률을 따로 둔다(r_lr)와 라운드별 감쇠(0.999). 본 구현은 두 단계에 같은 lr을 쓰고 감쇠하지 않는다.
"""

import numpy as np
import torch
import torch.nn as nn

from fl import evidence as EV
from fl import train as T
from fl.run import INIT_SEED, INSTS, PURPOSE, state_cpu


def uce_loss(logits, y, mode="all"):
    p = torch.sigmoid(logits)
    if mode == "pos":                                   # 양성 클래스 항만(공식 코드의 '정답 클래스 항만'에 대응)
        return -(torch.log(1 - p / 2) * y).sum() / y.sum().clamp(min=1)
    pt = torch.where(y > 0.5, p, 1 - p)
    return -torch.log(1 - pt / 2).mean()


def _local_sgd(model, data, idx, seed, device, nm, lr, loss="bce", bs=128, uce="all"):
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    idx = np.sort(np.asarray(idx))
    perm = torch.randperm(len(idx), generator=gen).numpy()
    offs = torch.randint(0, 1000 - T.WIN + 1, (len(idx),), generator=gen).numpy()
    for b in range(0, len(idx), bs):
        sel = perm[b:b + bs]
        if len(sel) < 2:                # 기록 1개짜리 마지막 배치는 BatchNorm 학습 불가(fl.train.local_train과 같게 건너뜀)
            continue
        rec = idx[sel]
        x = np.stack([data.X[i, :, o:o + T.WIN] for i, o in zip(rec, offs[sel])])
        x = torch.from_numpy((x - nm[0]) / nm[1]).float().to(device)
        y = torch.from_numpy(data.Y[rec]).to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        (uce_loss(out, y, uce) if loss == "uce" else nn.functional.binary_cross_entropy_with_logits(out, y)).backward()
        opt.step()
    return model


def _vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


def _buffers_avg(models, weights):
    tot = float(sum(weights))
    out = {}
    for name, b in models[0].named_buffers():
        if b.is_floating_point():
            out[name] = sum(dict(m.named_buffers())[name] * (w / tot) for m, w in zip(models, weights))
        else:
            out[name] = torch.stack([dict(m.named_buffers())[name] for m in models]).max(0).values
    return out


def osd_direction(G, gu):
    """망각 단계 갱신 방향 d(ω ← ω + η·d): −g_u를 남은 경사 행렬 G의 영공간에 사영하고 ‖g_u‖로 맞춘다(식 5–6)."""
    G = G.double()
    gram = (G @ G.T).cpu()
    coef = torch.linalg.pinv(gram).to(G.device) @ (G @ gu.double())
    d = -(gu.double() - G.T @ coef)                       # −g_u의 G 영공간 사영(식 6과 같은 해)
    return (d * (gu.norm().double() / d.norm())).float() if d.norm() > 0 else torch.zeros_like(gu)


def post_project(grads, ga, post_proj="conflict"):
    """사후 학습 경사(식 7): g_a = ω − ω⁰의 법평면에 사영하고 원래 노름으로 맞춘다. conflict는 g_i·g_a > 0일 때만."""
    proj = []
    for gi in grads:
        if post_proj == "all" or (gi @ ga) > 0:           # all: 공식 코드 방식(조건 없이 사영)
            gp = gi - (gi @ ga) / (ga @ ga) * ga
            gi = gp / gp.norm() * gi.norm()
        proj.append(gi)
    return torch.stack(proj).mean(0)


def fedosd(data, store, policy_wo, X_idx, start_state, base_seed, device, max_pool, rounds, norm,
           unlearn_rounds=5, post_rounds=10, lr=0.005, post_proj="conflict", uce="all"):
    nm = (norm["mean"], norm["std"])
    g = T.new_model(max_pool, device, INIT_SEED)
    g.load_state_dict(start_state)
    w0 = _vec(g).clone()
    commits, log = [], []
    S = {k: store.input_set(policy_wo, k, rounds) for k in INSTS}
    for t in range(1, unlearn_rounds + post_rounds + 1):
        w = _vec(g)
        locals_, grads, ws = [], [], []
        for k in INSTS:
            if len(S[k]) == 0:                  # 기관 탈퇴로 남은 기록이 없는 기관은 참여하지 않는다
                continue
            m = T.new_model(max_pool, device, INIT_SEED); m.load_state_dict(g.state_dict())
            m = _local_sgd(m, data, S[k], T.derive_seed(base_seed, f"osd{k}", t), device, nm, lr)
            locals_.append(m); grads.append((w - _vec(m)) / lr); ws.append(len(S[k]))
            commits.append({"C": {"k": k, "r": rounds, "root_S": EV.SortedMerkle(store.rids(S[k])).root,
                                  "n": len(S[k]), "sigma": policy_wo.version, "purpose": PURPOSE + f":fedosd-{t}"}})
        if t <= unlearn_rounds:
            mu = T.new_model(max_pool, device, INIT_SEED); mu.load_state_dict(g.state_dict())
            mu = _local_sgd(mu, data, X_idx, T.derive_seed(base_seed, "osd-u", t), device, nm, lr, loss="uce", uce=uce)
            gu = (w - _vec(mu)) / lr
            d = osd_direction(torch.stack(grads), gu)
            new_w = w + lr * d
            stage, conflicts = "unlearn", int(sum((d @ gi) < -1e-6 for gi in grads))
        else:
            new_w = w - lr * post_project(grads, w - w0, post_proj)
            stage, conflicts = "post", None
        torch.nn.utils.vector_to_parameters(new_w, g.parameters())
        for name, val in _buffers_avg(locals_, ws).items():
            dict(g.named_buffers())[name].copy_(val)
        log.append({"t": t, "stage": stage, "dist_to_w0": float((new_w - w0).norm()), "conflicts": conflicts})
    return {"model": g, "final_hash": T.model_hash(g)[0], "commits": commits,
            "forget_commit": {"root_X": EV.SortedMerkle(store.rids(X_idx)).root, "n": int(len(X_idx))}, "log": log}
