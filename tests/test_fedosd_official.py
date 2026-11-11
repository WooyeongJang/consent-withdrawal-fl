"""U4 FedOSD 재구현(fl/fedosd.py)과 공식 코드의 교차 검증(3장 반영 46번 (a), 2026-10-27).

공식 코드: github.com/zibinpan/FedOSD 커밋 41cc10635c6f2396795d0d9c6239ea181b7c9ee6 (MIT License, "Copyright (c) 2025 zibinpan").
아래 official_* 함수는 tfedplat/algorithm/unlearning/FedOSD.py와 UnlearnAlgorithm.py에서 계산 부분만 그대로 옮긴 것이다
(클래스·로그 코드 제외). 공식 갱신은 SGD 단계 ω ← ω − η·d(update_module)이고, 본 구현은 ω ← ω + η·d다.

  컨테이너 안에서: python tests/test_fedosd_official.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fl.fedosd import osd_direction, post_project, uce_loss  # noqa: E402


# ---- 공식 코드(FedOSD.py 34–59행) ----
def official_pinv(matrix):
    U, s, V = torch.svd(matrix)
    primary_sigma_indices = torch.where(s >= 1e-6)[0]
    s[primary_sigma_indices] = 1 / s[primary_sigma_indices]
    S = torch.diag(s)
    return V @ S @ U.T


def official_nearest_oth_d(gr_locals, gu):
    A = gr_locals
    A_T = A.T
    c = gu
    AAT_1 = official_pinv(A @ A_T)
    Ac = A @ c.reshape(-1, 1)
    AAT_1_Ac = AAT_1 @ Ac
    return c - (A_T @ AAT_1_Ac).reshape(-1)


def official_unlearn_step(gr_locals, gu):            # FedOSD.py 79–92행(가상 클라이언트 1개)
    d = official_nearest_oth_d(gr_locals, gu)
    return d / torch.norm(d) * torch.norm(gu)        # 공식: ω ← ω − η·d


def official_post_step(g_locals, ga):                # FedOSD.py 95–108행
    ga_norm = torch.norm(ga)
    g_locals = g_locals.clone()
    for idx in range(len(g_locals)):
        grad = g_locals[idx]
        g_norm = torch.norm(grad)
        g_locals[idx] = grad - grad @ ga / ga_norm ** 2 * ga
        g_locals[idx] = g_locals[idx] / torch.norm(g_locals[idx]) * g_norm
    weights = torch.Tensor([1 / len(g_locals)] * len(g_locals)).float()
    return weights @ g_locals                        # 공식: ω ← ω − η·d


def official_uce(pred, target):                      # UnlearnAlgorithm.UnLearningCELoss, reduction='mean'
    target_enc = F.one_hot(target, int(pred.shape[1]))
    pred = F.softmax(pred, dim=-1)
    return -torch.mean(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))


def cos(a, b):
    return float(a.double() @ b.double() / (a.double().norm() * b.double().norm()))


def main():
    torch.manual_seed(0)
    n, rep = 50_000, 20
    worst = {"unlearn_cos": 1.0, "unlearn_rel": 0.0, "post_all_rel": 0.0, "orth": 0.0}
    for _ in range(rep):
        G = torch.randn(3, n)                         # 남은 기관 3곳의 경사
        gu = torch.randn(n) + 0.5 * G[0]              # 망각 대상 경사(남은 경사와 일부 상관)
        ours = osd_direction(G, gu)                   # ω ← ω + η·ours
        off = -official_unlearn_step(G, gu)           # 같은 갱신이 되도록 부호를 맞춘다
        worst["unlearn_cos"] = min(worst["unlearn_cos"], cos(ours, off))
        worst["unlearn_rel"] = max(worst["unlearn_rel"], float((ours - off).norm() / off.norm()))
        worst["orth"] = max(worst["orth"], float((G.double() @ ours.double()).abs().max() / ours.double().norm()))
        ga = torch.randn(n)
        a = post_project(list(G), ga, "all")
        b = official_post_step(G, ga)
        worst["post_all_rel"] = max(worst["post_all_rel"], float((a - b).norm() / b.norm()))
    print(worst)
    assert worst["unlearn_cos"] > 1 - 1e-6 and worst["unlearn_rel"] < 1e-4, "망각 단계 방향 불일치"
    assert worst["orth"] < 1e-4, "남은 경사와 직교하지 않음"
    assert worst["post_all_rel"] < 1e-5, "사후 학습(공식 방식) 불일치"

    # 사후 학습 조건부 사영(본 구현 기본값): g_i·g_a ≤ 0인 경사는 그대로 둔다. 공식 방식과의 차이를 보인다.
    G2 = torch.stack([torch.randn(n), -torch.randn(n).abs()])
    ga2 = torch.randn(n).abs()                         # 두 번째 경사는 g_a와 음의 내적
    diff = float((post_project(list(G2), ga2, "conflict") - official_post_step(G2, ga2)).norm())
    assert diff > 0, "조건부 사영이 공식 방식과 같으면 안 된다(내적 ≤ 0인 경사 존재)"

    # UCE: 단일 레이블에서 공식 손실의 형태 −log(1 − p_y/2)를 확인하고, 본 구현 pos 모드가 같은 형태(sigmoid 확률)임을 확인
    z = torch.randn(64, 5)
    y = torch.randint(0, 5, (64,))
    p_soft = F.softmax(z, -1)[torch.arange(64), y]
    assert torch.allclose(official_uce(z, y), -torch.log(1 - p_soft / 2).mean(), atol=1e-6)
    yh = F.one_hot(y, 5).float()
    p_sig = torch.sigmoid(z)[torch.arange(64), y]
    assert torch.allclose(uce_loss(z, yh, "pos"), -torch.log(1 - p_sig / 2).mean(), atol=1e-6)
    print("fedosd official cross-check: OK")


if __name__ == "__main__":
    main()
