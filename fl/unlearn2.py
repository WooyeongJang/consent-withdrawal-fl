"""근사 영향 제거 방법 (3장 v1.6 8.3절 U2·U4).

U2 FedEraser(Liu et al., 2021, IWQoS; arXiv:2012.13891)를 개인 단위 철회에 맞게 조정한 재구현.
  - 원 논문: 대상 클라이언트 전체를 제거한다. 원 학습 중 Δt 라운드 간격으로 클라이언트 업데이트를 보존하고,
    보존 라운드마다 남은 클라이언트가 E_cali = r·E_loc 만큼 보정 학습해 업데이트 방향을 얻는다.
    보존된 업데이트의 크기(층별 노름)에 새 방향을 곱해 보정 업데이트를 만들고 집계해 모델을 재구성한다.
    논문 설정: r = 0.5, Δt = 2.
  - 본 연구의 조정(8.3 "재구현하며 그 사실을 명시"): 클라이언트를 빼는 대신, 철회 기록이 있는 기관을 포함한
    모든 기관이 철회 기록을 뺀 입력으로 보정 학습한다. BN 이동 통계는 보정 학습 결과를 가중 평균한다.
(주의) U4 본 실험은 fl/fedosd.py(FedOSD, Pan et al. 2025)를 쓴다. 아래 gradient_ascent는 FedOSD 확보 전 예비 실험용으로만 남긴다.
U4 경사 상승 기반 근사 언러닝 — **임시 구현(예비 실험 전용)**. Liu et al.(2025) 원문을 확보하지 못해, 철회 기록에 대한
  경사 상승(손실 최대화) 후 남은 데이터로 짧게 회복 학습하는 일반적인 방식으로 구현했다. 원문 확보 후 교체한다.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn

from fl import evidence as EV
from fl import train as T
from fl.run import INIT_SEED, INSTS, PURPOSE, state_cpu


def _float_keys(state):
    return [k for k, v in state.items() if v.is_floating_point() and "running" not in k]


def federaser(data, store, policy_wo, orig_dir, base_seed, device, max_pool, rounds, norm, delta_t=2, ratio=0.5,
              save_dir=None):
    """save_dir: 재구성 궤적(보존 라운드의 전역 상태 ckpt_r{t-1}과 보정 업데이트 upd_r{t}_k{k})을 원 학습과 같은 형식으로
    남긴다. 순차 철회를 차례별로 처리할 때 다음 요청의 FedEraser가 이 궤적을 '원 학습 기록'으로 쓴다(3장 반영 39번)."""
    nm = (norm["mean"], norm["std"])
    M = torch.load(os.path.join(orig_dir, "ckpt_r0.pt"))
    commits = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for t in range(1, rounds + 1, delta_t):
        old_global = torch.load(os.path.join(orig_dir, f"ckpt_r{t - 1}.pt"))
        if save_dir:
            torch.save(M, os.path.join(save_dir, f"ckpt_r{t - 1}.pt"))
        cal, ws = [], []
        for k in INSTS:
            p = os.path.join(orig_dir, f"upd_r{t}_k{k}.pt")
            if not os.path.exists(p):
                continue
            old = torch.load(p)["state"]
            S = store.input_set(policy_wo, k, t)
            if len(S) == 0:                     # 기관 전체가 빠진 경우(기관 탈퇴): 그 기관의 보관 갱신은 버린다
                continue
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(M)
            steps = math.ceil(math.ceil(len(S) / 128) * ratio)
            m, _ = T.local_train(m, data, k, t, base_seed, device, rounds, idx=S, norm=nm, max_steps=steps)
            new = state_cpu(m)
            out = {}
            for key in new:
                if key in _float_keys(new):
                    d_old, d_new = old[key] - old_global[key], new[key] - M[key]
                    n_new = d_new.norm()
                    out[key] = d_old.norm() * d_new / n_new if n_new > 0 else torch.zeros_like(d_new)
                else:
                    out[key] = new[key]
            cal.append(out); ws.append(len(S))
            if save_dir:
                torch.save({"state": {key: (M[key] + out[key] if key in _float_keys(M) else out[key]) for key in M},
                            "n": len(S)}, os.path.join(save_dir, f"upd_r{t}_k{k}.pt"))
            tree = EV.SortedMerkle(store.rids(S))
            commits.append({"C": {"k": k, "r": t, "root_S": tree.root, "n": len(S), "sigma": policy_wo.version,
                                  "purpose": PURPOSE + ":federaser-calibration"}})
        agg = T.fedavg(cal, ws)
        M = {key: (M[key] + agg[key] if key in _float_keys(M) else agg[key]) for key in M}
    g = T.new_model(max_pool, device, INIT_SEED)
    g.load_state_dict(M)
    return {"model": g, "final_hash": T.model_hash(g)[0], "commits": commits, "retained_rounds": len(range(1, rounds + 1, delta_t))}


def gradient_ascent(data, store, policy_wo, X_by_k, final_state, base_seed, device, max_pool, rounds, norm,
                    ga_epochs=1, ga_lr=5e-5, recover_rounds=2, recover_lr=1e-4, bs=128):
    """임시 U4: 철회 기록이 있는 기관이 그 기록으로 경사 상승 → FedAvg → 남은 데이터로 회복 학습."""
    nm = (norm["mean"], norm["std"])
    g = T.new_model(max_pool, device, INIT_SEED)
    g.load_state_dict(final_state)
    forget_commits, commits = [], []
    states, ws = [], []
    for k in INSTS:
        X = np.sort(np.asarray(X_by_k.get(k, []), dtype=np.int64))
        if len(X) == 0:
            continue
        m = T.new_model(max_pool, device, INIT_SEED)
        m.load_state_dict(g.state_dict())
        seed = T.derive_seed(base_seed, f"ga{k}", 0)
        gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
        opt = torch.optim.AdamW(m.parameters(), lr=ga_lr, weight_decay=0.0)
        m.train()
        for _ in range(ga_epochs):
            perm = torch.randperm(len(X), generator=gen).numpy()
            offs = torch.randint(0, 1000 - T.WIN + 1, (len(X),), generator=gen).numpy()
            for b in range(0, len(X), bs):
                sel = perm[b:b + bs]; rec = X[sel]
                x = np.stack([data.X[i, :, o:o + T.WIN] for i, o in zip(rec, offs[sel])])
                x = torch.from_numpy((x - nm[0]) / nm[1]).float().to(device)
                y = torch.from_numpy(data.Y[rec]).to(device)
                opt.zero_grad(set_to_none=True)
                (-nn.functional.binary_cross_entropy_with_logits(m(x), y)).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        states.append(state_cpu(m)); ws.append(len(X))
        forget_commits.append({"k": k, "root_X": EV.SortedMerkle(store.rids(X)).root, "n": int(len(X))})
    g.load_state_dict(T.fedavg(states, ws) if states else g.state_dict())
    for j in range(1, recover_rounds + 1):
        st, w = [], []
        for k in INSTS:
            S = store.input_set(policy_wo, k, rounds)
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(g.state_dict())
            m = _train_fixed_lr(m, data, S, T.derive_seed(base_seed, f"rec{k}", j), device, nm, recover_lr, bs)
            st.append(state_cpu(m)); w.append(len(S))
            commits.append({"C": {"k": k, "r": rounds, "root_S": EV.SortedMerkle(store.rids(S)).root, "n": len(S),
                                  "sigma": policy_wo.version, "purpose": PURPOSE + f":ga-recovery-{j}"}})
        g.load_state_dict(T.fedavg(st, w))
    return {"model": g, "final_hash": T.model_hash(g)[0], "commits": commits, "forget_commits": forget_commits}


def _train_fixed_lr(m, data, idx, seed, device, nm, lr, bs):
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    m.train()
    perm = torch.randperm(len(idx), generator=gen).numpy()
    offs = torch.randint(0, 1000 - T.WIN + 1, (len(idx),), generator=gen).numpy()
    for b in range(0, len(idx), bs):
        sel = perm[b:b + bs]; rec = idx[sel]
        x = np.stack([data.X[i, :, o:o + T.WIN] for i, o in zip(rec, offs[sel])])
        x = torch.from_numpy((x - nm[0]) / nm[1]).float().to(device)
        y = torch.from_numpy(data.Y[rec]).to(device)
        opt.zero_grad(set_to_none=True)
        nn.functional.binary_cross_entropy_with_logits(m(x), y).backward()
        opt.step()
    return m
