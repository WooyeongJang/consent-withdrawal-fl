"""연합학습 로컬 학습·집계·평가와 결정적 실행 설정 (3장 v1.6 3.4·5.2·8.2).

- 시드 도출 규칙(seed_rule): seed_{k,r} = int(SHA-256("seed|base|k|r")[:16], 16) mod 2^63
- 로컬 학습: 1 라운드 = 로컬 E 에폭. 배치 순서와 2.5초 구간 위치는 seed_{k,r}에서 만든 생성기로 정한다.
  옵티마이저(AdamW)는 라운드마다 새로 만든다(기관이 라운드 사이에 상태를 들고 있지 않음).
- 학습률: 기관별 one-cycle(코사인)을 전체 라운드에 걸쳐 적용한다. 스케줄 위치는 (라운드, 단계)로 정해지므로
  체크포인트에 별도 상태를 저장하지 않아도 된다(4.4절 동등 조건). E0 이후 확정한다.
- 집계: FedAvg(기록 수 가중 평균). 정수 버퍼(num_batches_tracked)는 최댓값.
- 평가: 반씩 겹치는 2.5초 구간 예측의 원소별 최댓값, macro AUC(벤치마크 방식).
"""

import hashlib
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from fl.model import ResNet1dWang

CATS = ["NORM", "MI", "STTC", "CD", "HYP"]
WIN = 250  # 2.5초 × 100Hz


def derive_seed(base, k, r):
    return int(hashlib.sha256(f"seed|{base}|{k}|{r}".encode()).hexdigest()[:16], 16) % (2 ** 63)


def configure(cond, threads=8):
    """cond: gpu_det(결정적 GPU, amax), cpu(CPU, 원 연산), gpu_nondet(대조군, 원 연산)."""
    torch.set_num_threads(threads)
    if cond == "gpu_det":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda"), "amax"
    if cond == "cpu":
        torch.use_deterministic_algorithms(True)
        return torch.device("cpu"), "adaptive"
    if cond == "gpu_nondet":
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda"), "adaptive"
    raise ValueError(cond)


def inst_keys():
    """기관 식별자. 환경변수 FL_K(기본 3)가 기관 수다. 데이터 캐시도 같은 K로 만든 것을 써야 한다(prepare_data --k)."""
    return tuple(str(i) for i in range(int(os.environ.get("FL_K", "3"))))


class Data:
    def __init__(self, cache):
        import csv
        sig = "c2021_100hz.npy" if os.path.exists(os.path.join(cache, "c2021_100hz.npy")) else "ptbxl_100hz.npy"   # 보조 데이터 캐시
        self.X = np.load(os.path.join(cache, sig), mmap_mode="r")
        if lead_subset() is not None:                    # 단일 유도 보조 조건(8.1): 지정 유도만 남긴다
            self.X = np.ascontiguousarray(self.X[:, lead_subset(), :])
        with open(os.path.join(cache, "meta.csv")) as f:
            rows = list(csv.DictReader(f))
        self.ecg = np.array([int(r["ecg_id"]) for r in rows])
        self.pid = np.array([int(float(r["patient_id"])) for r in rows])
        self.fold = np.array([int(r["strat_fold"]) for r in rows])
        self.inst = np.array([r["institution"] for r in rows])
        self.Y = np.array([[float(r[c] == "1") for c in CATS] for r in rows], dtype=np.float32)
        usable = np.array([r["has_label"] == "True" and r["excluded"] == "False" for r in rows])
        self.train_idx = {k: np.where(usable & (self.fold <= 8) & (self.inst == k))[0] for k in inst_keys()}
        self.val_idx = np.where(usable & (self.fold == 9))[0]
        self.test_idx = np.where(usable & (self.fold == 10))[0]
        # 표준화: 학습 집합 전체의 단일 평균·표준편차(벤치마크 방식). 연합 환경에서는 기관별 합·제곱합을
        # 모아 같은 값을 얻는다. 철회 레코드의 포함 여부는 8.1의 '정규화 통계 범위 커밋'에서 다룬다.
        s = n = ss = 0.0
        for k, idx in self.train_idx.items():
            x = np.asarray(self.X[idx], dtype=np.float64)
            s += x.sum(); ss += (x ** 2).sum(); n += x.size
        self.mean = s / n
        self.std = math.sqrt(ss / n - self.mean ** 2)


def model_hash(model):
    h = hashlib.sha256()
    per = {}
    for name, t in sorted(model.state_dict().items()):
        b = t.detach().cpu().contiguous().numpy().tobytes()
        h.update(name.encode()); h.update(b)
        per[name] = hashlib.sha256(b).hexdigest()[:16]
    return h.hexdigest(), per


def lr_at(step, total, max_lr=1e-2, pct=0.3, div=25.0, final_div=1e4):
    """fastai v1 fit_one_cycle과 같은 모양: 코사인 상승(max_lr/div→max_lr) 후 코사인 하강(→max_lr/div/final_div)."""
    up = int(total * pct)
    lo, end = max_lr / div, max_lr / div / final_div
    cos = lambda a, b, p: b + (a - b) / 2 * (1 + math.cos(math.pi * p))
    if step < up:
        return cos(lo, max_lr, step / max(up, 1)), cos(0.95, 0.85, step / max(up, 1))
    p = (step - up) / max(total - up, 1)
    return cos(max_lr, end, p), cos(0.85, 0.95, p)


def norm_stats(data, idx):
    """표준화 통계(단일 평균·표준편차). 정렬된 기록 순서로 float64 합산해 결정적으로 계산한다."""
    x = np.asarray(data.X[np.sort(idx)], dtype=np.float64)
    m = x.mean()
    return float(m), float(np.sqrt((x ** 2).mean() - m ** 2))


def local_train(model, data, k, r, base_seed, device, rounds, local_epochs=1, bs=128, idx=None, norm=None,
                max_steps=None):
    """idx: 이번 라운드의 입력 집합 S(정렬된 기록 위치). None이면 기관의 전체 학습 집합.
    norm: (mean, std). None이면 data.mean/std.
    max_steps: 에폭 안의 단계 수 상한(FedEraser 보정 학습 E_cali = 0.5 E_loc 등). None이면 전체."""
    idx = data.train_idx[k] if idx is None else np.sort(np.asarray(idx))
    mean, std = (data.mean, data.std) if norm is None else norm
    seed = derive_seed(base_seed, k, r)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # 드롭아웃(CPU·CUDA)
    steps_per_epoch = math.ceil(len(idx) / bs)
    total = rounds * local_epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.95, 0.99), weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    step = (r - 1) * local_epochs * steps_per_epoch
    for _ in range(local_epochs):
        perm = torch.randperm(len(idx), generator=g).numpy()
        offs = torch.randint(0, 1000 - WIN + 1, (len(idx),), generator=g).numpy()
        for b in range(steps_per_epoch if max_steps is None else min(max_steps, steps_per_epoch)):
            sel = perm[b * bs:(b + 1) * bs]
            if len(sel) < 2:            # 기록 1개짜리 마지막 배치는 BatchNorm 학습이 불가능하므로 건너뛴다(그런 실행은 이전에는 오류로 중단됨)
                continue
            rec = idx[sel]
            x = np.stack([data.X[i, :, o:o + WIN] for i, o in zip(rec, offs[sel])])
            x = torch.from_numpy((x - mean) / std).float().to(device)
            y = torch.from_numpy(data.Y[rec]).to(device)
            lr, mom = lr_at(step, total)
            for pg in opt.param_groups:
                pg["lr"], pg["betas"] = lr, (mom, 0.99)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            step += 1
    return model, float(loss.detach().cpu())


def fedavg(states, weights):
    tot = float(sum(weights))
    out = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            out[key] = sum(s[key].double() * (w / tot) for s, w in zip(states, weights)).to(states[0][key].dtype)
        else:
            out[key] = torch.stack([s[key] for s in states]).max(0).values
    return out


@torch.no_grad()
def predict(model, data, idx, device, bs=256, norm=None):
    mean, std = (data.mean, data.std) if norm is None else norm
    model.eval()
    starts = list(range(0, 1000 - WIN + 1, WIN // 2))  # 0,125,…,750
    out = []
    for b in range(0, len(idx), bs):
        rec = idx[b:b + bs]
        x = (np.asarray(data.X[rec], dtype=np.float32) - mean) / std
        p = None
        for s in starts:
            q = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x[:, :, s:s + WIN])).float().to(device)))
            p = q if p is None else torch.maximum(p, q)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def auc(y, s):
    """순위 기반 AUROC(동점은 평균 순위)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_auc(Y, P):
    per = [auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    return float(np.mean(per)), per


def lead_subset():
    """환경변수 FL_LEADS: 없으면 12유도 전체, "I"면 유도 I만(단일 유도 보조 조건, 웨어러블 성능으로 해석하지 않음)."""
    v = os.environ.get("FL_LEADS", "")
    return None if v in ("", "all") else [["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(x)
                                          for x in v.split(",")]


def new_model(max_pool, device, init_seed):
    torch.manual_seed(init_seed)
    sub = lead_subset()
    return ResNet1dWang(max_pool=max_pool, in_ch=12 if sub is None else len(sub)).to(device)


def now():
    return time.perf_counter()
