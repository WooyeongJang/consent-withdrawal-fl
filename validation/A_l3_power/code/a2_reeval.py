"""A2: 참조 U0 모델의 기록별 LiRA 통계량 재평가(학습 없음, 추론만).

시나리오마다 저장된 참조 체크포인트 U0_s1..5와 shadows.npz에서 기록별 통계량 φ와 offline 점수를 다시 계산한다.
X·Z·pool의 구성은 fl/e2.py(HEAD 4649db5)의 해당 줄을 그대로 따른다. 기존 모듈은 읽기 전용으로 import한다.
출력: <out>/<시나리오>.npz (phi[5,|pool|], offline[5,|pool|], pool, nX) 와 <out>/reeval_summary.json

  python validation/A_l3_power/code/a2_reeval.py --runs /runs --out /out --ptbxl /ptbxl --cache /cache/ptbxl_k3
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from fl.consent import demographics
from fl import train as T
from fl.exp_lira import evaluate, scores, stat
from fl.run import INIT_SEED, INSTS


def build_pool(sc, data, demo):
    """fl/e2.py main()의 X, Z, Zk, pool 구성(같은 난수)."""
    withdraw = {int(p): w for p, w in sc["withdraw"].items()}
    reconsent = {int(p): c for p, c in sc["reconsent"].items()}
    permanent = sorted(p for p in withdraw if p not in reconsent)
    pw = set(permanent)
    X = np.sort(np.array([int(i) for k in INSTS for i in data.train_idx[k] if int(data.pid[i]) in pw]))
    dist, val = sc["spec"].get("dist", {"type": "random"}), data.val_idx
    if "exit_inst" in sc["spec"]:
        elig = val[data.inst[val] == sc["spec"]["exit_inst"]]
    elif dist["type"] == "diag":
        elig = val[data.Y[val][:, T.CATS.index(dist["cat"])] == 1]
    elif dist["type"] == "agesex":
        elig = val[np.array([demo[int(data.ecg[i])][1] == dist["sex"] and demo[int(data.ecg[i])][0] >= dist["age_min"] for i in val])]
    elif dist["type"] == "inst":
        elig = val[data.inst[val] == dist["inst"]]
    else:
        elig = val
    Z = np.sort(np.random.default_rng(1).choice(elig, min(len(X), len(elig)), replace=False))
    Zk = np.array([z for z in Z if data.inst[z] in INSTS])
    return X, Zk


def main():
    ap = argparse.ArgumentParser()
    for a in ("--runs", "--out", "--ptbxl", "--cache"):
        ap.add_argument(a, required=True)
    ap.add_argument("--scenarios", default="results/consent")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--nref", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device, max_pool = T.configure("gpu_det")
    data = T.Data(args.cache)
    demo = {}
    with open(demographics(args.cache, args.ptbxl)) as f:
        for r in csv.DictReader(f):
            demo[int(r["ecg_id"])] = (float(r["age"]) if r["age"] else -1, int(float(r["sex"])) if r["sex"] else -1)
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__, "device": str(device),
            "image_digest": os.environ.get("IMAGE_DIGEST"), "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "argv": sys.argv, "scenarios": {}}
    names = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.runs, "E2-*")) if os.path.isdir(d))
    for name in names:
        t0 = time.time()
        sc = json.load(open(os.path.join(args.scenarios, f"{name}.json")))
        root = os.path.join(args.runs, name)
        X, Zk = build_pool(sc, data, demo)
        pool, nX = np.r_[X, Zk], len(X)
        z = np.load(os.path.join(root, "shadows.npz"))
        phis, inmask = z["phis"], z["inmask"]
        assert phis.shape[1] == len(pool), (name, phis.shape, len(pool))
        phi, off, ev_out, ck_sha = [], [], [], []
        for j in range(1, args.nref + 1):
            d = os.path.join(root, f"U0_s{j}")
            f = os.path.join(d, f"ckpt_r{args.rounds}.pt")
            ev = next(json.loads(l) for l in open(os.path.join(d, "events.jsonl")) if "NormStatsCommitted" in l)
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(torch.load(f))
            p = stat(m, data, pool, device, (ev["mean"], ev["std"]))
            _, o = scores(p, phis, inmask)
            phi.append(p); off.append(o); ev_out.append(evaluate(o, nX))
            ck_sha.append(hashlib.sha256(open(f, "rb").read()).hexdigest())
        np.savez(os.path.join(args.out, f"{name}.npz"), phi=np.array(phi), offline=np.array(off), pool=pool, nX=nX)
        meta["scenarios"][name] = {"nX": int(nX), "nZ": int(len(Zk)), "eval": ev_out, "ckpt_sha256": ck_sha,
                                   "wall_s": time.time() - t0}
        print(f"[{name}] nX={nX} nZ={len(Zk)} AUC={[round(e['auc'], 6) for e in ev_out]} ({time.time() - t0:.0f}s)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(meta, open(os.path.join(args.out, "reeval_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
