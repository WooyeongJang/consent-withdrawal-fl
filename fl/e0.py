"""E0 재현성 사전 실험 (3장 v1.6 8.4절, reports/2026-10-25_E0_설정근거.md 3.2절).

  python -m fl.e0 --mode unit  --cond gpu_det --run 0 --out OUT   # 기관 0, 1라운드
  python -m fl.e0 --mode chain --cond cpu     --run 0 --out OUT   # 세 기관, 5라운드 연쇄
  python -m fl.e0 --mode pilot --cond gpu_det --base-seed 1 --out OUT  # R=50 전체 학습과 성능

같은 base-seed로 반복 실행했을 때 업데이트(모델 상태) 해시가 비트 단위로 같은지 본다.
"""

import argparse
import json
import os
import platform
import socket

import torch

from fl import train as T

INIT_SEED = 20261025


def env_info(device):
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(), "host": socket.gethostname(), "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unit", "chain", "pilot"], required=True)
    ap.add_argument("--cond", choices=["gpu_det", "cpu", "gpu_nondet"], required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--chain-rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--vary-init", action="store_true", help="초기 가중치도 base-seed에서 도출")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device, max_pool = T.configure(args.cond, args.threads)
    init_seed = T.derive_seed(args.base_seed, "init", 0) if args.vary_init else INIT_SEED
    data = T.Data(args.cache)
    res = {"mode": args.mode, "cond": args.cond, "tag": args.tag, "run": args.run, "base_seed": args.base_seed,
           "init_seed": init_seed, "rounds_total": args.rounds, "env": env_info(device),
           "n_train": {k: int(len(v)) for k, v in data.train_idx.items()},
           "norm": {"mean": data.mean, "std": data.std}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.mode == "unit":
        m = T.new_model(max_pool, device, init_seed)
        res["start_hash"] = T.model_hash(m)[0]
        t = T.now()
        m, loss = T.local_train(m, data, "0", 1, args.base_seed, device, args.rounds)
        if device.type == "cuda":
            torch.cuda.synchronize()
        res["seconds"] = T.now() - t
        res["hash"], res["per_tensor"] = T.model_hash(m)
        res["last_loss"] = loss

    elif args.mode in ("chain", "pilot"):
        R = args.chain_rounds if args.mode == "chain" else args.rounds
        g = T.new_model(max_pool, device, init_seed)
        weights = [len(data.train_idx[k]) for k in T.inst_keys()]
        res["round_hashes"], res["round_seconds"], res["val_auc"] = [], [], {}
        for r in range(1, R + 1):
            t = T.now()
            states = []
            for k in T.inst_keys():
                m = T.new_model(max_pool, device, init_seed)
                m.load_state_dict(g.state_dict())
                m, _ = T.local_train(m, data, k, r, args.base_seed, device, args.rounds)
                states.append({n: v.detach().clone() for n, v in m.state_dict().items()})
            g.load_state_dict(T.fedavg(states, weights))
            if device.type == "cuda":
                torch.cuda.synchronize()
            res["round_seconds"].append(T.now() - t)
            res["round_hashes"].append(T.model_hash(g)[0])
            if args.mode == "pilot" and (r % 10 == 0 or r == R):
                res["val_auc"][r] = T.macro_auc(data.Y[data.val_idx], T.predict(g, data, data.val_idx, device))[0]
                print(f"round {r}: val macro AUC {res['val_auc'][r]:.4f}", flush=True)
        res["hash"] = res["round_hashes"][-1]
        if args.mode == "pilot":
            P = T.predict(g, data, data.test_idx, device)
            res["test_macro_auc"], res["test_auc_per_class"] = T.macro_auc(data.Y[data.test_idx], P)
            torch.save(g.state_dict(), os.path.join(args.out, f"pilot_{args.cond}_s{args.base_seed}.pt"))

    if device.type == "cuda":
        res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.mode}_{args.cond}{args.tag}_s{args.base_seed}_r{args.run}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in res if k in ("mode", "cond", "run", "hash", "seconds", "test_macro_auc")}))


if __name__ == "__main__":
    main()
