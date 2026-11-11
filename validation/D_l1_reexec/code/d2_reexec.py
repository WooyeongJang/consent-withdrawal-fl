"""D2: 유형 B 표본 학습 단위를 컨테이너(GPU, 결정적 모드)에서 다시 실행해 원장 커밋(hStart, rootS, hDelta)과 비교한다.

컨테이너 안에서(저장소 /w 읽기 전용, 데이터 /data, 실행 폴더 /runs 읽기 전용, 출력 /out):
  python /w/validation/D_l1_reexec/code/d2_reexec.py --units /w/validation/D_l1_reexec/results/d1_units.json --out /out/d2_reexec.json
끝난 단위는 /out/d2_reexec.json에 저장되므로 중단 뒤 다시 실행하면 이어서 돈다.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/w")
from fl import audit as AU  # noqa: E402
from fl import train as T  # noqa: E402
from fl.run import INSTS, Policy, Store, prog_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--runs", default="/runs")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--w", type=int, default=45)
    ap.add_argument("--frac", type=float, default=0.05)
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", "/w", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"# D2 reexec; start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; git HEAD {head}; image {os.environ.get('IMAGE_DIGEST')}; "
          f"script sha256 {hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16]}; torch {__import__('torch').__version__}", flush=True)
    device, max_pool = T.configure("gpu_det")
    data = T.Data(a.cache)
    store = Store(data)
    prog = prog_id("gpu_det")
    units = json.load(open(a.units))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"restarts": 0, "seeds": {}}
    if os.path.exists(a.out):
        res["restarts"] += 1
    # 철회 정책: E3-B와 같은 규칙(학습 환자 5%, rng 20261025, 라운드 w 효력). 입력 재구성 루트 = rootS 검사가 이 규칙을 독립적으로 확인한다.
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    cohort = sorted(int(p) for p in np.random.default_rng(20261025).choice(pids, int(len(pids) * a.frac), replace=False))
    policy = Policy(withdraw={p: a.w for p in cohort})
    scope = np.concatenate([store.input_set(policy, k, 1) for k in INSTS])
    mean, std = T.norm_stats(data, scope)
    norm = {"mean": float(mean), "std": float(std)}
    for s, info in units["seeds"].items():
        run_dir = os.path.join(a.runs, f"B_s{s}", "orig")
        ns = next(json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl")) if '"NormStatsCommitted"' in l)
        norm_match = abs(ns["mean"] - norm["mean"]) < 1e-12 and abs(ns["std"] - norm["std"]) < 1e-12
        sres = res["seeds"].setdefault(s, {"contract": info["contract"], "norm_recomputed": norm, "norm_matches_events_jsonl": norm_match, "units": {}})
        for u in info["units"]:
            key = f"{u['k']}:{u['r']}"
            if key in sres["units"]:
                continue
            commit = {"C": {"k": u["k"], "r": u["r"], "root_S": u["root_S"], "sigma": u["sigma"]},
                      "U": {"h_start": u["h_start"], "h_delta": u["h_delta"]}}
            t0 = time.time()
            v = AU.reexecute(data, store, policy, run_dir, commit, int(s), device, max_pool, a.rounds, norm, prog)
            sres["units"][key] = {"result": v["result"], "reason": v.get("reason", ""), "seconds": round(time.time() - t0, 2)}
            print(f"seed {s} k={u['k']} r={u['r']:2d}: {v['result']} {v.get('reason','')} ({sres['units'][key]['seconds']}s)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    tot = sum(len(v["units"]) for v in res["seeds"].values())
    ok = sum(1 for v in res["seeds"].values() for x in v["units"].values() if x["result"] == "통과")
    res["summary"] = {"units": tot, "pass": ok, "mismatch": tot - ok,
                      "reasons": {r: sum(1 for v in res["seeds"].values() for x in v["units"].values() if x["reason"] == r) for r in {x["reason"] for v in res["seeds"].values() for x in v["units"].values()}},
                      "seconds_total": round(sum(x["seconds"] for v in res["seeds"].values() for x in v["units"].values()), 1)}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("summary", res["summary"], flush=True)


if __name__ == "__main__":
    main()
