"""비트 단위 동일성 확인: 같은 시나리오·시드를 단일 프로세스 모사(fl.run.run_fl)로 돌려 주체별 컨테이너 결과와 비교한다.

  FL_K=3 python -m fl.dist.reference --cache /cache/ptbxl_k3 --dist /dist --runs /runs --scenario ... --seed 1 --rounds 50
표준화 통계는 (1) 단일 프로세스 방식(전체 기록 한 번에 평균)과 (2) 컨테이너 방식(기관별 합을 모음)의 값을 비교하고,
학습은 컨테이너 실행이 커밋한 값(2)을 넣어 비교한다.
"""

import argparse
import json
import os

import numpy as np

from fl import evidence as EV
from fl import train as T
from fl.dist.aggregator import scenario_policy
from fl.run import INSTS, Store, run_fl


def main():
    ap = argparse.ArgumentParser()
    for x in ("--cache", "--dist", "--runs", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--scenario", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=50)
    a = ap.parse_args()
    device, max_pool = T.configure("gpu_det")
    data = T.Data(a.cache)
    store = Store(data)
    policy = scenario_policy(a.scenario)
    pub = json.load(open(os.path.join(a.dist, "public.json")))
    scope = np.concatenate([store.input_set(policy, k, 1) for k in INSTS])
    m1, s1 = T.norm_stats(data, scope)
    nd = pub["norm"]
    norm = {"mean": nd["mean"], "std": nd["std"], "scope_root": EV.SortedMerkle(store.rids(scope)).root, "n": int(len(scope))}
    o = run_fl(data, store, policy, base_seed=a.seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=a.rounds,
               run_dir=os.path.join(a.runs, "reference"), label="ref", norm=norm)
    ref = {(c["C"]["k"], c["C"]["r"]): c for c in o["commits"]}
    diff = []
    for c in pub["commits"]:
        key = (c["C"]["k"], c["C"]["r"])
        x = ref[key]
        for f in ("root_S", "n", "sigma"):
            if x["C"][f] != c["C"][f]:
                diff.append([*key, "C." + f])
        for f in ("h_start", "h_delta"):
            if x["U"][f] != c["U"][f]:
                diff.append([*key, "U." + f])
        if x["B"] != c["B"]:
            diff.append([*key, "B"])
        if x["status"] != c["status"]:
            diff.append([*key, "status"])
    rh = {str(r): h for r, h in o["round_hashes"].items()}
    round_diff = [r for r, h in pub["round_hashes"].items() if r != "0" and rh.get(r) != h]
    out = {"units": len(pub["commits"]), "unit_diffs": diff, "round_diffs": round_diff,
           "final_equal": o["final_hash"] == pub["round_hashes"][str(a.rounds)],
           "norm_single": [m1, s1], "norm_dist": [nd["mean"], nd["std"]],
           "norm_rel_diff": [abs(m1 - nd["mean"]) / abs(m1), abs(s1 - nd["std"]) / s1],
           "norm_bit_equal": [m1 == nd["mean"], s1 == nd["std"]]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "unit_diffs"}), "unit_diffs", len(diff), flush=True)


if __name__ == "__main__":
    main()
