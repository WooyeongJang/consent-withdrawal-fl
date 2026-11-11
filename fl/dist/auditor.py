"""감사자 컨테이너: 데이터센터 k의 학습 기록 조각(/shard)과 저장소(/store)를 읽기 전용으로 보고,
공개 원장 사본(/pub/public.json)의 커밋을 검증한다(3장 알고리즘 3·4).

  DC_K=0 python -m fl.dist.auditor --shard /shard --store /store --pub /pub/public.json --out /out --p 0.3 --sample-seed 1
- L2: 정책으로 재구성한 입력 집합의 루트 = 커밋된 root_S (모든 라운드)
- L1: 표본 라운드에서 저장소의 출발 상태로 재실행해 업데이트 해시와 결속값을 비교
"""

import argparse
import json
import os

import numpy as np
import torch

from fl import evidence as EV
from fl import train as T
from fl.dist.dc import ShardData, ShardStore, policy_from
from fl.run import INIT_SEED, PURPOSE, prog_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="/shard")
    ap.add_argument("--store", default="/store")
    ap.add_argument("--pub", default="/pub/public.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--sample-seed", type=int, default=1)
    a = ap.parse_args()
    k = os.environ["DC_K"]
    device, max_pool = T.configure("gpu_det")
    data = ShardData(a.shard, k)
    store = ShardStore(data, k)
    pub = json.load(open(a.pub))
    policy = policy_from(pub["policy"])
    prog = prog_id("gpu_det")
    units = [c for c in pub["commits"] if c["C"]["k"] == k]
    rng = np.random.default_rng([a.sample_seed, int(k)])
    sample = {int(r) for r in rng.choice([c["C"]["r"] for c in units], max(1, int(round(a.p * len(units)))), replace=False)}
    l2, l1 = [], []
    for c in units:
        C, U, r = c["C"], c["U"], c["C"]["r"]
        S = store.input_set(policy, r)
        ok2 = EV.SortedMerkle(store.rids(S)).root == C["root_S"]
        l2.append({"r": r, "ok": ok2})
        if r not in sample:
            continue
        start = torch.load(os.path.join(a.store, f"global_r{r - 1}.pt"), weights_only=True)
        m = T.new_model(max_pool, device, INIT_SEED)
        m.load_state_dict(start)
        ok_start = T.model_hash(m)[0] == U["h_start"]
        m, _ = T.local_train(m, data, k, r, pub["base_seed"], device, pub["rounds"], idx=S,
                             norm=(pub["norm"]["mean"], pub["norm"]["std"]))
        h = T.model_hash(m)[0]
        B = EV.binding_value(k, r, U["h_start"], C["root_S"], C["sigma"], PURPOSE, prog, h)
        l1.append({"r": r, "start_ok": ok_start, "update_ok": h == U["h_delta"], "binding_ok": B == c["B"]})
    out = {"k": k, "units": len(units), "L2_pass": sum(x["ok"] for x in l2), "L1_sampled": len(l1),
           "L1_pass": sum(x["start_ok"] and x["update_ok"] and x["binding_ok"] for x in l1), "L1": l1}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"audit_dc{k}.json"), "w"), indent=1)
    print(f"[auditor dc{k}] L2 {out['L2_pass']}/{len(units)}  L1 {out['L1_pass']}/{len(l1)}", flush=True)


if __name__ == "__main__":
    main()
