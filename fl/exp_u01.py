"""시연 실험: 철회 → 처리(U0·U1) → 검증(L1·L2) (3장 v1.6 4.4·5.3·5.4절).

정보주체 A: 라운드 1부터 참여, 라운드 20에 철회 효력 → 표준화 통계 범위에 포함(4.4절 동등 조건 위반 사례)
정보주체 B: 라운드 10부터 참여, 라운드 30에 철회 효력 → 표준화 통계 범위 밖(동등 조건 충족 사례)
각 정보주체에 대해 U0(처음부터 재학습)와 U1(r₀ 직전 체크포인트에서 재수행)을 만들고 결과 해시를 비교한다.
원 학습의 계산 단위 일부를 L1 재실행으로 검증하고, 부정행위(커밋 밖 사용) 1건을 주입해 탐지되는지 본다.

  python -m fl.exp_u01 --out /out --runs /runs
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from fl import audit as A
from fl import train as T
from fl.run import INSTS, Policy, Store, run_fl


def pick_subjects(data):
    by = {}
    for i in data.train_idx["1"]:
        by.setdefault(int(data.pid[i]), []).append(int(i))
    cand = sorted(p for p, v in by.items() if len(v) >= 3)
    return cand[0], cand[1]


def evaluate(model, data, device, norm):
    P = T.predict(model, data, data.test_idx, device, norm=(norm["mean"], norm["std"]))
    return T.macro_auc(data.Y[data.test_idx], P)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--l1-samples", type=int, default=6)
    args = ap.parse_args()

    cond = "gpu_det"
    device, max_pool = T.configure(cond)
    data = T.Data(args.cache)
    store = Store(data)
    a, b = pick_subjects(data)
    policy = Policy(enroll={b: 10}, withdraw={a: 20, b: 30})
    R, seed = args.rounds, args.base_seed
    common = dict(base_seed=seed, device=device, max_pool=max_pool, cond=cond, rounds=R)
    res = {"subjects": {"A": {"patient_id": a, "enroll": 1, "withdraw": 20},
                        "B": {"patient_id": b, "enroll": 10, "withdraw": 30}},
           "rounds": R, "base_seed": seed}

    # 원 학습(철회가 반영되는 정상 연합학습)
    orig = run_fl(data, store, policy, run_dir=os.path.join(args.runs, "orig"), label="orig", **common)
    res["orig"] = {"final_hash": orig["final_hash"], "norm": orig["norm"], "prog": orig["prog"],
                   "test_macro_auc": evaluate(orig["model"], data, device, orig["norm"]),
                   "rejected": sum(c["status"] != "accepted" for c in orig["commits"])}

    # 처리 작업: U0·U1
    for tag, pid, r0 in (("A", a, 1), ("B", b, 10)):
        pol = policy.without([pid])
        u0 = run_fl(data, store, pol, run_dir=os.path.join(args.runs, f"U0_{tag}"), label=f"U0_{tag}", **common)
        start = torch.load(os.path.join(args.runs, "orig", f"ckpt_r{r0 - 1}.pt"))
        u1 = run_fl(data, store, pol, run_dir=os.path.join(args.runs, f"U1_{tag}"), label=f"U1_{tag}",
                    start_state=start, start_round=r0, norm=orig["norm"], **common)
        same_norm = u0["norm"]["mean"] == orig["norm"]["mean"] and u0["norm"]["std"] == orig["norm"]["std"]
        subj = {k: sorted(store.subject_rids(pid, k)) for k in INSTS}
        res[tag] = {
            "r0": r0,
            "norm_scope_includes_subject": not same_norm,
            "U0": {"final_hash": u0["final_hash"], "test_macro_auc": evaluate(u0["model"], data, device, u0["norm"])},
            "U1": {"final_hash": u1["final_hash"], "test_macro_auc": evaluate(u1["model"], data, device, orig["norm"]),
                   "replayed_rounds": R - r0 + 1},
            "U1_equals_U0": u1["final_hash"] == u0["final_hash"],
            "prefix_equal_to_orig": all(u0["round_hashes"][r] == orig["round_hashes"][r] for r in range(1, r0)),
            "L2_U0": A.verify_input_set(store, pol, u0, subj),
            "L2_U1": A.verify_input_set(store, pol, u1, subj),
        }
        print(tag, json.dumps({k: res[tag][k] for k in ("U1_equals_U0", "norm_scope_includes_subject")}), flush=True)

    # L1: 원 학습의 계산 단위 표본 재실행
    rng = random.Random(seed)
    sample = rng.sample(orig["commits"], args.l1_samples)
    res["L1_sample"] = [{"k": c["C"]["k"], "r": c["C"]["r"],
                         **A.reexecute(data, store, policy, os.path.join(args.runs, "orig"), c, seed, device,
                                       max_pool, R, orig["norm"], orig["prog"])} for c in sample]

    # 부정행위 주입: 기관 1, 라운드 25에서 철회된 A의 기록을 커밋에서 빼고 실제 학습에만 사용(T2)
    bad = run_fl(data, store, policy, run_dir=os.path.join(args.runs, "tamper"), label="tamper",
                 start_state=torch.load(os.path.join(args.runs, "orig", "ckpt_r24.pt")), start_round=25,
                 norm=orig["norm"], tamper={("1", 25): "hidden_withdrawn"}, log_every=0, **{**common, "rounds": R})
    c = next(c for c in bad["commits"] if c["C"]["k"] == "1" and c["C"]["r"] == 25)
    res["tamper_T2_hidden"] = {"commit_accepted": c["status"], "tampered": c["tampered"],
                               **A.reexecute(data, store, policy, os.path.join(args.runs, "tamper"), c, seed, device,
                                             max_pool, R, orig["norm"], orig["prog"])}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "u01_demo.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("orig", "tamper_T2_hidden")}, ensure_ascii=False, default=str)[:1500])


if __name__ == "__main__":
    main()
