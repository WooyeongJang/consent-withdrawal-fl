"""시연: 철회 → 처리 → 검증 → 배포 전 과정을 ConsentLedger(Kaia Kairos)에 기록 (3장 v1.6 3–6절).

학습은 6라운드로 줄인다(원장 거래 수를 줄이기 위한 시연 설정). 정보주체 A(처리 수준 3), B(처리 수준 1)가
라운드 4에 철회한다. 처리 작업은 U0(처음부터 재학습)이다. 결과 모델은 L1·L2·L3 검증 기록 후에만 배포된다.

  python -m fl.exp_onchain --out /out --runs /runs   (원장 서비스가 127.0.0.1:8600에서 실행 중이어야 함)
"""

import argparse
import json
import os
import random
import time

import numpy as np

from fl import audit as A
from fl import evidence as EV
from fl import train as T
from fl.exp_u01 import pick_subjects
from fl.ledger_client import Ledger, b32, text32
from fl.run import INSTS, Policy, Store, run_fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--withdraw-round", type=int, default=4)
    args = ap.parse_args()

    L = Ledger()
    L.deploy("ConsentLedger", label="onchain demo")          # 시연마다 새 배포(v3.1: 학습 설정 커밋은 배포당 한 번)
    info = L.info()
    t_start = time.time()
    device, max_pool = T.configure("gpu_det")
    data, R, w = T.Data(args.cache), args.rounds, args.withdraw_round
    store = Store(data)
    a, b = pick_subjects(data)
    subj = {"A": (a, 3), "B": (b, 1)}
    s_commit = {t: EV.H("subject-commit", EV.subject_secret(p)) for t, (p, _) in subj.items()}
    res = {"contract": info["contract"], "chain_id": info["chain_id"], "roles": info["roles"],
           "subjects": {t: {"patient_id": p, "level": lv, "subject_commit": s_commit[t]} for t, (p, lv) in subj.items()}}

    # 1. 정책 등록(플랫폼)
    for t, (p, lv) in subj.items():
        L.tx("platform", "registerPolicy", b32(s_commit[t]), 1, b32(EV.H("policy", p, lv, 1)), lv)

    # 2. 정규 라운드 1..w-1
    common = dict(base_seed=1, device=device, max_pool=max_pool, cond="gpu_det", rounds=R, log_every=1)
    # 같은 학습을 두 구간으로 나눠 실행: 1..w-1(정책 0), 철회 후 w..R(정책 1). 표준화 통계는 시작 시점에 고정
    policy = Policy(withdraw={a: w, b: w})
    run_a = run_fl(data, store, policy, run_dir=os.path.join(args.runs, "orig"), label="orig", ledger=L,
                   stop_round=w - 1, **common)
    # 3. 철회 요청(정보주체마다 새 계정 서명, 플랫폼 대납). δ=0 → 즉시 효력
    req = {}
    for t, (p, lv) in subj.items():
        root = EV.SortedMerkle(store.subject_rids(p)).root
        out = L.tx("subject", "requestWithdrawal", b32(s_commit[t]), 1, b32(root))
        req[t] = next(e["args"]["reqId"] for e in out["events"] if e["event"] == "WithdrawalRequested")
    # 4. 첫 사용 라운드 신고(4.2절). A·B 모두 라운드 1부터 학습에 쓰였다(기관 1)
    for t in subj:
        L.tx("dc1", "declareFirstUse", req[t], 1)
    # 5. 철회 이후 라운드 w..R
    import torch
    start = torch.load(os.path.join(args.runs, "orig", f"ckpt_r{w - 1}.pt"))
    run_b = run_fl(data, store, policy, run_dir=os.path.join(args.runs, "orig"), label="orig", ledger=L,
                   start_state=start, start_round=w, norm=run_a["norm"], **common)
    final = run_b["final_hash"]
    # 6. 배포 가능 여부(I2): 처리 수준 3 철회의 영향 범위(r ≥ 1)에 든 모델은 배포 불가
    res["deployable_after_withdrawal"] = {"final_round_model": L.view("deployable", b32(final)),
                                          "round1_model": L.view("deployable", b32(run_a["round_hashes"][1]))}
    L.tx("platform", "revoke", b32(final), text32("influence range r0=1 (subject A, level 3)"))
    # 7. 처리 작업 U0(처음부터 재학습, A만 처리 수준 3)
    pol_wo = policy.without([a])
    L.approve_method("U0", run_a["prog"])                   # v3.1: 승인된 (방법, prog)
    L.tx("aggregator", "createJob", text32("U0"), b32(run_a["prog"]), text32("init-rule"), 1)
    job = run_fl(data, store, pol_wo, run_dir=os.path.join(args.runs, "U0"), label="U0", log_every=0,
                 **{k: v for k, v in common.items() if k != "log_every"})
    L.tx("aggregator", "completeJob", 1, b32(job["final_hash"]))
    # 8. 감사: 정규 라운드 L1 표본(원장 seed로 선정), 처리 작업 L1·L2·L3
    seeds = [e["args"]["seed"] for x in L.log if x["fn"] == "sampleAudit" for e in x["events"]]
    units = run_a["commits"] + run_b["commits"]
    picks = []
    for r_idx, seed in enumerate(seeds, start=1):
        rng = random.Random(int(seed, 16))
        cand = [c for c in units if c["C"]["r"] == r_idx]
        picks.append(rng.choice(cand))
    l1_regular = []
    for c in picks:
        v = A.reexecute(data, store, policy, os.path.join(args.runs, "orig"), c, 1, device, max_pool, R,
                        run_a["norm"], run_a["prog"])
        L.tx("auditor", "recordVerification", b32(v["binding"]), 1, v["result"] == "통과", text32(v["reason"] or "ok"))
        l1_regular.append({"k": c["C"]["k"], "r": c["C"]["r"], "result": v["result"]})
    job_again = run_fl(data, store, pol_wo, run_dir=os.path.join(args.runs, "U0_audit"), label="U0_audit", log_every=0,
                       **{k: v for k, v in common.items() if k != "log_every"})
    l1_job = job_again["final_hash"] == job["final_hash"]
    l2_job = A.verify_input_set(store, pol_wo, job, {k: store.subject_rids(a, k) for k in INSTS})
    L.tx("auditor", "recordVerification", b32(job["final_hash"]), 1, l1_job, text32("job re-execution"))
    L.tx("auditor", "recordVerification", b32(job["final_hash"]), 2, l2_job["result"] == "통과", text32("input set"))
    deploy_before_l3 = L.view("deployable", b32(job["final_hash"]))
    L.tx("auditor", "recordVerification", b32(job["final_hash"]), 3, True,
         text32("U0 retraining with L1+L2 pass: L3 not required (5.5)"))
    # 9. 배포(I3) 10. 단말 대조(L4)
    L.tx("platform", "deploy", b32(job["final_hash"]), text32("verified job U0"))
    deployed = L.view("deployed")
    res.update({
        "rounds": R, "withdraw_round": w, "reqIds": req, "final_model_before_job": final, "job_output": job["final_hash"],
        "l1_regular_samples": l1_regular, "l1_job": l1_job, "l2_job": l2_job["result"],
        "deployable_job_before_L3": deploy_before_l3,
        "device_check_L4": {"deployed": deployed, "local_model_hash": "0x" + job["final_hash"],
                            "match": deployed.lower() == "0x" + job["final_hash"],
                            "revoked_final_model_rejected": not L.view("deployable", b32(final))},
        "pendingLevel2_after_deploy": L.view("pendingLevel2"),
        "wall_seconds": time.time() - t_start,
    })
    # 거래 통계
    by_fn = {}
    for x in L.log:
        d = by_fn.setdefault(x["fn"], {"n": 0, "gas": [], "latency": [], "fee_kei": 0, "failed": 0})
        d["n"] += 1
        if x.get("status") == 1:
            d["gas"].append(x.get("gas_used") or 0); d["latency"].append(x.get("latency_s") or 0); d["fee_kei"] += x.get("fee_kei") or 0
        else:
            d["failed"] += 1
    res["tx_stats"] = {fn: {"n": d["n"], "failed": d["failed"], "gas_mean": float(np.mean(d["gas"])) if d["gas"] else None,
                            "latency_mean_s": float(np.mean(d["latency"])) if d["latency"] else None,
                            "fee_kaia": d["fee_kei"] / 1e18} for fn, d in by_fn.items()}
    res["tx_total"] = len(L.log)
    res["fee_total_kaia"] = sum(x.get("fee_kei", 0) for x in L.log) / 1e18
    res["transactions"] = [{k: x.get(k) for k in ("fn", "role", "hash", "block", "status", "gas_used", "latency_s")} for x in L.log]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "onchain_demo.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: res[k] for k in ("contract", "deployable_after_withdrawal", "l1_regular_samples", "l1_job", "l2_job",
                                          "deployable_job_before_L3", "device_check_L4", "pendingLevel2_after_deploy",
                                          "tx_total", "fee_total_kaia", "wall_seconds")}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
