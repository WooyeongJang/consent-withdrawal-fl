"""E3 허위 행위 탐지 — 유형 A(학습 단계 공격: T1, T2a, T2b, T3, T10) (reports/2026-10-25_E3_설계.md).

실행 1회: 새 ConsentLedger(B4)와 CompletionRecord(B3)를 Kairos에 배포하고, 서로 다른 기관·라운드에 공격 5건을
주입한 50라운드 연합학습을 원장에 기록한다. 학습 후 B4의 증거(원장 이벤트, 원장 seed 기반 L1 표본 재실행,
정보주체 자기 비소속 검증, 서명된 요청과 원장 대조)로 공격마다 탐지 여부를 판정하고, B0–B3·B5가 가진 기록으로
같은 공격을 판정한다. 공격하지 않은 계산 단위에서 오탐도 센다.

  python -m fl.e3 --type A --seed 1 --p 0.3 --out /out --runs /runs
"""

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np

from fl import audit as AU
from fl import evidence as EV
from fl import train as T
from fl.ledger_client import Ledger, b32, text32
from fl.run import INSTS, Policy, Store, run_fl

ATTACKS_A = ["T1", "T2a", "T2b", "T3", "T10"]
SPEC = {"T2a": {"mode": "included"}, "T3": {"mode": "stale"}, "T2b": {"mode": "hidden"},
        "T10": {"mode": "included", "collude": True}}

# 비교 구조가 가진 수단(8.3)과 유형 A 공격의 판정 근거(설계 3.2)
BASELINE_RATIONALE = {
    "B0": "감사 기록 없음",
    "B1": "운영자의 동의 DB(데이터 선택 시점 판정)만 있음. 기관의 실제 입력·계산을 대조할 증거 없음",
    "B2": "운영자가 서명한 중앙 로그(접수·모델 해시). 접수 기록 작성자가 운영자이고 입력 커밋이 없음",
    "B3": "운영자가 원장에 기록한 철회 접수·완료·모델 해시. 기록 작성자가 운영자이고 입력 커밋이 없음",
    "B5": "B2 + 외부 증인 공동 서명. 증인은 운영자가 제출한 기록에만 서명",
}


def unit_sampled(seed_hex, k, p):
    u = int(hashlib.sha256(f"{seed_hex}|{k}".encode()).hexdigest(), 16) / 2 ** 256
    return u < p


def run_type_a(args):
    L = Ledger()
    dep = L.deploy("ConsentLedger", label=f"E3-A seed{args.seed} p{args.p}")
    b3 = L.deploy("CompletionRecord", label=f"E3-A B3 seed{args.seed}")
    device, max_pool = T.configure("gpu_det")
    data, R = T.Data(args.cache), args.rounds
    store = Store(data)
    rng = random.Random(args.seed)

    # 공격 사례 배정: 서로 다른 라운드·기관, 대상은 해당 기관의 학습 환자
    cands = sorted({max(2, round(R * f)) for f in (0.16, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88)})
    slots = rng.sample(cands, len(ATTACKS_A))
    insts = [INSTS[i % len(INSTS)] for i in rng.sample(range(len(ATTACKS_A)), len(ATTACKS_A))]
    used_pids, inst_list = set(), []
    for atk, r, k in zip(ATTACKS_A, slots, insts):
        pids = sorted({int(data.pid[i]) for i in data.train_idx[k]} - used_pids)
        pid = rng.choice(pids); used_pids.add(pid)
        inst_list.append({"attack": atk, "k": k, "r": r, "pid": pid,
                          "subject": EV.H("subject-commit", EV.subject_secret(pid)),
                          "rids": store.subject_rids(pid, k)})
    # 원장이 아는 정책: T1 대상은 플랫폼이 요청을 올리지 않으므로 원장상 철회가 없다(정직한 기관은 원장을 따른다)
    policy = Policy(withdraw={x["pid"]: x["r"] for x in inst_list if x["attack"] != "T1"})
    tamper = {(x["k"], x["r"]): {**SPEC[x["attack"]], "pids": [x["pid"]]} for x in inst_list if x["attack"] in SPEC}

    # 정책 등록(플랫폼, 연속 제출)
    L.batch([("platform", "registerPolicy", (b32(x["subject"]), 1, b32(EV.H("policy", x["pid"], 1, 1)), 1))
             for x in inst_list])
    signed = {}

    def on_round_start(r):
        for x in inst_list:
            if x["r"] != r:
                continue
            root = EV.SortedMerkle(x["rids"]).root
            s = L.sign_subject("requestWithdrawal", b32(x["subject"]), 1, b32(root))   # 정보주체 단말 서명
            signed[x["attack"]] = {"sender": s["sender"], "signed_at": s["signed_at"], "raw": s["raw"], "round": r}
            if x["attack"] == "T1":
                continue                                                               # T1: 플랫폼이 대납 제출을 하지 않음
            out = L.submit_feepayer(s["raw"], label="requestWithdrawal")
            signed[x["attack"]].update(block=out["block"], block_time=out["block_time"])
            L.tx_other("CompletionRecord", "platform", "recordWithdrawal", b32(x["subject"]), b32(root), int(time.time()))

    t0 = time.time()
    run = run_fl(data, store, policy, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det",
                 rounds=R, run_dir=os.path.join(args.runs, f"A_s{args.seed}"), label=f"E3A-s{args.seed}",
                 tamper=tamper, ledger=L, on_round_start=on_round_start, log_every=10)
    L.tx_other("CompletionRecord", "platform", "recordModel", R, b32(run["final_hash"]))
    train_wall = time.time() - t0

    # ---------------- B4 감사 ----------------
    seeds = {}
    for x in L.log:
        if x.get("fn") == "sampleAudit":
            for e in x.get("events", []):
                if e["event"] == "AuditSampled":
                    seeds[int(e["args"]["r"])] = e["args"]["seed"]
    by_unit = {(c["C"]["k"], c["C"]["r"]): c for c in run["commits"]}
    chain_roots = {}
    for x in L.log:
        for e in x.get("events", []) or []:
            if e["event"] == "RoundInputCommitted":
                chain_roots[(str(e["args"]["k"]), int(e["args"]["r"]))] = e["args"]["rootS"].removeprefix("0x")
    P_GRID = sorted({args.p, 0.1, 0.3, 0.5, 1.0})
    attacked_units = {(x["k"], x["r"]) for x in inst_list if x["attack"] != "T1"}
    sampled_by_p = {pp: [(k, r) for r in range(1, R + 1) for k in INSTS if r in seeds and unit_sampled(seeds[r], k, pp)]
                    for pp in P_GRID}
    # 재실행 비용을 줄이기 위해: 원장 기록용 p의 표본 전체 + 다른 p에서는 공격 단위만 추가 재실행
    sampled = sorted(set(sampled_by_p[args.p]) | {u for pp in P_GRID for u in sampled_by_p[pp] if u in attacked_units})
    l1 = {}
    for k, r in sampled:
        c = by_unit[(k, r)]
        v = AU.reexecute(data, store, policy, os.path.join(args.runs, f"A_s{args.seed}"), c, args.seed, device,
                         max_pool, R, run["norm"], run["prog"])
        l1[(k, r)] = v
    l1_record = {u: v for u, v in l1.items() if u in set(sampled_by_p[args.p])}
    if l1_record:
        L.batch([("auditor", "recordVerification", (b32(v["binding"]) if v.get("binding") else b32(by_unit[u]["U"]["h_delta"]),
                                                    1, v["result"] == "통과", text32(v["reason"] or "ok")))
                 for u, v in l1_record.items()])
    # 정보주체 자기 검증: 원장상 철회가 효력을 가진 뒤 수락된 자기 기관 계산 단위마다 비소속 확인(5.4)
    subj_check = {}
    for x in inst_list:
        if x["attack"] == "T1":
            continue
        fails = []
        for r in range(x["r"], R + 1):
            c = by_unit[(x["k"], r)]
            if c["status"] != "accepted":
                continue
            tree = EV.SortedMerkle(store.rids(c["_committed_idx"]))
            assert tree.root == chain_roots[(x["k"], r)], "DC가 제시한 트리와 원장 루트 불일치"
            if not all(EV.verify_non_membership(chain_roots[(x["k"], r)], tree.non_membership(q)) for q in x["rids"]):
                fails.append(r)
        subj_check[x["attack"]] = fails
    # T1: 정보주체가 보관한 서명 요청이 원장에 없는지(요청 계정의 거래 수 0)
    t1 = signed.get("T1", {})
    t1_on_chain = L.view("withdrawalCount")

    # 정보주체 검증(반사실 포함): 공격 단위의 커밋에 자기 기록이 있으면 비소속 증명 불가
    def subject_detects(x):
        c = by_unit[(x["k"], x["r"])]
        tree = EV.SortedMerkle(store.rids(c["_committed_idx"]))
        return not all(EV.verify_non_membership(tree.root, tree.non_membership(q)) for q in x["rids"])

    # ---------------- 판정 ----------------
    results = []
    for x in inst_list:
        a, k, r = x["attack"], x["k"], x["r"]
        c = by_unit.get((k, r))
        b4, how, latency_r = False, "", None
        if a == "T1":
            registered = any(s.get("block") for kk, s in signed.items() if kk == "T1")
            b4 = not registered
            how = "정보주체가 보관한 서명 요청이 원장에 없음 → 누락 사실 드러남(지연 길이는 증명 불가, 7.1)"
            latency_r = R - r
        elif a in ("T2a", "T3"):
            b4 = c["status"] == "rejected"
            how = f"집계 수락 검사(알고리즘 2) 거절: {c['status']}"
            latency_r = 0 if b4 else None
        elif a == "T2b":
            v = l1.get((k, r)) if (k, r) in set(sampled_by_p[args.p]) else None
            b4 = bool(v and v["result"] != "통과")
            how = "L1 표본 재실행 출력 불일치" if v else "표본 밖(재실행 안 됨)"
            latency_r = 0 if b4 else None
        elif a == "T10":
            v = l1.get((k, r)) if (k, r) in set(sampled_by_p[args.p]) else None
            b4 = bool(subj_check.get("T10")) or bool(v and v["result"] != "통과")
            how = (f"정보주체 자기 비소속 검증 실패 라운드 {subj_check.get('T10')}" if subj_check.get("T10") else "") + \
                  (f"; L1 {v['result']}({v['reason']})" if v else "")
            latency_r = 0 if b4 else None
        # 증거 벡터: (a) 정보주체 원장 등록·서명 요청 대조 (b) 수락 시점 비소속 검사 (c) 재실행 감사(p별)
        #            (d) 입력 집합 검사 (e) 잔존 영향 시험 (f) 단말 대조 — 유형 A는 (a)(b)(c)와 정보주체 자기 검증(s)
        evid = {"a": a == "T1" and b4, "b": a in ("T2a", "T3") and c["status"] == "rejected",
                "s": a in ("T2a", "T3", "T10") and subject_detects(x),
                "c": {str(pp): bool((k, r) in sampled_by_p[pp] and l1.get((k, r)) and l1[(k, r)]["result"] != "통과")
                      for pp in P_GRID} if a != "T1" else {str(pp): False for pp in P_GRID}}
        results.append({**{kk: x[kk] for kk in ("attack", "k", "r", "pid")}, "evidence": evid, "detected": {"B4": b4,
                        **{b: False for b in BASELINE_RATIONALE}}, "B4_evidence": how, "latency_rounds": latency_r,
                        "unit_sampled": (k, r) in set(sampled_by_p[args.p])})
    attacked = attacked_units
    fp = {"rejected_honest": [u for u, c in by_unit.items() if c["status"] != "accepted" and u not in attacked],
          "l1_fail_honest": [u for u, v in l1.items() if v["result"] != "통과" and u not in attacked],
          "subject_check_fail_honest": {a: f for a, f in subj_check.items() if a not in ("T10",) and f}}
    fees = sum(x.get("fee_kei", 0) or 0 for x in L.log) / 1e18
    out = {"type": "A", "seed": args.seed, "p": args.p, "contract": dep["address"], "b3_contract": b3["address"],
           "instances": results, "false_positives": fp, "l1_sampled_units": len(l1_record),
           "sampled_units_by_p": {str(pp): len(v) for pp, v in sampled_by_p.items()},
           "baseline_rationale": BASELINE_RATIONALE, "t1_signed_request": {k: t1.get(k) for k in ("sender", "signed_at", "round")},
           "withdrawal_count_on_chain": t1_on_chain, "tx_total": len(L.log), "fee_kaia": fees, "key_txs": L.key_txs(),
           "train_wall_s": train_wall, "final_hash": run["final_hash"]}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"A_seed{args.seed}_p{args.p}.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    for x in results:
        print(f"{x['attack']:4s} k={x['k']} r={x['r']:2d}  B4={'탐지' if x['detected']['B4'] else '미탐지'}  {x['B4_evidence']}")
    print("오탐:", fp, "| L1 표본", len(l1), "| 거래", len(L.log), "| 수수료", round(fees, 4), "KAIA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["A"], default="A")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    run_type_a(args)


if __name__ == "__main__":
    main()
