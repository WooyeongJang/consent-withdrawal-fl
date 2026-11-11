"""E3 허위 행위 탐지 — 유형 B(처리·배포 단계 공격: T4–T9) (reports/2026-10-25_E3_설계.md).

기반: 후기 철회 시나리오(학습 환자 5% 철회 집단, 처리 수준 3, 라운드 45 효력)를 Kairos에 기록하며 50라운드 학습한다.
같은 설정의 오프체인 원 학습(results/late)과 결정적으로 같아야 하며, 그래서 후기 조건 그림자 모델로 L3를 판정한다.
공격 사례: 처리 작업 시도마다 1건(T4·T5·T6·T7·T11), 배포 단계(T8), 증거 단계(T9 세 변형).
T11(적응형 위조, 2026-10-26 추가)은 fl/t11.py. 비교 판정으로 L3 단독과 허용 오차 재현 검증(ε>0)을 함께 기록한다.
각 사례는 정직한 공정과 같은 원장 절차(createJob·completeJob·recordVerification·deployable)를 거친다.
정직한 처리 작업(U0)과 정직한 단말 대조로 오탐을 센다.

  python -m fl.e3b --seed 1 --n 20 --out /out --runs /runs --lirarun /lirarun --latejson results/late/late_methods.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from fl import audit as AU
from fl import evidence as EV
from fl import fedosd as F
from fl import train as T
from fl.baselines import SignedLog
from fl.exp_lira import evaluate, scores, stat
from fl.l3 import tost
from fl.ledger_client import Ledger, b32, method_kind, text32
from fl.run import INIT_SEED, INSTS, Policy, Store, run_fl

BASE_B = {  # 유형 B 공격에 대한 비교 구조의 판정(가진 기록으로 볼 수 있는 것, 설계 3.2)
    "T4": {"B0": False, "B1": False, "B2": False, "B3": False, "B5": False},
    "T5": {"B0": False, "B1": False, "B2": False, "B3": False, "B5": False},
    "T6": {"B0": False, "B1": False, "B2": False, "B3": False, "B5": False},
    "T7": {"B0": False, "B1": False, "B2": False, "B3": False, "B5": False},
    "T8": {"B0": False, "B1": False, "B2": False, "B3": False, "B5": False},
    "T11": {"B0": False, "B1": False, "B2": False, "B3": False, "B5": False},
}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--out", "--runs", "--lirarun", "--latejson"):
        ap.add_argument(a, required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--w", type=int, default=45)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--attacks", default="T4,T5,T6,T7,T8,T9,T11")
    ap.add_argument("--eps", default="results/t11/t11_eps.json", help="T11 허용 오차 보정값")
    ap.add_argument("--tag", default="")
    ap.add_argument("--t11-variants", default="ab", help="T11 변형(a–f, 35번 (b)): 순서대로 돌아가며 주입")
    ap.add_argument("--n-t11", type=int, default=None, help="T11 주입 수(기본 --n)")
    args = ap.parse_args()
    t_start = time.time()
    rng = random.Random(args.seed)
    run_dir = os.path.join(args.runs, f"B_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    L = Ledger()
    dep = L.deploy("ConsentLedger", label=f"E3-B seed{args.seed}")
    b3 = L.deploy("CompletionRecord", label=f"E3-B B3 seed{args.seed}")
    device, max_pool = T.configure("gpu_det")
    data, R = T.Data(args.cache), args.rounds
    store = Store(data)
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    cohort = sorted(int(p) for p in np.random.default_rng(20261025).choice(pids, int(len(pids) * args.frac), replace=False))
    cs = set(cohort)
    inst_of = {int(data.pid[i]): k for k in INSTS for i in data.train_idx[k]}
    X_by_k = {k: [int(i) for i in data.train_idx[k] if int(data.pid[i]) in cs] for k in INSTS}
    X = np.sort(np.array([i for k in INSTS for i in X_by_k[k]]))
    Z = np.sort(np.random.default_rng(1).choice(data.val_idx, min(len(X), len(data.val_idx)), replace=False))
    pool = np.r_[X, np.array([z for z in Z if data.inst[z] in INSTS])]
    policy = Policy(withdraw={p: args.w for p in cohort})
    pol_wo = policy.without(cohort)
    subj = {k: sorted(store.rids(np.array(X_by_k[k], dtype=np.int64))) for k in INSTS}
    sc = {p: EV.H("subject-commit", EV.subject_secret(p)) for p in cohort}
    b2 = SignedLog(os.path.join(run_dir, "b2_log.jsonl"))
    b5 = SignedLog(os.path.join(run_dir, "b5_log.jsonl"), witness_key=os.urandom(32))
    logs = (b2, b5)

    # 정책 등록과 철회(정보주체마다 새 계정 서명 + 플랫폼 대납), 첫 사용 라운드 신고, B2·B3·B5 기록
    L.batch([("platform", "registerPolicy", (b32(sc[p]), 1, b32(EV.H("policy", p, 3, 1)), 3)) for p in cohort])
    reqs = {}

    def on_round_start(r):
        if r != args.w:
            return
        for p in cohort:
            root = EV.SortedMerkle(store.subject_rids(p)).root
            out = L.submit_feepayer(L.sign_subject("requestWithdrawal", b32(sc[p]), 1, b32(root))["raw"], "requestWithdrawal")
            reqs[p] = next(int(e["args"]["reqId"]) for e in out["events"] if e["event"] == "WithdrawalRequested")
            for lg in logs:
                lg.append("withdrawal_received", subject=sc[p], round=r)
            L.tx_other("CompletionRecord", "platform", "recordWithdrawal", b32(sc[p]), b32(root), int(time.time()))
        L.batch([(f"dc{inst_of[p]}", "declareFirstUse", (reqs[p], 1)) for p in cohort])

    run = run_fl(data, store, policy, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                 run_dir=os.path.join(run_dir, "orig"), label=f"E3B-s{args.seed}", ledger=L,
                 on_round_start=on_round_start, log_every=10)
    final = run["final_hash"]
    late = json.load(open(args.latejson))
    # 오프라인 후기 철회 실험(시드 1)과의 비트 동일 확인은 시드 1에서만 정의된다(28번 (c): 학습 시드 = --seed)
    res = {"type": "B", "seed": args.seed, "contract": dep["address"], "b3_contract": b3["address"], "cohort": len(cohort),
           "orig_identical_to_offline_late": final == late["methods"]["P1"]["final_hash"] if args.seed == 1 else None}
    for lg in logs:
        lg.append("model", round=R, hash=final)
    L.tx_other("CompletionRecord", "platform", "recordModel", R, b32(final))
    res["I2"] = {"final_deployable": L.view("deployable", b32(final)),
                 "round1_deployable": L.view("deployable", b32(run["round_hashes"][1]))}
    L.tx("platform", "revoke", b32(final), text32("influence r0=1"))

    # 정규 라운드 L1 표본(원장 seed, p)
    seeds = {int(e["args"]["r"]): e["args"]["seed"] for x in L.log if x.get("fn") == "sampleAudit"
             for e in x.get("events", []) if e["event"] == "AuditSampled"}
    from fl.e3 import unit_sampled
    sampled = [c for c in run["commits"] if c["C"]["r"] in seeds and unit_sampled(seeds[c["C"]["r"]], c["C"]["k"], args.p)]
    l1_train = [AU.reexecute(data, store, policy, os.path.join(run_dir, "orig"), c, args.seed, device, max_pool, R,
                             run["norm"], run["prog"])["result"] for c in sampled]
    res["training_L1"] = {"sampled": len(sampled), "fail": sum(v != "통과" for v in l1_train)}

    # 감사 기준값: 정직한 U0와 그 재실행(캐시), LiRA(후기 그림자)
    u0 = run_fl(data, store, pol_wo, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                run_dir=os.path.join(run_dir, "U0"), label="U0", log_every=0)
    u0_audit = run_fl(data, store, pol_wo, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                      run_dir=os.path.join(run_dir, "U0_audit"), label="U0_audit", log_every=0)["final_hash"]
    zf = np.load(os.path.join(args.lirarun, "shadows_late.npz"))
    phis, inmask = zf["phis"], zf["inmask"]
    u0_aucs = [late["methods"][f"U0_s{s}"]["lira_offline"]["auc"] for s in range(1, 6)]
    nm = (run["norm"]["mean"], run["norm"]["std"])

    l3_available = phis.shape[1] == len(pool)   # 그림자 모델은 후기 철회 집단(5%)용. 스모크 테스트에서는 L3 생략

    def l3_of(model, norm):
        if not l3_available:
            return {"result": "통과", "auc": None, "diff": None, "ci90": None, "skipped": True}
        _, off = scores(stat(model, data, pool, device, norm), phis, inmask)
        e = evaluate(off, len(X))
        return {**tost(e["auc"], e["ci"], u0_aucs), "auc": e["auc"]}

    round_models = set(run["round_hashes"].values())
    instances = []

    def job(attack, method, h_out, l1_ok, l2_ok, l3, why, extra=None):
        L.approve_method(method, run["prog"])                   # v3.1: 승인된 (방법, prog)만 작업 생성(34번 (a) iv)
        exact = method_kind(method) == 1
        out = L.tx("aggregator", "createJob", text32(method), b32(run["prog"]), text32("init-rule"), len(cohort))
        jid = next(int(e["args"]["jid"]) for e in out["events"] if e["event"] == "JobCreated")
        L.tx("aggregator", "completeJob", jid, b32(h_out))
        for lg in logs:
            lg.append("job_completed", method=method, hash=h_out)
        L.tx_other("CompletionRecord", "platform", "recordCompletion", b32(EV.H("cohort", len(cohort))), text32(method), b32(h_out))
        vs = [("auditor", "recordVerification", (b32(h_out), 1, l1_ok, text32("L1"))),
              ("auditor", "recordVerification", (b32(h_out), 2, l2_ok, text32("L2")))]
        if l3 is not None:                                      # 정확 방법이면 참고값, 근사 방법이면 배포 조건
            vs.append(("auditor", "recordVerification", (b32(h_out), 3, l3["result"] == "통과",
                                                         text32("L3 TOST" + (" (reference, exact method)" if exact else "")))))
        L.batch(vs)
        # 판정은 원장에 기록된 검증 결과로 한다(I3, v3: L1·L2 통과, 근사 방법은 L3 통과도). 조회는 일관성 확인용.
        recorded_ok = bool(l1_ok and l2_ok and (exact or (l3 is not None and l3["result"] == "통과")))
        deployable = L.view("deployable", b32(h_out))
        assert deployable == recorded_ok, f"원장 조회({deployable})와 기록된 검증({recorded_ok}) 불일치"
        inst = {"attack": attack, "method": method, "h_out": h_out, "L1": l1_ok, "L2": l2_ok,
                "L3": None if l3 is None else {k: l3[k] for k in ("result", "auc", "diff", "ci90")},
                "lineage_duplicate": h_out in round_models, "deployable": deployable,
                "detected": {"B4": (not recorded_ok) if attack != "honest" else None, **BASE_B.get(attack, {})},
                "B4_evidence": why, **(extra or {})}
        instances.append(inst)
        print(f"{attack:6s} {method:18s} L1={l1_ok} L2={l2_ok} L3={inst['L3']['result'] if inst['L3'] else 'exempt'} "
              f"dup={inst['lineage_duplicate']} → {'배포 불가(탐지)' if not deployable else '배포 가능'}", flush=True)
        return inst

    l2_u0 = AU.verify_input_set(store, pol_wo, u0, subj)["result"] == "통과"
    hashes_by_round = run["round_hashes"]
    A = set(args.attacks.split(","))
    for i in range(args.n if A & {"T4", "T7"} else 0):
        # T4: 재학습 없이 원 모델(P1)을 U0 결과로 등록
        job("T4", "U0", final, u0_audit == final, l2_u0, None, "L1 재실행 출력 불일치, 출력 = 이전 모델")
        # T7: 과거 라운드 모델을 U0 결과로 등록
        j = rng.randint(1, R - 1)
        job("T7", "U0", hashes_by_round[j], u0_audit == hashes_by_round[j], l2_u0, None,
            f"L1 출력 불일치, 계보 중복(M_{j})", {"disguised_round": j})
    for i in range(args.n if "T5" in A else 0):
        # T5: 약한 FedOSD(일부만 수행)를 완료로 선언
        tu, lr = rng.choice([(2, 0.005), (5, 0.005), (5, 0.025), (10, 0.025), (20, 0.025)])
        start = torch.load(os.path.join(run_dir, "orig", f"ckpt_r{R}.pt"))
        o = F.fedosd(data, store, pol_wo, X, start, args.seed, device, max_pool, R, run["norm"], tu, 10, lr)
        again = F.fedosd(data, store, pol_wo, X, start, args.seed, device, max_pool, R, run["norm"], tu, 10, lr)["final_hash"]
        l2 = AU.verify_input_set(store, pol_wo, o, subj)["result"] == "통과" and \
            o["forget_commit"]["root_X"] == EV.SortedMerkle(store.rids(X)).root
        job("T5", f"U4-FedOSD(Tu{tu},lr{lr})", o["final_hash"], again == o["final_hash"], l2, l3_of(o["model"], nm),
            "L3 동등성 검정 불통과(잔존 영향)", {"Tu": tu, "lr": lr})
    for i in range(args.n if "T6" in A else 0):
        # T6: 다른 데이터로 재학습 — 철회자 일부를 남기거나(include) 비철회자를 추가로 빼기(exclude)
        if i % 2 == 0:
            keep = rng.sample(cohort, max(1, int(len(cohort) * rng.uniform(0.05, 0.3))))
            bad = policy.without([p for p in cohort if p not in set(keep)])
            variant = f"include {len(keep)} withdrawn"
        else:
            others = rng.sample([p for p in pids if p not in cs], int(len(cohort) * rng.uniform(0.05, 0.3)))
            bad = policy.without(cohort + others)
            variant = f"exclude {len(others)} valid"
        o = run_fl(data, store, bad, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                   run_dir=os.path.join(run_dir, "T6"), label="T6", log_every=0)
        again = run_fl(data, store, bad, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                       run_dir=os.path.join(run_dir, "T6_audit"), label="T6a", log_every=0)["final_hash"]
        l2 = AU.verify_input_set(store, pol_wo, o, subj)["result"] == "통과"
        job("T6", "U0", o["final_hash"], again == o["final_hash"], l2, None, "L2 입력 집합 불일치", {"variant": variant})
    if "T11" in A:
        from fl import t11 as TM
        eps = json.load(open(args.eps))["eps"]
        eps_step = json.load(open(args.eps.replace("t11_eps.json", "t11_eps_step.json")))["eps_step"]
        EPS_GRID = [eps_step, 1e-5, 1e-4, 1e-3, 1e-2]     # ε_step(같은 GPU 비결정 보정), …, Zhang et al. 실험 수준(1e-3)
        retain = np.setdiff1d(np.concatenate([data.train_idx[k] for k in INSTS]), X)
        u0_state = {k: v.detach().cpu() for k, v in u0["model"].state_dict().items()}
        nm_u0 = (u0["norm"]["mean"], u0["norm"]["std"])
        start = torch.load(os.path.join(run_dir, "orig", f"ckpt_r{R}.pt"))
        p1_state = {k: v.detach().cpu() for k, v in run["model"].state_dict().items()}
        VARS = args.t11_variants
        cache = {}

        def round_check(claimed_dir, norm):
            """라운드 단위 허용 오차 비교: 철회 이전 무작위 5라운드. 시드 규칙 검증자(S_wo 배치)와 제출 배치 검증자(이웃 치환 배치)."""
            if "nbr" not in cache:
                cache["nbr"] = TM.neighbor_map(data, X, retain, device)
            rs = sorted(rng.sample(range(1, args.w), 5))
            seed_rule = TM.round_deviation(data, store, pol_wo, claimed_dir, rs, norm, args.seed, device, max_pool, R,
                                          os.path.join(run_dir, "T11_roundcheck"))
            dv, pol_s, tam = TM.substitute_setup(data, cohort, cache["nbr"], args.w)
            claimed = TM.round_deviation(dv, store, pol_s, claimed_dir, rs, norm, args.seed, device, max_pool, R,
                                        os.path.join(run_dir, "T11_roundcheck"), tamper=tam)
            return {"seed_rule": seed_rule, "claimed_batch": claimed}

        def grind_candidates():                      # T11e: 기준 시드 후보 8개로 정직하게 재학습(실행당 한 번)
            if "e" not in cache:
                cands = []
                for c in range(1, 9):
                    o = run_fl(data, store, pol_wo, base_seed=args.seed * 1000 + c, device=device, max_pool=max_pool,
                               cond="gpu_det", rounds=R, run_dir=os.path.join(run_dir, f"T11e_c{c}"), label="T11e", log_every=0)
                    st = {k: v.detach().cpu() for k, v in o["model"].state_dict().items()}
                    cands.append({"seed": args.seed * 1000 + c, "model": o["model"], "hash": o["final_hash"], "norm": o["norm"],
                                  "dist_to_P1": TM.rel_dist(st, p1_state)})
                cache["e"] = cands
            return cache["e"]

        for i in range(args.n_t11 or args.n):
            v = VARS[i % len(VARS)]
            # 모든 변형: 정직한 U0 입력 커밋을 제시하고 결과를 "U0"라고 주장한다
            if v == "c":
                # T11c: 원 궤적(P1) 기록 위조(Adv-F형). 라운드마다 상대 γ = 10⁻³ 섭동, 재계산 없음
                cdir = os.path.join(run_dir, f"T11c_{i}")
                st = TM.perturb_trajectory(os.path.join(run_dir, "orig"), R, 1e-3, 300 + i, cdir)
                mdl = T.new_model(max_pool, device, 0); mdl.load_state_dict(st)
                h_out, l3 = T.model_hash(mdl)[0], l3_of(mdl, nm)
                dev = round_check(cdir, run["norm"])
                extra = {"variant": "c", "gamma": 1e-3, "round_dev": dev}
            elif v == "d":
                # T11d: 이웃 치환 재학습(Adv-R형). 잊을 기록 자리에 같은 레이블 최근접 이웃 신호
                if "nbr" not in cache:
                    cache["nbr"] = TM.neighbor_map(data, X, retain, device)
                o = TM.forge_substitute(data, store, cohort, cache["nbr"], R, args.w, device, max_pool, args.seed,
                                       os.path.join(run_dir, "T11d"))
                mdl, h_out = o["model"], o["final_hash"]
                l3 = l3_of(mdl, (o["norm"]["mean"], o["norm"]["std"]))
                l2_d = AU.verify_input_set(store, pol_wo, o, subj)["result"] == "통과"
                dev = round_check(os.path.join(run_dir, "T11d"), o["norm"])
                extra = {"variant": "d", "round_dev": dev, "L2_override": l2_d,
                         "same_label_neighbors": int(sum((data.Y[a_] == data.Y[b_]).all() for a_, b_ in cache["nbr"].items()))}
            elif v == "e":
                # T11e: 시드 고르기. 후보 4개 중 원 모델에 가장 가까운 정직한 재학습을 고른다(커밋된 기준 시드와 다름)
                pick = min(rng.sample(grind_candidates(), 4), key=lambda c_: c_["dist_to_P1"])
                mdl, h_out = pick["model"], pick["hash"]
                l3 = l3_of(mdl, (pick["norm"]["mean"], pick["norm"]["std"]))
                extra = {"variant": "e", "chosen_seed": pick["seed"], "dist_to_P1": pick["dist_to_P1"],
                         "u0_dist_to_P1": TM.rel_dist(u0_state, p1_state),
                         "candidate_dist_to_P1": [c_["dist_to_P1"] for c_ in grind_candidates()]}
            elif v == "f":
                # T11f: 미승인 코드(원 모델 1라운드 미세조정)를 U0로 선언. f1: 실제 prog를 밝힘 → 원장이 작업 생성 거부.
                # f2: 승인된 prog라고 속임 → L1 불일치
                o = run_fl(data, store, pol_wo, base_seed=args.seed, device=device, max_pool=max_pool, cond="gpu_det",
                           rounds=R + 1, start_state=start, start_round=R + 1, norm=run["norm"],
                           run_dir=os.path.join(run_dir, "T11f"), label="T11f", log_every=0)
                mdl, h_out, l3 = o["model"], o["final_hash"], l3_of(o["model"], nm)
                sub = "f1" if (i // len(VARS)) % 2 == 0 else "f2"
                extra = {"variant": "f", "sub": sub}
                if sub == "f1":
                    fake = EV.H("prog", "finetune-from-original", run["prog"])
                    out = L.tx("aggregator", "createJob", text32("U0"), b32(fake), text32("init-rule"), len(cohort), expect_ok=False)
                    refused = out.get("status") != 1
                    d = TM.rel_dist({k: v_.detach().cpu() for k, v_ in mdl.state_dict().items()}, u0_state)
                    instances.append({"attack": "T11", "method": "U0", "h_out": h_out, "L1": True, "L2": True, "L3": None,
                                      "ledger_refused": refused, "deployable": False, "lineage_duplicate": False,
                                      "detected": {"B4": refused, **BASE_B["T11"]},
                                      "B4_evidence": "원장: 승인되지 않은 (방법, prog)로 처리 작업 생성 거부",
                                      **extra, "rel_dist_to_U0": d, "eps_end": eps,
                                      "counterfactual": {"L3_only": l3["result"] != "통과", "tolerance_end": d > eps,
                                                         "name_only_approval": False}})
                    print(f"T11    U0 (prog 미승인)        → {'원장 거부(탐지)' if refused else '생성됨'}", flush=True)
                    continue
            elif v == "a":
                mdl, steps, l3 = TM.forge_l3match(data, X, pool, retain, start, nm, device, max_pool, phis, inmask,
                                                 lambda m_: l3_of(m_, nm), seed=200 + i)
                h_out = T.model_hash(mdl)[0]
                extra = {"variant": "a", "steps": steps}
            else:                                           # v == "b"
                nk = rng.choice([1, 5, 25, 100, len(cohort)])
                o = TM.forge_inject(data, store, cohort, rng.sample(cohort, nk), 1, os.path.join(run_dir, "U0"), u0["norm"],
                                   R, device, max_pool, os.path.join(run_dir, "T11b"), base_seed=args.seed)
                mdl, h_out = o["model"], o["final_hash"]
                l3 = l3_of(mdl, nm_u0)
                extra = {"variant": "b", "n_inject": nk}
            d = TM.rel_dist({k: v.detach().cpu() for k, v in mdl.state_dict().items()}, u0_state)
            # 비교 판정: L3 단독, 최종 모델 허용 오차 비교(ε_end), 라운드 단위 허용 오차 비교(T11b만 정의됨, ε 격자)
            cf = {"L3_only": l3["result"] != "통과", "tolerance_end": d > eps}
            if extra["variant"] == "b":
                cf.update({f"tolerance_step_{e:g}": d > e for e in EPS_GRID})
            if extra["variant"] in ("c", "d"):
                for vk in ("seed_rule", "claimed_batch"):       # 두 검증자 모형 × ε 격자(ε = 0 포함)
                    mx = max(extra["round_dev"][vk].values())
                    cf.update({f"tolerance_step_{vk}_{e:g}": mx > e for e in [0.0] + EPS_GRID})
            if extra["variant"] == "e":
                cf["L1_without_config_commit"] = False    # 제시한 시드로 재실행하면 같은 결과(정직한 계산) → 설정 커밋 없이는 통과
            if extra["variant"] == "f":
                cf["name_only_approval"] = u0_audit != h_out   # 이름만 승인(v3)이어도 prog를 속이면 L1이 잡는다
            extra.update({"rel_dist_to_U0": d, "eps_end": eps, "counterfactual": cf})
            job("T11", "U0", h_out, u0_audit == h_out, extra.pop("L2_override", l2_u0), l3,
                "L1 비트 단위 재실행 불일치(선언한 U0 계산과 다름)", extra)
    # 정직한 처리 작업(오탐 확인) → 배포
    honest = job("honest", "U0", u0["final_hash"], u0_audit == u0["final_hash"], l2_u0, None, "정직한 U0")
    if honest["deployable"]:
        L.tx("platform", "deploy", b32(u0["final_hash"]), text32("verified honest U0"))
    deployed = L.view("deployed")
    # T8: 단말에 검증되지 않은(이전) 모델 전달 → 단말 대조(L4)
    olds = ([final] + [hashes_by_round[rng.randint(1, R)] for _ in range(args.n - 1)]) if "T8" in A else []
    for h in olds:
        ok = (("0x" + h) == deployed.lower()) and L.view("deployable", b32(h))
        instances.append({"attack": "T8", "delivered": h, "device_accepts": ok,
                          "detected": {"B4": not ok, **BASE_B["T8"]}, "B4_evidence": "단말 대조: 배포 허용 모델 아님"})
    honest_device = (("0x" + u0["final_hash"]) == deployed.lower()) and L.view("deployable", b32(u0["final_hash"]))
    # T9: (a) 수락 기록 누락 후 집계 (b) 감사자에게 원장과 다른 커밋 제시 (c) 중앙 로그 사후 수정
    chain_roots = {(str(e["args"]["k"]), int(e["args"]["r"])): e["args"]["rootS"].removeprefix("0x")
                   for x in L.log for e in (x.get("events") or []) if e["event"] == "RoundInputCommitted"}
    for i in range(args.n if "T9" in A else 0):
        v = "abc"[i % 3]
        if v == "a":
            r = 1000 + i
            L.tx("platform", "openRound", r, int(time.time()) + 3600, b32(final))
            L.batch([(f"dc{k}", f, (int(k), r, b32(EV.H("x", i, k)), 1, 1, text32("p")) if f == "commitInput"
                      else (int(k), r, b32(EV.H("d", i, k)), b32(final))) for k in INSTS for f in ("commitInput", "commitUpdate")])
            L.batch([("aggregator", "decide", (int(k), r, True, text32("ok"))) for k in INSTS[:2]])
            out = L.tx("aggregator", "aggregate", r, b32(EV.H("m", i)), b32(EV.H("m", i)), expect_ok=False)
            b4 = out.get("status") != 1
            det = {"B4": b4, "B0": False, "B1": False, "B2": False, "B3": False, "B5": False}
            why = "원장 상태 전이 불가(수락 기록 누락 시 집계 거부)"
        elif v == "b":
            k, r = rng.choice(list(chain_roots))
            forged = EV.H("forged", i)
            b4 = forged != chain_roots[(k, r)]
            det = {"B4": b4, "B0": False, "B1": False, "B2": False, "B3": False, "B5": True}
            why = "기관이 제시한 커밋과 원장 기록 교차 대조 불일치(B5는 증인 사본 대조)"
        else:
            for lg in logs:
                lg.save()
            tampered = [dict(e) for e in b2.entries]
            idx = rng.randrange(len(tampered))
            body = json.loads(tampered[idx]["body"]); body["tampered"] = True
            tampered[idx]["body"] = json.dumps(body, sort_keys=True)
            det = {"B4": True, "B0": False, "B1": False, "B2": not b2.verify(tampered), "B3": True,
                   "B5": not b5.matches_witness(tampered)}
            why = "원장 기록은 변경 불가(사후 수정 시도 자체가 불가능)"
        instances.append({"attack": "T9", "variant": v, "detected": det, "B4_evidence": why})

    fees = sum(x.get("fee_kei", 0) or 0 for x in L.log) / 1e18
    res.update({"instances": instances, "false_positives": {"honest_job_deployable": honest["deployable"],
                "honest_device_accepts": honest_device, "training_L1_fail": res["training_L1"]["fail"]},
                "tx_total": len(L.log), "fee_kaia": fees, "key_txs": L.key_txs(), "wall_s": time.time() - t_start})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"B_seed{args.seed}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    from collections import Counter
    c = Counter((x["attack"], x["detected"]["B4"]) for x in instances if x["attack"] != "honest")
    print("B4 탐지:", dict(c), "| 오탐:", res["false_positives"], "| 동일 재현:", res["orig_identical_to_offline_late"],
          "| 거래", len(L.log), "| 수수료", round(fees, 3))


if __name__ == "__main__":
    main()
