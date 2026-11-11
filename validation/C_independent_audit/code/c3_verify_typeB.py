"""C-1·C-3·C-4(유형 B: T4–T9, T11, 정직한 작업): 원장 이벤트, 저장된 체크포인트 해시, 데이터 분할로 판정을 다시 도출한다.

근거
 (1) Kairos 이벤트(ConsentLedger·CompletionRecord), 최신 상태의 읽기 호출(eth_call).
 (2) 저장된 체크포인트의 모델 해시와 상대 거리(results/ckpt_hashes.json, c2_hash_ckpts.py).
 (3) 저장된 실행 폴더의 events.jsonl(작업의 입력 커밋은 원장에 올리지 않으므로 여기에만 있다).
 (4) 실행 로그 JSON의 수치(AUC, CI, 거리, 편차). 판정 필드(detected, L1, L2, L3.result, deployable, counterfactual)는 비교 대상.

  python3 c3_verify_typeB.py
"""
import collections
import json
import statistics
import time
from pathlib import Path

import c_lib as L

OUT = L.EXP / "results"
RUNS = Path.home() / "data/runs/final_e3b"
DELTA, Z90 = 0.05, 1.6448536269514722
FEDOSD = [(2, 0.005), (5, 0.005), (5, 0.025), (10, 0.025), (20, 0.025)]
METHODS = {L.text32("U0"): "U0", **{L.text32(f"U4-FedOSD(Tu{tu},lr{lr})"): f"U4-FedOSD(Tu{tu},lr{lr})" for tu, lr in FEDOSD}}


def load_events(table, addr):
    logs = json.load(open(L.RAW / "merged" / f"{addr.lower()}.json"))
    ev = [L.decode_log(table, l) for l in logs]
    assert all(e is not None for e in ev)
    ev.sort(key=lambda e: (e["block"], e["logIndex"]))
    return ev


def events_file_roots(path):
    out = {}
    for line in open(path):
        e = json.loads(line)
        if e["event"] == "RoundInputCommitted":
            out[(e["k"], e["r"])] = (e["root_S"], e["n"])
    return out


def call_result(out):
    return out.get("result"), json.dumps(out.get("error")) if "error" in out else None


def main():
    hdr = L.run_header(__file__)
    abi, _, _ = L.compile_abi("ConsentLedger.sol", "ConsentLedger")
    abi3, _, _ = L.compile_abi("CompletionRecord.sol", "CompletionRecord")
    table, table3 = L.event_table(abi), L.event_table(abi3)
    meta = L.Meta()
    cand = L.candidate_progs()
    scope_root = L.merkle_root([meta.rid[e] for k in meta.insts for _, _, e in meta.train[k]])
    idx = json.load(open(OUT / "chain_index.json"))
    ck = json.load(open(OUT / "ckpt_hashes.json"))["seeds"]
    late = json.load(open(L.REPO / "results/late/late_methods.json"))
    u0_refs = [late["methods"][f"U0_s{s}"]["lira_offline"]["auc"] for s in range(1, 6)]
    mu, sd = statistics.mean(u0_refs), statistics.stdev(u0_refs)
    eps_end = json.load(open(L.REPO / "results/t11/t11_eps.json"))["eps"]
    eps_step = json.load(open(L.REPO / "results/t11/t11_eps_step.json"))["eps_step"]
    by_addr = {c["address"]: c for c in idx["contracts"]}
    runs = []
    for c in sorted((x for x in idx["contracts"] if x["type"] == "B" and x["which"] == "contract"), key=lambda x: x["seed"]):
        s = c["seed"]
        h = ck[str(s)]
        ev = load_events(table, c["address"])
        by = collections.defaultdict(list)
        for e in ev:
            by[e["event"]].append(e)
        pos = lambda e: (e["block"], e["logIndex"])
        res = {"seed": s, "contract": c["address"]}
        # --- 설정 커밋과 prog ---
        cfg = by["TrainingConfigCommitted"][0]["args"]["config"]
        progs = [p for p in cand if L.H("config", s, L.SEED_RULE, p, scope_root) == cfg]
        res["config_commit_progs"] = [{"prog": p, **cand[p]} for p in progs]
        res["job_progs"] = sorted({e["args"]["prog"] for e in by["JobCreated"]})
        res["prog_consistent"] = res["job_progs"] == progs
        # --- 철회 집단(원장에서 복원) ---
        opened = sorted((pos(e), e["args"]["r"]) for e in by["RoundOpened"])
        cohort, wr = [], collections.Counter()
        for e in by["WithdrawalRequested"]:
            pid = meta.subject_commit.get(e["args"]["subject"])
            cohort.append(pid)
            wr[max(r for p, r in opened if p < pos(e))] += 1
        res["cohort"] = len(cohort)
        res["cohort_all_identified"] = None not in cohort and len(set(cohort)) == len(cohort)
        res["withdrawal_rounds"] = {str(k): v for k, v in wr.items()}
        res["level3_withdrawals"] = sum(e["args"]["level"] == 3 for e in by["WithdrawalRequested"])
        res["first_use_declared"] = {str(k): v for k, v in collections.Counter(e["args"]["r0"] for e in by["FirstUseDeclared"]).items()}
        cs = set(cohort)
        # --- 정규 라운드: 계보 해시와 입력 루트 ---
        agg = {e["args"]["r"]: e["args"] for e in by["RoundAggregated"]}
        res["lineage_hash_match"] = sum(agg[r]["hOut"] == h["orig"][str(r)] for r in range(1, 51))
        res["start_model_match"] = next(e["args"]["startModel"] for e in by["RoundOpened"] if e["args"]["r"] == 1) == h["orig"]["0"]
        inp = {(str(e["args"]["k"]), e["args"]["r"]): e["args"] for e in by["RoundInputCommitted"] if e["args"]["r"] <= 50}
        root_all = {k: (L.merkle_root(meta.input_rids(k, lambda p: True)), len(meta.input_rids(k, lambda p: True))) for k in meta.insts}
        root_wo = {k: (L.merkle_root(meta.input_rids(k, lambda p: p not in cs)), len(meta.input_rids(k, lambda p: p not in cs))) for k in meta.insts}
        res["round_roots_match"] = sum((a["rootS"], a["n"]) == (root_all[k] if r < 45 else root_wo[k]) for (k, r), a in inp.items())
        res["round_units"] = len(inp)
        res["rejected_round_units"] = sum(1 for e in by["UpdateRejected"] if e["args"]["r"] <= 50)
        seeds = {e["args"]["r"]: "0x" + e["args"]["seed"] for e in by["AuditSampled"]}
        res["sampled_p0.3"] = sum(L.unit_sampled(seeds[r], k, 0.3) for (k, r) in inp)
        # --- 배포 전이 ---
        dch = [(e["args"]["model"], e["args"]["state"]) for e in by["DeploymentChanged"]]
        res["deployment_changes"] = dch
        res["revoked_final"] = (h["orig"]["50"], 2) in dch
        deployed_evt = next((m for m, st in reversed(dch) if st == 1), None)
        res["deployed_is_honest_U0"] = deployed_evt == h["final"]["U0"]["hash"]
        dep_latest, _ = call_result(L.eth_call(c["address"], "deployed()"))
        res["deployed_latest_call"] = dep_latest[2:] if dep_latest else None
        # --- 작업 ---
        approved = {(e["args"]["method"], e["args"]["prog"]): e["args"]["kind"] for e in by["MethodApproved"]}
        res["approved_pairs"] = sorted(f"{METHODS.get(m, m)}|{p[:8]}|kind{k}" for (m, p), k in approved.items())
        jobs = {}
        seq = [e for e in ev if e["event"] in ("JobCreated", "JobCompleted", "VerificationRecorded", "DeploymentChanged")]
        ver = {}                       # 컨트랙트 상태 재연: verification[target][level]
        pending = res["level3_withdrawals"]     # pendingLevel2(수준 2 이상 미해소 철회 수)
        cur = None
        for e in seq:
            a = e["args"]
            if e["event"] == "JobCreated":
                cur = a["jid"]
                jobs[cur] = {"jid": cur, "method": METHODS.get(a["method"], a["method"]), "kind": a["kind"], "covers": a["covers"],
                             "prog": a["prog"], "ver": {}}
            elif e["event"] == "JobCompleted":
                jobs[a["jid"]]["hOut"] = a["hOut"]
            elif e["event"] == "VerificationRecorded":
                ver.setdefault(a["target"], {})[a["level"]] = a["passed"]
                j = jobs.get(cur)
                if j is not None and j.get("hOut") == a["target"]:
                    j["ver"][a["level"]] = a["passed"]
                    v = ver[a["target"]]
                    j["deployable_replay"] = bool(v.get(1) and v.get(2) and (j["kind"] == 1 or v.get(3))) and j["covers"] == pending
            elif e["event"] == "DeploymentChanged" and a["state"] == 1:
                pending = 0
        res["jobs"] = len(jobs)
        res["jobs_completed"] = sum("hOut" in j for j in jobs.values())
        h_audit, h_u0, h_p1 = h["final"]["U0_audit"]["hash"], h["final"]["U0"]["hash"], h["orig"]["50"]
        res["U0_equals_U0_audit"] = h_u0 == h_audit
        res["U0_chain_rounds_equal_audit"] = h["U0_rounds"] == h["U0_audit_rounds"] and len(h["U0_rounds"]) == 51
        lineage = {h["orig"][str(r)]: r for r in range(1, 51)}
        known = {h_p1: "P1(final)", h_u0: "U0"}
        for name, v in h["final"].items():
            if name.startswith(("T11", "T6")) and "hash" in v:
                known.setdefault(v["hash"], name)
        # 작업 입력 커밋(원장 밖): 정직한 U0의 입력 루트가 기대 집합(철회자를 처음부터 뺀 집합)과 같은가 → L2
        u0_roots = events_file_roots(RUNS / f"B_s{s}/U0/events.jsonl")
        l2_u0 = len(u0_roots) == 150 and all(v == root_wo[k] for (k, r), v in u0_roots.items())
        res["L2_U0_input_roots_match"] = l2_u0
        t6_roots = events_file_roots(RUNS / f"B_s{s}/T6/events.jsonl")
        res["T6_last_roots_mismatch_units"] = sum(v != root_wo[k] for (k, r), v in t6_roots.items())
        res["T6_last_L1_reexec_equal"] = h["final"]["T6"]["hash"] == h["final"]["T6_audit"]["hash"]
        t11d_roots = events_file_roots(RUNS / f"B_s{s}/T11d/events.jsonl")
        res["T11d_roots_mismatch_units"] = sum(v != root_wo[k] for (k, r), v in t11d_roots.items())
        # --- 실행 로그 JSON과 순서 맞춤(비교용: 유형 이름과 수치) ---
        j = json.load(open(L.REPO / "results/final/e3" / f"B_seed{s}.json"))
        with_job = [x for x in j["instances"] if x["attack"] in ("T4", "T5", "T6", "T7", "T11", "honest") and not x.get("ledger_refused")]
        res["json_jobs"] = len(with_job)
        rows = []
        for jid, x in zip(sorted(jobs), with_job):
            jb = jobs[jid]
            attack = x["attack"] + (x.get("variant", "") if x["attack"] == "T11" else "")
            # 독립 도출
            if jb["method"] == "U0":
                l1 = jb["hOut"] == h_audit                      # 선언한 U0 계산의 저장된 재실행 출력과 비교
                l1_src = "ckpt"
            else:
                l1, l1_src = None, "none"                       # FedOSD 산출물 미보존
            if attack == "T6":
                last_t6 = jb["hOut"] == h["final"]["T6"]["hash"]
                l1, l1_src = (res["T6_last_L1_reexec_equal"], "ckpt") if last_t6 else (None, "none")
                l2, l2_src = (res["T6_last_roots_mismatch_units"] == 0, "events.jsonl") if last_t6 else (None, "none")
            elif attack == "T11d":
                l2, l2_src = res["T11d_roots_mismatch_units"] == 0, "events.jsonl"
            elif jb["method"] == "U0":
                l2, l2_src = l2_u0, "events.jsonl"              # 정직한 U0 입력 커밋을 제시
            else:
                l2, l2_src = None, "none"
            l3 = None
            if x.get("L3") and x["L3"].get("ci90"):
                lo, hi = x["L3"]["ci90"]
                cls = "pass" if (lo > -DELTA and hi < DELTA) else "exceeds" if lo > 0 else "inconclusive"
                se = (hi - lo) / (2 * Z90)
                l3 = {"class": cls, "diff_recomputed": x["L3"]["auc"] - mu, "diff_stored": x["L3"]["diff"],
                      "se_implied": se, "se_ge_ref_part": se >= sd / 5 ** 0.5 - 1e-12}
            rows.append({"jid": jid, "attack": attack, "method_chain": jb["method"], "kind": jb["kind"], "hOut": jb["hOut"],
                         "hOut_equals_json": jb["hOut"] == x["h_out"], "known_artifact": known.get(jb["hOut"]),
                         "lineage_round": lineage.get(jb["hOut"]), "chain_ver": jb["ver"], "deployable_replay": jb.get("deployable_replay"),
                         "L1_indep": l1, "L1_src": l1_src, "L2_indep": l2, "L2_src": l2_src, "L3_indep": l3,
                         "json": {"L1": x["L1"], "L2": x["L2"], "L3": (x["L3"] or {}).get("result"), "deployable": x["deployable"],
                                  "B4": x["detected"]["B4"], "lineage_duplicate": x["lineage_duplicate"]},
                         "variant_extra": {k: x.get(k) for k in ("sub", "rel_dist_to_U0", "round_dev", "counterfactual", "n_inject",
                                                                 "chosen_seed", "disguised_round") if k in x}})
        res["job_rows"] = rows
        # --- T11f1(원장이 작업 생성을 거부): 승인 목록과 최신 상태 조회 ---
        f1 = [x for x in j["instances"] if x.get("ledger_refused") is not None]
        fake = L.H("prog", "finetune-from-original", progs[0]) if progs else None
        mk, err = call_result(L.eth_call(c["address"], "methodKind(bytes32,bytes32)", [L.text32("U0"), fake]))
        res["T11f1"] = {"n": len(f1), "fake_prog_pair_in_MethodApproved": (L.text32("U0"), fake) in approved,
                        "methodKind_latest": int(mk, 16) if mk else err, "json_refused": sum(bool(x["ledger_refused"]) for x in f1),
                        "h_out_equals_T11f_ckpt": sum(x["h_out"] == h["final"]["T11f"]["hash"] for x in f1)}
        # --- T8(L4): 단말에 전달된 모델과 원장의 배포 상태 ---
        t8 = [x for x in j["instances"] if x["attack"] == "T8"]
        t8rows = []
        for x in t8:
            d = x["delivered"]
            out, err = call_result(L.eth_call(c["address"], "deployable(bytes32)", [d]))
            t8rows.append({"delivered": d, "in_lineage": d in lineage, "equals_deployed_event": d == deployed_evt,
                           "deployable_latest": (int(out, 16) != 0) if out else err, "json_B4": x["detected"]["B4"]})
        res["T8"] = t8rows
        out, err = call_result(L.eth_call(c["address"], "deployable(bytes32)", [h_u0]))
        res["honest_deployable_latest"] = (int(out, 16) != 0) if out else err
        # --- T9a: 수락 기록이 빠진 라운드의 집계 ---
        aggregator = "0x" + call_result(L.eth_call(c["address"], "aggregator()"))[0][-40:]
        t9a = []
        for r in sorted(e["args"]["r"] for e in by["RoundOpened"] if e["args"]["r"] >= 1000):
            n_in = sum(e["args"]["r"] == r for e in by["RoundInputCommitted"])
            n_acc = sum(e["args"]["r"] == r for e in by["UpdateAccepted"])
            n_rej = sum(e["args"]["r"] == r for e in by["UpdateRejected"])
            out, err = call_result(L.eth_call(c["address"], "aggregate(uint32,bytes32,bytes32)", [r, L.H("probe", r), L.H("probe", r)], frm=aggregator))
            t9a.append({"r": r, "inputs": n_in, "accepted": n_acc, "rejected": n_rej, "aggregated_event": r in agg,
                        "aggregate_call_now": "reverts" if err else "succeeds", "revert_has_decisions": bool(err and ("decisions" in err or "6465636973696f6e73" in err))})
        res["T9a"] = t9a
        t9 = collections.Counter(x["variant"] for x in j["instances"] if x["attack"] == "T9")
        res["T9_json_variants"] = dict(t9)
        res["T9_json_B4"] = dict(collections.Counter(f'{x["variant"]}:{x["detected"]["B4"]}' for x in j["instances"] if x["attack"] == "T9"))
        chain_roots = {a["rootS"] for a in inp.values()}
        res["T9b_forged_not_on_chain"] = sum(L.H("forged", i) not in chain_roots for i in range(20) if i % 3 == 1)
        # --- CompletionRecord(B3) ---
        ev3 = load_events(table3, by_addr_b3(idx, s))
        res["b3_events"] = dict(collections.Counter(e["event"] for e in ev3))
        res["json"] = {"training_L1": j["training_L1"], "false_positives": j["false_positives"], "I2": j["I2"],
                       "orig_identical_to_offline_late": j["orig_identical_to_offline_late"], "cohort": j["cohort"]}
        res["P1_final_equals_offline_late"] = h_p1 == late["methods"]["P1"]["final_hash"] if s == 1 else None
        res["rel_dist_P1_U0"] = h["final"]["P1"]["rel_dist_to_U0"]
        runs.append(res)
        print("seed", s, "jobs", res["jobs"], "lineage", res["lineage_hash_match"], "roots", res["round_roots_match"], "sampled", res["sampled_p0.3"],
              "prog_ok", res["prog_consistent"], flush=True)

    # ---------------- 집계 ----------------
    allrows = [dict(r, seed=x["seed"]) for x in runs for r in x["job_rows"]]
    types = ["T4", "T5", "T6", "T7", "T11a", "T11b", "T11c", "T11d", "T11e", "T11f", "honest"]
    per = {}
    for t in types:
        rs = [r for r in allrows if r["attack"] == t]
        per[t] = {"n_jobs": len(rs), "distinct_hOut": len({(r["seed"], r["hOut"]) for r in rs}),
                  "hOut_equals_json": sum(r["hOut_equals_json"] for r in rs),
                  "chain_L1_pass": sum(r["chain_ver"].get(1) is True for r in rs),
                  "chain_L2_pass": sum(r["chain_ver"].get(2) is True for r in rs),
                  "chain_L3_recorded": sum(3 in r["chain_ver"] for r in rs), "chain_L3_pass": sum(r["chain_ver"].get(3) is True for r in rs),
                  "not_deployable_replay": sum(r["deployable_replay"] is False for r in rs),
                  "L1_indep_available": sum(r["L1_indep"] is not None for r in rs),
                  "L1_indep_equals_chain": sum(r["L1_indep"] is not None and r["L1_indep"] == r["chain_ver"].get(1) for r in rs),
                  "L1_indep_fail": sum(r["L1_indep"] is False for r in rs),
                  "L2_indep_available": sum(r["L2_indep"] is not None for r in rs),
                  "L2_indep_equals_chain": sum(r["L2_indep"] is not None and r["L2_indep"] == r["chain_ver"].get(2) for r in rs),
                  "L3_exceeds_from_stats": sum(bool(r["L3_indep"]) and r["L3_indep"]["class"] == "exceeds" for r in rs),
                  "L3_pass_from_stats": sum(bool(r["L3_indep"]) and r["L3_indep"]["class"] == "pass" for r in rs),
                  "L3_diff_recomputed_ok": sum(bool(r["L3_indep"]) and abs(r["L3_indep"]["diff_recomputed"] - r["L3_indep"]["diff_stored"]) < 1e-12 for r in rs),
                  "json_fields_equal_chain": sum(r["json"]["L1"] == r["chain_ver"].get(1) and r["json"]["L2"] == r["chain_ver"].get(2)
                                                 and r["json"]["deployable"] == r["deployable_replay"] for r in rs),
                  "artifact_identified": sum(r["known_artifact"] is not None or r["lineage_round"] is not None for r in rs),
                  "lineage_duplicate": sum(r["lineage_round"] is not None for r in rs)}
    # T11 표(보충 Table S7): 약한 검증자 열을 저장된 수치에서 다시 센다
    t11 = {}
    for v in "abcdef":
        inst = []
        for x in runs:
            jj = json.load(open(L.REPO / "results/final/e3" / f"B_seed{x['seed']}.json"))
            hh = ck[str(x["seed"])]["final"]
            f_l3 = next((i["L3"] for i in jj["instances"] if i["attack"] == "T11" and i.get("variant") == "f" and i.get("L3")), None)
            for i in jj["instances"]:
                if i["attack"] != "T11" or i.get("variant") != v:
                    continue
                l3 = i.get("L3") or (f_l3 if v == "f" else None)
                lo, hi = l3["ci90"]
                d_json = i["rel_dist_to_U0"]
                name = next((n for n, w in hh.items() if w.get("hash") == i["h_out"]), None)
                d_ck = hh[name]["rel_dist_to_U0"] if name else None
                d = d_ck if d_ck is not None else d_json
                row = {"L3_only": not (lo > -DELTA and hi < DELTA), "final_eps": d > eps_end, "d_from_ckpt": d_ck is not None,
                       "d_ckpt_vs_json_relerr": None if d_ck is None else abs(d_ck - d_json) / d_json,
                       "json_cf": i["counterfactual"], "json_B4": i["detected"]["B4"]}
                if v == "b":
                    row["eps_r_le_1e-3"] = all(d > e for e in (eps_step, 1e-5, 1e-4, 1e-3))
                    row["eps_r_1e-2"] = d > 1e-2
                if v in "cd":
                    mx_s = max(i["round_dev"]["seed_rule"].values())
                    mx_c = max(i["round_dev"]["claimed_batch"].values())
                    row["eps_r_le_1e-3"] = all(mx_s > e for e in (eps_step, 1e-5, 1e-4, 1e-3))
                    row["eps_r_1e-2"] = mx_s > 1e-2
                    row["submitted_batch_eps0"] = mx_c > 0.0
                    row["submitted_batch_any"] = any(mx_c > e for e in (0.0, eps_step, 1e-5, 1e-4, 1e-3, 1e-2))
                inst.append(row)
        t11[v] = {"n": len(inst), **{k: sum(bool(r.get(k)) for r in inst) for k in ("L3_only", "final_eps", "eps_r_le_1e-3", "eps_r_1e-2",
                                                                                     "submitted_batch_eps0", "submitted_batch_any")},
                  "d_from_ckpt": sum(r["d_from_ckpt"] for r in inst),
                  "max_relerr_ckpt_vs_json": max((r["d_ckpt_vs_json_relerr"] for r in inst if r["d_ckpt_vs_json_relerr"] is not None), default=None),
                  "L3_only_equals_json": sum(r["L3_only"] == r["json_cf"]["L3_only"] for r in inst),
                  "final_eps_equals_json": sum(r["final_eps"] == r["json_cf"]["tolerance_end"] for r in inst)}
    b4_t11 = {v: per[f"T11{v}"]["not_deployable_replay"] for v in "abcde"}
    b4_t11["f"] = per["T11f"]["not_deployable_replay"] + sum(x["T11f1"]["n"] for x in runs
                                                             if not x["T11f1"]["fake_prog_pair_in_MethodApproved"] and x["T11f1"]["methodKind_latest"] == 0)
    agg = {"per_type": per, "t11_table": t11, "t11_B4": b4_t11, "eps_end": eps_end, "eps_step": eps_step,
           "jobs_total": sum(x["jobs"] for x in runs), "jobs_completed_total": sum(x["jobs_completed"] for x in runs),
           "b3_completion_records": sum(x["b3_events"].get("CompletionRecorded", 0) for x in runs),
           "b3_events": [x["b3_events"] for x in runs],
           "injected_jobs_deployable_replay": sum(r["deployable_replay"] is True for r in allrows if r["attack"] != "honest"),
           "honest_jobs_deployable_replay": sum(r["deployable_replay"] is True for r in allrows if r["attack"] == "honest"),
           "T8": {"n": sum(len(x["T8"]) for x in runs), "distinct": len({(x["seed"], r["delivered"]) for x in runs for r in x["T8"]}),
                  "not_equal_deployed": sum(not r["equals_deployed_event"] for x in runs for r in x["T8"]),
                  "not_deployable_latest": sum(r["deployable_latest"] is False for x in runs for r in x["T8"]),
                  "in_lineage": sum(r["in_lineage"] for x in runs for r in x["T8"])},
           "T9a": {"n": sum(len(x["T9a"]) for x in runs),
                   "no_aggregate_event": sum(not r["aggregated_event"] for x in runs for r in x["T9a"]),
                   "inputs3_accepted2": sum(r["inputs"] == 3 and r["accepted"] == 2 and r["rejected"] == 0 for x in runs for r in x["T9a"]),
                   "aggregate_call_reverts_now": sum(r["aggregate_call_now"] == "reverts" for x in runs for r in x["T9a"]),
                   "revert_reason_decisions": sum(r["revert_has_decisions"] for x in runs for r in x["T9a"])},
           "T9_json_variants": {v: sum(x["T9_json_variants"].get(v, 0) for x in runs) for v in "abc"},
           "T9b_forged_not_on_chain": sum(x["T9b_forged_not_on_chain"] for x in runs),
           "lineage_hash_match": sum(x["lineage_hash_match"] for x in runs), "round_roots_match": sum(x["round_roots_match"] for x in runs),
           "round_units": sum(x["round_units"] for x in runs), "sampled_p0.3": [x["sampled_p0.3"] for x in runs],
           "sampled_p0.3_total": sum(x["sampled_p0.3"] for x in runs),
           "json_training_L1_sampled": [x["json"]["training_L1"]["sampled"] for x in runs],
           "U0_equals_U0_audit": sum(x["U0_equals_U0_audit"] for x in runs),
           "U0_all_rounds_equal_audit": sum(x["U0_chain_rounds_equal_audit"] for x in runs),
           "L2_U0_input_roots_match": sum(x["L2_U0_input_roots_match"] for x in runs),
           "T6_last": {"L1_reexec_equal": sum(x["T6_last_L1_reexec_equal"] for x in runs),
                       "roots_mismatch_units": [x["T6_last_roots_mismatch_units"] for x in runs]},
           "T11d_roots_mismatch_units": [x["T11d_roots_mismatch_units"] for x in runs],
           "prog_consistent_runs": sum(x["prog_consistent"] for x in runs),
           "config_commit_code_commits": sorted({y["commit"] for x in runs for y in x["config_commit_progs"]}),
           "deployed_is_honest_U0": sum(x["deployed_is_honest_U0"] for x in runs), "revoked_final": sum(x["revoked_final"] for x in runs),
           "deployed_latest_equals_U0": sum(x["deployed_latest_call"] == ck[str(x["seed"])]["final"]["U0"]["hash"] for x in runs),
           "honest_deployable_latest": sum(x["honest_deployable_latest"] is True for x in runs),
           "cohort": [x["cohort"] for x in runs], "cohort_identified": sum(x["cohort_all_identified"] for x in runs),
           "withdrawal_rounds": [x["withdrawal_rounds"] for x in runs],
           "rel_dist_P1_U0": [x["rel_dist_P1_U0"] for x in runs], "P1_seed1_equals_offline_late": runs[0]["P1_final_equals_offline_late"],
           "T11f1": [x["T11f1"] for x in runs], "u0_refs_mean_sd": [mu, sd]}
    json.dump({"header": hdr, "aggregate": agg, "runs": runs}, open(OUT / "typeB_verdicts.json", "w"), indent=1, ensure_ascii=False, default=str)
    with open(L.EXP / "logs" / f"c3_verify_typeB_{time.strftime('%Y%m%dT%H%M%S')}.log", "w") as f:
        f.write(json.dumps(hdr) + "\n" + json.dumps(agg, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in agg.items() if k not in ("per_type", "t11_table", "T11f1", "b3_events")}, ensure_ascii=False, default=str))
    for t, v in per.items():
        print(t, v)
    for v, x in t11.items():
        print("T11" + v, x)
    print(agg["T11f1"])


def by_addr_b3(idx, seed):
    return next(c["address"] for c in idx["contracts"] if c["type"] == "B" and c["which"] == "b3" and c["seed"] == seed)


if __name__ == "__main__":
    main()
