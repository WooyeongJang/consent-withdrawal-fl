"""C-1·C-2·C-4(유형 A: T1, T2a, T2b, T3, T10): 원장 이벤트와 데이터 분할만으로 판정을 다시 도출한다.

근거: (1) Kairos 이벤트(results/raw/merged/*.json, c1_fetch_chain.py가 받음), (2) 기관 분할 캐시 meta.csv.
실행 로그 JSON(results/final/e3/A_seed*.json)의 instances·detected·unit_sampled 등은 마지막에 비교 대상으로만 읽는다.

도출 순서(실행마다)
 1. 철회 정책: WithdrawalRequested의 subject를 데이터의 모든 학습 환자에 대한 H("subject-commit", k_s)와 대조해 환자를 찾고,
    효력 라운드는 그 로그보다 원장 순서가 앞선 마지막 RoundOpened의 r로 둔다. listRoot는 환자 rid의 머클 루트와 대조한다.
 2. 단위 (k, r)마다 정직한 입력 집합의 루트를 다시 계산해 원장의 rootS와 비교한다. 다르면 철회자 기록을 넣은 루트와 대조한다.
 3. 주입 유형을 원장에서 식별한다: T1 = 정책은 있으나 철회 요청이 원장에 없는 정보주체, T3 = σ가 낡은 단위,
    T2a = 철회 기록이 커밋에 들어 있고 거절된 단위, T10 = 철회 기록이 커밋에 들어 있는데 수락된 단위,
    T2b = 남은 철회(커밋은 정직).
 4. L1 표본: AuditSampled의 seed로 Algorithm S3 2행을 다시 계산한다(p = 0.1, 0.3, 0.5, 1.0).
 5. L1 기록: VerificationRecorded(level 1)의 target을 결속값 B(k, r)(prog는 유형 B 원장의 JobCreated.prog) 또는 H(Δ)와 맞춘다.
 6. 환자 자기 검사: 철회 뒤 수락된 자기 기관 단위의 rootS에 환자 rid의 소속 증명이 검증되면(비소속 증명 불가) 탐지.

  python3 c3_verify_typeA.py
"""
import collections
import json
import time

import c_lib as L

OUT = L.EXP / "results"
P_GRID = (0.1, 0.3, 0.5, 1.0)
REASONS = {L.text32(s): s for s in ("ok", "출력 불일치", "입력 재구성 실패", "출발 상태 불일치", "stale_snapshot", "policy_conflict")}


def load_events(table, addr):
    logs = json.load(open(L.RAW / "merged" / f"{addr.lower()}.json"))
    ev = [L.decode_log(table, l) for l in logs]
    assert all(e is not None for e in ev), "알 수 없는 이벤트"
    ev.sort(key=lambda e: (e["block"], e["logIndex"]))
    return ev


def main():
    hdr = L.run_header(__file__)
    abi, binrt, _ = L.compile_abi("ConsentLedger.sol", "ConsentLedger")
    table = L.event_table(abi)
    meta = L.Meta()
    idx = json.load(open(OUT / "chain_index.json"))
    # prog 후보: 유형 B 원장의 JobCreated.prog(같은 코드·이미지로 실행했다면 같아야 한다)
    progs = set()
    for c in idx["contracts"]:
        if c["type"] == "B" and c["which"] == "contract":
            progs |= {e["args"]["prog"] for e in load_events(table, c["address"]) if e["event"] == "JobCreated"}
    progs_chainB = sorted(progs)
    cand = L.candidate_progs()                      # git 이력에서 다시 만든 prog 후보
    scope_root = L.merkle_root([meta.rid[e] for k in meta.insts for _, _, e in meta.train[k]])
    root_cache = {}

    def root_of(k, excl, incl=()):
        key = (k, frozenset(excl), frozenset(incl))
        if key not in root_cache:
            ex = set(excl) - set(incl)
            rids = meta.input_rids(k, lambda p: p not in ex)
            root_cache[key] = (L.merkle_root(rids), len(rids))
        return root_cache[key]

    runs = []
    for c in sorted((x for x in idx["contracts"] if x["type"] == "A" and x["which"] == "contract"), key=lambda x: x["seed"]):
        ev = load_events(table, c["address"])
        by = collections.defaultdict(list)
        for e in ev:
            by[e["event"]].append(e)
        pos = lambda e: (e["block"], e["logIndex"])
        opened = sorted((pos(e), e["args"]["r"]) for e in by["RoundOpened"])
        R = max(r for _, r in opened)
        # 1. 철회 정책
        registered = [e["args"]["subject"] for e in by["PolicyRegistered"]]
        wd = {}
        for e in by["WithdrawalRequested"]:
            s = e["args"]["subject"]
            pid = meta.subject_commit.get(s)
            r_w = max((r for p, r in opened if p < pos(e)), default=None)
            k = meta.inst_of.get(pid)
            wd[pid] = {"subject": s, "round": r_w, "k": k,
                       "listRoot_ok": pid is not None and e["args"]["listRoot"] == L.merkle_root(meta.rids_of(pid, k))}
        unknown_subjects = [s for s in registered if s not in meta.subject_commit]
        no_request = [meta.subject_commit[s] for s in registered if s in meta.subject_commit and meta.subject_commit[s] not in wd]
        # 2. 단위별 루트 분류
        inp = {(str(e["args"]["k"]), e["args"]["r"]): e["args"] for e in by["RoundInputCommitted"]}
        upd = {(str(e["args"]["k"]), e["args"]["r"]): e["args"] for e in by["UpdateCommitted"]}
        dec = {}
        for e in by["UpdateAccepted"]:
            dec[(str(e["args"]["k"]), e["args"]["r"])] = ("accepted", None, e)
        for e in by["UpdateRejected"]:
            dec[(str(e["args"]["k"]), e["args"]["r"])] = ("rejected", REASONS.get(e["args"]["reason"], e["args"]["reason"]), e)
        units, anomalous = {}, []
        for (k, r), a in sorted(inp.items(), key=lambda x: (x[0][1], x[0][0])):
            excl = [p for p, w in wd.items() if w["k"] == k and w["round"] is not None and w["round"] <= r]
            honest, n_h = root_of(k, excl)
            cls, who = "unknown", None
            if a["rootS"] == honest and a["n"] == n_h:
                cls = "honest"
            else:
                for p in excl:
                    rt, n = root_of(k, excl, [p])
                    if a["rootS"] == rt and a["n"] == n:
                        cls, who = "withdrawn_included", p
                        break
            units[(k, r)] = {"class": cls, "who": who, "sigma": a["sigma"], "decision": dec[(k, r)][0], "reason": dec[(k, r)][1]}
            if cls != "honest" or a["sigma"] != 1 or dec[(k, r)][0] != "accepted":
                anomalous.append((k, r))
        # 수락 버전 기록: UpdateAccepted.withdrawalVersion == 그 로그보다 앞선 WithdrawalEffective 수
        eff_pos = sorted(pos(e) for e in by["WithdrawalEffective"])
        ver_ok = all(e["args"]["withdrawalVersion"] == sum(1 for p in eff_pos if p < pos(e)) for e in by["UpdateAccepted"])
        # 3. 유형 식별
        found = {}
        for (k, r) in anomalous:
            u = units[(k, r)]
            if u["class"] == "withdrawn_included" and u["sigma"] == 0:
                found.setdefault("T3", []).append((k, r, u["who"]))
            elif u["class"] == "withdrawn_included" and u["decision"] == "rejected":
                found.setdefault("T2a", []).append((k, r, u["who"]))
            elif u["class"] == "withdrawn_included" and u["decision"] == "accepted":
                found.setdefault("T10", []).append((k, r, u["who"]))
            else:
                found.setdefault("unclassified", []).append((k, r, u))
        used = {x[2] for v in found.values() for x in v if isinstance(x[2], int)}
        rest = [p for p in wd if p not in used]
        found["T2b"] = [(wd[p]["k"], wd[p]["round"], p) for p in rest]
        found["T1"] = [(meta.inst_of[p], None, p) for p in no_request]
        # 4. 표본
        seeds = collections.defaultdict(list)
        for e in by["AuditSampled"]:
            seeds[e["args"]["r"]].append("0x" + e["args"]["seed"])
        multi_seed_rounds = [r for r, v in seeds.items() if len(v) != 1]
        sampled = {p: {(k, r) for (k, r) in inp if r in seeds and L.unit_sampled(seeds[r][0], k, p)} for p in P_GRID}
        # 5. L1 기록과 단위의 대응
        cfg = by["TrainingConfigCommitted"][0]["args"]["config"]
        progs = [p for p in cand if L.H("config", c["seed"], L.SEED_RULE, p, scope_root) == cfg]   # 설정 커밋으로 prog·기준 시드 확인
        cfg_first = pos(by["TrainingConfigCommitted"][0]) < min(pos(e) for e in by["RoundOpened"] + by["WithdrawalRequested"])
        tgt = {}
        for prog in progs:
            for (k, r), a in inp.items():
                u = upd[(k, r)]
                tgt[L.binding(k, r, u["hStart"], a["rootS"], a["sigma"], prog, u["hDelta"])] = ((k, r), "B", prog)
        for (k, r), u in upd.items():
            tgt.setdefault(u["hDelta"], ((k, r), "hDelta", None))
        l1, unmatched, prog_used = {}, [], set()
        for e in by["VerificationRecorded"]:
            a = e["args"]
            m = tgt.get(a["target"])
            if a["level"] != 1 or m is None:
                unmatched.append(a)
                continue
            l1[m[0]] = {"passed": a["passed"], "reason": REASONS.get(a["reason"], a["reason"]), "keyed_by": m[1]}
            if m[2]:
                prog_used.add(m[2])
        attacked = {(k, r) for t in ("T2a", "T2b", "T3", "T10") for (k, r, _) in found.get(t, [])}
        honest_sampled = sampled[0.3] - attacked
        l1_set_ok = set(l1) == sampled[0.3]
        honest_fail = sorted(u for u in honest_sampled if not l1.get(u, {}).get("passed", False))
        rejected_honest = sorted(u for u, x in units.items() if x["decision"] == "rejected" and u not in attacked)
        # 6. 환자 자기 검사(철회 뒤 수락된 자기 기관 단위)
        self_fail = {}
        for t in ("T2a", "T3", "T10", "T2b"):
            for (k, r0, p) in found.get(t, []):
                fails = []
                for r in range(r0, R + 1):
                    if (k, r) not in inp or units[(k, r)]["decision"] != "accepted":
                        continue
                    excl = [q for q, w in wd.items() if w["k"] == k and w["round"] <= r]
                    incl = [units[(k, r)]["who"]] if units[(k, r)]["class"] == "withdrawn_included" else []
                    rids = meta.input_rids(k, lambda q: q not in (set(excl) - set(incl)))
                    leaves, levels = L.merkle_levels(rids)
                    assert levels[-1][0] == inp[(k, r)]["rootS"]
                    if any(L.membership_verifies(inp[(k, r)]["rootS"], leaves, levels, q) for q in meta.rids_of(p, k)):
                        fails.append(r)
                self_fail[t] = fails
        # 반사실: 공격 단위의 커밋에 환자 기록이 있는가(수락 여부와 무관, 증거 s)
        s_evid = {t: all(units[(k, r)]["class"] == "withdrawn_included" for (k, r, _) in found.get(t, [])) and bool(found.get(t))
                  for t in ("T2a", "T3", "T10")}
        # 판정(원고의 탐지 근거)
        det = {}
        det["T1"] = {"B4": len(found["T1"]) == 1 and len(wd) == len(registered) - 1, "basis": "정책 등록 5건 중 철회 요청이 원장에 없는 정보주체 1명"}
        for t in ("T2a", "T3"):
            x = found.get(t, [])
            det[t] = {"B4": len(x) == 1 and units[(x[0][0], x[0][1])]["decision"] == "rejected" if x else False,
                      "reason": units[(x[0][0], x[0][1])]["reason"] if x else None,
                      "l1_sampled_fail": {str(p): bool(x and (x[0][0], x[0][1]) in sampled[p]) for p in P_GRID},
                      "l1_record": l1.get((x[0][0], x[0][1])) if x else None}
        x = found.get("T2b", [])
        u2b = (x[0][0], x[0][1]) if len(x) == 1 else None
        det["T2b"] = {"unit": u2b, "sampled": {str(p): bool(u2b in sampled[p]) for p in P_GRID}, "l1_record": l1.get(u2b),
                      "B4": bool(u2b in sampled[0.3] and u2b in l1 and not l1[u2b]["passed"]),
                      "B6": bool(u2b in sampled[0.3] and u2b in l1 and not l1[u2b]["passed"])}
        x = found.get("T10", [])
        u10 = (x[0][0], x[0][1]) if len(x) == 1 else None
        det["T10"] = {"unit": u10, "self_check_fail_rounds": self_fail.get("T10"), "sampled": {str(p): bool(u10 in sampled[p]) for p in P_GRID},
                      "l1_record": l1.get(u10),
                      "B4": bool(self_fail.get("T10")) or bool(u10 in l1 and not l1[u10]["passed"]),
                      "without_self_check": bool(u10 in sampled[0.3] and u10 in l1 and not l1[u10]["passed"])}
        # ---- 비교 대상(실행 로그 JSON) ----
        j = json.load(open(L.REPO / "results/final/e3" / c["file"]))
        cmp = {}
        for x in j["instances"]:
            t = x["attack"]
            mine = found.get(t, [])
            loc_ok = len(mine) == 1 and mine[0][2] == x["pid"] and mine[0][0] == x["k"] and (t == "T1" or mine[0][1] == x["r"])
            cmp[t] = {"location_match": loc_ok, "json_B4": x["detected"]["B4"], "mine_B4": det[t]["B4"],
                      "json_unit_sampled": x["unit_sampled"],
                      "mine_unit_sampled": None if t == "T1" else ((x["k"], x["r"]) in sampled[0.3]),
                      "json_c": x["evidence"]["c"], "json_s": x["evidence"]["s"], "json_b": x["evidence"]["b"]}
        t1_sender = j["t1_signed_request"]["sender"]
        nonce = int(L.rpc("eth_getTransactionCount", [t1_sender, "latest"], cache_key=f"nonce/{t1_sender.lower()}")["result"], 16)
        runs.append({"seed": c["seed"], "contract": c["address"], "rounds": R, "n_units": len(inp),
                     "registered": len(registered), "withdrawals_on_chain": len(wd), "unknown_subjects": unknown_subjects,
                     "withdrawals": {str(p): w for p, w in wd.items()}, "found": {t: [list(map(str, y)) for y in v] for t, v in found.items()},
                     "anomalous_units": [list(u) for u in anomalous], "accept_version_matches_effective_count": ver_ok,
                     "multi_seed_rounds": multi_seed_rounds, "sampled_counts": {str(p): len(v) for p, v in sampled.items()},
                     "l1_records": len(l1), "l1_records_unmatched": unmatched, "l1_record_set_equals_sample_p0.3": l1_set_ok,
                     "prog_matched": sorted(prog_used), "config_commit_progs": [{"prog": p, **cand[p]} for p in progs],
                     "config_commit_before_round1_and_withdrawals": cfg_first, "honest_sampled_p0.3": len(honest_sampled), "honest_l1_fail": honest_fail,
                     "rejected_honest": rejected_honest, "self_check_fail": self_fail, "s_evidence": s_evid, "detected": det,
                     "t1_sender_nonce_latest": nonce,
                     "compare_json": cmp, "json_l1_sampled_units": j["l1_sampled_units"], "json_sampled_units_by_p": j["sampled_units_by_p"],
                     "json_withdrawal_count_on_chain": j["withdrawal_count_on_chain"], "json_false_positives": j["false_positives"]})
        print("seed", c["seed"], {t: len(v) for t, v in found.items()}, "sampled", runs[-1]["sampled_counts"], "l1", len(l1),
              "prog", [p[:8] for p in prog_used], "honest_fail", len(honest_fail), flush=True)

    # ---- 집계 ----
    n = len(runs)
    agg = {"runs": n,
           "B4": {t: sum(bool(r["detected"][t]["B4"]) for r in runs) for t in ("T1", "T2a", "T2b", "T3", "T10")},
           "B6_T2b": sum(bool(r["detected"]["T2b"]["B6"]) for r in runs),
           "T2b_by_p": {str(p): sum(r["detected"]["T2b"]["sampled"][str(p)] for r in runs) for p in P_GRID},
           "T10_l1_by_p": {str(p): sum(r["detected"]["T10"]["sampled"][str(p)] for r in runs) for p in P_GRID},
           "T10_without_self_check": sum(r["detected"]["T10"]["without_self_check"] for r in runs),
           "T2a_sampled_p0.3": sum(r["detected"]["T2a"]["l1_sampled_fail"]["0.3"] for r in runs),
           "T3_sampled_p0.3": sum(r["detected"]["T3"]["l1_sampled_fail"]["0.3"] for r in runs),
           "types_found_exactly_once_per_run": all(all(len(r["found"].get(t, [])) == 1 for t in ("T1", "T2a", "T2b", "T3", "T10"))
                                                   and not r["found"].get("unclassified") for r in runs),
           "location_match_all": {t: sum(r["compare_json"][t]["location_match"] for r in runs) for t in ("T1", "T2a", "T2b", "T3", "T10")},
           "B4_equals_json": {t: sum(r["compare_json"][t]["json_B4"] == r["compare_json"][t]["mine_B4"] for r in runs)
                              for t in ("T1", "T2a", "T2b", "T3", "T10")},
           "unit_sampled_equals_json": sum(r["compare_json"][t]["json_unit_sampled"] == r["compare_json"][t]["mine_unit_sampled"]
                                           for r in runs for t in ("T2a", "T2b", "T3", "T10")),
           "sampled_units_p0.3_total": sum(r["sampled_counts"]["0.3"] for r in runs),
           "sampled_counts_equal_json": sum(r["sampled_counts"] == {k: v for k, v in r["json_sampled_units_by_p"].items()} for r in runs),
           "l1_record_set_equals_sample": sum(r["l1_record_set_equals_sample_p0.3"] for r in runs),
           "honest_sampled_total": sum(r["honest_sampled_p0.3"] for r in runs),
           "honest_l1_fail_total": sum(len(r["honest_l1_fail"]) for r in runs),
           "rejected_honest_total": sum(len(r["rejected_honest"]) for r in runs),
           "accept_version_ok_runs": sum(r["accept_version_matches_effective_count"] for r in runs),
           "listRoot_ok": sum(w["listRoot_ok"] for r in runs for w in r["withdrawals"].values()),
           "withdrawals_total": sum(r["withdrawals_on_chain"] for r in runs),
           "t1_sender_nonce_zero": sum(r["t1_sender_nonce_latest"] == 0 for r in runs),
           "progs_from_typeB_chain": progs_chainB,
           "config_commit_reconstructed_runs": sum(len(r["config_commit_progs"]) == 1 for r in runs),
           "config_commit_first_runs": sum(r["config_commit_before_round1_and_withdrawals"] for r in runs),
           "config_commit_code_commits": sorted({x["commit"] for r in runs for x in r["config_commit_progs"]}),
           "l1_records_total": sum(r["l1_records"] for r in runs),
           "l1_records_unmatched_total": sum(len(r["l1_records_unmatched"]) for r in runs),
           "l1_fail_records": collections.Counter(f"{t}:{r['detected'][t]['l1_record']['reason']}" for r in runs for t in ("T2a", "T2b", "T3", "T10")
                                                  if r["detected"][t].get("l1_record") and not r["detected"][t]["l1_record"]["passed"]),
           "prog_matched_runs": sum(bool(r["prog_matched"]) for r in runs),
           "self_check_T10_fail_runs": sum(bool(r["self_check_fail"].get("T10")) for r in runs),
           "self_check_other_fail_runs": sum(bool(r["self_check_fail"].get(t)) for r in runs for t in ("T2a", "T3", "T2b")),
           "units_total": sum(r["n_units"] for r in runs),
           "units_honest_root": sum(r["n_units"] - len([u for u in r["anomalous_units"]]) for r in runs)}
    json.dump({"header": hdr, "aggregate": agg, "runs": runs}, open(OUT / "typeA_verdicts.json", "w"), indent=1, ensure_ascii=False, default=str)
    with open(L.EXP / "logs" / f"c3_verify_typeA_{time.strftime('%Y%m%dT%H%M%S')}.log", "w") as f:
        f.write(json.dumps(hdr) + "\n" + json.dumps(agg, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(agg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
