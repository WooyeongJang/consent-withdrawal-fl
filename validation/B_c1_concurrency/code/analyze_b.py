"""실험 B 분석: 로컬(B-L) 3회 실행과 테스트넷(B-T) 본 실행의 시행 기록을 요약하고 사전 가설 H-B1..H-B5를 판정한다.

  python3 validation/B_c1_concurrency/code/analyze_b.py
"""
import collections
import hashlib
import json
import math
import os
import statistics
import time

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return rows


def q(v, p):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * p
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def hyp(trials, version):
    """사전 가설 판정에 쓰는 수. trials: 제외되지 않은 시행."""
    t = [x for x in trials if x["version"] == version]
    w_first = [x for x in t if x["ledger_class"].startswith("W<D")]
    d_first = [x for x in t if x["ledger_class"].startswith("D<W")]
    vc = lambda x: x["vc"] if "vc" in x else x["D"]["vc"]                       # noqa: E731
    acc_sent = lambda x: x["accept_sent"] if "accept_sent" in x else x["D"]["accept_sent"]   # noqa: E731
    stale_read = [x for x in w_first if acc_sent(x) and vc(x) < x["v_incl"]]     # 수락을 보냈고, 읽은 버전이 포함 시점보다 낡음
    accepted = [x for x in t if x["accepted"]]
    improper = [x for x in t if x["improper_acceptance"]]
    out = {"n": len(t), "ledger_class": dict(collections.Counter(x["ledger_class"] for x in t)),
           "W_reflected": sum(all(s == 1 for s in (x["W_status"] if "W_status" in x else [x["W"]["status"]])) for x in t),
           "accept_sent": sum(bool(acc_sent(x)) for x in t), "accepted": len(accepted),
           "stale_read_with_W_first": len(stale_read),
           "stale_read_accepted": sum(x["accepted"] for x in stale_read),
           "stale_read_reverted": sum(x["reverted"] for x in stale_read),
           "stale_acceptance": sum(x["stale_acceptance"] for x in t),
           "stale_acceptance_same_inst": sum(x["stale_acceptance"] and x["relation"] == "same" for x in t),
           "stale_acceptance_other_inst": sum(x["stale_acceptance"] and x["relation"] == "other" for x in t),
           "improper_acceptance": len(improper),
           "reverted": sum(x["reverted"] for x in t),
           "revert_reasons": dict(collections.Counter((x.get("revert_reason") or "n/a")[-13:] for x in t if x["reverted"])),
           "unnecessary_revert": sum(x["unnecessary_revert"] for x in t),
           "reverted_fresh_read": sum(x["reverted"] and not (vc(x) < x["v_incl"]) for x in t),
           "followup": dict(collections.Counter(f"{x.get('followup')}:{x.get('followup_status')}" for x in t if x["reverted"])),
           "attempts_mean_when_reverted": statistics.mean([x["attempts"] for x in t if x["reverted"] and "attempts" in x]) if any(x["reverted"] for x in t) else None,
           # 탐지기
           "accepted_with_det": sum(x["det"] is not None for x in accepted),
           "version_recorded_consistent": sum(bool(x["det"] and x["det"]["version_consistent"]) for x in accepted),
           "Dpos_TP": sum(bool(x["det"] and x["det"]["flag_pos"]) for x in improper),
           "Dpos_FN": sum(not (x["det"] and x["det"]["flag_pos"]) for x in improper),
           "Dpos_FP": sum(bool(x["det"] and x["det"]["flag_pos"]) and not x["improper_acceptance"] for x in t),
           "Dtime_TP": sum(bool(x["det"] and x["det"]["flag_time"]) for x in improper),
           "Dtime_FP": sum(bool(x["det"] and x["det"]["flag_time"]) and not x["improper_acceptance"] for x in t),
           "Dtime_FP_by_class": dict(collections.Counter(x["ledger_class"] for x in t if x["det"] and x["det"]["flag_time"] and not x["improper_acceptance"])),
           "D_first": len(d_first), "D_first_accepted": sum(x["accepted"] for x in d_first),
           "D_first_flag_pos": sum(bool(x["det"] and x["det"]["flag_pos"]) for x in d_first),
           "D_first_same_block_same_inst": sum(x["ledger_class"] == "D<W same block" and x["relation"] == "same" for x in t),
           "D_first_same_block_same_inst_flag_time": sum(bool(x["det"] and x["det"]["flag_time"]) for x in t
                                                         if x["ledger_class"] == "D<W same block" and x["relation"] == "same")}
    return out


def local():
    runs = []
    for i in (1, 2, 3):
        p = os.path.join(RES, f"local_run{i}.jsonl")
        if os.path.exists(p):
            runs.append(load(p))
    if not runs:
        return None
    key = lambda rows: hashlib.sha256(json.dumps([[r[k] for k in sorted(r) if k not in ("block_ts", "start_block", "end_block", "vc_read_block", "W_pos", "D_pos")]
                                                  for r in rows], sort_keys=True, default=str).encode()).hexdigest()   # noqa: E731
    digests = [key(r) for r in runs]
    # 블록 번호·시각까지 포함한 전체 일치 여부도 본다
    full = [hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in runs]
    rows = runs[0]
    trials = [r for r in rows if r["type"] == "trial"]
    live = [r for r in rows if r["type"] == "liveness"]
    out = {"runs": len(runs), "outcome_digests": digests, "reproduced": len(set(digests)) == 1, "full_record_identical": len(set(full)) == 1,
           "n_trials": len(trials), "by_version": {v: hyp(trials, v) for v in ("v3.1", "v3.2")}, "cells": {}, "gas": {}, "liveness": {}}
    for v in ("v3.1", "v3.2"):
        for c in ("L1", "L2", "L3", "L4", "L5", "L6"):
            for rel in ("same", "other"):
                t = [x for x in trials if x["version"] == v and x["cond"] == c and x["relation"] == rel]
                out["cells"][f"{v}|{c}|{rel}"] = {
                    "n": len(t), "ledger_class": dict(collections.Counter(x["ledger_class"] for x in t)),
                    "accepted": sum(x["accepted"] for x in t), "stale_acceptance": sum(x["stale_acceptance"] for x in t),
                    "improper_acceptance": sum(x["improper_acceptance"] for x in t), "reverted": sum(x["reverted"] for x in t),
                    "unnecessary_revert": sum(x["unnecessary_revert"] for x in t),
                    "flag_pos": sum(bool(x["det"] and x["det"]["flag_pos"]) for x in t),
                    "flag_time": sum(bool(x["det"] and x["det"]["flag_time"]) for x in t),
                    "followup_ok": sum(x.get("followup_status") == 1 for x in t)}
        t = [x for x in trials if x["version"] == v]
        g = lambda xs: {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs)} if xs else None   # noqa: E731
        out["gas"][v] = {"W": g([w for x in t for w in x["W_gas"]]),
                         "D_accept_ok": g([x["D_gas"] for x in t if x["accepted"]]),
                         "D_reject_ok": g([x["D_gas"] for x in t if x["D_status"] == 1 and not x["accepted"]]),
                         "D_reverted": g([x["D_gas"] for x in t if x["reverted"]]),
                         "followup": g([x["followup_gas"] for x in t if "followup_gas" in x])}
        for qq in (0.1, 0.5, 0.9):
            a = [x["attempts"] for x in live if x["version"] == v and x["q"] == qq]
            if a:
                out["liveness"][f"{v}|q={qq}"] = {"n": len(a), "mean_attempts": statistics.mean(a), "max": max(a),
                                                  "capped_at_20": sum(1 for x in live if x["version"] == v and x["q"] == qq and not x["accepted"]),
                                                  "expected_1_over_1_minus_q": 1 / (1 - qq)}
    return out


def testnet(v):
    p = os.path.join(RES, f"testnet_main_{v}.jsonl")
    if not os.path.exists(p):
        return None
    rows = load(p)
    meta = next(r for r in rows if r["type"] == "meta")
    summ = next((r for r in rows if r["type"] == "summary"), None)
    allt = [r for r in rows if r["type"] == "trial"]
    if summ is None or not allt:                      # 아직 실행 중
        return None
    t = [r for r in allt if "excluded" not in r]
    h = hyp(t, v)
    lat_w, lat_d = [x["latency_W_s"] for x in t], [x["latency_D_s"] for x in t]
    inv = sum(1 for x in t if (x["submit_order"] == "W first") != x["ledger_class"].startswith("W<D"))
    probes = [pr for x in t for pr in x.get("probes_after_W", [])]
    stale_probe = sum(1 for x in t for pr in x.get("probes_after_W", []) if "head" in pr and pr["head"] < x["W"]["block"])
    same_ts = sum(1 for x in t if len(x["block_ts"]) == 2 and len(set(x["block_ts"].values())) == 1)
    out = {"deploy": meta["deploy"], "started": meta["started"], "finished": summ and summ["finished"], "planned": meta["n_planned"],
           "run": len(allt), "excluded": [{"tid": r["tid"], "reason": r["excluded"]} for r in allt if "excluded" in r],
           "analysed": len(t), **h,
           "by_class_relation": {f"{c}|{rel}": {"n": sum(1 for x in t if x["ledger_class"] == c and x["relation"] == rel),
                                                "accepted": sum(x["accepted"] for x in t if x["ledger_class"] == c and x["relation"] == rel),
                                                "stale_acceptance": sum(x["stale_acceptance"] for x in t if x["ledger_class"] == c and x["relation"] == rel),
                                                "improper": sum(x["improper_acceptance"] for x in t if x["ledger_class"] == c and x["relation"] == rel),
                                                "reverted": sum(x["reverted"] for x in t if x["ledger_class"] == c and x["relation"] == rel),
                                                "rejected_by_fresh_read": sum((not x["D"]["accept_sent"]) for x in t if x["ledger_class"] == c and x["relation"] == rel)}
                                 for c in ("W<D other block", "W<D same block", "D<W same block", "D<W other block") for rel in ("same", "other")},
           "stale_acceptance_ci95": wilson(h["stale_acceptance"], len(t)), "improper_acceptance_ci95": wilson(h["improper_acceptance"], len(t)),
           "latency_W_s": {"median": q(lat_w, 0.5), "p95": q(lat_w, 0.95), "max": max(lat_w)},
           "latency_D_s": {"median": q(lat_d, 0.5), "p95": q(lat_d, 0.95), "max": max(lat_d)},
           "latency_blocks_W": dict(collections.Counter(x["latency_W_blocks"] for x in t)),
           "latency_blocks_D": dict(collections.Counter(x["latency_D_blocks"] for x in t)),
           "order_inverted": inv, "order_inverted_rate": inv / len(t),
           "probes": len(probes), "probe_errors": sum("error" in pr for pr in probes), "probe_stale_head": stale_probe,
           "different_blocks_same_timestamp": same_ts,
           "gas": {"W_median": statistics.median(x["W"]["gas_used"] for x in t), "D_accept_median": statistics.median([x["D"]["gas_used"] for x in t if x["accepted"]] or [0]),
                   "D_reverted_median": statistics.median([x["D"]["gas_used"] for x in t if x["reverted"]] or [0]),
                   "setup_median": statistics.median(x["setup_gas"] for x in t)},
           "fee_kaia": summ and summ["fee_kaia_receipts"], "payer_balance_diff": summ and summ["payer_balance_before"] - summ["payer_balance_after"],
           "trial_duration_s_median": statistics.median(x["t_end"] - x["t_start"] for x in t)}
    return out


def main():
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "local": local(), "testnet": {v: testnet(v) for v in ("v3.1", "v3.2")}}
    json.dump(out, open(os.path.join(RES, "b_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
