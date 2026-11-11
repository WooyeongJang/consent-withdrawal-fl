"""E0: 유형 A 컨트랙트 20개의 원장 이벤트에서 궤적 해시·표본 단위·커밋·거절 단위·주입 위치를 모은다(실험 C의 c_lib·캐시, 표준 라이브러리만).

출력 results/e0_chain.json: seeds[s] = {contract, start_model, h_out{r: hash}, rejected[(k, r)], sampled[{k, r, root_S, sigma, n, h_start, h_delta}],
  injections{T1: pid, T2a/T2b/T3/T10: [k, r, pid]}, json_l1_sampled_units}
  python3 validation/E_typeA_regen/code/e0_chain.py
"""
import collections
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
sys.path.insert(0, str(REPO / "validation/C_independent_audit/code"))
import c_lib as L  # noqa: E402

P = 0.3


def main():
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"# E0 chain; start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; git HEAD {head}; script sha256 {hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16]}", flush=True)
    abi, _, _ = L.compile_abi("ConsentLedger.sol", "ConsentLedger")
    table = L.event_table(abi)
    idx = json.load(open(REPO / "validation/C_independent_audit/results/chain_index.json"))
    verd = {r["seed"]: r for r in json.load(open(REPO / "validation/C_independent_audit/results/typeA_verdicts.json"))["runs"]}
    out = {"p": P, "seeds": {}}
    tot = tot_json = 0
    for c in sorted((x for x in idx["contracts"] if x["type"] == "A" and x["which"] == "contract"), key=lambda x: x["seed"]):
        s = c["seed"]
        logs = json.load(open(L.RAW / "merged" / f"{c['address'].lower()}.json"))
        ev = [L.decode_log(table, lg) for lg in logs]
        assert all(e is not None for e in ev)
        by = collections.defaultdict(list)
        for e in ev:
            by[e["event"]].append(e)
        h_out = {int(e["args"]["r"]): e["args"]["hOut"] for e in by["RoundAggregated"] if int(e["args"]["r"]) <= 50}
        start = next(e["args"]["startModel"] for e in by["RoundOpened"] if int(e["args"]["r"]) == 1)
        inp = {(str(e["args"]["k"]), int(e["args"]["r"])): e["args"] for e in by["RoundInputCommitted"] if int(e["args"]["r"]) <= 50}
        upd = {(str(e["args"]["k"]), int(e["args"]["r"])): e["args"] for e in by["UpdateCommitted"] if int(e["args"]["r"]) <= 50}
        rej = sorted((str(e["args"]["k"]), int(e["args"]["r"])) for e in by["UpdateRejected"] if int(e["args"]["r"]) <= 50)
        seeds = {int(e["args"]["r"]): "0x" + e["args"]["seed"] for e in by["AuditSampled"]}
        assert len(h_out) == 50 and set(inp) == set(upd) and len(inp) == 150, (s, len(h_out), len(inp), len(upd))
        sampled = []
        for (k, r), a in sorted(inp.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if r in seeds and L.unit_sampled(seeds[r], k, P):
                u = upd[(k, r)]
                sampled.append({"k": k, "r": r, "root_S": a["rootS"], "sigma": int(a["sigma"]), "n": int(a["n"]),
                                "h_start": u["hStart"], "h_delta": u["hDelta"]})
        found = verd[s]["found"]
        inj = {"T1": int(found["T1"][0][2])}
        for t in ("T2a", "T2b", "T3", "T10"):
            k, r, pid = found[t][0]
            inj[t] = [str(k), int(r), int(pid)]
        rec = json.load(open(REPO / f"results/final/e3/A_seed{s}_p0.3.json"))
        out["seeds"][str(s)] = {"contract": c["address"], "start_model": start, "h_out": {str(r): h for r, h in sorted(h_out.items())},
                                "rejected": rej, "sampled": sampled, "n_sampled": len(sampled), "injections": inj,
                                "json_l1_sampled_units": rec["l1_sampled_units"], "json_final_hash": rec["final_hash"]}
        tot += len(sampled); tot_json += rec["l1_sampled_units"]
        inj_sampled = sum(1 for u in sampled if any((u["k"], u["r"]) == (v[0], v[1]) for t, v in inj.items() if t != "T1"))
        print(f"seed {s:2d} {c['address']}: hOut 50, rejected {len(rej)}, sampled {len(sampled)} (json {rec['l1_sampled_units']}), injected among sampled {inj_sampled}, final==hOut50 {h_out[50] == rec['final_hash']}", flush=True)
    out["total_sampled"], out["total_json"] = tot, tot_json
    (EXP / "results").mkdir(exist_ok=True)
    json.dump(out, open(EXP / "results/e0_chain.json", "w"), indent=1)
    print(f"total sampled {tot} (json {tot_json})", flush=True)


if __name__ == "__main__":
    main()
