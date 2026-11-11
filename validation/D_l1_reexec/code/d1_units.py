"""D1: 원장 이벤트에서 유형 B(시드 1–5)의 표본 학습 단위와 커밋을 모은다(실험 C의 c_lib·캐시 사용, 표준 라이브러리만).

출력 results/d1_units.json: 시드별 {contract, units: [{k, r, root_S, sigma, n, h_start, h_delta}], n_sampled, json_training_L1_sampled,
events_jsonl_match(원장 커밋과 실행 폴더 events.jsonl의 같은 (k, r) 커밋 일치 수)}.
  python3 validation/D_l1_reexec/code/d1_units.py
"""
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
sys.path.insert(0, str(REPO / "validation/C_independent_audit/code"))
import c_lib as L  # noqa: E402

RUNS = Path(os.environ.get("E3B_RUNS", str(Path.home() / "data/runs/final_e3b")))
P = 0.3


def main():
    t0 = time.time()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"# D1 units; start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; git HEAD {head}; script sha256 {hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16]}", flush=True)
    abi, _, _ = L.compile_abi("ConsentLedger.sol", "ConsentLedger")
    table = L.event_table(abi)
    idx = json.load(open(REPO / "validation/C_independent_audit/results/chain_index.json"))
    out = {"p": P, "seeds": {}}
    total = 0
    for c in sorted((x for x in idx["contracts"] if x["type"] == "B" and x["which"] == "contract"), key=lambda x: x["seed"]):
        s = c["seed"]
        logs = json.load(open(L.RAW / "merged" / f"{c['address'].lower()}.json"))
        ev = [L.decode_log(table, lg) for lg in logs]
        assert all(e is not None for e in ev)
        by = collections.defaultdict(list)
        for e in ev:
            by[e["event"]].append(e)
        # k는 fl 코드의 기관 키(문자열 "0","1","2")로 맞춘다(원장 디코더는 정수를 준다; 1차 실행의 KeyError 원인, DEVIATIONS D절)
        inp = {(str(e["args"]["k"]), int(e["args"]["r"])): e["args"] for e in by["RoundInputCommitted"] if int(e["args"]["r"]) <= 50}
        upd = {(str(e["args"]["k"]), int(e["args"]["r"])): e["args"] for e in by["UpdateCommitted"] if int(e["args"]["r"]) <= 50}
        seeds = {int(e["args"]["r"]): "0x" + e["args"]["seed"] for e in by["AuditSampled"]}
        assert set(inp) == set(upd) and len(inp) == 150, (s, len(inp), len(upd))
        units = []
        for (k, r), a in sorted(inp.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if r in seeds and L.unit_sampled(seeds[r], k, P):
                u = upd[(k, r)]
                units.append({"k": k, "r": r, "root_S": a["rootS"], "sigma": int(a["sigma"]), "n": int(a["n"]),
                              "h_start": u["hStart"], "h_delta": u["hDelta"]})
        # 실행 폴더 events.jsonl(실험 기록)과의 일치 — 입력으로 쓰지 않고 일치 수만 기록
        ej = {}
        for line in open(RUNS / f"B_s{s}" / "orig" / "events.jsonl"):
            e = json.loads(line)
            if e.get("event") == "RoundInputCommitted":
                ej.setdefault((str(e["k"]), int(e["r"])), {})["root_S"] = e["root_S"]
            elif e.get("event") == "UpdateCommitted":
                ej.setdefault((str(e["k"]), int(e["r"])), {}).update(h_start=e["h_start"], h_delta=e["h_delta"])
        match = sum(1 for u in units if ej.get((u["k"], u["r"]), {}).get("root_S") == u["root_S"]
                    and ej[(u["k"], u["r"])].get("h_start") == u["h_start"] and ej[(u["k"], u["r"])].get("h_delta") == u["h_delta"])
        rec = json.load(open(REPO / f"results/final/e3/B_seed{s}.json"))["training_L1"]
        out["seeds"][str(s)] = {"contract": c["address"], "units": units, "n_sampled": len(units), "rounds_with_seed": len(seeds),
                                "json_training_L1_sampled": rec["sampled"], "json_training_L1_fail": rec["fail"],
                                "events_jsonl_match": match}
        total += len(units)
        print(f"seed {s} {c['address']}: units 150, AuditSampled rounds {len(seeds)}, sampled {len(units)} (json {rec['sampled']}), events.jsonl match {match}/{len(units)}", flush=True)
    out["total_sampled"] = total
    out["total_json"] = sum(v["json_training_L1_sampled"] for v in out["seeds"].values())
    (EXP / "results").mkdir(exist_ok=True)
    json.dump(out, open(EXP / "results/d1_units.json", "w"), indent=1)
    print(f"total sampled {total} (json {out['total_json']}); {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
