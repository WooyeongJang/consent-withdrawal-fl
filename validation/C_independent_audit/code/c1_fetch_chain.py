"""C-1 준비: E3 최종 실행 25회의 컨트랙트 이벤트를 Kairos에서 읽어 저장한다(읽기 전용).

컨트랙트 주소는 results/final/e3/*.json의 contract·b3_contract 필드에서 가져오고(판정 필드는 읽지 않는다),
배포 블록은 ~/data/ledger/state/deployments.jsonl에서 찾는다. 로그는 5,000블록 단위로 나눠 읽는다.

  python3 c1_fetch_chain.py
"""
import json
import sys
import time
from pathlib import Path

import c_lib as L

OUT = L.EXP / "results"
LOG = L.EXP / "logs" / f"c1_fetch_chain_{time.strftime('%Y%m%dT%H%M%S')}.log"
CHUNK = 5000
SPAN = {"A": 10000, "B": 45000}        # 배포 뒤 읽을 블록 수(실행 시간보다 넉넉히)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def main():
    say(json.dumps(L.run_header(__file__)))
    dep = {}
    for l in open(Path.home() / "data/ledger/state/deployments.jsonl"):
        d = json.loads(l)
        dep[d["address"].lower()] = d
    runs = []
    for f in sorted((L.REPO / "results/final/e3").glob("*.json")):
        j = json.load(open(f))
        runs.append({"file": f.name, "type": j["type"], "seed": j["seed"], "contract": j["contract"], "b3": j["b3_contract"]})
    head = int(L.rpc("eth_blockNumber", [])["result"], 16)
    say("head block", head, "runs", len(runs))
    index = []
    for r in runs:
        for which in ("contract", "b3"):
            addr = r[which]
            d = dep[addr.lower()]
            b0 = d["block"]
            logs, empty_tail = [], 0
            for lo in range(b0, b0 + SPAN[r["type"]], CHUNK):
                hi = min(lo + CHUNK - 1, head)
                out = L.rpc("eth_getLogs", [{"address": addr, "fromBlock": hex(lo), "toBlock": hex(hi)}],
                            cache_key=f"logs/{addr.lower()}_{lo}_{hi}")
                if "error" in out:
                    say("ERROR getLogs", addr, lo, hi, out["error"])
                    sys.exit(1)
                logs += out["result"]
            code = L.rpc("eth_getCode", [addr, "latest"], cache_key=f"code/{addr.lower()}")["result"]
            last = max((int(x["blockNumber"], 16) for x in logs), default=None)
            index.append({**r, "which": which, "address": addr, "deploy_block": b0, "deploy_label": d.get("label"),
                          "deploy_time": d.get("time"), "n_logs": len(logs), "last_log_block": last,
                          "tail_margin_blocks": None if last is None else b0 + SPAN[r["type"]] - last,
                          "code_keccak": L.keccak256(bytes.fromhex(code[2:])).hex(), "code_len": len(code) // 2 - 1})
            (L.RAW / "merged").mkdir(exist_ok=True)
            (L.RAW / "merged" / f"{addr.lower()}.json").write_text(json.dumps(logs))
            say(r["file"], which, addr, "logs", len(logs), "last", last, "margin", index[-1]["tail_margin_blocks"])
    json.dump({"head_block_at_fetch": head, "contracts": index}, open(OUT / "chain_index.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
