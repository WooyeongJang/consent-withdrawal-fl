"""Kairos 공개 RPC 응답 지연과 속도 제한 관찰 (3장 v1.6 8.2절: RPC 제한이 E4의 병목인지 기록).

공용 인프라에 부담을 주지 않도록 요청 수를 제한한다(순차 100건, 동시 5·10·20개에서 각 60건).
조회 요청(eth_blockNumber)만 보내며 거래는 보내지 않는다.

    python kairos_rpc_probe.py [--out reports/kairos_rpc_probe.json]
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RPC = "https://public-en-kairos.node.kaia.io"
BODY = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}).encode()
RATE_HEADERS = ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "ratelimit-limit",
                "ratelimit-remaining", "ratelimit-reset")


def call(_=None):
    req = urllib.request.Request(RPC, data=BODY, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            status = r.status if "result" in body else f"rpc_error:{body.get('error', {}).get('code')}"
            headers = {k.lower(): v for k, v in r.headers.items() if k.lower() in RATE_HEADERS}
    except urllib.error.HTTPError as e:
        status, headers = e.code, {k.lower(): v for k, v in e.headers.items() if k.lower() in RATE_HEADERS}
    except Exception as e:  # 연결 오류, 시간 초과
        status, headers = type(e).__name__, {}
    return status, (time.perf_counter() - t) * 1000, headers


def summarize(results, wall):
    ok = [ms for s, ms, _ in results if s == 200]
    q = statistics.quantiles(ok, n=20) if len(ok) >= 20 else []
    return {
        "requests": len(results),
        "status_counts": {str(k): v for k, v in Counter(s for s, _, _ in results).items()},
        "latency_ms_p50": round(statistics.median(ok), 1) if ok else None,
        "latency_ms_p95": round(q[18], 1) if q else None,
        "latency_ms_max": round(max(ok), 1) if ok else None,
        "throughput_rps": round(len(results) / wall, 1),
        "rate_limit_headers_seen": sorted({k for _, _, h in results for k in h}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = {"rpc": RPC, "method": "eth_blockNumber", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    t = time.perf_counter()
    seq = [call() for _ in range(100)]
    out["sequential"] = summarize(seq, time.perf_counter() - t)
    out["concurrent"] = {}
    for c in (5, 10, 20):
        time.sleep(5)
        t = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(call, range(60)))
        out["concurrent"][str(c)] = summarize(res, time.perf_counter() - t)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
