"""실험 C 독립 검증기 공용 모듈(표준 라이브러리만 사용).

fl/audit.py, fl/e3.py, fl/e3b.py, fl/l3.py, scripts/summarize_*.py, scripts/l3_*.py,
scripts/recount_l3_deploy.py, reports/audit/*.py 를 import하거나 복사하지 않는다.

정의의 출처
- keccak-256, ABI 이벤트 부호화: Ethereum 규격(직접 구현).
- 이벤트 이름·인자: contracts/ConsentLedger.sol, contracts/CompletionRecord.sol 을 solc로 컴파일한 ABI.
- 해시 H(구분자 0x1f를 붙인 SHA-256), rid, 정렬 머클 트리의 잎·마디 접두사, 결속값 B의 필드 순서,
  표본 추출의 문자열 부호화(seed|k): 원고에는 수식만 있고 바이트 수준 부호화가 없어
  fl/evidence.py, fl/e3.py(unit_sampled), fl/run.py 를 읽어 정의만 옮겼다(코드를 실행·import하지 않음).
"""
import csv
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
RAW = EXP / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
RPC = os.environ.get("KAIROS_RPC", "https://public-en-kairos.node.kaia.io")
DATA_CACHE = Path(os.environ.get("PTBXL_CACHE", str(Path.home() / "data/cache/ptbxl_k3")))

# ---------------- keccak-256 ----------------
_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
       0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
       0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
       0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
       0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
_M = (1 << 64) - 1


def _rol(x, n):
    return ((x << n) | (x >> (64 - n))) & _M if n else x


def _f(A):
    for rc in _RC:
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol(C[(x + 1) % 5], 1) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol(A[x][y], _ROT[x][y])
        A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        A[0][0] ^= rc
    return A


def keccak256(data: bytes) -> bytes:
    rate = 136
    p = bytearray(data)
    p.append(0x01)
    while len(p) % rate:
        p.append(0)
    p[-1] |= 0x80
    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(p), rate):
        blk = p[off:off + rate]
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(blk[8 * i:8 * i + 8], "little")
        A = _f(A)
    out = b"".join(A[i % 5][i // 5].to_bytes(8, "little") for i in range(4))
    return out


assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
assert keccak256(b"Transfer(address,address,uint256)").hex().startswith("ddf252ad1be2c89b")

# ---------------- RPC (읽기 전용, 캐시) ----------------
_last = [0.0]


def rpc(method, params, cache_key=None, retries=6, pause=0.35):
    """읽기 전용 호출. cache_key가 있으면 원시 응답을 results/raw/에 저장하고 다시 묻지 않는다."""
    assert method in ("eth_getLogs", "eth_call", "eth_getCode", "eth_blockNumber", "eth_getTransactionReceipt",
                      "eth_getTransactionCount", "eth_getBlockByNumber", "eth_chainId", "eth_getTransactionByHash")
    f = RAW / f"{cache_key}.json" if cache_key else None
    if f is not None and f.exists():
        return json.loads(f.read_text())
    err = None
    for a in range(retries):
        wait = pause - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(RPC, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": "validation-c/1.0"})
            out = json.load(urllib.request.urlopen(req, timeout=90))
            if "error" in out and a < retries - 1 and "revert" not in json.dumps(out["error"]).lower() \
                    and "missing trie node" not in json.dumps(out["error"]):
                err = out["error"]
                time.sleep(1.5 * (a + 1))
                continue
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(out))
            return out
        except Exception as e:  # 네트워크 오류: 물러났다가 다시
            err = repr(e)
            time.sleep(2.0 * (a + 1))
    raise RuntimeError(f"rpc {method} failed: {err}")


# ---------------- ABI ----------------
def compile_abi(sol, name):
    out = subprocess.run([str(REPO / "tools/bin/solc"), "--evm-version", "prague", "--optimize", "--combined-json",
                          "abi,bin-runtime,hashes", str(REPO / "contracts" / sol)], check=True, capture_output=True, text=True).stdout
    c = next(v for k, v in json.loads(out)["contracts"].items() if k.endswith(":" + name))
    abi = c["abi"] if isinstance(c["abi"], list) else json.loads(c["abi"])
    return abi, c["bin-runtime"], c["hashes"]


def event_table(abi):
    t = {}
    for e in abi:
        if e.get("type") != "event":
            continue
        sig = f"{e['name']}({','.join(i['type'] for i in e['inputs'])})"
        t["0x" + keccak256(sig.encode()).hex()] = e
    return t


def _word(t, w):
    if t == "bytes32":
        return w.hex()
    if t == "bool":
        return int.from_bytes(w, "big") != 0
    if t == "address":
        return "0x" + w[12:].hex()
    return int.from_bytes(w, "big")


def decode_log(table, lg):
    e = table.get(lg["topics"][0])
    if e is None:
        return None
    data = bytes.fromhex(lg["data"][2:])
    topics = [bytes.fromhex(t[2:]) for t in lg["topics"][1:]]
    args, ti, di = {}, 0, 0
    for i in e["inputs"]:
        if i["indexed"]:
            args[i["name"]] = _word(i["type"], topics[ti]); ti += 1
        else:
            args[i["name"]] = _word(i["type"], data[32 * di:32 * di + 32]); di += 1
    return {"event": e["name"], "args": args, "block": int(lg["blockNumber"], 16), "txIndex": int(lg["transactionIndex"], 16),
            "logIndex": int(lg["logIndex"], 16), "tx": lg["transactionHash"]}


def selector(sig):
    return keccak256(sig.encode())[:4].hex()


def enc_word(v):
    if isinstance(v, bool):
        return (b"\x00" * 31 + (b"\x01" if v else b"\x00")).hex()
    if isinstance(v, int):
        return v.to_bytes(32, "big").hex()
    v = v.removeprefix("0x")
    return v.rjust(64, "0")


def eth_call(to, sig, args=(), frm=None, block="latest"):
    tx = {"to": to, "data": "0x" + selector(sig) + "".join(enc_word(a) for a in args)}
    if frm:
        tx["from"] = frm
    key = "call/" + to.lower() + "_" + hashlib.sha256(json.dumps([tx, block], sort_keys=True).encode()).hexdigest()[:24]
    return rpc("eth_call", [tx, block], cache_key=key)


# ---------------- 증거 해시(정의는 fl/evidence.py에서 옮김) ----------------
STUDY_SALT = "dissertation-ptbxl-v1.0.3"
SEED_RULE = "seed_{k,r} = int(SHA256('seed|base|k|r')[:16],16) mod 2^63"
PURPOSE = "ptbxl-superdiag-train"


def H(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode()); h.update(b"\x1f")
    return h.hexdigest()


def text32(s):
    return hashlib.sha256(s.encode()).hexdigest()


def merkle_levels(items):
    leaves = sorted(set(items))
    lv = [H("leaf", x) for x in leaves] or [H("empty")]
    levels = [lv]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([H("node", cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i]) for i in range(0, len(cur), 2)])
    return leaves, levels


def merkle_root(items):
    return merkle_levels(items)[1][-1][0]


def membership_verifies(root, leaves, levels, x):
    """x의 소속 증명(잎 → 루트 경로)이 주어진 루트로 검증되는지. 비소속 증명이 불가능함을 보이는 데 쓴다."""
    import bisect
    i = bisect.bisect_left(leaves, x)
    if i == len(leaves) or leaves[i] != x:
        return False
    h, idx = H("leaf", x), i
    for lv in levels[:-1]:
        j = idx ^ 1
        sib = lv[j] if j < len(lv) else lv[idx]
        h = H("node", h, sib) if idx % 2 == 0 else H("node", sib, h)
        idx //= 2
    return h == root


def binding(k, r, h_start, root_s, sigma, prog, h_delta):
    return H("B", k, r, h_start, root_s, sigma, PURPOSE, prog, SEED_RULE, h_delta)


def unit_sampled(seed_hex_with_0x, k, p):
    """Algorithm S3 2행: Hash(η_r ‖ k)/2^256 < p. 부호화는 SHA-256("0x<seed>|<k>")."""
    return int(hashlib.sha256(f"{seed_hex_with_0x}|{k}".encode()).hexdigest(), 16) / 2 ** 256 < p


# ---------------- 데이터(기관 분할 캐시의 meta.csv만 읽는다) ----------------
class Meta:
    def __init__(self, cache=DATA_CACHE, insts=("0", "1", "2")):
        rows = list(csv.DictReader(open(Path(cache) / "meta.csv")))
        self.insts = insts
        self.train = {k: [] for k in insts}      # 기관별 (행 번호, pid, ecg_id)
        by_pid = {}
        for i, r in enumerate(rows):
            if r["has_label"] == "True" and r["excluded"] == "False" and int(r["strat_fold"]) <= 8 and r["institution"] in insts:
                pid, ecg = int(float(r["patient_id"])), int(r["ecg_id"])
                self.train[r["institution"]].append((i, pid, ecg))
                by_pid.setdefault(pid, []).append(ecg)
        self.rid = {}                              # ecg_id → rid
        self.inst_of = {}
        for pid, ecgs in by_pid.items():
            ks = H("ks", STUDY_SALT, pid)
            for j, e in enumerate(sorted(ecgs)):
                self.rid[e] = H("rid", ks, j)
        for k in insts:
            for _, pid, _ in self.train[k]:
                self.inst_of[pid] = k
        self.pids = sorted(by_pid)
        self.subject_commit = {H("subject-commit", H("ks", STUDY_SALT, p)): p for p in self.pids}
        self.ecgs_of = by_pid

    def rids_of(self, pid, k=None):
        if k is not None and self.inst_of[pid] != k:
            return []
        return sorted(self.rid[e] for e in self.ecgs_of[pid])

    def input_rids(self, k, valid):
        """기관 k의 학습 기록 중 valid(pid)가 참인 기록의 rid."""
        return [self.rid[e] for _, pid, e in self.train[k] if valid(pid)]


def candidate_progs(since="2026-10-24", until="2026-10-31"):
    """prog = H("prog", fl/*.py의 코드 해시, 이미지 digest, 조건, torch 판, 스레드 수)를 git 이력에서 다시 만든다.
    (정의는 fl/run.py prog_id·code_hash를 읽고 옮김.) 반환: {prog: {"commit", "time"}} — 같은 prog는 가장 이른 커밋을 남긴다."""
    tc = json.load(open(REPO / "tools/toolchain.json"))["training_image"]
    out = {}
    lines = subprocess.run(["git", "-C", str(REPO), "log", "--reverse", "--format=%H %cI", f"--since={since}", f"--until={until}", "--", "fl"],
                           capture_output=True, text=True).stdout.split("\n")
    for line in lines:
        if not line.strip():
            continue
        h, when = line.split()
        names = subprocess.run(["git", "-C", str(REPO), "ls-tree", "--name-only", h, "fl/"], capture_output=True, text=True).stdout.split()
        hh = hashlib.sha256()
        for n in sorted(x for x in names if x.endswith(".py")):
            hh.update(n.split("/")[-1].encode())
            hh.update(subprocess.run(["git", "-C", str(REPO), "show", f"{h}:{n}"], capture_output=True).stdout)
        prog = H("prog", hh.hexdigest(), tc["repo_digest"], "gpu_det", tc["torch"], 8)
        out.setdefault(prog, {"commit": h, "time": when, "code_hash": hh.hexdigest()})
    return out


def run_header(script):
    """실행 기록 첫머리: 시각, git HEAD, 스크립트 SHA-256, 환경."""
    import platform
    import sys
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"datetime": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": head, "script": str(script),
            "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
            "lib_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version.split()[0], "platform": platform.platform(), "rpc": RPC}
