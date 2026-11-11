"""실험 B-L: 로컬 EVM(py-evm, Prague)에서 철회(W)와 수락(D)의 순서를 통제한 동시성 시험.

- 컨트랙트는 results/artifacts/의 컴파일 산출물(원본 소스 그대로)을 쓴다.
- 모든 거래는 역할 계정이 미리 서명한 raw 거래다. 자동 채굴을 끄고, 하네스가 전달 순서와 블록 경계를 정한다.
- 한 시행 = 새 라운드 r의 단위 (k, r) 하나와 철회 K건(L6 외에는 1건).

  <venv>/python validation/B_c1_concurrency/code/local_harness.py --out validation/B_c1_concurrency/results/local_run1.jsonl
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from eth_abi import encode
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "..", "results", "artifacts")
GAS = 700_000
TAG_POLICY = Web3.keccak(text="ConsentLedger.v3.2.policy")
TAG_WITHDRAW = Web3.keccak(text="ConsentLedger.v3.2.withdraw")
# eth-tester는 자동 채굴을 끄면 대기 거래를 최신 상태 기준으로 검증하므로, 한 블록에는 계정당 거래 하나만 넣을 수 있다.
# 그래서 철회 제출 계정(relay)을 셋 두고, 준비 거래를 여러 블록에 나눈다.
ROLE_NAMES = ("platform", "aggregator", "auditor", "dc0", "dc1", "dc2", "relay0", "relay1", "relay2")


def b32(s):
    return Web3.keccak(text=s)


class World:
    """컨트랙트 하나(v3.1 또는 v3.2)와 역할 계정, nonce, 블록 제어."""

    def __init__(self, version):
        self.version = version
        self.prov = EthereumTesterProvider()
        self.w3 = w3 = Web3(self.prov)
        self.t = self.prov.ethereum_tester
        self.sent = []                                       # 아직 채굴되지 않은 거래의 해시
        self.evlog = []                                      # 채굴 직후 영수증에서 읽은 이벤트
        self.c = None
        self.roles = {n: Account.from_key(Web3.keccak(text=f"validation-B-local-{n}")) for n in ROLE_NAMES}
        for a in self.roles.values():                       # 로컬 체인 전용 계정에 가스비 지급
            w3.eth.send_transaction({"from": w3.eth.accounts[0], "to": a.address, "value": 10 ** 20})
        self.nonce = {a.address: 0 for a in self.roles.values()}
        self.chain = w3.eth.chain_id
        art = json.load(open(os.path.join(ART, f"{version}.json")))
        self.abi = art["abi"]
        self._event_names = [x["name"] for x in self.abi if x.get("type") == "event"]
        ctor = w3.eth.contract(abi=self.abi, bytecode=art["bin"]).constructor(
            self.roles["aggregator"].address, self.roles["auditor"].address,
            [self.roles[f"dc{i}"].address for i in range(3)], 0)
        h = self._send_raw(self._sign("platform", ctor.build_transaction(self._txp("platform", gas=6_000_000))))
        rc = w3.eth.get_transaction_receipt(h)
        assert rc.status == 1
        self.c = w3.eth.contract(address=rc.contractAddress, abi=self.abi)
        assert w3.eth.get_code(rc.contractAddress).hex().removeprefix("0x") == art["bin_runtime"], "배포 코드가 컴파일 결과와 다르다"
        self.deploy_block = rc.blockNumber
        self.sent = []
        self.mined(self.tx("platform", "commitTrainingConfig", b32("cfg")))
        self._collect()
        self.round = 0
        self.t.disable_auto_mine_transactions()
        self.ver_cache = {}
        self._cache_version()

    # ---- 거래
    def _txp(self, role, gas=GAS):
        a = self.roles[role].address
        n = self.nonce[a]; self.nonce[a] += 1
        return {"from": a, "nonce": n, "gas": gas, "maxFeePerGas": 10 ** 10, "maxPriorityFeePerGas": 10 ** 9, "chainId": self.chain}

    def _sign(self, role, tx):
        return self.roles[role].sign_transaction(tx).raw_transaction

    def _send_raw(self, raw):
        h = self.w3.eth.send_raw_transaction(raw)
        self.sent.append(h)
        return h

    def sign(self, role, fn, *args):
        """서명만 한다(전달은 나중에). 반환: raw 거래."""
        return self._sign(role, getattr(self.c.functions, fn)(*args).build_transaction(self._txp(role)))

    def tx(self, role, fn, *args):
        return self._send_raw(self.sign(role, fn, *args))

    def deliver(self, raw):
        return self._send_raw(raw)

    def mine(self, n=1):
        for _ in range(n):
            self.t.mine_blocks(1)
            self._cache_version()
            self._collect()

    def _collect(self):
        """방금 채굴된 거래의 영수증에서 이벤트를 읽는다. eth-tester는 오래된 거래의 영수증·로그 조회가 체인 길이에 비례해
        느려지므로(1·2차 실행 중단, DEVIATIONS.md) 머리 블록 근처에서 바로 읽어 둔다. 이벤트를 내는 것은 거래뿐이라 내용은 같다."""
        from web3.logs import DISCARD
        if self.c is None:
            self.sent = []
            return
        blk = self.w3.eth.get_block("latest")
        for h in self.sent:
            rc = self.w3.eth.get_transaction_receipt(h)
            if rc.status != 1 or not rc.logs:
                continue
            for ev in self._event_names:
                for e in getattr(self.c.events, ev)().process_receipt(rc, errors=DISCARD):
                    self.evlog.append({"event": e.event, "args": {k: (v.hex() if isinstance(v, (bytes, bytearray)) else v) for k, v in e.args.items()},
                                       "block": e.blockNumber, "log_index": e.logIndex, "tx_index": e.transactionIndex, "ts": blk.timestamp})
        self.sent = []

    def mined(self, h):
        return self.w3.eth.get_transaction_receipt(h)

    def head(self):
        return self.w3.eth.block_number

    # ---- 조회
    def read_version(self, block=None):
        """집계 주체가 읽는 철회 버전. v3.1: effectiveCount, v3.2: versionAt(그 블록의 시각)."""
        blk = self.w3.eth.get_block("latest" if block is None else block)
        if self.version == "v3.1":
            return self.c.functions.effectiveCount().call(block_identifier=blk.number)
        return self.c.functions.versionAt(blk.timestamp).call(block_identifier=blk.number)

    def _cache_version(self):
        self.ver_cache[self.head()] = self.read_version()

    def sig_v32(self, acct, tag, subject, version, value, level):
        h = Web3.keccak(encode(["uint256", "address", "bytes32", "bytes32", "uint32", "bytes32", "uint8"],
                               [self.chain, self.c.address, tag, subject, version, value, level]))
        return acct.sign_message(encode_defunct(primitive=h)).signature

    # ---- 프로토콜 단계
    def register(self, pid, home):
        subject, rids = C.subject_of(pid)
        key = Account.from_key(Web3.keccak(text=f"validation-B-subjectkey-{pid}"))
        pc = b32(f"policy-{pid}")
        if self.version == "v3.1":
            raw = self.sign("platform", "registerPolicy", subject, 1, pc, 3)
        else:
            raw = self.sign("platform", "registerPolicy", subject, 1, pc, 3, key.address, home,
                            self.sig_v32(key, TAG_POLICY, subject, 1, pc, 3))
        return {"pid": pid, "subject": subject, "rids": rids, "key": key, "home": home, "raw_register": raw}

    def sign_withdraw(self, s, relay=0):
        root = b32(f"list-{s['pid']}")
        if self.version == "v3.1":
            return self.sign(f"relay{relay}", "requestWithdrawal", s["subject"], 1, root)
        return self.sign(f"relay{relay}", "requestWithdrawal", s["subject"], 1, root, self.sig_v32(s["key"], TAG_WITHDRAW, s["subject"], 1, root, 0))

    def sign_decide(self, k, r, accept, vc, reason=b"\x00" * 32):
        if self.version == "v3.1":
            return self.sign("aggregator", "decide", k, r, accept, reason)
        return self.sign("aggregator", "decide", k, r, accept, reason, vc)

    def events(self):
        return sorted(self.evlog, key=lambda e: (e["block"], e["tx_index"], e["log_index"]))

    def revert_reason(self, k, r, vc):
        """되돌려진 수락을 같은 인자로 call해 사유를 얻는다(현재 머리 상태)."""
        try:
            args = (k, r, True, b"\x00" * 32) + (() if self.version == "v3.1" else (vc,))
            self.c.functions.decide(*args).call({"from": self.roles["aggregator"].address})
            return None
        except Exception as e:
            return str(e)[:120]


def pos(rc):
    return (rc.blockNumber, rc.transactionIndex)


def run_trial(w, p, trees, subject_rids):
    k = p["k"]
    w.round += 1
    r = w.round
    K = p.get("K", 1)
    raw_open = w.sign("platform", "openRound", r, C.FAR, b32(f"start{r}"))      # 같은 계정의 거래는 서명 순서대로 전달한다
    home_rel = k if p["relation"] == "same" else (k + 1) % 3
    subs = [w.register(10_000_000 * (1 if w.version == "v3.1" else 2) + p["tid"] * 10 + i, home_rel if i == 0 else (k + 2) % 3) for i in range(K)]
    related = subs[0] if p["relation"] == "same" else None
    tree = C.build_set(f"{w.version}-{p['tid']}", related["rids"] if related else [])
    trees[tree.root] = tree
    for s in subs:
        subject_rids[s["subject"].hex()] = s["rids"]
    # 준비 블록(계정당 블록마다 거래 하나): [라운드 개시, 입력 커밋] → [정책 등록 1, 갱신 커밋] → [정책 등록 2..K]
    sigma = w.ver_cache[w.head()]
    hs = [w.deliver(raw_open), w.tx(f"dc{k}", "commitInput", k, r, C.root_bytes(tree), len(tree), sigma, b32("purpose"))]
    w.mine()
    hs += [w.deliver(subs[0]["raw_register"]), w.tx(f"dc{k}", "commitUpdate", k, r, b32(f"delta-{w.version}-{p['tid']}"), b32(f"start{r}"))]
    w.mine()
    for s in subs[1:]:
        hs.append(w.deliver(s["raw_register"])); w.mine()
    assert all(w.mined(h).status == 1 for h in hs), "준비 거래 실패"
    start_block = w.head()
    rec = {**p, "r": r, "sigma": sigma, "start_block": start_block}
    cond = p["cond"]
    hW, hD, vc, accept_sent = [], None, None, True
    if cond == "L1":
        hW = [w.deliver(w.sign_withdraw(subs[0]))]; w.mine()
        vc = w.read_version()                                    # 철회 확정 뒤의 새 조회
        accept_sent = related is None                             # 정직한 집계 주체: 비소속 검사 실패(같은 기관)면 거절
        hD = w.deliver(w.sign_decide(k, r, accept_sent, vc if accept_sent else 0, b32("withdrawn record in S")))
        w.mine()
        rec["vc_read_block"] = w.head() - 1
    elif cond == "L2":
        vc = w.read_version(); rec["vc_read_block"] = w.head()
        hD = w.deliver(w.sign_decide(k, r, True, vc)); w.mine()
        w.mine(p["gap"] - 1)
        hW = [w.deliver(w.sign_withdraw(subs[0]))]; w.mine()
    elif cond in ("L3", "L5"):
        vc = w.read_version(); rec["vc_read_block"] = w.head()
        rD, rW = w.sign_decide(k, r, True, vc), w.sign_withdraw(subs[0])
        same = True if cond == "L5" else p["same_block"]
        first, second = (rW, rD) if p["first"] == "W" else (rD, rW)
        h1 = w.deliver(first)
        if not same:
            w.mine()
        h2 = w.deliver(second); w.mine()
        hW, hD = ([h1], h2) if p["first"] == "W" else ([h2], h1)
    elif cond == "L4":
        if p["signed_first"] == "D":                              # D를 먼저 서명·제출했지만 W가 먼저 포함된다
            vc = w.read_version(); rec["vc_read_block"] = w.head()
            rD = w.sign_decide(k, r, True, vc); rW = w.sign_withdraw(subs[0])
            hW = [w.deliver(rW)]
            if not p["same_block"]:
                w.mine()
            hD = w.deliver(rD); w.mine()
        else:                                                     # W를 먼저 서명·제출했지만 D가 먼저 포함된다
            rW = w.sign_withdraw(subs[0])
            vc = w.read_version(); rec["vc_read_block"] = w.head()
            rD = w.sign_decide(k, r, True, vc)
            hD = w.deliver(rD)
            if not p["same_block"]:
                w.mine()
            hW = [w.deliver(rW)]; w.mine()
    elif cond == "L6":
        h0 = w.head()
        vc = w.ver_cache[h0]; rec["vc_read_block"] = h0           # L블록 전 상태에서 읽은 값
        raws = [w.sign_withdraw(s, relay=i) for i, s in enumerate(subs)]
        for b in range(1, p["lag"] + 1):
            hW += [w.deliver(raws[i]) for i in range(K) if p["w_blocks"][i] == b]
            w.mine()
        assert w.read_version(h0) == vc, "과거 블록 상태 조회와 캐시가 다르다"
        hD = w.deliver(w.sign_decide(k, r, True, vc)); w.mine()
    rcW, rcD = [w.mined(h) for h in hW], w.mined(hD)
    w_first = min(pos(x) for x in rcW)
    rec.update({"vc": vc, "accept_sent": accept_sent, "W_status": [x.status for x in rcW], "W_pos": [list(pos(x)) for x in rcW],
                "W_gas": [x.gasUsed for x in rcW], "D_status": rcD.status, "D_pos": list(pos(rcD)), "D_gas": rcD.gasUsed,
                "ledger_class": C.ledger_class(w_first, pos(rcD)),
                "block_ts": {str(b): w.w3.eth.get_block(b).timestamp for b in sorted({x.blockNumber for x in rcW} | {rcD.blockNumber})}})
    # 참값(하네스): D보다 원장 순서가 앞선 효력 철회 수
    n_before = sum(1 for x in rcW if x.status == 1 and pos(x) < pos(rcD))
    v_incl = sigma + n_before                                     # 준비 블록 이후 이 컨트랙트의 다른 철회는 없다
    rec["v_incl"] = v_incl
    accepted = bool(rcD.status == 1 and accept_sent)
    rec["accepted"] = accepted
    rec["stale_acceptance"] = bool(accepted and vc < v_incl)
    related_before = bool(related is not None and rcW[0].status == 1 and pos(rcW[0]) < pos(rcD))
    rec["improper_acceptance"] = bool(accepted and related_before)
    rec["reverted"] = bool(rcD.status == 0)
    rec["unnecessary_revert"] = bool(rcD.status == 0 and not related_before)
    if rcD.status == 0:
        rec["revert_reason"] = w.revert_reason(k, r, vc)
        v2 = w.read_version()                                      # 정직한 후속: 새로 읽고 다시 결정
        acc2 = not related_before
        h2 = w.deliver(w.sign_decide(k, r, acc2, v2 if acc2 else 0, b32("withdrawn record in S"))); w.mine()
        rc2 = w.mined(h2)
        rec.update({"followup": "accept" if acc2 else "reject", "followup_status": rc2.status, "followup_gas": rc2.gasUsed, "attempts": 2})
    else:
        rec["attempts"] = 1
    rec["end_block"] = w.head()
    return rec


def liveness(version, q, n, seed):
    """2차: 블록마다 무관한 철회가 확률 q로 도착하고 조회가 1블록 늦을 때, 수락까지의 제출 횟수(상한 20)."""
    import random
    rng = random.Random(f"{seed}-live-{version}-{q}")
    w = World(version)
    out = []
    pid = 30_000_000
    for i in range(n):
        w.round += 1
        r, k = w.round, 0
        tree = C.build_set(f"live-{version}-{q}-{i}", [])
        w.tx("platform", "openRound", r, C.FAR, b32(f"start{r}"))
        w.tx("dc0", "commitInput", k, r, C.root_bytes(tree), len(tree), w.ver_cache[w.head()], b32("purpose"))
        w.mine()
        w.tx("dc0", "commitUpdate", k, r, b32(f"live-delta-{q}-{i}"), b32(f"start{r}"))
        w.mine()
        attempts = 0
        while attempts < 20:
            attempts += 1
            vc = w.read_version()                              # 머리 블록 h의 상태
            if rng.random() < q:                               # 블록 h+1에 무관한 철회가 먼저 들어간다
                pid += 1
                s = w.register(pid, 2)
                w.deliver(s["raw_register"]); w.deliver(w.sign_withdraw(s))
            hD = w.deliver(w.sign_decide(k, r, True, vc)); w.mine()
            if w.mined(hD).status == 1:
                break
        out.append({"q": q, "version": version, "i": i, "attempts": attempts, "accepted": bool(w.mined(hD).status == 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=C.N_PER_CELL)
    ap.add_argument("--skip-liveness", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    head = subprocess.run(["git", "-C", C.ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import eth, eth_tester, web3
    print(f"# local_harness start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} git {head} seed {C.SEED} n/cell {args.n} "
          f"web3 {web3.__version__} py-evm {eth.__version__} script_sha256 {C.sha256_file(__file__)} common_sha256 {C.sha256_file(C.__file__)}", flush=True)
    plan = C.local_plan(n=args.n)
    worlds, trees, rids = {}, {}, {}
    recs = []
    for p in plan:
        w = worlds.get(p["version"]) or worlds.setdefault(p["version"], World(p["version"]))
        recs.append(run_trial(w, p, trees, rids))
        if len(recs) % 100 == 0:
            print(f"  {len(recs)}/{len(plan)} trials ({time.time() - t0:.0f}s)", flush=True)
    # 사후 탐지기: 컨트랙트별 전체 이벤트
    det = {v: C.detect(w.events(), trees, rids) for v, w in worlds.items()}
    for rec in recs:
        d = det[rec["version"]].get((rec["k"], rec["r"]))
        rec["det"] = None if d is None else {"version_recorded": d["version_recorded"], "eff_before_pos": d["eff_before_pos"],
                                              "version_consistent": d["version_consistent"],
                                              "flag_pos": bool(d["viol_pos"]), "flag_time": bool(d["viol_time"])}
    live = []
    if not args.skip_liveness:
        for version in ("v3.1", "v3.2"):
            for q in (0.1, 0.5, 0.9):
                live += liveness(version, q, 50, C.SEED)
                print(f"  liveness {version} q={q} done ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out, "w") as f:
        for rec in recs:
            f.write(json.dumps({"type": "trial", **rec}) + "\n")
        for x in live:
            f.write(json.dumps({"type": "liveness", **x}) + "\n")
    # 재현 비교용 요약 벡터(시각·해시처럼 실행마다 달라질 수 있는 값은 뺀다)
    key = [(r["tid"], r["ledger_class"], r["D_status"], r["W_status"], r["accepted"], r["stale_acceptance"], r["improper_acceptance"],
            r["reverted"], r["unnecessary_revert"], r["attempts"], r["D_gas"], r["W_gas"], r["vc"], r["v_incl"],
            None if r["det"] is None else (r["det"]["version_recorded"], r["det"]["flag_pos"], r["det"]["flag_time"])) for r in recs]
    key += [(x["version"], x["q"], x["i"], x["attempts"]) for x in live]
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    print(f"# done {len(recs)} trials, {len(live)} liveness, outcome_sha256 {digest}, wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
