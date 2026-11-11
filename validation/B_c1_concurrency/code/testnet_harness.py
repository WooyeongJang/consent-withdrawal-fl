"""실험 B-T: Kaia Kairos 테스트넷에서 철회(W)와 수락(D)을 독립된 두 제출자가 동시에 내는 관측 시험.

- 컨트랙트는 results/artifacts/의 컴파일 산출물(원본 소스 그대로)을 새로 배포해 쓴다. ledger/ 서비스는 쓰지 않는다.
- 역할 계정이 서명하고 플랫폼 계정이 수수료를 대납한다(실험과 같은 방식, web3py-ext 2.1.0). 키 값은 출력·기록하지 않는다.
- 포함 순서는 통제하지 않는다. 영수증의 (블록, 거래 색인)으로 사후에 나눈다.
- 실행 환경: 원장 컨테이너 이미지 dissertation-ledger:0.5.

  python testnet_harness.py --version v3.1 --n 150 --out /out/testnet_v3.1.jsonl --budget-kaia 5
"""
import argparse
import json
import os
import sys
import threading
import time

from cytoolz import merge
from eth_abi import encode
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from web3.exceptions import TransactionNotFound
from web3py_ext import extend  # noqa: F401
from web3py_ext.transaction.transaction import TxType, empty_tx

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

RPC = os.environ.get("RPC", "https://public-en-kairos.node.kaia.io")
CHAIN_ID = 1001
KEYS = os.environ.get("KEYS", "/keys")
ART = os.environ.get("ART", os.path.join(os.path.dirname(__file__), "..", "results", "artifacts"))
TAG_POLICY = Web3.keccak(text="ConsentLedger.v3.2.policy")
TAG_WITHDRAW = Web3.keccak(text="ConsentLedger.v3.2.withdraw")
GAS = {"openRound": 150_000, "registerPolicy": 350_000, "commitInput": 250_000, "commitUpdate": 250_000,
       "requestWithdrawal": 400_000, "decide": 250_000, "commitTrainingConfig": 150_000}
ZERO32 = b"\x00" * 32


def b32(s):
    return Web3.keccak(text=s)


def as_hex(h):
    return h if isinstance(h, str) else "0x" + bytes(h).hex().removeprefix("0x")


def new_w3():
    return Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))


def retry(fn, tries=6, wait=0.5):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


class Net:
    def __init__(self, version):
        self.version = version
        self.w3 = new_w3()
        self.roles = {}
        for f in os.listdir(KEYS):
            if f.endswith(".json"):
                j = json.load(open(os.path.join(KEYS, f)))
                self.roles[j["role"]] = Account.from_key(j["private_key"])
        self.payer = self.roles["platform"]
        self.nonce = {}
        self.gas_price = retry(lambda: self.w3.eth.gas_price)
        self.fee_kei = 0
        self.fee_lock = threading.Lock()
        art = json.load(open(os.path.join(ART, f"{version}.json")))
        self.abi, self.bin, self.runtime = art["abi"], art["bin"], art["bin_runtime"]
        self.c = None
        self.txlog = []

    def next_nonce(self, acct):
        a = acct.address
        if a not in self.nonce:
            self.nonce[a] = retry(lambda: self.w3.eth.get_transaction_count(a, "pending"))
        n = self.nonce[a]; self.nonce[a] += 1
        return n

    def resync(self):
        self.nonce = {}

    def sign(self, acct, data, gas, to=None, deploy=False, nonce=None):
        """역할 계정 서명 + 플랫폼 대납 서명까지 끝낸 raw 거래."""
        ttype = TxType.FEE_DELEGATED_SMART_CONTRACT_DEPLOY if deploy else TxType.FEE_DELEGATED_SMART_CONTRACT_EXECUTION
        tx = merge(empty_tx(ttype), {"from": acct.address, "nonce": self.next_nonce(acct) if nonce is None else nonce, "chainId": CHAIN_ID,
                                     "gasPrice": self.gas_price, "gas": gas, "data": data}, {} if deploy else {"to": to})
        signed = Account.sign_transaction(tx, acct.key)
        return Account.sign_transaction_as_feepayer(signed.rawTransaction, self.payer.address, self.payer.key).rawTransaction

    def call_data(self, fn, *args):
        return getattr(self.c.functions, fn)(*args)._encode_transaction_data()

    def sign_fn(self, role_or_acct, fn, *args, nonce=None):
        acct = self.roles[role_or_acct] if isinstance(role_or_acct, str) else role_or_acct
        return self.sign(acct, self.call_data(fn, *args), GAS[fn], to=self.c.address, nonce=nonce)

    def send(self, raw, w3=None):
        return as_hex((w3 or self.w3).eth.send_raw_transaction(raw))

    def receipt(self, h, w3=None, timeout=90):
        w3 = w3 or self.w3
        t0 = time.time()
        while True:
            try:
                rc = w3.eth.get_transaction_receipt(h)
                if rc is not None:
                    with self.fee_lock:
                        self.fee_kei += rc.gasUsed * rc.effectiveGasPrice
                    return rc
            except TransactionNotFound:
                pass
            except Exception:
                pass
            if time.time() - t0 > timeout:
                raise TimeoutError(f"receipt timeout {h}")
            time.sleep(0.1)

    def send_wait(self, raw, label):
        h = self.send(raw)
        rc = self.receipt(h)
        self.txlog.append({"label": label, "hash": h, "block": rc.blockNumber, "status": rc.status, "gas_used": rc.gasUsed})
        return h, rc

    def deploy(self):
        ctor = self.w3.eth.contract(abi=self.abi, bytecode=self.bin).constructor(
            self.roles["aggregator"].address, self.roles["auditor"].address, [self.roles[f"dc{i}"].address for i in range(3)], 0)
        h, rc = self.send_wait(self.sign(self.payer, ctor.data_in_transaction, 8_000_000, deploy=True), "deploy")
        assert rc.status == 1, "deploy failed"
        code = retry(lambda: self.w3.eth.get_code(rc.contractAddress, block_identifier=rc.blockNumber))
        self.c = self.w3.eth.contract(address=rc.contractAddress, abi=self.abi)
        info = {"address": rc.contractAddress, "tx": h, "block": rc.blockNumber, "gas_used": rc.gasUsed,
                "code_matches_local_compile": bytes(code).hex() == self.runtime, "code_bytes": len(bytes(code))}
        h2, rc2 = self.send_wait(self.sign_fn("platform", "commitTrainingConfig", b32("validation-B-config")), "commitTrainingConfig")
        assert rc2.status == 1
        return info

    def read_version(self, w3=None, block="latest"):
        """집계 주체의 조회. v3.1: effectiveCount, v3.2: versionAt(조회한 블록의 시각). 반환 (버전, 조회 시 머리 블록)."""
        w3 = w3 or self.w3
        c = w3.eth.contract(address=self.c.address, abi=self.abi)
        blk = w3.eth.get_block(block)
        if self.version == "v3.1":
            return c.functions.effectiveCount().call(block_identifier=block), blk.number
        return c.functions.versionAt(blk.timestamp).call(block_identifier=block), blk.number

    def read_version_pinned(self, block):
        """영수증 블록에 고정한 조회(노드가 그 블록을 알 때까지 다시 시도)."""
        return retry(lambda: self.read_version(block=block), tries=40, wait=0.25)

    def sig_v32(self, key, tag, subject, version, value, level):
        h = Web3.keccak(encode(["uint256", "address", "bytes32", "bytes32", "uint32", "bytes32", "uint8"],
                               [CHAIN_ID, self.c.address, tag, subject, version, value, level]))
        return Account.sign_message(encode_defunct(primitive=h), key.key).signature

    def events(self, from_block, to_block):
        topics = {}
        for ev in (x for x in self.abi if x.get("type") == "event"):
            sig = f"{ev['name']}({','.join(i['type'] for i in ev['inputs'])})"
            topics[Web3.keccak(text=sig).hex().removeprefix("0x")] = ev["name"]
        out, ts = [], {}
        b = from_block
        while b <= to_block:
            e = min(b + 499, to_block)
            logs = retry(lambda: self.w3.eth.get_logs({"address": self.c.address, "fromBlock": b, "toBlock": e}), tries=8, wait=1.0)
            for lg in logs:
                name = topics.get(bytes(lg["topics"][0]).hex().removeprefix("0x"))
                if not name:
                    continue
                d = getattr(self.c.events, name)().process_log(lg)
                if d.blockNumber not in ts:
                    ts[d.blockNumber] = retry(lambda: self.w3.eth.get_block(d.blockNumber)).timestamp
                out.append({"event": name, "args": {k: (bytes(v).hex() if isinstance(v, (bytes, bytearray)) else v) for k, v in d.args.items()},
                            "block": d.blockNumber, "tx_index": d.transactionIndex, "log_index": d.logIndex, "ts": ts[d.blockNumber]})
            b = e + 1
        return sorted(out, key=lambda x: (x["block"], x["tx_index"], x["log_index"]))


def submitter(net, at, make_raw, res, name):
    """독립 제출자: 자기 연결로 정해진 시각에 (필요하면 조회하고) 서명·제출한 뒤 영수증을 기다린다."""
    try:
        w3 = new_w3()
        head0 = w3.eth.block_number                     # 연결을 미리 연다
        time.sleep(max(0.0, at - time.time()))
        raw, extra = make_raw(w3)
        res["head_at_submit"] = w3.eth.block_number
        res["t_submit"] = time.time()
        h = net.send(raw, w3)
        res["t_sent"] = time.time()
        rc = net.receipt(h, w3)
        res.update({"t_receipt": time.time(), "hash": h, "block": rc.blockNumber, "tx_index": rc.transactionIndex,
                    "status": rc.status, "gas_used": rc.gasUsed, "fee_kei": rc.gasUsed * rc.effectiveGasPrice, **extra})
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"


def run_trial(net, p, r, trees, subject_rids):
    k = p["k"]
    rec = {**p, "r": r, "t_start": time.time()}
    pid = (31 if net.version == "v3.1" else 32) * 1_000_000 + p["tid"]
    subject, rids = C.subject_of(pid)
    skey = Account.create()                             # 정보주체 단말의 키(테스트넷 전용, 저장하지 않는다)
    home = k if p["relation"] == "same" else (k + 1) % 3
    related = p["relation"] == "same"
    tree = C.build_set(f"T-{net.version}-{p['tid']}", rids if related else [])
    trees[tree.root] = tree
    subject_rids[subject.hex()] = rids
    pc = b32(f"policy-{pid}")
    # ---- 준비: 라운드 개시·정책 등록(플랫폼) → 입력·갱신 커밋(데이터센터 k)
    raws = [net.sign_fn("platform", "openRound", r, C.FAR, b32(f"start{r}"))]
    if net.version == "v3.1":
        raws.append(net.sign_fn("platform", "registerPolicy", subject, 1, pc, 3))
    else:
        raws.append(net.sign_fn("platform", "registerPolicy", subject, 1, pc, 3, skey.address, home, net.sig_v32(skey, TAG_POLICY, subject, 1, pc, 3)))
    hs = [net.send(x) for x in raws]
    rcs = [net.receipt(h) for h in hs]
    assert all(x.status == 1 for x in rcs), "setup(platform) failed"
    sigma, _ = net.read_version_pinned(max(x.blockNumber for x in rcs))
    dc = f"dc{k}"
    raws = [net.sign_fn(dc, "commitInput", k, r, C.root_bytes(tree), len(tree), sigma, b32("purpose")),
            net.sign_fn(dc, "commitUpdate", k, r, b32(f"delta-{net.version}-{p['tid']}"), b32(f"start{r}"))]
    hs = [net.send(x) for x in raws]
    rcs2 = [net.receipt(h) for h in hs]
    assert all(x.status == 1 for x in rcs2), "setup(dc) failed"
    setup_block = max(x.blockNumber for x in rcs2)
    rec.update({"sigma": sigma, "setup_block": setup_block, "setup_gas": sum(x.gasUsed for x in rcs + rcs2)})
    # ---- 경합: 독립된 두 제출자
    root = b32(f"list-{pid}")
    wacct = Account.create()                            # 철회 요청마다 새 제출 계정(수수료는 플랫폼 대납)
    if net.version == "v3.1":
        raw_w = net.sign(wacct, net.call_data("requestWithdrawal", subject, 1, root), GAS["requestWithdrawal"], to=net.c.address, nonce=0)
    else:
        raw_w = net.sign(wacct, net.call_data("requestWithdrawal", subject, 1, root, net.sig_v32(skey, TAG_WITHDRAW, subject, 1, root, 0)),
                         GAS["requestWithdrawal"], to=net.c.address, nonce=0)
    agg = net.roles["aggregator"]
    agg_nonce = net.next_nonce(agg)

    def make_d(w3):
        t_read = time.time()
        vc, blk = net.read_version(w3, "latest")        # 제출 직전 "latest" 조회
        # 정직한 집계 주체: 읽은 버전에 이 단위 소속 철회가 보이면(비소속 검사 실패) 거절한다. 그 밖에는 읽은 버전으로 수락한다.
        accept = not (related and vc > sigma)
        args = (k, r, accept, ZERO32 if accept else b32("withdrawn record in S")) + (() if net.version == "v3.1" else (vc if accept else 0,))
        return (net.sign(agg, net.call_data("decide", *args), GAS["decide"], to=net.c.address, nonce=agg_nonce),
                {"vc": vc, "vc_read_block": blk, "accept_sent": accept, "t_read": t_read})

    t0 = time.time() + 1.0
    off = p["offset_s"]
    at_w, at_d = (t0, t0 + off) if off >= 0 else (t0 - off, t0)
    W, D = {}, {}
    tw = threading.Thread(target=submitter, args=(net, at_w, lambda w3: (raw_w, {}), W, "W"))
    td = threading.Thread(target=submitter, args=(net, at_d, make_d, D, "D"))
    tw.start(); td.start(); tw.join(); td.join()
    rec["W"], rec["D"] = W, D
    if "error" in W or "error" in D:
        rec["excluded"] = f"W: {W.get('error')} | D: {D.get('error')}"
        net.resync()
        return rec
    # ---- 지연된 조회 측정: W 영수증 직후 독립 연결 3개의 "latest"
    probes = []
    for _ in range(3):
        try:
            v, blk = net.read_version(new_w3(), "latest")
            probes.append({"version": v, "head": blk})
        except Exception as e:
            probes.append({"error": str(e)[:80]})
    rec["probes_after_W"] = probes
    w_pos, d_pos = (W["block"], W["tx_index"]), (D["block"], D["tx_index"])
    blocks = sorted({W["block"], D["block"]})
    rec["block_ts"] = {str(b): retry(lambda: net.w3.eth.get_block(b)).timestamp for b in blocks}
    rec["ledger_class"] = C.ledger_class(w_pos, d_pos)
    w_before = bool(W["status"] == 1 and w_pos < d_pos)
    v_incl = sigma + (1 if w_before else 0)
    accepted = bool(D["status"] == 1 and D["accept_sent"])
    rec.update({"v_incl": v_incl, "accepted": accepted, "stale_acceptance": bool(accepted and D["vc"] < v_incl),
                "improper_acceptance": bool(accepted and related and w_before), "reverted": bool(D["status"] == 0),
                "unnecessary_revert": bool(D["status"] == 0 and not (related and w_before)),
                "submit_order": "W first" if W["t_submit"] < D["t_submit"] else "D first",
                "latency_W_s": W["t_receipt"] - W["t_submit"], "latency_D_s": D["t_receipt"] - D["t_submit"],
                "latency_W_blocks": W["block"] - W["head_at_submit"], "latency_D_blocks": D["block"] - D["head_at_submit"]})
    if D["status"] == 0:
        # 정직한 후속: 영수증 블록에 고정해 다시 읽고 다시 결정
        v2, _ = net.read_version_pinned(max(W["block"], D["block"]))
        acc2 = not (related and w_before)
        args = (k, r, acc2, ZERO32 if acc2 else b32("withdrawn record in S")) + (() if net.version == "v3.1" else (v2 if acc2 else 0,))
        try:
            h2 = net.send(net.sign_fn("aggregator", "decide", *args))
            rc2 = net.receipt(h2)
            rec.update({"followup": "accept" if acc2 else "reject", "followup_status": rc2.status, "followup_gas": rc2.gasUsed,
                        "followup_hash": h2, "followup_vc": v2, "attempts": 2})
        except Exception as e:
            rec.update({"followup_error": str(e)[:200]}); net.resync()
    else:
        rec["attempts"] = 1
    rec["t_end"] = time.time()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=("v3.1", "v3.2"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-kaia", type=float, default=5.0)
    ap.add_argument("--label", default="main")
    args = ap.parse_args()
    import web3
    print(f"# testnet_harness start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} version {args.version} n {args.n} label {args.label} seed {C.SEED} "
          f"rpc {RPC} web3 {web3.__version__} script_sha256 {C.sha256_file(__file__)} common_sha256 {C.sha256_file(C.__file__)}", flush=True)
    net = Net(args.version)
    bal0 = net.w3.eth.get_balance(net.payer.address)
    print(f"# roles {sorted(net.roles)} payer {net.payer.address} balance {bal0 / 1e18:.4f} KAIA client {net.w3.client_version} gasPrice {net.gas_price}", flush=True)
    dep = net.deploy()
    print(f"# deployed {json.dumps(dep)}", flush=True)
    assert dep["code_matches_local_compile"], "배포된 코드가 로컬 컴파일 결과와 다르다"
    plan = C.testnet_plan(args.version, args.n)
    trees, rids, recs = {}, {}, []
    out = open(args.out, "w")
    out.write(json.dumps({"type": "meta", "version": args.version, "label": args.label, "deploy": dep, "payer": net.payer.address,
                          "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_planned": args.n}) + "\n"); out.flush()
    for i, p in enumerate(plan):
        if net.fee_kei / 1e18 > args.budget_kaia:
            print(f"# budget exceeded ({net.fee_kei / 1e18:.3f} KAIA) — stop at trial {i}", flush=True)
            break
        try:
            rec = run_trial(net, p, i + 1, trees, rids)
        except Exception as e:
            rec = {**p, "r": i + 1, "excluded": f"setup: {type(e).__name__}: {str(e)[:200]}"}
            net.resync()
        recs.append(rec)
        if (i + 1) % 10 == 0:
            done = [x for x in recs if "excluded" not in x]
            print(f"  {i + 1}/{len(plan)} trials, excluded {len(recs) - len(done)}, stale acc {sum(x['stale_acceptance'] for x in done)}, "
                  f"reverted {sum(x['reverted'] for x in done)}, fee {net.fee_kei / 1e18:.3f} KAIA", flush=True)
    last = retry(lambda: net.w3.eth.block_number)
    evs = net.events(dep["block"], last)
    det = C.detect(evs, trees, rids)
    for rec in recs:
        d = det.get((rec["k"], rec["r"]))
        rec["det"] = None if d is None else {"version_recorded": d["version_recorded"], "eff_before_pos": d["eff_before_pos"],
                                              "version_consistent": d["version_consistent"], "flag_pos": bool(d["viol_pos"]),
                                              "flag_time": bool(d["viol_time"]), "tau": d["tau"]}
        out.write(json.dumps({"type": "trial", **rec}) + "\n")
    bal1 = net.w3.eth.get_balance(net.payer.address)
    out.write(json.dumps({"type": "summary", "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": len(evs), "last_block": last,
                          "fee_kaia_receipts": net.fee_kei / 1e18, "payer_balance_before": bal0 / 1e18, "payer_balance_after": bal1 / 1e18,
                          "txlog": net.txlog}) + "\n")
    out.close()
    json.dump(evs, open(args.out.replace(".jsonl", "_events.json"), "w"))
    print(f"# done {len(recs)} trials, events {len(evs)}, fee {net.fee_kei / 1e18:.4f} KAIA (balance diff {(bal0 - bal1) / 1e18:.4f})", flush=True)


if __name__ == "__main__":
    main()
