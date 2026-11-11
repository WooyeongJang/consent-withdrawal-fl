"""원장 연동 서비스: ConsentLedger(Kaia Kairos)에 역할 계정 서명 + 플랫폼 수수료 대납 거래를 보낸다.

  POST /deploy   {"delta": 0}                        → 컨트랙트 배포(플랫폼), 주소 저장
  POST /tx       {"role": "dc0", "fn": "commitInput", "args": [...]}
                 role: platform | aggregator | auditor | dc0.. | subject(요청마다 새 계정, 6.1절)
                 → {"hash","block","status","gas_used","fee_kei","latency_s","events":[...]}
  POST /view     {"fn": "deployable", "args": [...]} → {"result": ...}
  GET  /info                                         → 주소, 역할 계정, 플랫폼 잔액

bytes32 인자는 "0x" + 64자리 16진수 문자열로 받는다. 직렬 처리(단일 스레드)로 계정별 nonce 충돌을 막는다.
알려진 web3py-ext 2.1.0 문제(reports/2026-10-25_Kairos_연구계정.md 4절): fill_transaction은 대납을 고려하지 않으므로
필드를 직접 채우고, send_raw_transaction의 반환값(문자열)을 정규화한다.
"""

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from cytoolz import merge
from hexbytes import HexBytes
from eth_account import Account
from web3 import Web3
from web3.logs import DISCARD
from web3py_ext import extend  # noqa: F401
from web3py_ext.transaction.transaction import TxType, empty_tx

RPC = os.environ.get("RPC", "https://public-en-kairos.node.kaia.io")
CHAIN_ID = int(os.environ.get("CHAIN_ID", "1001"))
KEYS = os.environ.get("KEYS", "/keys")
STATE = os.environ.get("STATE", "/state")
ARTIFACT = os.environ.get("ARTIFACT", "/artifact/ConsentLedger.json")
ART_DIR = os.path.dirname(ARTIFACT)
GAS_CAP = 1_000_000
# 일괄 제출(파이프라인)에서는 앞 거래가 반영되기 전이라 가스 추정이 실패할 수 있으므로 고정 한도를 쓴다.
# 값은 2026-10-25 시연의 실측 평균 × 약 1.6(reports/2026-10-25_원장연동_시연.md 3절). 미사용 가스는 청구되지 않는다.
FIXED_GAS = {"openRound": 100_000, "commitInput": 110_000, "commitUpdate": 100_000, "decide": 115_000,
             "aggregate": 110_000, "sampleAudit": 70_000, "recordVerification": 100_000}
LOCKS = {}
DEPLOY_GAS_CAP = 8_000_000

w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 60}))
art = json.load(open(ARTIFACT))
ABI, BIN = art["abi"], art["bin"]
ARTS = {f[:-5]: json.load(open(os.path.join(ART_DIR, f))) for f in os.listdir(ART_DIR) if f.endswith(".json")}
OTHER = {}   # 이름 → 배포된 보조 컨트랙트(예: CompletionRecord, 비교군 B3)
ROLES = {json.load(open(os.path.join(KEYS, f)))["role"]: Account.from_key(json.load(open(os.path.join(KEYS, f)))["private_key"])
         for f in os.listdir(KEYS) if f.endswith(".json")}
PAYER = ROLES["platform"]
state_file = os.path.join(STATE, "deployment.json")
contract = None
if os.path.exists(state_file):
    contract = w3.eth.contract(address=json.load(open(state_file))["address"], abi=ABI)


def as_hex(h):
    return h if isinstance(h, str) else "0x" + bytes(h).hex().removeprefix("0x")


def conv_args(fn, args):
    """ABI 타입에 맞게 인자를 변환한다(bytes32는 16진수 문자열 → bytes)."""
    spec = next(x for x in ABI if x.get("type") == "function" and x["name"] == fn and len(x["inputs"]) == len(args))
    out = []
    for a, inp in zip(args, spec["inputs"]):
        t = inp["type"]
        if t == "bytes32":
            out.append(bytes.fromhex(a.removeprefix("0x")))
        elif t.startswith("uint") or t.startswith("int"):
            out.append(int(a))
        elif t == "bool":
            out.append(bool(a))
        elif t == "address[]":
            out.append([Web3.to_checksum_address(x) for x in a])
        elif t == "address":
            out.append(Web3.to_checksum_address(a))
        else:
            out.append(a)
    return out


def decode_events(receipt):
    evs = []
    for ev in (x for x in ABI if x.get("type") == "event"):
        for e in getattr(contract.events, ev["name"])().process_receipt(receipt, errors=DISCARD):
            evs.append({"event": e.event, "args": {k: (as_hex(v) if isinstance(v, (bytes, bytearray)) else v)
                                                   for k, v in e.args.items()}})
    return evs


def send_fee_delegated(sender, data, to=None, deploy=False):
    est_tx = {"from": sender.address, "data": data}
    if to:
        est_tx["to"] = to
    try:
        gas = min(int(w3.eth.estimate_gas(est_tx) * 1.3) + 30_000, DEPLOY_GAS_CAP if deploy else GAS_CAP)
    except Exception as e:  # 실패 사유(revert 메시지)를 그대로 돌려준다
        raise RuntimeError(f"estimate_gas: {e}")
    ttype = TxType.FEE_DELEGATED_SMART_CONTRACT_DEPLOY if deploy else TxType.FEE_DELEGATED_SMART_CONTRACT_EXECUTION
    tx = merge(empty_tx(ttype), {"from": sender.address, "nonce": w3.eth.get_transaction_count(sender.address),
                                 "chainId": CHAIN_ID, "gasPrice": w3.eth.gas_price, "gas": gas, "data": data},
               {} if deploy else {"to": to})
    signed = Account.sign_transaction(tx, sender.key)
    fd = Account.sign_transaction_as_feepayer(signed.rawTransaction, PAYER.address, PAYER.key)
    t0 = time.time()
    h = as_hex(w3.eth.send_raw_transaction(fd.rawTransaction))
    rc = w3.eth.wait_for_transaction_receipt(h, timeout=180, poll_latency=0.1)
    return h, rc, time.time() - t0, gas


def _lock(addr):
    return LOCKS.setdefault(addr, threading.Lock())


def _send_pipelined(role, items):
    """같은 계정의 거래를 nonce 순서로 연달아 보내고 영수증을 모은다."""
    sender = ROLES[role]
    out = []
    with _lock(sender.address):
        nonce = w3.eth.get_transaction_count(sender.address, "pending")
        gas_price = w3.eth.gas_price
        sent = []
        for i, it in enumerate(items):
            fn = getattr(contract.functions, it["fn"])(*conv_args(it["fn"], it["args"]))
            data = fn._encode_transaction_data()
            gas = FIXED_GAS.get(it["fn"], GAS_CAP)
            tx = merge(empty_tx(TxType.FEE_DELEGATED_SMART_CONTRACT_EXECUTION),
                       {"from": sender.address, "to": contract.address, "nonce": nonce + i, "chainId": CHAIN_ID,
                        "gasPrice": gas_price, "gas": gas, "data": data})
            signed = Account.sign_transaction(tx, sender.key)
            with _lock(PAYER.address + ":sign"):
                fd = Account.sign_transaction_as_feepayer(signed.rawTransaction, PAYER.address, PAYER.key)
            t0 = time.time()
            sent.append((it, as_hex(w3.eth.send_raw_transaction(fd.rawTransaction)), t0, gas))
        for it, h, t0, gas in sent:
            rc = w3.eth.wait_for_transaction_receipt(h, timeout=180, poll_latency=0.1)
            out.append({"idx": it["idx"], "role": role, "fn": it["fn"], "hash": h, "block": rc.blockNumber,
                        "status": rc.status, "gas_used": rc.gasUsed, "gas_limit": gas,
                        "fee_kei": rc.gasUsed * rc.effectiveGasPrice, "latency_s": time.time() - t0,
                        "sender": sender.address, "events": decode_events(rc)})
    return out


def run_batch(txs):
    by_role = {}
    for i, x in enumerate(txs):
        by_role.setdefault(x["role"], []).append({**x, "idx": i})
    with ThreadPoolExecutor(len(by_role)) as ex:
        parts = list(ex.map(lambda kv: _send_pipelined(*kv), by_role.items()))
    return sorted((r for p in parts for r in p), key=lambda r: r["idx"])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/info":
            return self.reply(200, {"contract": contract.address if contract else None, "chain_id": w3.eth.chain_id,
                                    "roles": {r: a.address for r, a in ROLES.items()},
                                    "platform_balance_kaia": float(w3.from_wei(w3.eth.get_balance(PAYER.address), "ether"))})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        global contract
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        try:
            if self.path == "/deploy" and req.get("name", "ConsentLedger") != "ConsentLedger":
                name = req["name"]
                c0 = w3.eth.contract(abi=ARTS[name]["abi"], bytecode=ARTS[name]["bin"])
                h, rc, lat, gas = send_fee_delegated(ROLES[req.get("role", "platform")], c0.constructor().data_in_transaction, deploy=True)
                if rc.status != 1:
                    return self.reply(400, {"error": "deploy failed", "hash": h})
                OTHER[name] = w3.eth.contract(address=rc.contractAddress, abi=ARTS[name]["abi"])
                with open(os.path.join(STATE, "deployments.jsonl"), "a") as f:
                    f.write(json.dumps({"name": name, "address": rc.contractAddress, "tx": h, "block": rc.blockNumber,
                                        "label": req.get("label", ""), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
                return self.reply(200, {"address": rc.contractAddress, "hash": h, "gas_used": rc.gasUsed, "latency_s": lat})
            if self.path == "/tx_other":
                c = OTHER[req["name"]]
                sender = ROLES[req["role"]]
                args = [bytes.fromhex(a[2:]) if isinstance(a, str) and a.startswith("0x") and len(a) == 66 else a
                        for a in req["args"]]
                data = getattr(c.functions, req["fn"])(*args)._encode_transaction_data()
                h, rc, lat, gas = send_fee_delegated(sender, data, to=c.address)
                evs = []
                for ev in (x for x in ARTS[req["name"]]["abi"] if x.get("type") == "event"):
                    for e in getattr(c.events, ev["name"])().process_receipt(rc, errors=DISCARD):
                        evs.append({"event": e.event, "args": {k: (as_hex(v) if isinstance(v, (bytes, bytearray)) else v) for k, v in e.args.items()}})
                return self.reply(200, {"hash": h, "block": rc.blockNumber, "status": rc.status, "gas_used": rc.gasUsed,
                                        "fee_kei": rc.gasUsed * rc.effectiveGasPrice, "latency_s": lat, "events": evs})
            if self.path == "/deploy":
                c0 = w3.eth.contract(abi=ABI, bytecode=BIN)
                data = c0.constructor(ROLES["aggregator"].address, ROLES["auditor"].address,
                                      [ROLES[f"dc{i}"].address for i in range(int(req.get("k", 3)))], int(req.get("delta", 0))).data_in_transaction
                h, rc, lat, gas = send_fee_delegated(PAYER, data, deploy=True)
                if rc.status != 1 or len(w3.eth.get_code(rc.contractAddress)) == 0:
                    return self.reply(400, {"error": "deploy failed", "hash": h, "gas_used": rc.gasUsed, "gas_limit": gas})
                contract = w3.eth.contract(address=rc.contractAddress, abi=ABI)
                rec = {"address": rc.contractAddress, "tx": h, "block": rc.blockNumber, "label": req.get("label", ""),
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                json.dump(rec, open(state_file, "w"))
                with open(os.path.join(STATE, "deployments.jsonl"), "a") as f:   # 실험마다 새 컨트랙트(배포 이력)
                    f.write(json.dumps(rec) + "\n")
                return self.reply(200, {"address": rc.contractAddress, "hash": h, "gas_used": rc.gasUsed, "latency_s": lat})
            if self.path == "/tx":
                role = req["role"]
                if role == "subject":
                    sender = Account.create()   # 철회 요청마다 새 계정(6.1절). 키는 요청 기록과 함께 보관
                    with open(os.path.join(STATE, "subject_accounts.jsonl"), "a") as f:
                        f.write(json.dumps({"address": sender.address, "private_key": sender.key.hex()}) + "\n")
                else:
                    sender = ROLES[role]
                fn = getattr(contract.functions, req["fn"])(*conv_args(req["fn"], req["args"]))
                data = fn._encode_transaction_data()
                h, rc, lat, gas = send_fee_delegated(sender, data, to=contract.address)
                return self.reply(200, {"hash": h, "block": rc.blockNumber, "status": rc.status, "gas_used": rc.gasUsed,
                                        "gas_limit": gas, "fee_kei": rc.gasUsed * rc.effectiveGasPrice,
                                        "latency_s": lat, "sender": sender.address, "events": decode_events(rc)})
            if self.path == "/sign_subject":
                # 정보주체 단말: 요청마다 새 계정으로 서명만 한 거래를 만든다(대납 서명 전). 단말이 서명 시각과 함께 보관한다.
                sender = Account.create()
                with open(os.path.join(STATE, "subject_accounts.jsonl"), "a") as f:
                    f.write(json.dumps({"address": sender.address, "private_key": sender.key.hex()}) + "\n")
                fn = getattr(contract.functions, req["fn"])(*conv_args(req["fn"], req["args"]))
                tx = merge(empty_tx(TxType.FEE_DELEGATED_SMART_CONTRACT_EXECUTION),
                           {"from": sender.address, "to": contract.address, "nonce": 0, "chainId": CHAIN_ID,
                            "gasPrice": w3.eth.gas_price, "gas": 250_000, "data": fn._encode_transaction_data()})
                signed = Account.sign_transaction(tx, sender.key)
                return self.reply(200, {"raw": as_hex(signed.rawTransaction), "sender": sender.address,
                                        "signed_at": time.time()})
            if self.path == "/submit_feepayer":
                # 플랫폼: 정보주체가 서명한 거래에 대납 서명을 붙여 제출한다(T1에서는 이 시점을 늦춘다).
                raw = HexBytes(bytes.fromhex(req["raw"].removeprefix("0x")))
                with _lock(PAYER.address + ":sign"):
                    fd = Account.sign_transaction_as_feepayer(raw, PAYER.address, PAYER.key)
                t0 = time.time()
                h = as_hex(w3.eth.send_raw_transaction(fd.rawTransaction))
                rc = w3.eth.wait_for_transaction_receipt(h, timeout=180, poll_latency=0.1)
                blk = w3.eth.get_block(rc.blockNumber)
                return self.reply(200, {"hash": h, "block": rc.blockNumber, "block_time": blk.timestamp,
                                        "status": rc.status, "gas_used": rc.gasUsed,
                                        "fee_kei": rc.gasUsed * rc.effectiveGasPrice, "latency_s": time.time() - t0,
                                        "events": decode_events(rc)})
            if self.path == "/batch":
                return self.reply(200, {"results": run_batch(req["txs"])})
            if self.path == "/view":
                # 공개 RPC는 여러 노드로 분산되어 있어 직전 거래가 반영되지 않은 노드가 응답할 수 있다.
                # block을 주면 그 블록의 상태로 조회하고, 노드가 아직 그 블록을 모르면 다시 시도한다(읽기-쓰기 일관성).
                call = getattr(contract.functions, req["fn"])(*conv_args(req["fn"], req.get("args", [])))
                blk = req.get("block")
                for attempt in range(60):
                    try:
                        res = call.call(block_identifier=int(blk)) if blk else call.call()
                        break
                    except Exception:
                        if not blk or attempt == 59:
                            raise
                        time.sleep(0.5)
                return self.reply(200, {"result": as_hex(res) if isinstance(res, (bytes, bytearray)) else res})
            self.reply(404, {"error": "not found"})
        except Exception as e:
            self.reply(400, {"error": str(e)[:500]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8600"))
    print(f"ledger service on 127.0.0.1:{port}, contract={contract.address if contract else None}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
