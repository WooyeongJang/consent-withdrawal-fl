"""Kairos 컨트랙트 배포·실행 연결 시험 (3장 v1.6 3.2·6.3절, 8.2절).

contracts/probe/Probe.sol을 컴파일하고, 발신 계정 서명 + 수수료 대납 계정 대납으로
(1) FEE_DELEGATED_SMART_CONTRACT_DEPLOY 1건, (2) FEE_DELEGATED_SMART_CONTRACT_EXECUTION 1건
(requestWithdrawal → WithdrawalRequested 이벤트)을 보내고 지연·가스·이벤트를 기록한다.
연결 시험용이며 이 거래를 실험의 진본성 증거로 쓰지 않는다.

    python kairos_contract_probe.py --solc PATH [--evm-version prague] [--out reports/kairos_contract_probe.json]
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from cytoolz import merge
from eth_account import Account
from web3 import Web3
from web3py_ext import extend  # noqa: F401  Account에 Kaia 서명 함수를 붙인다
from web3py_ext.transaction.transaction import TxType, empty_tx

RPC = "https://public-en-kairos.node.kaia.io"
CHAIN_ID = 1001
KEY_DIR = Path(os.path.expanduser(os.environ.get("KAIROS_KEY_DIR", "~/.config/dissertation/kairos")))
SRC = Path(__file__).resolve().parent.parent / "contracts" / "probe" / "Probe.sol"


def load(role):
    return Account.from_key(json.loads((KEY_DIR / f"{role}.json").read_text())["private_key"])


def compile_probe(solc, evm):
    out = subprocess.run([solc, "--evm-version", evm, "--combined-json", "abi,bin", str(SRC)],
                         check=True, capture_output=True, text=True).stdout
    c = json.loads(out)["contracts"]
    c = next(v for k, v in c.items() if k.endswith(":Probe"))
    abi = c["abi"] if isinstance(c["abi"], list) else json.loads(c["abi"])
    return abi, "0x" + c["bin"]


def as_hex(h):
    # web3py-ext는 거래 해시를 bytes가 아닌 문자열로 돌려준다
    return h if isinstance(h, str) else "0x" + bytes(h).hex().removeprefix("0x")


def send_fee_delegated(w3, tx_type, fields, sender, payer, gas):
    tx = merge(empty_tx(tx_type), {
        "from": sender.address,
        "nonce": w3.eth.get_transaction_count(sender.address),
        "chainId": CHAIN_ID,
        "gasPrice": w3.eth.gas_price,
        "gas": gas,
    }, fields)
    signed = Account.sign_transaction(tx, sender.key)
    fd = Account.sign_transaction_as_feepayer(signed.rawTransaction, payer.address, payer.key)
    t0 = time.time()
    h = as_hex(w3.eth.send_raw_transaction(fd.rawTransaction))
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=120, poll_latency=0.1)
    t1 = time.time()
    ktx = w3.provider.make_request("kaia_getTransactionByHash", [h])["result"]
    blk = w3.eth.get_block(rcpt.blockNumber)
    return rcpt, {
        "tx_hash": h,
        "tx_type": ktx.get("type"),
        "status": rcpt.status,
        "block_number": rcpt.blockNumber,
        "sender": ktx.get("from"),
        "fee_payer_in_tx": ktx.get("feePayer"),
        "gas_limit": gas,
        "gas_used": rcpt.gasUsed,
        "fee_kei": rcpt.gasUsed * rcpt.effectiveGasPrice,
        "submit_to_receipt_s": round(t1 - t0, 3),
        "submit_to_block_timestamp_s": round(blk.timestamp - t0, 3),
        "explorer": f"https://kairos.kaiascan.io/tx/{h}",
    }


def estimate(w3, payer, **kw):
    # eth_estimateGas는 대납을 고려하지 않으므로 잔액이 있는 대납 계정을 from으로 추정하고 여유를 둔다
    return int(w3.eth.estimate_gas({"from": payer.address, **kw}) * 1.5) + 50_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solc", required=True)
    ap.add_argument("--evm-version", default="prague")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    assert w3.eth.chain_id == CHAIN_ID, "체인 ID가 Kairos(1001)가 아님"
    sender, payer = load("sender"), load("fee_payer")
    solc_version = subprocess.run([args.solc, "--version"], capture_output=True, text=True).stdout.split()[-1]

    abi, bytecode = compile_probe(args.solc, args.evm_version)
    _, bytecode_osaka = compile_probe(args.solc, "osaka")
    payer_before = w3.eth.get_balance(payer.address)

    # 1. 대납 배포
    rcpt, deploy = send_fee_delegated(
        w3, TxType.FEE_DELEGATED_SMART_CONTRACT_DEPLOY, {"data": bytecode},
        sender, payer, estimate(w3, payer, data=bytecode))
    address = rcpt.contractAddress
    deploy["contract_address"] = address
    deploy["runtime_code_bytes"] = len(w3.eth.get_code(address))

    # 2. 대납 실행: requestWithdrawal → WithdrawalRequested
    probe = w3.eth.contract(address=address, abi=abi)
    args_ = (Web3.keccak(text="probe-subject"), 1, 3, Web3.keccak(text="probe-withdrawal-list"))
    data = probe.encodeABI(fn_name="requestWithdrawal", args=args_)
    rcpt2, execute = send_fee_delegated(
        w3, TxType.FEE_DELEGATED_SMART_CONTRACT_EXECUTION, {"to": address, "data": data},
        sender, payer, estimate(w3, payer, to=address, data=data))
    ev = probe.events.WithdrawalRequested().process_receipt(rcpt2)
    execute["events"] = [{
        "event": e.event,
        "subject": e.args.subject.hex(),
        "version": e.args.version,
        "level": e.args.level,
        "listRoot": e.args.listRoot.hex(),
        "matches_input": (e.args.subject, e.args.version, e.args.level, e.args.listRoot) == args_,
    } for e in ev]

    result = {
        "rpc": RPC,
        "chain_id": CHAIN_ID,
        "client": w3.provider.make_request("web3_clientVersion", [])["result"],
        "solc": solc_version,
        "evm_version": args.evm_version,
        "bytecode_same_as_osaka": bytecode == bytecode_osaka,
        "deploy": deploy,
        "execute": execute,
        "fee_payer_balance_change_kei": w3.eth.get_balance(payer.address) - payer_before,
        "sender_balance_kei": w3.eth.get_balance(sender.address),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
