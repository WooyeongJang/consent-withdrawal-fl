"""Kairos 수수료 대납 거래 연결 시험 (3장 v1.6 3.2절 대납 구조, 8.2절 구현 환경).

발신 계정이 서명하고 수수료 대납 계정이 대납 서명한 FEE_DELEGATED_VALUE_TRANSFER(값 0, 자기 자신에게)
1건을 보내고, 블록 포함·수수료 부담 주체·지연을 기록한다.
연결 시험용이며 이 거래를 실험의 진본성 증거로 쓰지 않는다(reports/common/2026-10-25_Kairos_연구계정.md).

실행 환경: web3py-ext 2.1.0(web3 6.3.x 고정) + setuptools<81
    python kairos_fee_delegation_probe.py [--out reports/kairos_fd_probe.json]
"""

import argparse
import json
import os
import time
from pathlib import Path

from cytoolz import merge
from eth_account import Account
from web3 import Web3
from web3py_ext import extend  # noqa: F401  Account에 Kaia 서명 함수를 붙인다
from web3py_ext.transaction.transaction import TxType, empty_tx

RPC = "https://public-en-kairos.node.kaia.io"
CHAIN_ID = 1001
GAS_LIMIT = 100_000  # 대납 값 전송의 고유 가스(약 31,000)보다 넉넉하게
KEY_DIR = Path(os.path.expanduser(os.environ.get("KAIROS_KEY_DIR", "~/.config/dissertation/kairos")))


def load(role):
    return json.loads((KEY_DIR / f"{role}.json").read_text())


def rpc(w3, method, params):
    return w3.provider.make_request(method, params)["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tx-hash", default=None, help="새로 보내지 않고 이미 보낸 거래만 기록")
    args = ap.parse_args()

    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    assert w3.eth.chain_id == CHAIN_ID, "체인 ID가 Kairos(1001)가 아님"
    sender, payer = load("sender"), load("fee_payer")
    s_acct, p_acct = Account.from_key(sender["private_key"]), Account.from_key(payer["private_key"])
    assert s_acct.address == sender["address"] and p_acct.address == payer["address"]

    balances = lambda: {r: w3.eth.get_balance(a) for r, a in (("sender", s_acct.address), ("fee_payer", p_acct.address))}
    measured = {}
    if args.tx_hash:
        tx_hash = args.tx_hash
    else:
        before = balances()
        # fill_transaction은 eth_estimateGas로 가스를 추정하는데, 이 호출은 대납을 고려하지 않아
        # 잔액이 없는 발신자에게 "insufficient funds"를 낸다. 그래서 필드를 직접 채운다.
        tx = merge(empty_tx(TxType.FEE_DELEGATED_VALUE_TRANSFER), {
            "from": s_acct.address,
            "to": s_acct.address,
            "value": 0,
            "nonce": w3.eth.get_transaction_count(s_acct.address),
            "chainId": CHAIN_ID,
            "gasPrice": w3.eth.gas_price,
            "gas": GAS_LIMIT,
        })
        signed = Account.sign_transaction(tx, s_acct.key)
        fd_signed = Account.sign_transaction_as_feepayer(signed.rawTransaction, p_acct.address, p_acct.key)

        t0 = time.time()
        tx_hash = w3.eth.send_raw_transaction(fd_signed.rawTransaction)
        w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120, poll_latency=0.2)
        t1 = time.time()
        after = balances()
        measured = {
            "balance_change_kei": {r: after[r] - before[r] for r in before},
            "submit_to_receipt_s": round(t1 - t0, 3),
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        }
    # web3py-ext는 거래 해시를 bytes가 아닌 문자열로 돌려준다
    tx_hash = tx_hash if isinstance(tx_hash, str) else "0x" + bytes(tx_hash).hex().removeprefix("0x")

    receipt = w3.eth.get_transaction_receipt(tx_hash)
    block = w3.eth.get_block(receipt.blockNumber)
    kaia_tx = rpc(w3, "kaia_getTransactionByHash", [tx_hash])
    fee = receipt.gasUsed * receipt.effectiveGasPrice

    result = {
        "rpc": RPC,
        "chain_id": CHAIN_ID,
        "tx_type": kaia_tx.get("type"),
        "tx_hash": tx_hash,
        "status": receipt.status,
        "block_number": receipt.blockNumber,
        "block_timestamp": block.timestamp,
        "sender": kaia_tx.get("from"),
        "fee_payer_in_tx": kaia_tx.get("feePayer"),
        "gas": int(kaia_tx["gas"], 16),
        "gas_used": receipt.gasUsed,
        "effective_gas_price_kei": receipt.effectiveGasPrice,
        "fee_kei": fee,
        "balances_now_kei": balances(),
        **measured,
        "explorer": f"https://kairos.kaiascan.io/tx/{tx_hash}",
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
