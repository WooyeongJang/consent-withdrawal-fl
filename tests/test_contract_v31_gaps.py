"""v3.1(실험에 쓴 판)에서 구멍이 실제로 열려 있음을 확인한다(로컬 py-evm). 실행: <venv>/python tests/test_contract_v31_gaps.py

각 경로는 tests/test_contract_v3_2.py에서 v3.2가 막는 공격과 짝을 이룬다. 성공하면 "열림"으로 표시한다.
"""
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from test_contract import b32, compile_ledger

FAR = 2 ** 40
GAS = {"gas": 1_000_000}


def setup():
    prov = EthereumTesterProvider()
    w3 = Web3(prov)
    platform, agg, aud, dc0, dc1, dc2, subj, other = w3.eth.accounts[:8]
    abi, bytecode = compile_ledger()
    tx = w3.eth.contract(abi=abi, bytecode=bytecode).constructor(agg, aud, [dc0, dc1, dc2], 0).transact({"from": platform})
    c = w3.eth.contract(address=w3.eth.get_transaction_receipt(tx).contractAddress, abi=abi)
    c.functions.commitTrainingConfig(b32("cfg")).transact({"from": platform})
    return prov, w3, c, (platform, agg, aud, (dc0, dc1, dc2), other)


def run_round(c, r, roles, model, accept=True, open_=True):
    platform, agg, aud, dcs, _ = roles
    if open_:
        c.functions.openRound(r, FAR, b32(f"start{r}")).transact({"from": platform})
    for k, dc in enumerate(dcs):
        c.functions.commitInput(k, r, b32(f"root{k}{r}{model}"), 10, 1, b32("p")).transact({"from": dc})
        c.functions.commitUpdate(k, r, b32(f"d{k}{r}{model}"), b32(f"start{r}")).transact({"from": dc})
        c.functions.decide(k, r, accept, b32("")).transact({"from": agg})
    c.functions.aggregate(r, b32(model), b32("st")).transact({"from": agg})


def main():
    res = {}
    # (1) 제3자 철회·재전송: 아무 계정이나 같은 (subject, version)을 반복 제출 → 철회 수가 계속 오른다
    prov, w3, c, roles = setup()
    platform, agg, aud, dcs, other = roles
    s = b32("s")
    c.functions.registerPolicy(s, 1, b32("p"), 3).transact({"from": platform})
    for _ in range(3):
        c.functions.requestWithdrawal(s, 1, b32("list")).transact({"from": other})
    res["replayed withdrawal (3 submissions)"] = c.functions.pendingLevel2().call() == 3
    # 재동의 위조: 플랫폼 혼자 새 정책 버전 등록
    c.functions.registerPolicy(s, 2, b32("forged re-consent"), 1).transact({"from": platform})
    res["platform-only re-consent"] = c.functions.policies(s).call()[0] == 2

    # (2) 수락 경합: 한 블록 안에서 철회가 먼저 들어가도 낡은 조회로 한 수락이 통과
    prov, w3, c, roles = setup()
    platform, agg, aud, dcs, other = roles
    c.functions.registerPolicy(s, 1, b32("p"), 3).transact({"from": platform})
    c.functions.openRound(1, FAR, b32("start1")).transact({"from": platform})
    c.functions.commitInput(0, 1, b32("r"), 10, 1, b32("p")).transact({"from": dcs[0]})
    c.functions.commitUpdate(0, 1, b32("d"), b32("start1")).transact({"from": dcs[0]})
    t = prov.ethereum_tester
    t.disable_auto_mine_transactions()
    tw = c.functions.requestWithdrawal(s, 1, b32("list")).transact({"from": other, **GAS})
    td = c.functions.decide(0, 1, True, b32("")).transact({"from": agg, **GAS})
    t.mine_blocks(1)
    t.enable_auto_mine_transactions()
    rw, rd = w3.eth.get_transaction_receipt(tw), w3.eth.get_transaction_receipt(td)
    res["acceptance after in-block withdrawal"] = (rw.blockNumber == rd.blockNumber and rw.transactionIndex < rd.transactionIndex
                                                   and rd.status == 1)

    # (3) r₀ 미신고·다른 기관의 늦은 신고, (4) 해소 뒤 범위 해제, (5) 라운드 재사용
    prov, w3, c, roles = setup()
    platform, agg, aud, dcs, other = roles
    for r in (1, 2, 3, 4):
        run_round(c, r, roles, f"M{r}")
    c.functions.registerPolicy(s, 1, b32("p"), 3).transact({"from": platform})
    c.functions.requestWithdrawal(s, 1, b32("list")).transact({"from": other})
    res["undeclared r0: in-scope model deployable"] = c.functions.deployable(b32("M4")).call()
    c.functions.declareFirstUse(0, 4).transact({"from": dcs[0]})       # 소속이 아닌 기관이 실제(2)보다 늦게 신고
    res["late r0 by another institution (M2, M3 deployable)"] = (c.functions.deployable(b32("M2")).call()
                                                                 and c.functions.deployable(b32("M3")).call())
    # (5) 라운드 0을 다시 열어 범위 안 모델 M4를 round 0으로 재등록 → 배포 가능
    run_round(c, 0, roles, "M4")
    res["round reuse re-registers M4 as round 0"] = (c.functions.models(b32("M4")).call()[0] == 0
                                                     and c.functions.deployable(b32("M4")).call())
    # (4) 해소 뒤 옛 범위 모델이 다시 배포 가능
    prov, w3, c, roles = setup()
    platform, agg, aud, dcs, other = roles
    for r in (1, 2, 3):
        run_round(c, r, roles, f"M{r}")
    c.functions.registerPolicy(s, 1, b32("p"), 3).transact({"from": platform})
    c.functions.requestWithdrawal(s, 1, b32("list")).transact({"from": other})
    c.functions.declareFirstUse(0, 2).transact({"from": dcs[1]})
    P = b32("prog")
    c.functions.approveMethod(b32("U0"), P, 1).transact({"from": platform})
    c.functions.createJob(b32("U0"), P, b32("M1"), 1).transact({"from": agg})
    c.functions.completeJob(1, b32("J1")).transact({"from": agg})
    for lv in (1, 2):
        c.functions.recordVerification(b32("J1"), lv, True, b32("")).transact({"from": aud})
    c.functions.deploy(b32("J1"), b32("job")).transact({"from": platform})
    res["after resolution: old in-scope model deployable"] = c.functions.deployable(b32("M3")).call()

    for k, v in res.items():
        print(f"  {'open' if v else 'CLOSED?'}  {k}")
    assert all(res.values()), "v3.1에서 열려 있어야 할 경로가 닫혀 있다"
    print(f"v3.1 gaps: {sum(res.values())}/{len(res)} open")


if __name__ == "__main__":
    main()
