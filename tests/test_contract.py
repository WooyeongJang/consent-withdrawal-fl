"""ConsentLedger 상태 전이 단위 테스트(로컬 py-evm, Prague). 실행: <venv>/python tests/test_contract.py

필요: web3, eth-tester[py-evm], tools/bin/solc(tools/get_solc.sh)
"""
import json
import subprocess
from pathlib import Path

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

ROOT = Path(__file__).resolve().parent.parent


def compile_ledger():
    out = subprocess.run([str(ROOT / "tools/bin/solc"), "--evm-version", "prague", "--optimize",
                          "--combined-json", "abi,bin", str(ROOT / "contracts/ConsentLedger.sol")],
                         check=True, capture_output=True, text=True).stdout
    c = next(v for k, v in json.loads(out)["contracts"].items() if k.endswith(":ConsentLedger"))
    return c["abi"] if isinstance(c["abi"], list) else json.loads(c["abi"]), c["bin"]


def b32(s):
    return Web3.keccak(text=s)


def reverts(fn, sender, msg):
    try:
        fn.transact({"from": sender})
    except Exception as e:
        assert msg in str(e), f"expected revert '{msg}', got {e}"
        return
    raise AssertionError(f"expected revert '{msg}'")


def main():
    w3 = Web3(EthereumTesterProvider())
    platform, agg, aud, dc0, dc1, dc2, subj, other = w3.eth.accounts[:8]
    abi, bytecode = compile_ledger()
    tx = w3.eth.contract(abi=abi, bytecode=bytecode).constructor(agg, aud, [dc0, dc1, dc2], 0).transact({"from": platform})
    c = w3.eth.contract(address=w3.eth.get_transaction_receipt(tx).contractAddress, abi=abi)
    now = w3.eth.get_block("latest").timestamp

    # 정책 등록과 버전
    s = b32("subject-1")
    c.functions.registerPolicy(s, 1, b32("p1"), 3).transact({"from": platform})
    reverts(c.functions.registerPolicy(s, 3, b32("p"), 3), platform, "version")
    reverts(c.functions.registerPolicy(s, 2, b32("p"), 3), other, "role")

    # 학습 설정 커밋(v3.1): 없으면 라운드 개시 불가, 한 번만, 플랫폼만
    reverts(c.functions.openRound(1, now + 10 ** 6, b32("start1")), platform, "no config")
    reverts(c.functions.commitTrainingConfig(b32("cfg")), other, "role")
    c.functions.commitTrainingConfig(b32("cfg")).transact({"from": platform})
    reverts(c.functions.commitTrainingConfig(b32("cfg2")), platform, "config")

    # 정규 라운드 1–3
    for r in (1, 2, 3):
        c.functions.openRound(r, now + 10 ** 6, b32(f"start{r}")).transact({"from": platform})
        for k, dc in enumerate((dc0, dc1, dc2)):
            reverts(c.functions.commitUpdate(k, r, b32("d"), b32("s")), dc, "no input")
            reverts(c.functions.commitInput(k, r, b32("root"), 10, 1, b32("purpose")), other, "dc")
            c.functions.commitInput(k, r, b32(f"root{k}{r}"), 10, 1, b32("purpose")).transact({"from": dc})
            reverts(c.functions.decide(k, r, True, b32("")), agg, "no update")
            c.functions.commitUpdate(k, r, b32(f"d{k}{r}"), b32(f"start{r}")).transact({"from": dc})
        reverts(c.functions.aggregate(r, b32(f"M{r}"), b32("st")), agg, "decisions")
        reverts(c.functions.sampleAudit(r), aud, "not aggregated")
        for k in range(3):
            c.functions.decide(k, r, True, b32("")).transact({"from": agg})
        reverts(c.functions.decide(0, r, False, b32("x")), agg, "decided")
        c.functions.aggregate(r, b32(f"M{r}"), b32("st")).transact({"from": agg})
        c.functions.sampleAudit(r).transact({"from": aud})
    assert c.functions.deployable(b32("M3")).call()
    c.functions.deploy(b32("M3"), b32("aggregated")).transact({"from": platform})

    # 마감 이후 입력 커밋 거부
    c.functions.openRound(4, now - 1, b32("start4")).transact({"from": platform})
    reverts(c.functions.commitInput(0, 4, b32("root"), 10, 1, b32("p")), dc0, "round")

    # 처리 수준 3 철회(δ=0 → 즉시 효력), 첫 사용 라운드 r₀=2 신고
    c.functions.requestWithdrawal(s, 1, b32("listroot")).transact({"from": subj})
    assert c.functions.pendingLevel2().call() == 1 and c.functions.effectiveCount().call() == 1
    reverts(c.functions.declareFirstUse(0, 2), other, "dc")
    c.functions.declareFirstUse(0, 2).transact({"from": dc1})
    # I2: r₀ 이후 모델은 배포 불가, 이전 모델(M1)은 안전 버전
    assert not c.functions.deployable(b32("M3")).call() and not c.functions.deployable(b32("M2")).call()
    assert c.functions.deployable(b32("M1")).call()
    reverts(c.functions.deploy(b32("M3"), b32("x")), platform, "not deployable")
    c.functions.revoke(b32("M3"), b32("influence r0=2")).transact({"from": platform})
    assert c.functions.deployed().call() == b"\x00" * 32
    c.functions.deploy(b32("M1"), b32("rollback to safe version")).transact({"from": platform})

    # 처리 방법 승인(v3): 미승인 방법은 작업 생성 불가, 승인은 플랫폼만
    P = b32("prog-v1")
    reverts(c.functions.createJob(b32("U2"), P, b32("init"), 1), agg, "method")
    reverts(c.functions.approveMethod(b32("U2"), P, 2), agg, "role")
    reverts(c.functions.approveMethod(b32("U2"), P, 3), platform, "kind")
    c.functions.approveMethod(b32("U2"), P, 2).transact({"from": platform})       # 근사
    c.functions.approveMethod(b32("U0"), P, 1).transact({"from": platform})       # 정확
    # (iv) 승인되지 않은 코드(prog)로는 승인된 방법 이름이어도 작업을 만들 수 없다
    reverts(c.functions.createJob(b32("U0"), b32("prog-finetune"), b32("init"), 1), agg, "method")

    # 근사 방법 작업: L1·L2·L3 통과 전에는 배포 불가(I3), 검증은 감사자만(I4)
    reverts(c.functions.createJob(b32("U2"), P, b32("init"), 2), agg, "covers")
    c.functions.createJob(b32("U2"), P, b32("init"), 1).transact({"from": agg})
    c.functions.completeJob(1, b32("J1")).transact({"from": agg})
    assert not c.functions.deployable(b32("J1")).call()
    reverts(c.functions.recordVerification(b32("J1"), 1, True, b32("")), agg, "role")
    c.functions.recordVerification(b32("J1"), 1, True, b32("")).transact({"from": aud})
    c.functions.recordVerification(b32("J1"), 2, True, b32("")).transact({"from": aud})
    assert not c.functions.deployable(b32("J1")).call(), "L3 없이 배포 불가"
    c.functions.recordVerification(b32("J1"), 3, False, b32("residual above margin")).transact({"from": aud})
    assert not c.functions.deployable(b32("J1")).call(), "L3 불통과면 배포 불가"
    c.functions.recordVerification(b32("J1"), 3, True, b32("")).transact({"from": aud})
    assert c.functions.deployable(b32("J1")).call()
    c.functions.deploy(b32("J1"), b32("verified job")).transact({"from": platform})
    assert c.functions.pendingLevel2().call() == 0 and c.functions.deployed().call() == b32("J1")
    assert c.functions.deployable(b32("J1")).call(), "배포된 처리 작업 결과는 배포 허용을 유지(단말 대조 L4)"
    # 새 처리 수준 3 철회가 생기면 J1도 다시 막힌다(새 철회자의 데이터로 학습된 모델)
    s2 = b32("subject-2")
    c.functions.registerPolicy(s2, 1, b32("p2"), 3).transact({"from": platform})
    c.functions.requestWithdrawal(s2, 1, b32("list2")).transact({"from": subj})
    assert not c.functions.deployable(b32("J1")).call(), "새 미해소 철회가 생기면 배포 불가"

    # 정확 방법 작업(v3, 44번 (a)): L1·L2 통과면 L3 기록 없이도, L3 불통과(판정 불가)여도 배포 허용
    c.functions.createJob(b32("U0"), P, b32("init"), 1).transact({"from": agg})
    c.functions.completeJob(2, b32("J2")).transact({"from": agg})
    c.functions.recordVerification(b32("J2"), 1, True, b32("")).transact({"from": aud})
    assert not c.functions.deployable(b32("J2")).call(), "L2 없이 배포 불가"
    c.functions.recordVerification(b32("J2"), 2, True, b32("")).transact({"from": aud})
    assert c.functions.deployable(b32("J2")).call(), "정확 방법은 L1·L2로 완료"
    c.functions.recordVerification(b32("J2"), 3, False, b32("inconclusive: precision")).transact({"from": aud})
    assert c.functions.deployable(b32("J2")).call(), "정확 방법의 L3는 참고값"
    c.functions.recordVerification(b32("J2"), 1, False, b32("re-execution mismatch")).transact({"from": aud})
    assert not c.functions.deployable(b32("J2")).call(), "L1 불일치면 정확 방법도 배포 불가"

    # 이벤트 수 집계
    names = ["PolicyRegistered", "WithdrawalRequested", "WithdrawalEffective", "RoundInputCommitted", "UpdateCommitted",
             "UpdateAccepted", "RoundAggregated", "FirstUseDeclared", "JobCreated", "JobCompleted", "AuditSampled",
             "VerificationRecorded", "DeploymentChanged", "MethodApproved", "TrainingConfigCommitted"]
    counts = {n: len(getattr(c.events, n)().get_logs(from_block=0)) for n in names}
    assert counts["RoundInputCommitted"] == 9 and counts["UpdateAccepted"] == 9 and counts["DeploymentChanged"] == 4
    assert counts["WithdrawalRequested"] == 2 and counts["JobCreated"] == 2 and counts["MethodApproved"] == 2
    # 철회 요청 뒤에는 설정 커밋 불가(새 배포)
    tx = w3.eth.contract(abi=abi, bytecode=bytecode).constructor(agg, aud, [dc0, dc1, dc2], 0).transact({"from": platform})
    c2 = w3.eth.contract(address=w3.eth.get_transaction_receipt(tx).contractAddress, abi=abi)
    c2.functions.registerPolicy(s, 1, b32("p1"), 3).transact({"from": platform})
    c2.functions.requestWithdrawal(s, 1, b32("listroot")).transact({"from": subj})
    reverts(c2.functions.commitTrainingConfig(b32("cfg")), platform, "after withdrawal")
    print("contract tests: OK", counts)


if __name__ == "__main__":
    main()
