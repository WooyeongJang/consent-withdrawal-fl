"""ConsentLedger v3.2 단위 테스트(로컬 py-evm, Prague). 실행: <venv>/python tests/test_contract_v3_2.py

v3.1의 구멍(3장 반영 50–52·54·55번, reports/2026-10-29_IEEE_기술타당성_검토.md Part 2)이 닫혔는지 확인한다.
각 절의 공격은 v3.1에서는 성공하던 경로다. 필요: web3, eth-tester[py-evm], eth-account, tools/bin/solc
"""
import json
import subprocess
from pathlib import Path

from eth_abi import encode
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

ROOT = Path(__file__).resolve().parent.parent
TAG_POLICY = Web3.keccak(text="ConsentLedger.v3.2.policy")
TAG_WITHDRAW = Web3.keccak(text="ConsentLedger.v3.2.withdraw")
FAR = 2 ** 40
GAS = {"gas": 1_000_000}                               # 자동 채굴을 끈 동안 가스 추정(대기 상태 기준)을 건너뛴다


def compile_ledger():
    out = subprocess.run([str(ROOT / "tools/bin/solc"), "--evm-version", "prague", "--optimize",
                          "--combined-json", "abi,bin", str(ROOT / "contracts/ConsentLedger_v3_2.sol")],
                         check=True, capture_output=True, text=True).stdout
    c = next(v for k, v in json.loads(out)["contracts"].items() if k.endswith(":ConsentLedgerV32"))
    return c["abi"] if isinstance(c["abi"], list) else json.loads(c["abi"]), c["bin"]


def b32(s):
    return Web3.keccak(text=s)


def sign(acct, chain_id, contract, tag, subject, version, value, level):
    """정보주체 단말의 서명: 지갑 표준(EIP-191) personal_sign으로 만든다. 컨트랙트의 digest와 독립적으로 계산한다."""
    h = Web3.keccak(encode(["uint256", "address", "bytes32", "bytes32", "uint32", "bytes32", "uint8"],
                           [chain_id, contract, tag, subject, version, value, level]))
    return acct.sign_message(encode_defunct(primitive=h)).signature


def reverts(fn, sender, msg):
    try:
        fn.transact({"from": sender})
    except Exception as e:
        assert msg in str(e), f"expected revert '{msg}', got {e}"
        return
    raise AssertionError(f"expected revert '{msg}'")


class Env:
    def __init__(self, delta=0, config=True):
        self.prov = EthereumTesterProvider()
        self.w3 = w3 = Web3(self.prov)
        self.platform, self.agg, self.aud, self.dc0, self.dc1, self.dc2, self.anyone, self.other = w3.eth.accounts[:8]
        abi, bytecode = compile_ledger()
        tx = w3.eth.contract(abi=abi, bytecode=bytecode).constructor(
            self.agg, self.aud, [self.dc0, self.dc1, self.dc2], delta).transact({"from": self.platform})
        self.c = w3.eth.contract(address=w3.eth.get_transaction_receipt(tx).contractAddress, abi=abi)
        self.chain = w3.eth.chain_id
        self.dcs = (self.dc0, self.dc1, self.dc2)
        self.round = 0
        if config:
            self.c.functions.commitTrainingConfig(b32("cfg")).transact({"from": self.platform})

    def sig_policy(self, acct, s, v, pc, lv):
        return sign(acct, self.chain, self.c.address, TAG_POLICY, s, v, pc, lv)

    def sig_withdraw(self, acct, s, v, root):
        return sign(acct, self.chain, self.c.address, TAG_WITHDRAW, s, v, root, 0)

    def register(self, acct, s, v=1, level=3, home=1, signer=None):
        pc = b32(f"policy-{v}")
        self.c.functions.registerPolicy(s, v, pc, level, acct.address, home,
                                        self.sig_policy(signer or acct, s, v, pc, level)).transact({"from": self.platform})

    def withdraw(self, acct, s, v=1, sender=None, signer=None):
        root = b32(f"list-{s.hex()}-{v}")
        return self.c.functions.requestWithdrawal(s, v, root, self.sig_withdraw(signer or acct, s, v, root)).transact(
            {"from": sender or self.anyone})

    def version(self):
        return self.c.functions.versionAt(self.w3.eth.get_block("latest").timestamp).call()

    def run_round(self, model=None):
        """정규 라운드 하나: 세 기관 입력·업데이트 커밋 → 현재 버전으로 수락 → 집계."""
        c, r = self.c, self.round + 1
        c.functions.openRound(r, FAR, b32(f"start{r}")).transact({"from": self.platform})
        for k, dc in enumerate(self.dcs):
            c.functions.commitInput(k, r, b32(f"root{k}{r}"), 10, 1, b32("purpose")).transact({"from": dc})
            c.functions.commitUpdate(k, r, b32(f"d{k}{r}"), b32(f"start{r}")).transact({"from": dc})
        for k in range(3):
            c.functions.decide(k, r, True, b32(""), self.version()).transact({"from": self.agg})
        c.functions.aggregate(r, model or b32(f"M{r}"), b32("st")).transact({"from": self.agg})
        self.round = r

    def deployable(self, name):
        return self.c.functions.deployable(b32(name)).call()


def test_authenticity():
    """(1) 철회 진정성: v3.1에서는 누구나 남의 철회를 몇 번이든 낼 수 있었고, 플랫폼 혼자 재동의를 등록할 수 있었다."""
    e = Env()
    c = e.c
    alice, mallory = Account.create(), Account.create()
    s = b32("subject-alice")
    pc = b32("policy-1")
    # 첫 등록: 정보주체가 커밋하는 키로 서명해야 한다(소유 증명). 다른 키의 서명이나 없는 키는 거부
    reverts(c.functions.registerPolicy(s, 1, pc, 3, alice.address, 1, e.sig_policy(mallory, s, 1, pc, 3)), e.platform, "sig")
    reverts(c.functions.registerPolicy(s, 1, pc, 3, alice.address, 7, e.sig_policy(alice, s, 1, pc, 3)), e.platform, "key")
    reverts(c.functions.registerPolicy(s, 1, pc, 3, alice.address, 1, b"\x01" * 64), e.platform, "sig")
    e.register(alice, s)
    assert c.functions.policies(s).call()[2] == alice.address
    # 다른 키로 서명한 철회(제3자 위조) → 거부
    root = b32("list")
    reverts(c.functions.requestWithdrawal(s, 1, root, e.sig_withdraw(mallory, s, 1, root)), e.anyone, "sig")
    # 서명을 다른 목록 루트에 옮겨 붙이기 → 거부
    reverts(c.functions.requestWithdrawal(s, 1, b32("other-list"), e.sig_withdraw(alice, s, 1, root)), e.anyone, "sig")
    # 정상: 정보주체 서명, 제출 계정은 아무나(새 계정·수수료 대납 유지)
    c.functions.requestWithdrawal(s, 1, root, e.sig_withdraw(alice, s, 1, root)).transact({"from": e.other})
    assert c.functions.effectiveCount().call() == 1 and c.functions.pendingLevel2().call() == 1
    # 재전송: 같은 서명·같은 버전을 다시 제출 → 거부(v3.1에서는 effectiveCount·pendingLevel2가 계속 올랐다)
    reverts(c.functions.requestWithdrawal(s, 1, root, e.sig_withdraw(alice, s, 1, root)), e.anyone, "withdrawn")
    assert c.functions.effectiveCount().call() == 1 and c.functions.pendingLevel2().call() == 1
    # 재동의 위조: 플랫폼이 정보주체 서명 없이(자기 키·다른 키로) 새 버전 등록 → 거부
    pk2 = b32("policy-2")
    plat_acct = Account.create()
    reverts(c.functions.registerPolicy(s, 2, pk2, 3, alice.address, 1, e.sig_policy(plat_acct, s, 2, pk2, 3)), e.platform, "sig")
    # 키 교체 시도 → 거부
    reverts(c.functions.registerPolicy(s, 2, pk2, 3, mallory.address, 1, e.sig_policy(mallory, s, 2, pk2, 3)), e.platform, "key")
    # 정보주체가 서명한 재동의는 받고, 새 버전에서는 다시 철회할 수 있다
    e.register(alice, s, v=2)
    e.withdraw(alice, s, v=2)
    assert c.functions.effectiveCount().call() == 2
    # 다른 컨트랙트(재배포)의 서명은 쓸 수 없다: 서명 대상에 컨트랙트 주소와 체인 ID가 들어간다
    abi, bytecode = compile_ledger()
    tx = e.w3.eth.contract(abi=abi, bytecode=bytecode).constructor(e.agg, e.aud, list(e.dcs), 0).transact({"from": e.platform})
    c2 = e.w3.eth.contract(address=e.w3.eth.get_transaction_receipt(tx).contractAddress, abi=abi)
    sig1 = sign(alice, e.chain, c2.address, TAG_POLICY, s, 1, pc, 3)
    c2.functions.registerPolicy(s, 1, pc, 3, alice.address, 1, sig1).transact({"from": e.platform})
    reverts(c2.functions.requestWithdrawal(s, 1, root, e.sig_withdraw(alice, s, 1, root)), e.anyone, "sig")
    # 컨트랙트의 digest가 지갑 표준 서명 대상과 같은지 직접 대조
    h = Web3.keccak(encode(["uint256", "address", "bytes32", "bytes32", "uint32", "bytes32", "uint8"],
                           [e.chain, c.address, TAG_WITHDRAW, s, 1, root, 0]))
    assert c.functions.digest(TAG_WITHDRAW, s, 1, root, 0).call() == Web3.keccak(b"\x19Ethereum Signed Message:\n32" + h)
    print("  authenticity: OK")


def test_acceptance_race():
    """(2) 수락 경합: 조회(v_c)와 수락 거래 사이에 효력이 생긴 철회. v3.1은 기록만 하고 거절하지 않았다."""
    e = Env()
    c, w3 = e.c, e.w3
    alice = Account.create()
    s = b32("subject-race")
    e.register(alice, s)
    e.run_round()                                       # 라운드 1 정상
    r = 2
    c.functions.openRound(r, FAR, b32("start2")).transact({"from": e.platform})
    for k, dc in enumerate(e.dcs):
        c.functions.commitInput(k, r, b32(f"root{k}"), 10, 1, b32("p")).transact({"from": dc})
        c.functions.commitUpdate(k, r, b32(f"d{k}"), b32("start2")).transact({"from": dc})
    vc = e.version()
    assert vc == 0
    # 한 블록 안에서 철회 거래가 수락 거래보다 먼저 들어가는 경우(블록 안 순서)
    tester = e.prov.ethereum_tester
    tester.disable_auto_mine_transactions()
    root = b32("race-list")
    tw = c.functions.requestWithdrawal(s, 1, root, e.sig_withdraw(alice, s, 1, root)).transact({"from": e.anyone, **GAS})
    td = c.functions.decide(0, r, True, b32(""), vc).transact({"from": e.agg, **GAS})
    tester.mine_blocks(1)
    tester.enable_auto_mine_transactions()
    rw, rd = w3.eth.get_transaction_receipt(tw), w3.eth.get_transaction_receipt(td)
    assert rw.blockNumber == rd.blockNumber and rw.transactionIndex < rd.transactionIndex, "같은 블록, 철회가 먼저"
    assert rw.status == 1 and rd.status == 0, "낡은 v_c로 한 수락은 되돌려져야 한다"
    assert c.functions.decision(c.functions.key(0, r).call()).call() == 0
    # 집계 주체는 새 버전으로 비소속을 다시 확인한 뒤 결정한다(여기서는 그 기관에 철회자가 있다고 보고 거절)
    reverts(c.functions.decide(0, r, True, b32(""), vc), e.agg, "stale version")
    c.functions.decide(0, r, False, b32("withdrawn record in S"), 0).transact({"from": e.agg})
    c.functions.decide(1, r, True, b32(""), 1).transact({"from": e.agg})
    # 수락이 먼저 들어간 경우는 그대로 유효하다(원장 순서가 기준)
    c.functions.decide(2, r, True, b32(""), 1).transact({"from": e.agg})
    assert c.functions.rounds(r).call()[3] == 2
    # 거절 결정에는 버전 검사를 하지 않는다
    print("  acceptance race: OK")


def test_effective_time_version():
    """(2) δ > 0: 효력 버전은 효력 시각으로 센다. effectuate를 아무도 부르지 않아도 v_c 검사가 작동한다."""
    e = Env(delta=100)
    c, tester = e.c, e.prov.ethereum_tester
    alice = Account.create()
    s = b32("subject-delta")
    e.register(alice, s)
    e.run_round()
    c.functions.openRound(2, FAR, b32("start2")).transact({"from": e.platform})
    for k, dc in enumerate(e.dcs):
        c.functions.commitInput(k, 2, b32(f"root{k}"), 10, 1, b32("p")).transact({"from": dc})
        c.functions.commitUpdate(k, 2, b32(f"d{k}"), b32("start2")).transact({"from": dc})
    e.withdraw(alice, s)
    c.functions.decide(0, 2, True, b32(""), 0).transact({"from": e.agg})          # 유예 중: 아직 버전 0
    tester.time_travel(e.w3.eth.get_block("latest").timestamp + 101)
    assert c.functions.effectiveCount().call() == 0, "effectuate는 부르지 않았다"
    reverts(c.functions.decide(1, 2, True, b32(""), 0), e.agg, "stale version")
    c.functions.decide(1, 2, True, b32(""), 1).transact({"from": e.agg})
    print("  effective-time version: OK")


def test_first_use_and_resolution():
    """(3) r₀ 기본값·신고 권한·정정 방향, (4) 해소 뒤 범위 유지. v3.1에서는 미신고·늦은 신고·해소 뒤 배포가 가능했다."""
    e = Env()
    c = e.c
    alice, bob = Account.create(), Account.create()
    sa, sb = b32("subject-a"), b32("subject-b")
    e.register(alice, sa, home=1)
    e.register(bob, sb, home=2)
    for _ in range(4):
        e.run_round()                                   # M1–M4
    e.withdraw(alice, sa)
    # 미신고: r₀ = 1로 간주 → 정규 라운드 모델 전부 배포 불가(v3.1에서는 모두 배포 가능)
    assert c.functions.scopeStart().call() == 1
    assert not any(e.deployable(f"M{r}") for r in range(1, 5))
    # 소속 기관이 아닌 데이터센터의 신고 → 거부(v3.1은 어느 데이터센터든 먼저 신고하면 확정)
    reverts(c.functions.declareFirstUse(0, 3), e.dc0, "dc")
    reverts(c.functions.declareFirstUse(0, 3), e.other, "dc")
    c.functions.declareFirstUse(0, 3).transact({"from": e.dc1})
    assert e.deployable("M1") and e.deployable("M2") and not e.deployable("M3")
    # 첫 신고 뒤 r₀를 올리는 정정(범위 축소) → 거부, 낮추는 정정은 허용
    reverts(c.functions.declareFirstUse(0, 4), e.dc1, "raise")
    c.functions.declareFirstUse(0, 2).transact({"from": e.dc1})
    assert e.deployable("M1") and not e.deployable("M2")
    # 처리 작업으로 해소
    P = b32("prog")
    c.functions.approveMethod(b32("U0"), P, 1).transact({"from": e.platform})
    c.functions.createJob(b32("U0"), P, b32("M1"), 1).transact({"from": e.agg})
    c.functions.completeJob(1, b32("J1")).transact({"from": e.agg})
    for lv in (1, 2):
        c.functions.recordVerification(b32("J1"), lv, True, b32("")).transact({"from": e.aud})
    c.functions.deploy(b32("J1"), b32("verified exact job")).transact({"from": e.platform})
    assert c.functions.pendingLevel2().call() == 0 and c.functions.tainted(0).call() == [2, 4]
    # (4) 해소 뒤에도 오염 구간(M2–M4)은 배포 불가(v3.1에서는 해소 즉시 다시 배포 가능)
    assert e.deployable("M1") and not e.deployable("M2") and not e.deployable("M4")
    reverts(c.functions.deploy(b32("M3"), b32("x")), e.platform, "not deployable")
    # 해소 뒤 새 라운드는 정상
    e.run_round()                                       # M5
    assert e.deployable("M5")
    # 다른 정보주체 B가 큰 r₀로 철회해도 A의 옛 범위 모델은 계속 배포 불가(v3.1 경로 ③)
    e.withdraw(bob, sb)
    c.functions.declareFirstUse(1, 5).transact({"from": e.dc2})
    assert not e.deployable("M2") and not e.deployable("M5") and e.deployable("M1")
    # 해소된 A의 요청을 뒤늦게 정정해도 현재 범위에는 영향이 없다(소급 반영은 범위 밖)
    c.functions.declareFirstUse(0, 1).transact({"from": e.dc1})
    assert c.functions.scopeStart().call() == 5
    print("  first use and resolution: OK")


def test_lineage():
    """(5) 라운드 번호 재사용·모델 계보 덮어쓰기. v3.1에서는 openRound(0) 뒤 범위 안 모델을 round 0으로 재등록할 수 있었다."""
    e = Env()
    c = e.c
    alice = Account.create()
    s = b32("subject-lineage")
    e.register(alice, s)
    reverts(c.functions.openRound(0, FAR, b32("x")), e.platform, "round order")
    reverts(c.functions.openRound(2, FAR, b32("x")), e.platform, "round order")
    for _ in range(3):
        e.run_round()
    reverts(c.functions.openRound(2, FAR, b32("x")), e.platform, "round order")       # 재사용
    reverts(c.functions.openRound(1, FAR, b32("x")), e.platform, "round order")       # 역행
    reverts(c.functions.openRound(5, FAR, b32("x")), e.platform, "round order")       # 건너뛰기
    # 범위 안 모델 M3을 새 라운드의 출력으로 재등록 → 거부
    e.withdraw(alice, s)
    c.functions.declareFirstUse(0, 3).transact({"from": e.dc1})
    assert not e.deployable("M3")
    c.functions.openRound(4, FAR, b32("start4")).transact({"from": e.platform})
    for k, dc in enumerate(e.dcs):
        c.functions.commitInput(k, 4, b32(f"root{k}4"), 10, 1, b32("p")).transact({"from": dc})
        c.functions.commitUpdate(k, 4, b32(f"d{k}4"), b32("start4")).transact({"from": dc})
        c.functions.decide(k, 4, True, b32(""), e.version()).transact({"from": e.agg})
    reverts(c.functions.aggregate(4, b32("M3"), b32("st")), e.agg, "dup model")
    reverts(c.functions.aggregate(4, b"\x00" * 32, b32("st")), e.agg, "dup model")
    assert c.functions.models(b32("M3")).call()[0] == 3, "M3의 계보(round 3)는 그대로"
    c.functions.aggregate(4, b32("M4"), b32("st")).transact({"from": e.agg})
    # 처리 작업 출력으로 기존 모델 해시 등록 → 거부
    P = b32("prog")
    c.functions.approveMethod(b32("U0"), P, 1).transact({"from": e.platform})
    c.functions.createJob(b32("U0"), P, b32("M2"), 1).transact({"from": e.agg})
    reverts(c.functions.completeJob(1, b32("M1")), e.agg, "dup model")
    print("  lineage: OK")


def test_regression():
    """v3.1의 정상 흐름(설정 커밋, 라운드, 역할, 승인 방법, 근사·정확 작업의 배포 조건)이 그대로 성립하는지."""
    e = Env()
    c = e.c
    alice, bob = Account.create(), Account.create()
    s, s2 = b32("subject-1"), b32("subject-2")
    e.register(alice, s)
    reverts(c.functions.registerPolicy(s, 3, b32("p"), 3, alice.address, 1, e.sig_policy(alice, s, 3, b32("p"), 3)), e.platform, "version")
    reverts(c.functions.registerPolicy(s, 2, b32("p"), 3, alice.address, 1, e.sig_policy(alice, s, 2, b32("p"), 3)), e.other, "role")
    reverts(c.functions.commitTrainingConfig(b32("cfg2")), e.platform, "config")
    for _ in range(3):
        e.run_round()
    reverts(c.functions.decide(0, 3, False, b32("x"), 0), e.agg, "decided")
    c.functions.sampleAudit(3).transact({"from": e.aud})
    c.functions.deploy(b32("M3"), b32("aggregated")).transact({"from": e.platform})
    e.withdraw(alice, s)
    c.functions.declareFirstUse(0, 2).transact({"from": e.dc1})
    assert e.deployable("M1") and not e.deployable("M2")
    P = b32("prog-v1")
    c.functions.approveMethod(b32("U2"), P, 2).transact({"from": e.platform})
    c.functions.approveMethod(b32("U0"), P, 1).transact({"from": e.platform})
    reverts(c.functions.createJob(b32("U0"), b32("prog-finetune"), b32("init"), 1), e.agg, "method")
    reverts(c.functions.createJob(b32("U2"), P, b32("init"), 2), e.agg, "covers")
    c.functions.createJob(b32("U2"), P, b32("init"), 1).transact({"from": e.agg})
    c.functions.completeJob(1, b32("J1")).transact({"from": e.agg})
    for lv in (1, 2):
        c.functions.recordVerification(b32("J1"), lv, True, b32("")).transact({"from": e.aud})
    assert not e.deployable("J1"), "근사 방법은 L3 없이 배포 불가"
    c.functions.recordVerification(b32("J1"), 3, True, b32("")).transact({"from": e.aud})
    c.functions.deploy(b32("J1"), b32("verified job")).transact({"from": e.platform})
    assert e.deployable("J1")
    e.register(bob, s2, home=2)
    e.withdraw(bob, s2)
    assert not e.deployable("J1"), "새 미해소 철회가 생기면 배포 불가"
    c.functions.createJob(b32("U0"), P, b32("init"), 1).transact({"from": e.agg})
    c.functions.completeJob(2, b32("J2")).transact({"from": e.agg})
    for lv in (1, 2):
        c.functions.recordVerification(b32("J2"), lv, True, b32("")).transact({"from": e.aud})
    assert e.deployable("J2"), "정확 방법은 L1·L2로 완료"
    # 설정 커밋 없이는 라운드 개시 불가, 철회 요청 뒤에는 설정 커밋 불가(새 배포)
    e2 = Env(config=False)
    reverts(e2.c.functions.openRound(1, FAR, b32("s")), e2.platform, "no config")
    e2.register(alice, s)
    e2.withdraw(alice, s)
    reverts(e2.c.functions.commitTrainingConfig(b32("cfg")), e2.platform, "after withdrawal")
    print("  regression: OK")


def main():
    for t in (test_authenticity, test_acceptance_race, test_effective_time_version, test_first_use_and_resolution,
              test_lineage, test_regression):
        t()
    print("contract v3.2 tests: OK")


if __name__ == "__main__":
    main()
