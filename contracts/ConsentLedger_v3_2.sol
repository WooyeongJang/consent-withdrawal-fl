// SPDX-License-Identifier: MIT
// ConsentLedger v3.2(2026-10-30, 설계 이탈 기록 10-30, 3장 반영 59번). v3.1(`ConsentLedger.sol`, 모든 실험에 사용)의 구멍을 닫는다.
// 실험(E1–E4)은 v3.1로 수행했고, 이 판은 로컬 단위 테스트(tests/test_contract_v3_2.py)로만 확인한다.
//   (1) 철회 진정성: 첫 정책 등록 때 정보주체 키를 커밋하고, 철회와 재동의(새 정책 버전)에 그 키의 서명을 요구한다.
//       정책 버전당 철회는 한 번만 받는다(재전송 방지). 제출 계정과 수수료 대납은 그대로 자유롭다.
//   (2) 수락 경합: decide(수락)에 집계 주체가 조회한 철회 버전 v_c를 넘기고, 실행 시점의 효력 버전과 다르면 되돌린다.
//       효력 버전은 효력 시각(tReq + δ) ≤ 현재 시각인 요청 수로 세므로 effectuate 호출에 기대지 않는다.
//   (3) r₀: 신고 전에는 r₀ = 1로 간주한다. 신고는 정책 등록 때 정한 소속 기관만, 첫 신고 뒤에는 낮추는 정정만 받는다.
//   (4) 해소 뒤 범위: 처리 작업 배포로 해소할 때 [영향 시작 라운드, 마지막 라운드]를 오염 구간으로 영구 기록한다.
//   (5) 계보: 라운드 번호는 직전 + 1만 열 수 있고, 이미 등록된 모델 해시는 다시 등록할 수 없다.
// 범위 밖: 정규 라운드 집계의 정확성(감사자의 재계산과 h_start 사슬 확인), 해소 뒤 r₀ 정정의 소급 반영.
pragma solidity ^0.8.30;

contract ConsentLedgerV32 {
    // ---------------- 역할 ----------------
    address public platform;
    address public aggregator;
    address public auditor;
    mapping(uint32 => address) public dataCenter;
    uint32 public nDataCenters;
    uint64 public delta;

    modifier only(address who) { require(msg.sender == who, "role"); _; }
    modifier onlyDC(uint32 k) { require(msg.sender == dataCenter[k] && msg.sender != address(0), "dc"); _; }

    // ---------------- 이벤트 ----------------
    event SubjectKeyCommitted(bytes32 indexed subject, address key, uint32 home);
    event PolicyRegistered(bytes32 indexed subject, uint32 version, bytes32 policyCommit, uint8 level);
    event WithdrawalRequested(uint256 indexed reqId, bytes32 indexed subject, uint32 version, uint8 level, bytes32 listRoot);
    event WithdrawalEffective(uint256 indexed reqId, uint64 tEff);
    event ReceiptRootPosted(uint32 indexed k, uint32 date, bytes32 root);
    event RoundOpened(uint32 indexed r, uint64 deadline, bytes32 startModel);
    event RoundInputCommitted(uint32 indexed k, uint32 indexed r, bytes32 rootS, uint32 n, uint32 sigma, bytes32 purpose);
    event UpdateCommitted(uint32 indexed k, uint32 indexed r, bytes32 hDelta, bytes32 hStart, bytes32 hC);
    event UpdateAccepted(uint32 indexed k, uint32 indexed r, uint64 tau, uint256 withdrawalVersion);
    event UpdateRejected(uint32 indexed k, uint32 indexed r, uint64 tau, bytes32 reason);
    event RoundAggregated(uint32 indexed r, uint32 accepted, bytes32 hOut, bytes32 stateHash);
    event FirstUseDeclared(uint256 indexed reqId, uint32 r0);
    event ScopeResolved(uint32 fromRound, uint32 toRound);
    event TrainingConfigCommitted(bytes32 config, uint64 committedAt);
    event MethodApproved(bytes32 indexed method, bytes32 indexed prog, uint8 kind);
    event JobCreated(uint256 indexed jid, bytes32 method, bytes32 prog, bytes32 hStart, uint256 covers, uint8 kind);
    event JobCompleted(uint256 indexed jid, bytes32 hOut);
    event AuditSampled(uint32 indexed r, bytes32 seed, uint64 sampledAt);
    event VerificationRecorded(bytes32 indexed target, uint8 level, bool passed, bytes32 reason);
    event DeploymentChanged(bytes32 indexed model, uint8 state, bytes32 basis);
    event RetentionStarted(uint256 indexed reqId, uint32 k);
    event RetentionDestroyed(uint256 indexed reqId, uint32 k);

    // ---------------- 상태 ----------------
    struct Policy { uint32 version; uint8 level; address key; uint32 home; uint32 withdrawn; }   // withdrawn: 마지막으로 철회한 정책 버전
    mapping(bytes32 => Policy) public policies;

    struct Withdrawal { bytes32 subject; uint8 level; uint64 tReq; uint64 tEff; uint32 r0; bool effective; bool declared; uint32 epoch; }
    Withdrawal[] public withdrawals;                   // tEff는 요청 순서대로 단조 증가한다(δ가 상수)
    uint256 public effectiveCount;

    struct Round { uint64 deadline; uint32 inputs; uint32 decided; uint32 accepted; bool aggregated; }
    mapping(uint32 => Round) public rounds;
    uint32 public lastRound;
    mapping(bytes32 => bytes32) public inputCommit;
    mapping(bytes32 => bytes32) public updateCommit;
    mapping(bytes32 => uint8) public decision;

    struct Model { uint32 round; bool registered; uint256 jobId; }
    mapping(bytes32 => Model) public models;
    uint8 constant EXACT = 1;
    uint8 constant APPROX = 2;
    mapping(bytes32 => mapping(bytes32 => uint8)) public methodKind;
    bytes32 public trainingConfig;
    struct Job { bytes32 method; bytes32 prog; bytes32 hStart; bytes32 hOut; uint256 covers; bool completed; bool resolved; uint8 kind; }
    Job[] public jobs;
    mapping(bytes32 => mapping(uint8 => uint8)) public verification;

    // 영향 범위: 미해소 처리 수준 2 이상 철회. epoch는 해소할 때마다 오른다(w.epoch == epoch이면 미해소).
    uint256 public pendingLevel2;
    uint256 public undeclaredPending;                  // 미해소이면서 r₀ 미신고(r₀ = 1로 간주)
    uint32 public minDeclaredR0 = type(uint32).max;    // 미해소이면서 신고된 r₀의 최솟값
    uint32 public epoch;
    struct Range { uint32 lo; uint32 hi; }
    Range[] public tainted;                            // 해소된 영향 범위(정규 라운드 모델 영구 배포 불가)

    bytes32 public deployed;
    mapping(bytes32 => bool) public revoked;

    bytes32 constant TAG_POLICY = keccak256("ConsentLedger.v3.2.policy");
    bytes32 constant TAG_WITHDRAW = keccak256("ConsentLedger.v3.2.withdraw");
    uint256 constant HALF_N = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0;

    constructor(address _aggregator, address _auditor, address[] memory dcs, uint64 _delta) {
        platform = msg.sender;
        aggregator = _aggregator;
        auditor = _auditor;
        for (uint32 i = 0; i < dcs.length; i++) dataCenter[i] = dcs[i];
        nDataCenters = uint32(dcs.length);
        delta = _delta;
    }

    function key(uint32 k, uint32 r) public pure returns (bytes32) { return keccak256(abi.encode(k, r)); }

    // ---------------- 서명(정보주체 키) ----------------
    // 서명 대상: H(체인 ID, 컨트랙트 주소, 용도, subject, 정책 버전, 값, 수준)에 EIP-191 접두사를 붙인 해시.
    function digest(bytes32 tag, bytes32 subject, uint32 version, bytes32 value, uint8 level) public view returns (bytes32) {
        bytes32 h = keccak256(abi.encode(block.chainid, address(this), tag, subject, version, value, level));
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", h));
    }

    function _signer(bytes32 d, bytes calldata sig) internal pure returns (address a) {
        require(sig.length == 65, "sig");
        bytes32 r; bytes32 s; uint8 v;
        assembly {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        require(uint256(s) <= HALF_N, "sig");          // 서명 가변성 차단
        if (v < 27) v += 27;
        a = ecrecover(d, v, r, s);
        require(a != address(0), "sig");
    }

    // ---------------- 동의·철회 ----------------
    // 첫 버전: 플랫폼이 정보주체 키와 소속 기관을 등록하고, 정보주체가 그 키로 서명해 소유를 증명한다.
    // 이후 버전(재동의 등): 같은 키의 서명이 있어야 한다. 키 교체는 v3.2에서 지원하지 않는다.
    function registerPolicy(bytes32 subject, uint32 version, bytes32 policyCommit, uint8 level,
                            address subjectKey, uint32 home, bytes calldata sig) external only(platform) {
        require(level >= 1 && level <= 3, "level");
        Policy storage p = policies[subject];
        require(version == p.version + 1, "version");
        if (version == 1) {
            require(subjectKey != address(0) && home < nDataCenters, "key");
            p.key = subjectKey;
            p.home = home;
            emit SubjectKeyCommitted(subject, subjectKey, home);
        } else {
            require(subjectKey == p.key && home == p.home, "key");
        }
        require(_signer(digest(TAG_POLICY, subject, version, policyCommit, level), sig) == p.key, "sig");
        p.version = version;
        p.level = level;
        emit PolicyRegistered(subject, version, policyCommit, level);
    }

    // 단말이 요청마다 새 계정으로 제출하고 플랫폼이 수수료를 대납해도 된다. 진정성은 정보주체 키의 서명으로 확인한다.
    function requestWithdrawal(bytes32 subject, uint32 version, bytes32 listRoot, bytes calldata sig) external returns (uint256 reqId) {
        Policy storage p = policies[subject];
        require(p.version != 0 && version == p.version, "policy");
        require(p.withdrawn < version, "withdrawn");
        require(_signer(digest(TAG_WITHDRAW, subject, version, listRoot, 0), sig) == p.key, "sig");
        p.withdrawn = version;
        uint64 tEff = uint64(block.timestamp) + delta;
        withdrawals.push(Withdrawal(subject, p.level, uint64(block.timestamp), tEff, 0, false, false, 0));
        reqId = withdrawals.length - 1;
        emit WithdrawalRequested(reqId, subject, version, p.level, listRoot);
        if (delta == 0) _effectuate(reqId);
    }

    function effectuate(uint256 reqId) external {
        require(block.timestamp >= withdrawals[reqId].tEff && !withdrawals[reqId].effective, "not yet");
        _effectuate(reqId);
    }

    function _effectuate(uint256 reqId) internal {
        Withdrawal storage w = withdrawals[reqId];
        w.effective = true;
        w.epoch = epoch;
        effectiveCount++;
        if (w.level >= 2) { pendingLevel2++; undeclaredPending++; }
        emit WithdrawalEffective(reqId, w.tEff);
    }

    // 시각 t의 철회 기록 버전: 효력 시각이 t 이하인 요청 수(tEff 단조 증가 → 이분 탐색).
    function versionAt(uint64 t) public view returns (uint256 lo) {
        uint256 hi = withdrawals.length;
        while (lo < hi) {
            uint256 mid = (lo + hi) / 2;
            if (withdrawals[mid].tEff <= t) lo = mid + 1; else hi = mid;
        }
    }

    // ---------------- 데이터센터 ----------------
    function postReceiptRoot(uint32 k, uint32 date, bytes32 root) external onlyDC(k) {
        emit ReceiptRootPosted(k, date, root);
    }

    function commitTrainingConfig(bytes32 config) external only(platform) {
        require(trainingConfig == bytes32(0) && config != bytes32(0), "config");
        require(withdrawals.length == 0, "after withdrawal");
        trainingConfig = config;
        emit TrainingConfigCommitted(config, uint64(block.timestamp));
    }

    function openRound(uint32 r, uint64 deadline, bytes32 startModel) external only(platform) {
        require(trainingConfig != bytes32(0), "no config");
        require(r == lastRound + 1, "round order");    // 라운드 번호 재사용·역행·건너뛰기 금지
        lastRound = r;
        rounds[r].deadline = deadline;
        emit RoundOpened(r, deadline, startModel);
    }

    function commitInput(uint32 k, uint32 r, bytes32 rootS, uint32 n, uint32 sigma, bytes32 purpose) external onlyDC(k) {
        Round storage rd = rounds[r];
        require(rd.deadline != 0 && block.timestamp <= rd.deadline, "round");
        bytes32 kr = key(k, r);
        require(inputCommit[kr] == bytes32(0), "dup");
        inputCommit[kr] = keccak256(abi.encode(k, r, rootS, n, sigma, purpose));
        rd.inputs++;
        emit RoundInputCommitted(k, r, rootS, n, sigma, purpose);
    }

    function commitUpdate(uint32 k, uint32 r, bytes32 hDelta, bytes32 hStart) external onlyDC(k) {
        bytes32 kr = key(k, r);
        require(inputCommit[kr] != bytes32(0), "no input");
        require(updateCommit[kr] == bytes32(0), "dup");
        updateCommit[kr] = hDelta;
        emit UpdateCommitted(k, r, hDelta, hStart, inputCommit[kr]);
    }

    // 소속 기관만 신고한다. 첫 신고는 기본값 1을 대신하고, 그 뒤에는 낮추는 정정만 받는다.
    function declareFirstUse(uint256 reqId, uint32 r0) external {
        Withdrawal storage w = withdrawals[reqId];
        require(msg.sender == dataCenter[policies[w.subject].home], "dc");
        require(w.effective && r0 > 0, "r0");
        bool open = w.level >= 2 && w.epoch == epoch;  // 미해소 영향 범위에 속하는가
        if (!w.declared) {
            w.declared = true;
            if (open) undeclaredPending--;
        } else {
            require(r0 < w.r0, "raise");
        }
        w.r0 = r0;
        if (open && r0 < minDeclaredR0) minDeclaredR0 = r0;
        emit FirstUseDeclared(reqId, r0);
    }

    // 미해소 영향 범위의 시작 라운드(미신고가 있으면 1).
    function scopeStart() public view returns (uint32) {
        return undeclaredPending > 0 ? 1 : minDeclaredR0;
    }

    // ---------------- 집계 주체 ----------------
    function decide(uint32 k, uint32 r, bool accept, bytes32 reason, uint256 vc) external only(aggregator) {
        bytes32 kr = key(k, r);
        require(updateCommit[kr] != bytes32(0), "no update");
        require(decision[kr] == 0, "decided");
        decision[kr] = accept ? 1 : 2;
        rounds[r].decided++;
        if (accept) {
            require(vc == versionAt(uint64(block.timestamp)), "stale version");   // 조회 이후 효력이 생긴 철회가 있으면 되돌림
            rounds[r].accepted++;
            emit UpdateAccepted(k, r, uint64(block.timestamp), vc);
        } else {
            emit UpdateRejected(k, r, uint64(block.timestamp), reason);
        }
    }

    function aggregate(uint32 r, bytes32 hOut, bytes32 stateHash) external only(aggregator) {
        Round storage rd = rounds[r];
        require(!rd.aggregated && rd.decided == rd.inputs && rd.inputs > 0, "decisions");
        require(hOut != bytes32(0) && !models[hOut].registered, "dup model");
        rd.aggregated = true;
        models[hOut] = Model(r, true, 0);
        emit RoundAggregated(r, rd.accepted, hOut, stateHash);
    }

    function approveMethod(bytes32 method, bytes32 prog, uint8 kind) external only(platform) {
        require(kind == EXACT || kind == APPROX, "kind");
        methodKind[method][prog] = kind;
        emit MethodApproved(method, prog, kind);
    }

    function createJob(bytes32 method, bytes32 prog, bytes32 hStart, uint256 covers) external only(aggregator) returns (uint256 jid) {
        require(pendingLevel2 > 0 && covers == pendingLevel2, "covers");
        uint8 kind = methodKind[method][prog];
        require(kind != 0, "method");
        jobs.push(Job(method, prog, hStart, bytes32(0), covers, false, false, kind));
        jid = jobs.length;
        emit JobCreated(jid, method, prog, hStart, covers, kind);
    }

    function completeJob(uint256 jid, bytes32 hOut) external only(aggregator) {
        Job storage j = jobs[jid - 1];
        require(!j.completed, "done");
        require(hOut != bytes32(0) && !models[hOut].registered, "dup model");
        j.completed = true; j.hOut = hOut;
        models[hOut] = Model(0, true, jid);
        emit JobCompleted(jid, hOut);
    }

    // ---------------- 감사자 ----------------
    function sampleAudit(uint32 r) external only(auditor) returns (bytes32 seed) {
        require(rounds[r].aggregated, "not aggregated");
        seed = keccak256(abi.encode(blockhash(block.number - 1), r, address(this)));
        emit AuditSampled(r, seed, uint64(block.timestamp));
    }

    function recordVerification(bytes32 target, uint8 level, bool passed, bytes32 reason) external only(auditor) {
        require(level >= 1 && level <= 4, "level");
        verification[target][level] = passed ? 1 : 2;
        emit VerificationRecorded(target, level, passed, reason);
    }

    // ---------------- 배포 ----------------
    function isTainted(uint32 r) public view returns (bool) {
        for (uint256 i = 0; i < tainted.length; i++) if (r >= tainted[i].lo && r <= tainted[i].hi) return true;
        return false;
    }

    function deployable(bytes32 model) public view returns (bool) {
        Model memory m = models[model];
        if (!m.registered || revoked[model]) return false;
        if (m.jobId > 0) {
            Job memory j = jobs[m.jobId - 1];
            bool verified = verification[model][1] == 1 && verification[model][2] == 1 &&
                            (j.kind == EXACT || verification[model][3] == 1);
            return verified && (j.resolved ? pendingLevel2 == 0 : j.covers == pendingLevel2);
        }
        // 정규 라운드 모델: 해소된 오염 구간 밖이고, 미해소 영향 범위(r ≥ 시작 라운드) 밖이어야 한다
        if (isTainted(m.round)) return false;
        return pendingLevel2 == 0 || m.round < scopeStart();
    }

    function deploy(bytes32 model, bytes32 basis) external only(platform) {
        require(deployable(model), "not deployable");
        Model memory m = models[model];
        if (m.jobId > 0 && !jobs[m.jobId - 1].resolved) {   // 영향 제거가 검증된 모델로 교체 → 철회 해소
            jobs[m.jobId - 1].resolved = true;
            uint32 lo = scopeStart();
            if (lo <= lastRound) {                          // 해소된 영향 범위는 지우지 않고 오염 구간으로 남긴다
                tainted.push(Range(lo, lastRound));
                emit ScopeResolved(lo, lastRound);
            }
            pendingLevel2 = 0;
            undeclaredPending = 0;
            minDeclaredR0 = type(uint32).max;
            epoch++;
        }
        deployed = model;
        emit DeploymentChanged(model, 1, basis);
    }

    function revoke(bytes32 model, bytes32 basis) external only(platform) {
        require(models[model].registered && !revoked[model], "model");
        revoked[model] = true;
        if (deployed == model) deployed = bytes32(0);
        emit DeploymentChanged(model, 2, basis);
    }

    // ---------------- 보존·파기 ----------------
    function retention(uint256 reqId, uint32 k, bool destroyed) external onlyDC(k) {
        if (destroyed) emit RetentionDestroyed(reqId, k); else emit RetentionStarted(reqId, k);
    }

    function withdrawalCount() external view returns (uint256) { return withdrawals.length; }
}
