// SPDX-License-Identifier: MIT
// 3장 v1.6 6.3절 이벤트 스키마와 상태 전이(불변식 I2·I3·I4의 원장 측 강제).
// 원장에는 레코드 단위 정보를 올리지 않고 루트·해시·커밋만 기록한다(6.1절).
// v3(2026-10-27, 3장 반영 44번 (a)): 승인된 처리 방법 목록과 방법 종류. 정확 방법(처음부터 재학습, 동등 조건을
// 충족한 체크포인트 재수행)의 결과는 L1·L2 통과로 배포 허용, 근사 방법은 L3 통과도 요구한다(알고리즘 7의 4단계).
// v3.1(2026-10-27, 3장 반영 34번 (a)): L1(ε=0)의 전제를 원장이 강제한다.
//   (ii) 학습 설정 커밋 H(기준 시드, 시드 규칙, prog, 표준화 범위 루트)를 첫 라운드와 첫 철회 요청보다 먼저 기록한다
//        (시드 고르기 방지, 라운드 시드는 이 커밋의 기준 시드에서 결정적으로 도출).
//   (iv) 처리 방법은 (방법, prog) 쌍으로 승인한다. 처리 작업은 승인된 쌍으로만 만들 수 있다.
pragma solidity ^0.8.30;

contract ConsentLedger {
    // ---------------- 역할 ----------------
    address public platform;
    address public aggregator;
    address public auditor;
    mapping(uint32 => address) public dataCenter;   // k -> 데이터센터 계정
    uint32 public nDataCenters;
    uint64 public delta;                            // 처리 유예 시간 δ(초)

    modifier only(address who) { require(msg.sender == who, "role"); _; }
    modifier onlyDC(uint32 k) { require(msg.sender == dataCenter[k] && msg.sender != address(0), "dc"); _; }

    // ---------------- 이벤트(6.3절) ----------------
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
    struct Policy { uint32 version; uint8 level; }
    mapping(bytes32 => Policy) public policies;

    struct Withdrawal { bytes32 subject; uint8 level; uint64 tReq; uint64 tEff; uint32 r0; bool effective; }
    Withdrawal[] public withdrawals;
    uint256 public effectiveCount;                  // 철회 기록 버전(수락 검사에 쓴 철회 기록의 길이)

    struct Round { uint64 deadline; uint32 inputs; uint32 decided; uint32 accepted; bool aggregated; }
    mapping(uint32 => Round) public rounds;
    mapping(bytes32 => bytes32) public inputCommit;   // key(k,r) -> H(C)
    mapping(bytes32 => bytes32) public updateCommit;  // key(k,r) -> hDelta
    mapping(bytes32 => uint8) public decision;        // key(k,r) -> 1 수락, 2 거절

    // 모델 계보와 배포(4.2–4.3절)
    struct Model { uint32 round; bool registered; uint256 jobId; }   // jobId > 0이면 처리 작업 결과
    mapping(bytes32 => Model) public models;
    uint8 constant EXACT = 1;                          // 정확 방법: L1·L2로 완료(L3는 참고값)
    uint8 constant APPROX = 2;                         // 근사 방법: L3 통과까지 요구
    mapping(bytes32 => mapping(bytes32 => uint8)) public methodKind;   // (방법, prog) → 종류(0이면 미승인)
    bytes32 public trainingConfig;                     // 학습 설정 커밋(34번 (a))
    struct Job { bytes32 method; bytes32 prog; bytes32 hStart; bytes32 hOut; uint256 covers; bool completed; bool resolved; uint8 kind; }
    Job[] public jobs;                                 // jid = index + 1
    mapping(bytes32 => mapping(uint8 => uint8)) public verification;  // target -> level -> 1 통과, 2 불통과
    uint256 public pendingLevel2;                      // 처리 수준 2 이상, 아직 처리 작업 배포로 해소되지 않은 철회 수
    uint32 public minR0 = type(uint32).max;            // 해소되지 않은 철회의 최소 영향 시작 라운드
    bytes32 public deployed;                           // 현재 배포 허용 모델(단말 대조용)
    mapping(bytes32 => bool) public revoked;

    constructor(address _aggregator, address _auditor, address[] memory dcs, uint64 _delta) {
        platform = msg.sender;
        aggregator = _aggregator;
        auditor = _auditor;
        for (uint32 i = 0; i < dcs.length; i++) dataCenter[i] = dcs[i];
        nDataCenters = uint32(dcs.length);
        delta = _delta;
    }

    function key(uint32 k, uint32 r) public pure returns (bytes32) { return keccak256(abi.encode(k, r)); }

    // ---------------- 동의·철회(3.1–3.2절) ----------------
    function registerPolicy(bytes32 subject, uint32 version, bytes32 policyCommit, uint8 level) external only(platform) {
        require(level >= 1 && level <= 3, "level");
        require(version == policies[subject].version + 1, "version");
        policies[subject] = Policy(version, level);
        emit PolicyRegistered(subject, version, policyCommit, level);
    }

    // 정보주체 단말이 요청마다 새 계정으로 서명하고 플랫폼이 수수료를 대납한다.
    function requestWithdrawal(bytes32 subject, uint32 version, bytes32 listRoot) external returns (uint256 reqId) {
        Policy memory p = policies[subject];
        require(p.version != 0 && version == p.version, "policy");
        uint64 tEff = uint64(block.timestamp) + delta;
        withdrawals.push(Withdrawal(subject, p.level, uint64(block.timestamp), tEff, 0, false));
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
        effectiveCount++;
        if (w.level >= 2) pendingLevel2++;
        emit WithdrawalEffective(reqId, w.tEff);
    }

    // ---------------- 데이터센터(알고리즘 1) ----------------
    function postReceiptRoot(uint32 k, uint32 date, bytes32 root) external onlyDC(k) {
        emit ReceiptRootPosted(k, date, root);
    }

    // 학습 설정 커밋: 한 번만, 첫 철회 요청 이전에. 라운드 개시는 이 커밋 뒤에만 가능하다.
    function commitTrainingConfig(bytes32 config) external only(platform) {
        require(trainingConfig == bytes32(0) && config != bytes32(0), "config");
        require(withdrawals.length == 0, "after withdrawal");
        trainingConfig = config;
        emit TrainingConfigCommitted(config, uint64(block.timestamp));
    }

    function openRound(uint32 r, uint64 deadline, bytes32 startModel) external only(platform) {
        require(trainingConfig != bytes32(0), "no config");
        require(rounds[r].deadline == 0, "opened");
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

    function declareFirstUse(uint256 reqId, uint32 r0) external {
        // 데이터센터가 신고하고 감사자가 이전 커밋에 대한 비소속 증명으로 확인한다(4.2절, 오프체인).
        bool isDC = false;
        for (uint32 i = 0; i < nDataCenters; i++) if (msg.sender == dataCenter[i]) isDC = true;
        require(isDC, "dc");
        Withdrawal storage w = withdrawals[reqId];
        require(w.effective && w.r0 == 0 && r0 > 0, "r0");
        w.r0 = r0;
        if (w.level >= 2 && r0 < minR0) minR0 = r0;
        emit FirstUseDeclared(reqId, r0);
    }

    // ---------------- 집계 주체(알고리즘 2) ----------------
    function decide(uint32 k, uint32 r, bool accept, bytes32 reason) external only(aggregator) {
        bytes32 kr = key(k, r);
        require(updateCommit[kr] != bytes32(0), "no update");
        require(decision[kr] == 0, "decided");
        decision[kr] = accept ? 1 : 2;
        rounds[r].decided++;
        if (accept) {
            rounds[r].accepted++;
            emit UpdateAccepted(k, r, uint64(block.timestamp), effectiveCount);
        } else {
            emit UpdateRejected(k, r, uint64(block.timestamp), reason);
        }
    }

    function aggregate(uint32 r, bytes32 hOut, bytes32 stateHash) external only(aggregator) {
        Round storage rd = rounds[r];
        require(!rd.aggregated && rd.decided == rd.inputs && rd.inputs > 0, "decisions");
        rd.aggregated = true;
        models[hOut] = Model(r, true, 0);
        emit RoundAggregated(r, rd.accepted, hOut, stateHash);
    }

    // 처리 방법 승인(플랫폼). 체크포인트 재수행은 동등 조건 충족 여부에 따라 서로 다른 방법 식별자로 등록하고,
    // 정확 방법으로 등록된 작업의 동등 조건(표준화 범위·출발 상태)은 감사자가 L2에서 확인한다.
    function approveMethod(bytes32 method, bytes32 prog, uint8 kind) external only(platform) {
        require(kind == EXACT || kind == APPROX, "kind");
        methodKind[method][prog] = kind;
        emit MethodApproved(method, prog, kind);
    }

    function createJob(bytes32 method, bytes32 prog, bytes32 hStart, uint256 covers) external only(aggregator) returns (uint256 jid) {
        require(pendingLevel2 > 0 && covers == pendingLevel2, "covers");   // 처리 수준 3인 철회가 있어야 함(간이: 수준 2 이상)
        uint8 kind = methodKind[method][prog];
        require(kind != 0, "method");
        jobs.push(Job(method, prog, hStart, bytes32(0), covers, false, false, kind));
        jid = jobs.length;
        emit JobCreated(jid, method, prog, hStart, covers, kind);
    }

    function completeJob(uint256 jid, bytes32 hOut) external only(aggregator) {
        Job storage j = jobs[jid - 1];
        require(!j.completed, "done");
        j.completed = true; j.hOut = hOut;
        models[hOut] = Model(0, true, jid);
        emit JobCompleted(jid, hOut);
    }

    // ---------------- 감사자(알고리즘 3–5, 7) ----------------
    function sampleAudit(uint32 r) external only(auditor) returns (bytes32 seed) {
        require(rounds[r].aggregated, "not aggregated");          // 커밋 게시 이후(A7)
        seed = keccak256(abi.encode(blockhash(block.number - 1), r, address(this)));
        emit AuditSampled(r, seed, uint64(block.timestamp));
    }

    function recordVerification(bytes32 target, uint8 level, bool passed, bytes32 reason) external only(auditor) {
        require(level >= 1 && level <= 4, "level");
        verification[target][level] = passed ? 1 : 2;              // I4: 단계별 검증 결과로만 갱신
        emit VerificationRecorded(target, level, passed, reason);
    }

    // ---------------- 배포(4.3절, I2·I3) ----------------
    function deployable(bytes32 model) public view returns (bool) {
        Model memory m = models[model];
        if (!m.registered || revoked[model]) return false;
        if (m.jobId > 0) {
            // 처리 작업 결과: L1·L2 통과 기록, 근사 방법이면 L3 통과 기록도 있어야 한다(I3, 44번 (a)).
            // 배포 전: 작업이 미해소 철회 전부를 다뤄야 한다. 배포로 해소된 뒤: 새 미해소 철회(처리 수준 2 이상)가
            // 생기기 전까지 허용을 유지한다(새 철회자의 데이터로 학습된 모델이므로 새 철회가 생기면 다시 막힌다).
            Job memory j = jobs[m.jobId - 1];
            bool verified = verification[model][1] == 1 && verification[model][2] == 1 &&
                            (j.kind == EXACT || verification[model][3] == 1);
            return verified && (j.resolved ? pendingLevel2 == 0 : j.covers == pendingLevel2);
        }
        // 정규 라운드 모델: 해소되지 않은 처리 수준 2 이상 철회의 영향 범위(r ≥ r₀) 밖이어야 한다(I2)
        return pendingLevel2 == 0 || m.round < minR0;
    }

    function deploy(bytes32 model, bytes32 basis) external only(platform) {
        require(deployable(model), "not deployable");
        Model memory m = models[model];
        if (m.jobId > 0 && !jobs[m.jobId - 1].resolved) {       // 영향 제거가 검증된 모델로 교체 → 철회 해소
            jobs[m.jobId - 1].resolved = true;
            pendingLevel2 = 0;
            minR0 = type(uint32).max;
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

    // ---------------- 보존·파기(6.4절) ----------------
    function retention(uint256 reqId, uint32 k, bool destroyed) external onlyDC(k) {
        if (destroyed) emit RetentionDestroyed(reqId, k); else emit RetentionStarted(reqId, k);
    }

    function withdrawalCount() external view returns (uint256) { return withdrawals.length; }
}
