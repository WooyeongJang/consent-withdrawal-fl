// SPDX-License-Identifier: MIT
// 비교군 B3: 블록체인 기반 완료 기록(3장 8.3). 선행 설계(Kasiraju et al., 2025; Zuo et al., 2025)가 공통으로 제공하는
// "철회와 처리 완료, 모델 해시의 원장 기록" 기능만 같은 Kairos에서 재구성한다. 기록 주체는 운영자다.
pragma solidity ^0.8.30;

contract CompletionRecord {
    address public operator;
    event WithdrawalRecorded(bytes32 indexed subject, bytes32 listRoot, uint64 receivedAt);
    event CompletionRecorded(bytes32 indexed subject, bytes32 method, bytes32 modelHash);
    event ModelRecorded(uint32 indexed round, bytes32 modelHash);

    constructor() { operator = msg.sender; }
    modifier onlyOperator() { require(msg.sender == operator, "operator"); _; }

    function recordWithdrawal(bytes32 subject, bytes32 listRoot, uint64 receivedAt) external onlyOperator {
        emit WithdrawalRecorded(subject, listRoot, receivedAt);
    }
    function recordCompletion(bytes32 subject, bytes32 method, bytes32 modelHash) external onlyOperator {
        emit CompletionRecorded(subject, method, modelHash);
    }
    function recordModel(uint32 round, bytes32 modelHash) external onlyOperator {
        emit ModelRecorded(round, modelHash);
    }
}
