// SPDX-License-Identifier: MIT
// Kairos 연결 시험용 컨트랙트. 3장 6.3절 이벤트 스키마의 형태만 흉내 내며 상태 전이 규칙은 없다.
pragma solidity ^0.8.30;

contract Probe {
    event WithdrawalRequested(bytes32 indexed subject, uint32 version, uint8 level, bytes32 listRoot);
    event RoundInputCommitted(uint32 indexed k, uint32 indexed r, bytes32 rootS, uint32 n, uint32 sigma, bytes32 purpose);

    function requestWithdrawal(bytes32 subject, uint32 version, uint8 level, bytes32 listRoot) external {
        emit WithdrawalRequested(subject, version, level, listRoot);
    }

    function commitInput(uint32 k, uint32 r, bytes32 rootS, uint32 n, uint32 sigma, bytes32 purpose) external {
        emit RoundInputCommitted(k, r, rootS, n, sigma, purpose);
    }
}
