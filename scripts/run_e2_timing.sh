#!/usr/bin/env bash
# E2 기준 시나리오 처리 시간 단독 재측정(2026-10-30 결정 (나)).
# 최종 E2는 E1·E3와 GPU를 함께 쓰며 돌아 처리 시간이 부풀었다. E4와 집계가 끝난 뒤 다른 GPU 작업 없이
# E2-base 시드 1–10을 새 디렉터리에서 다시 돌리고, 모델 해시가 results/final/e2와 같은지 확인한 뒤 시간만 쓴다.
#   bash scripts/run_e2_timing.sh && python3 scripts/check_e2_timing.py
cd "$(dirname "$0")/.."
F=${F:-results/final}
busy=$(docker ps --format '{{.Names}} {{.Image}}' | grep -v -E "dissertation-ledger${IGNORE_CONTAINERS:+|$IGNORE_CONTAINERS}" || true)
[ -z "$busy" ] || { echo "다른 컨테이너가 실행 중이다 — 단독 측정이 아니다:"; echo "$busy"; exit 1; }
SEEDS="$(seq -s ' ' 1 10)" bash scripts/run_e2.sh "$F/e2_timing" final_e2_timing E2-base 2>&1 | tee "$F/run_e2_timing.log"
