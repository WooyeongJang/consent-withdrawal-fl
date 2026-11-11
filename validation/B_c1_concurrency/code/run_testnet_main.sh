#!/usr/bin/env bash
# B-T 본 실행: v3.1 150회 → v3.2 150회(역할 계정을 공유하므로 차례로 실행). 저장소 루트에서 실행한다.
set -u
cd "$(dirname "$0")/../../.."
run_t() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD":/w:ro \
  -v "$HOME/.config/dissertation/kairos/exp":/keys:ro -v "$PWD/validation/B_c1_concurrency/results":/out \
  --entrypoint python dissertation-ledger:0.5 -W ignore /w/validation/B_c1_concurrency/code/testnet_harness.py "$@"; }
for v in v3.1 v3.2; do
  run_t --version $v --n 150 --out /out/testnet_main_$v.jsonl --budget-kaia 4.5 --label main > validation/B_c1_concurrency/logs/testnet_main_$v.log 2>&1
  echo "exit $? $(date -Is)" >> validation/B_c1_concurrency/logs/testnet_main_$v.log
done
