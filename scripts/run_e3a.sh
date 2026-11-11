#!/usr/bin/env bash
# E3 유형 A 본 실행: 시드마다 새 컨트랙트, 50라운드, 공격 5건. 이미 결과가 있으면 건너뛴다(재시작 가능).
set -uo pipefail
cd "$(dirname "$0")/.."
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
OUT=${OUT:-$PWD/results/e3}; RUNS=${RUNS:-$HOME/data/runs/e3}; P=${P:-0.3}
mkdir -p "$OUT" "$RUNS"
for s in $(seq ${FROM:-1} ${TO:-20}); do
  [ -f "$OUT/A_seed${s}_p${P}.json" ] && { echo "skip seed $s"; continue; }
  echo "== seed $s $(date +%T)"
  docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
    -v "$PWD":/w:ro -v "$HOME/data/cache/ptbxl_k3":/data:ro -v "$OUT":/out -v "$RUNS":/runs -w /w "$IMG" \
    python -m fl.e3 --type A --seed "$s" --p "$P" --out /out --runs /runs 2>&1 | grep -E "^(T1|T2|T3|T10|오탐)|Traceback|Error"
  rm -rf "$RUNS/A_s${s}"          # 체크포인트 정리(디스크 절약). 원장 기록과 결과 JSON은 남는다
done
echo "== done $(date +%T)"
