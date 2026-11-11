#!/usr/bin/env bash
# E0 재현성 사전 실험 전체 실행 (reports/2026-10-25_E0_설정근거.md 3.2절).
# 각 실행은 새 컨테이너(독립 프로세스)에서 한다. 이미지는 tools/toolchain.json의 digest로 고정한다.
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
CACHE=${CACHE:-$HOME/data/cache/ptbxl_k3}
OUT=${OUT:-$PWD/results/e0}
REPS=${REPS:-10}
mkdir -p "$OUT"
run() { docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp --ipc=host \
          -v "$PWD":/w:ro -v "$CACHE":/data:ro -v "$OUT":/out -w /w "$IMG" python -m fl.e0 "$@" --out /out 2>&1 | tail -1; }

echo "== 1. 같은 시드 반복: 기관 0, 1라운드 × ${REPS}회 × 3조건"
for cond in gpu_det cpu gpu_nondet; do for i in $(seq 1 "$REPS"); do run --mode unit --cond "$cond" --run "$i"; done; done
echo "== 2. CPU 스레드 수 변경(8→4) 2회"
for i in 1 2; do run --mode unit --cond cpu --threads 4 --tag _t4 --run "$i"; done
echo "== 3. 라운드 연쇄: 세 기관 × 5라운드 × 2회 (gpu_det, cpu)"
for cond in gpu_det cpu; do for i in 1 2; do run --mode chain --cond "$cond" --run "$i"; done; done
echo "== 4. 성능 분산 예비 측정: R=50 전체 학습 × 시드 5개 (gpu_det)"
for s in 1 2 3 4 5; do run --mode pilot --cond gpu_det --base-seed "$s" --run 1; done
echo "== 완료"
