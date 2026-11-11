#!/usr/bin/env bash
# E2 실행: run_e2.sh <결과 디렉터리> <runs 하위 디렉터리> <시나리오 이름…> -- <fl.e2 추가 인자>
# 예: SEEDS="1 2 3 4 5" bash scripts/run_e2.sh results/e2 e2 E2-base E2-rate01
# 결과 이름에 접미사가 붙는 모드(--sequential → -perwave, --osd-variants → -osdcheck)는 SUFFIX로 건너뛰기 검사를 맞춘다.
# 보조 데이터: FL_K=4 CACHE=c2021 bash scripts/run_e2.sh results/e2_c2021 e2_c2021 C2021-base C2021-mid
cd "$(dirname "$0")/.."
OUT=$1; RUNS=$2; shift 2
NAMES=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do NAMES+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
mkdir -p "$OUT" "$HOME/data/runs/$RUNS"
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved'
for n in "${NAMES[@]}"; do for s in ${SEEDS:-1}; do
  [ -f "$OUT/${n}${SUFFIX:-}_s${s}.json" ] && { echo "skip $n s$s"; continue; }
  echo "== $n s$s $(date +%T)"
  docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
    -v "$PWD":/w:ro -e FL_K="${FL_K:-3}" -e FL_LEADS="${FL_LEADS:-}" -v "$HOME/data/cache":/cache:ro -v "$PWD/$OUT":/out -v "$HOME/data/runs/$RUNS":/runs \
    -v "$HOME/data/physionet.org/files/ptb-xl/1.0.3":/ptbxl:ro -w /w "$IMG" \
    python -m fl.e2 --scenario "results/consent/$n.json" --seed "$s" --out /out --runs /runs --ptbxl /ptbxl --cache "/cache/${CACHE:-ptbxl_k${FL_K:-3}}" "$@" \
    2>&1 | grep -vE "$FILTER"; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "== FAILED $n s$s"; exit 1; }
done; done
echo "== all done $(date +%T)"
