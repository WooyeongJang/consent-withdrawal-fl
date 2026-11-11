#!/usr/bin/env bash
# E4: 미시 측정 → K=3·5·10 원장 실행(R=10, 정보주체 100명, 철회 20명)
cd "$(dirname "$0")/.."
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
OUT=${OUT:-results/e4}; mkdir -p "$OUT" "$HOME/data/runs/e4"
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved'
run() { docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" -e FL_K="$1" --ipc=host \
  -v "$PWD":/w:ro -v "$HOME/data/cache":/cache:ro -v "$PWD/$OUT":/out -v "$HOME/data/runs/e4":/runs -w /w "$IMG" \
  python -m fl.e4 --cache "/cache/ptbxl_k$1" --out /out --runs /runs "${@:2}" 2>&1 | grep -vE "$FILTER"; return ${PIPESTATUS[0]}; }
echo "== micro $(date +%T)"; run 3 --mode micro || exit 1
for k in ${KS:-3 5 10}; do echo "== K$k $(date +%T)"; run $k --mode chain --rounds ${R:-10} --subjects ${N:-100} --withdraw ${W:-20} || exit 1; done
echo "== done $(date +%T)"
