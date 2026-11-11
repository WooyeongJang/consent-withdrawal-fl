#!/usr/bin/env bash
# T11 오프라인 예비 실험: ε 보정(비결정 U0 2회) → T11a·T11b 예비 사례
cd "$(dirname "$0")/.."
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
mkdir -p results/t11 "$HOME/data/runs/t11"
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved'
run() { docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
  -v "$PWD":/w:ro -v "$HOME/data/cache/ptbxl_k3":/data:ro -v "$PWD/results/t11":/out -v "$HOME/data/runs/t11":/runs \
  -v "$HOME/data/runs/e3b":/e3b:ro -v "$HOME/data/runs/lira":/lirarun:ro -w /w "$IMG" python -m fl.exp_t11 \
  --out /out --runs /runs --e3b /e3b "$@" 2>&1 | grep -vE "$FILTER"; return ${PIPESTATUS[0]}; }
echo "== pilot $(date +%T)"
run --mode pilot --lirarun /lirarun --latejson results/late/late_methods.json --n ${N:-4} || exit 1
echo "== eps $(date +%T)"
run --mode eps || exit 1
echo "== done $(date +%T)"
