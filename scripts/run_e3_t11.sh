#!/usr/bin/env bash
# E3 T11(적응형 위조) 원장 실행: 스모크(정직한 작업 배포·단말 대조 확인) → 본 실행(시드 1, T11 20건)
cd "$(dirname "$0")/.."
test -f results/t11/t11_eps.json || { echo "ε 보정값 없음(scripts/run_t11_pilot.sh 먼저)"; exit 1; }
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
mkdir -p results/e3 results/e3_smoke "$HOME/data/runs/e3b_t11" "$HOME/data/runs/e3b_smoke"
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved'
run() { docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
  -v "$PWD":/w:ro -v "$HOME/data/cache/ptbxl_k3":/data:ro -v "$PWD/$1":/out -v "$HOME/data/runs/$2":/runs \
  -v "$HOME/data/runs/lira":/lirarun:ro -w /w "$IMG" python -m fl.e3b --out /out --runs /runs --lirarun /lirarun \
  --latejson results/late/late_methods.json "${@:3}" 2>&1 | grep -vE "$FILTER"; return ${PIPESTATUS[0]}; }
echo "== T11 smoke $(date +%T)"
rm -f results/e3_smoke/B_seed902_T11.json
run results/e3_smoke e3b_smoke --seed 902 --n 2 --rounds 10 --w 8 --frac 0.005 --attacks T11 --tag _T11 || { echo "== smoke FAILED"; exit 1; }
python3 -c "
import json,sys; r=json.load(open('results/e3_smoke/B_seed902_T11.json')); f=r['false_positives']
sys.exit(0 if f['honest_job_deployable'] and f['honest_device_accepts'] else 1)" || { echo "== smoke: honest check failed"; exit 1; }
echo "== T11 full $(date +%T)"
run results/e3 e3b_t11 --seed 1 --n 20 --attacks T11 --tag _T11
echo "== T11 done $(date +%T)"
