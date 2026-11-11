#!/usr/bin/env bash
# 유형 A 본 실행이 끝나면 유형 B 스모크 테스트(소규모) → 성공 시 B 본 실행.
# 원장 서비스는 최근 배포 컨트랙트로 거래를 보내므로 A와 동시에 실행하지 않는다.
cd "$(dirname "$0")/.."
until grep -q "== done" results/e3_runA.log; do sleep 60; done
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
mkdir -p results/e3 results/e3_smoke "$HOME/data/runs/e3b" "$HOME/data/runs/e3b_smoke"
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved'
run() { docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
  -v "$PWD":/w:ro -v "$HOME/data/cache/ptbxl_k3":/data:ro -v "$PWD/$1":/out -v "$HOME/data/runs/$2":/runs \
  -v "$HOME/data/runs/lira":/lirarun:ro -w /w "$IMG" python -m fl.e3b --out /out --runs /runs --lirarun /lirarun \
  --latejson results/late/late_methods.json "${@:3}" 2>&1 | grep -vE "$FILTER"; return ${PIPESTATUS[0]}; }
echo "== B smoke $(date +%T)"
rm -f results/e3_smoke/B_seed902.json
run results/e3_smoke e3b_smoke --seed 902 --n 1 --rounds 10 --w 8 --frac 0.005 || { echo "== B smoke FAILED"; exit 1; }
python3 -c "
import json,sys; r=json.load(open('results/e3_smoke/B_seed902.json')); f=r['false_positives']
sys.exit(0 if f['honest_job_deployable'] and f['honest_device_accepts'] else 1)" || { echo "== B smoke: honest job/device check failed"; exit 1; }
echo "== B full $(date +%T)"
run results/e3 e3b --seed 1 --n 20
echo "== B done $(date +%T)"
