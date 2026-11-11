#!/usr/bin/env bash
# 주체별 컨테이너 실행(3장 8.2, 3장 반영 42번 (a)).
#   TAG=eq_base SCEN=results/consent/E2-base.json SEED=1 R=50 bash scripts/run_dist.sh
# 환경변수: OUT_DIR(기본 results/dist/$TAG), FL_K(기본 3), CACHE(기본 ptbxl_k$FL_K), DELAY_MS, RATE_MBPS(인위적 지연·대역폭), P(감사 표본 비율), AUDIT=0(감사 생략), REF=1(단일 프로세스 비교)
# 컨테이너: fed_dc{k}(자기 조각 읽기 전용 + 자기 저장소), fed_agg(데이터 없음), fed_aud{k}(조각·저장소 읽기 전용)
set -uo pipefail
cd "$(dirname "$0")/.."
K=${FL_K:-3}; CACHE=${CACHE:-ptbxl_k$K}; TAG=${TAG:?}; SEED=${SEED:-1}; R=${R:-50}; SCEN=${SCEN:-}
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
SPLIT=$HOME/data/split/$CACHE; RUN=$HOME/data/runs/dist/$TAG; OUT=${OUT_DIR:-$PWD/results/dist/$TAG}
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved|docker run --gpus'
U="--user $(id -u):$(id -g) -e HOME=/tmp -e IMAGE_DIGEST=$IMG -e FL_K=$K -e FL_LEADS=${FL_LEADS:-} --ipc=host"
mkdir -p "$RUN/agg" "$OUT/audit"
cleanup() { for c in $(docker ps -aq --filter "name=^fed_"); do docker rm -f "$c" >/dev/null; done; }
trap cleanup EXIT
cleanup
# 0. 데이터센터별 조각(없을 때만, 플랫폼 준비 단계)
if [ ! -f "$SPLIT/dc$((K-1))/X.npy" ]; then
  mkdir -p "$SPLIT"
  docker run --rm --gpus all $U -v "$PWD":/w:ro -v "$HOME/data/cache":/cache:ro -v "$SPLIT":/split -w /w "$IMG" \
    python -m fl.dist.split --cache "/cache/$CACHE" --out /split 2>&1 | grep -vE "$FILTER"
fi
docker network inspect fednet >/dev/null 2>&1 || docker network create --internal fednet >/dev/null
# 1. 데이터센터 컨테이너
DCS=""
for k in $(seq 0 $((K-1))); do
  mkdir -p "$RUN/store_dc$k"; rm -f "$RUN/store_dc$k"/*
  docker run -d --name fed_dc$k --network fednet --gpus all $U -e DC_K=$k \
    -v "$PWD":/w:ro -v "$SPLIT/dc$k":/shard:ro -v "$RUN/store_dc$k":/store -w /w "$IMG" \
    python -m fl.dist.dc --shard /shard --store /store --port 9000 >/dev/null
  DCS="$DCS${DCS:+,}http://fed_dc$k:9000"
done
# 2. 집계 주체(데이터 연결 없음)
echo "== aggregator $(date +%T)"
docker run --rm --name fed_agg --network fednet --gpus all $U -v "$PWD":/w:ro -v "$OUT":/out -v "$RUN/agg":/ckpt -w /w "$IMG" \
  python -m fl.dist.aggregator ${SCEN:+--scenario $SCEN} --seed $SEED --rounds $R --out /out --dcs "$DCS" \
  --delay-ms ${DELAY_MS:-0} --rate-mbps ${RATE_MBPS:-0} 2>&1 | grep -vE "$FILTER"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "== FAILED aggregator"; for k in $(seq 0 $((K-1))); do docker logs fed_dc$k 2>&1 | tail -5; done; exit 1; }
for k in $(seq 0 $((K-1))); do docker rm -f fed_dc$k >/dev/null; done
# 3. 감사자(데이터센터마다, 조각·저장소·공개 원장 사본 읽기 전용)
[ "${AUDIT:-1}" = 1 ] && echo "== auditors $(date +%T)"
[ "${AUDIT:-1}" = 1 ] && for k in $(seq 0 $((K-1))); do
  docker run --rm --name fed_aud$k --network none --gpus all $U -e DC_K=$k -v "$PWD":/w:ro -v "$SPLIT/dc$k":/shard:ro \
    -v "$RUN/store_dc$k":/store:ro -v "$OUT":/pub:ro -v "$OUT/audit":/out -w /w "$IMG" \
    python -m fl.dist.auditor --shard /shard --store /store --pub /pub/public.json --out /out --p ${P:-0.3} 2>&1 | grep -vE "$FILTER"
done
# 4. (선택) 단일 프로세스 모사와 비트 단위 비교
if [ "${REF:-0}" = 1 ]; then
  echo "== reference $(date +%T)"
  docker run --rm --gpus all $U -v "$PWD":/w:ro -v "$HOME/data/cache":/cache:ro -v "$OUT":/dist -v "$RUN":/runs -w /w "$IMG" \
    python -m fl.dist.reference --cache "/cache/$CACHE" --dist /dist --runs /runs --out /dist/reference.json \
    ${SCEN:+--scenario $SCEN} --seed $SEED --rounds $R 2>&1 | grep -vE "$FILTER"
fi
echo "== done $(date +%T)"
