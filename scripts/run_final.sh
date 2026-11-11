#!/usr/bin/env bash
# 최종 재실행(reports/2026-10-27_최종재실행_계획.md). 결과는 results/final/ 아래에만 쓴다(예비 결과는 덮어쓰지 않는다).
#   bash scripts/run_final.sh all            # 전체(아래 순서)
#   bash scripts/run_final.sh <단계>          # e0 | e1 | e2main | e2aux | e3 | e4 | summary
# 순서: e0(단독) → [e2main → e2aux] ∥ [e1 → e3] → e4(단독) → summary
#   - 원장 서비스는 가장 최근 배포 컨트랙트로 거래를 보내므로 원장 단계(e1 Kairos, e3, e4)는 한 줄로만 돈다.
#   - 단계마다 관문 검사(계획 5절). 실패하면 그 자리에서 멈춘다. 모든 단계는 이미 있는 결과를 건너뛰므로 다시 불러도 된다.
set -uo pipefail
cd "$(dirname "$0")/.."
F=${F:-results/final}   # 별도 재실행(검수)은 F=<다른 폴더>로 덮어쓴다
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
PLATFORM=${PLATFORM:-0x02F6771C74ab530301d77641955dF630290cA5c4}   # 수수료 대납 계정(잔액 확인용). 다른 계정이면 덮어쓴다
RPC=https://public-en-kairos.node.kaia.io
MIN_KAIA=${MIN_KAIA:-15}
FILTER='^$|NVIDIA|=====|Copyright|license|governed|https|container image|PyTorch Version|Container image|NOTE|^  [A-Z]|PyTorch ==|rights reserved'
mkdir -p "$F"

log() { echo "== [final] $* $(date '+%F %T')"; }
die() { log "중단: $*"; exit 1; }
inbox() {   # inbox [docker 옵션…] -- <명령…>
  local opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
    -v "$PWD":/w -w /w "${opts[@]}" "$IMG" "$@" 2>&1 | grep -vE "$FILTER"; return "${PIPESTATUS[0]}"; }

balance() { curl -s -X POST "$RPC" -H 'content-type: application/json' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_getBalance\",\"params\":[\"$PLATFORM\",\"latest\"]}" |
  python3 -c 'import sys,json;print(int(json.load(sys.stdin)["result"],16)/1e18)'; }
gate_kaia() { local b; b=$(balance) || die "잔액 조회 실패"; log "KAIA 잔액 $b"
  python3 -c "import sys;sys.exit(0 if float('$b')>=$MIN_KAIA else 1)" || die "KAIA 잔액 $b < $MIN_KAIA (계획 3절)"; }
gate_solo() {   # 시간 측정 단계(E4)는 다른 학습 컨테이너가 없을 때만
  local n; n=$(docker ps -q --filter "ancestor=$IMG" | wc -l)
  [ "$n" -eq 0 ] || die "E4는 단독 실행 — 학습 이미지 컨테이너 ${n}개가 돌고 있다"; }

stage_e0() {
  log "E0 시작"
  [ -f "$F/E0.json" ] && { log "E0 결과 있음 — 건너뜀"; return 0; }
  OUT=$PWD/$F/e0 bash scripts/run_e0.sh || die "E0 실행 실패"
  python3 scripts/summarize_e0.py "$F/e0" "$F/E0.json" || die "E0 집계 실패"
  python3 - "$F/E0.json" <<'EOF' || die "E0 관문: gpu_det 비트 단위 재현 불성립(A6) — 원인 보고 후 결정"
import json, sys
g = {(x["mode"], x["cond"]): x for x in json.load(open(sys.argv[1]))["groups"]}
ok = g[("unit", "gpu_det")]["distinct_hashes"] == 1 and all(g[("chain", "gpu_det")]["rounds_identical"])
print("E0 관문(gpu_det 단위·연쇄 해시 일치):", ok); sys.exit(0 if ok else 1)
EOF
  log "E0 완료"
}

stage_e1() {
  log "E1 시뮬레이션"
  [ -f "$F/e1/E1.json" ] || inbox -v "$HOME/data/cache/ptbxl_k3":/data:ro -- python -m fl.e1 --cache /data --out "$F/e1" \
    > "$F/e1_run.log" || die "E1 시뮬레이션 실패"
  gate_kaia
  log "E1 Kairos 확인"
  mkdir -p "$HOME/data/runs/final_e1chain"
  [ -f "$F/e1_chain/E1_chain.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('$F/e1_chain/E1_chain.json'))))")" -eq 7 ] ||
    inbox -v "$HOME/data/cache/ptbxl_k3":/data:ro -v "$HOME/data/runs/final_e1chain":/runs -- \
      python -m fl.e1chain --cache /data --out "$F/e1_chain" --runs /runs | tee "$F/e1_chain_run.log" || die "E1 Kairos 실패"
  log "E1 완료"
}

E2_MAIN="E2-base E2-rate01 E2-rate10 E2-rate20 E2-early E2-mid E2-rate01-early E2-rate01-mid E2-rate10-early E2-rate10-mid
         E2-rate20-early E2-rate20-mid E2-diagMI E2-F65 E2-inst2 E2-seq3 E2-reconsent E2-exit2 E2-latejoin"
stage_e2main() {
  log "E2 주 데이터(19 × 10)"
  # shellcheck disable=SC2086
  SEEDS="$(seq -s ' ' 1 10)" bash scripts/run_e2.sh "$F/e2" final_e2 $E2_MAIN || die "E2 주 데이터 실패"
  log "E2 주 데이터 완료"
}

stage_e2aux() {
  log "E2 보조 조건(n = 5)"
  local S="1 2 3 4 5"
  SEEDS="$S" FL_K=5 bash scripts/run_e2.sh "$F/e2_k5" final_e2_k5 E2-K5-base || die "K5 실패"
  SEEDS="$S" FL_LEADS=I bash scripts/run_e2.sh "$F/e2_leadI" final_e2_leadI E2-base E2-mid || die "유도 I 실패"
  SEEDS="$S" FL_K=4 CACHE=c2021 bash scripts/run_e2.sh "$F/e2_c2021" final_e2_c2021 C2021-base C2021-mid || die "C2021 실패"
  SEEDS="$S" SUFFIX=-perwave bash scripts/run_e2.sh "$F/e2_perwave" final_e2_perwave E2-seq3 -- --sequential || die "차례별 처리 실패"
  SEEDS="$S" SUFFIX=-osdcheck bash scripts/run_e2.sh "$F/e2_osdcheck" final_e2_osdcheck E2-base -- --osd-variants || die "FedOSD 점검 실패"
  log "E2 보조 조건 완료"
}

e3b() {   # e3b <결과 디렉터리> <runs 이름> <fl.e3b 인자…>
  mkdir -p "$1" "$HOME/data/runs/$2"
  docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
    -v "$PWD":/w:ro -v "$HOME/data/cache/ptbxl_k3":/data:ro -v "$PWD/$1":/out -v "$HOME/data/runs/$2":/runs \
    -v "$HOME/data/runs/lira":/lirarun:ro -w /w "$IMG" python -m fl.e3b --out /out --runs /runs --lirarun /lirarun \
    --latejson results/late/late_methods.json "${@:3}" 2>&1 | grep -vE "$FILTER"; return "${PIPESTATUS[0]}"; }
gate_b() { python3 - "$1" <<'EOF'
import json, sys
f = json.load(open(sys.argv[1]))["false_positives"]
ok = f["honest_job_deployable"] and f["honest_device_accepts"] and f.get("training_L1_fail", 0) == 0
print("E3-B 관문(정직한 배포·단말 대조·정규 학습 L1):", ok, f); sys.exit(0 if ok else 1)
EOF
}
B_ARGS="--attacks T4,T5,T6,T7,T8,T9,T11 --t11-variants abcdef --n 20 --n-t11 24"
CALIB="b45934b87ffb666955ef86c332ac6b922a7130f40d0a94b1a6bde2c26ca09d56  $HOME/data/runs/lira/shadows_late.npz
6e143f989e4ec8f18882a5533cce0bd2d3895ffb29fca55fb56453237235e236  results/late/late_methods.json
fa52963262bdddb7c28f011a4befdda45cd02d93a4ece34097c8e755e452ee90  results/t11/t11_eps.json"
stage_e3() {
  echo "$CALIB" | sha256sum -c --quiet || die "E3-B 보정 입력 해시 불일치(계획 0-7)"
  log "E3 유형 A(시드 1–20)"
  gate_kaia
  OUT=$PWD/$F/e3 RUNS=$HOME/data/runs/final_e3 FROM=1 TO=20 bash scripts/run_e3a.sh || die "E3 유형 A 실패"
  python3 - "$F/e3" <<'EOF' || die "E3-A 관문: 정직한 단위 거절·L1 실패·단말 대조 실패 발생"
import glob, json, sys
fs = sorted(glob.glob(f"{sys.argv[1]}/A_seed*_p0.3.json")); bad = []
for p in fs:
    f = json.load(open(p))["false_positives"]
    if f["rejected_honest"] or f["l1_fail_honest"] or any(f["subject_check_fail_honest"].values()):
        bad.append(p)
print("E3-A 결과", len(fs), "개, 관문 실패", bad); sys.exit(0 if len(fs) == 20 and not bad else 1)
EOF
  log "E3 유형 B 스모크"
  rm -f "$F/e3_smoke/B_seed902.json"
  # shellcheck disable=SC2086
  e3b "$F/e3_smoke" final_e3b_smoke $B_ARGS --seed 902 --n 1 --n-t11 6 --rounds 10 --w 8 --frac 0.01 || die "B 스모크 실패"
  gate_b "$F/e3_smoke/B_seed902.json" || die "B 스모크 관문 실패"
  for s in 1 2 3 4 5; do
    [ -f "$F/e3/B_seed$s.json" ] && { gate_b "$F/e3/B_seed$s.json" || die "B 시드 $s 관문 실패(기존 결과)"; continue; }
    gate_kaia
    log "E3 유형 B 시드 $s"
    # shellcheck disable=SC2086
    e3b "$F/e3" final_e3b --seed "$s" $B_ARGS || die "B 시드 $s 실행 실패"
    gate_b "$F/e3/B_seed$s.json" || die "B 시드 $s 관문 실패 — 실행 폐기 후 원인 보고(계획 5절)"
  done
  log "E3 완료"
}

stage_e4() {
  gate_solo; gate_kaia
  log "E4 미시·원장(단독)"
  [ -f "$F/e4/E4_K10.json" ] || OUT=$F/e4 bash scripts/run_e4.sh || die "E4 실패"
  gate_solo
  log "E4 주체별 컨테이너: 단일 프로세스와의 비트 단위 동일성"
  [ -f "$F/dist/eq_base/reference.json" ] || OUT_DIR=$PWD/$F/dist/eq_base TAG=final_eq_base SCEN=results/consent/E2-base.json \
    SEED=1 R=50 REF=1 bash scripts/run_dist.sh || die "주체별 컨테이너 실패"
  python3 - "$F/dist/eq_base/reference.json" <<'EOF2' || die "주체별 컨테이너 관문: 단일 프로세스 모사와 불일치"
import json, sys
r = json.load(open(sys.argv[1])); ok = r["final_equal"] and not r["unit_diffs"] and not r["round_diffs"]
print("주체별 컨테이너 = 단일 프로세스(최종·라운드·단위 해시):", ok); sys.exit(0 if ok else 1)
EOF2
  log "E4 지연·대역폭 민감도(예비 실행과 같은 격자: R=10, 지연 0·50·200ms × 대역폭 무제한·100·10Mbps)"
  for k in 3 10; do for d in 0 50 200; do for bw in 0 100 10; do
    t=lat_K${k}_d${d}_b${bw}; [ -f "$F/dist/$t/aggregator.json" ] && continue
    OUT_DIR=$PWD/$F/dist/$t FL_K=$k TAG=final_$t SEED=1 R=10 AUDIT=0 DELAY_MS=$d RATE_MBPS=$bw bash scripts/run_dist.sh || die "$t 실패"
  done; done; done
  log "E4 완료"
}

stage_summary() {
  log "집계"
  python3 scripts/summarize_e0.py "$F/e0" "$F/E0.json" > "$F/E0_summary.md"
  for d in e2 e2_k5 e2_leadI e2_c2021 e2_perwave e2_osdcheck; do
    inbox -- python scripts/summarize_e2.py "$F/$d" "$F/E2_$d.json" > "$F/E2_${d}_summary.md" || die "$d 집계 실패"
  done
  python3 scripts/summarize_e3.py "$F/e3" "$F/E3.json" > "$F/E3_summary.md" || die "E3 집계 실패"
  log "집계 완료 — 재현성 기록은 결과 확인 후 scripts/repro_manifest.py로 다시 만든다"
}

case "${1:-all}" in
  e0|e1|e2main|e2aux|e3|e4|summary) "stage_$1" ;;
  all)
    git diff --quiet HEAD -- fl contracts scripts tests tools || die "작업 트리에 커밋되지 않은 코드 변경이 있다(코드 동결, 계획 0-1)"
    git describe --exact-match --tags HEAD 2>/dev/null | grep -q '^final-rerun-v1$' || die "HEAD가 final-rerun-v1 태그가 아니다"
    stage_e0
    ( stage_e2main && stage_e2aux ) > "$F/run_e2_branch.log" 2>&1 & P_E2=$!
    ( stage_e1 && stage_e3 ) > "$F/run_ledger_branch.log" 2>&1 & P_L=$!
    wait $P_L || die "원장 갈래(E1·E3) 실패 — $F/run_ledger_branch.log"
    wait $P_E2 || die "E2 갈래 실패 — $F/run_e2_branch.log"
    stage_e4
    stage_summary
    log "전체 완료" ;;
  *) die "알 수 없는 단계: $1" ;;
esac
