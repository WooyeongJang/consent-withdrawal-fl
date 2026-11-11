#!/usr/bin/env bash
# PhysioNet/CinC Challenge 2021 v1.0.3에서 사용 코호트만 내려받고 SHA256으로 검증한다.
# PhysioNet 서버가 연결당 수 KB/s로 느리고 클라우드 미러가 없어, 파일 목록을 SHA256SUMS에서
# 만들고 적당한 병렬도로 이어받기(-c)한다. 중단 후 다시 실행하면 이어서 받는다.
#
#   bash fetch_c2021.sh [ROOT] [JOBS]
set -euo pipefail

ROOT=${1:-$HOME/data/physionet.org/files/challenge-2021/1.0.3}
JOBS=${2:-12}
BASE=https://physionet.org/files/challenge-2021/1.0.3
COHORTS="ningbo georgia chapman_shaoxing cpsc_2018 cpsc_2018_extra"
mkdir -p "$ROOT"
cd "$ROOT"

# 1. SHA256SUMS.txt(약 17MB)를 Range 요청으로 나누어 병렬로 받는다.
SIZE=$(curl -sI "$BASE/SHA256SUMS.txt" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
if [ "$(stat -c %s SHA256SUMS.txt 2>/dev/null || echo 0)" != "$SIZE" ]; then
  PARTS=16
  CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
  mkdir -p .sums_parts
  for i in $(seq 0 $((PARTS - 1))); do
    s=$((i * CHUNK)); e=$((s + CHUNK - 1)); [ $e -ge $SIZE ] && e=$((SIZE - 1))
    want=$((e - s + 1))
    ( until [ "$(stat -c %s .sums_parts/$i 2>/dev/null || echo 0)" = "$want" ]; do
        curl -s --retry 5 -r $s-$e -o .sums_parts/$i "$BASE/SHA256SUMS.txt" || sleep 5
      done ) &
  done
  wait
  cat $(for i in $(seq 0 $((PARTS - 1))); do echo .sums_parts/$i; done) > SHA256SUMS.txt
  rm -rf .sums_parts
fi
[ "$(stat -c %s SHA256SUMS.txt)" = "$SIZE" ] || { echo "SHA256SUMS 크기 불일치" >&2; exit 1; }

# 2. 사용 코호트의 파일 목록과 검사합 부분집합
pattern=$(echo $COHORTS | sed 's/ /|/g')
grep -E " training/($pattern)/" SHA256SUMS.txt > SHA256SUMS.cohorts.txt
echo "대상 파일 $(wc -l < SHA256SUMS.cohorts.txt)개"

# 3. 아직 검증되지 않은 파일만 병렬로 이어받기. 받은 뒤 바로 검사합을 확인해 깨진 파일은 지운다.
fetch_one() {
  sum=$1; path=$2
  if [ -f "$path" ] && echo "$sum  $path" | sha256sum -c --status 2>/dev/null; then return 0; fi
  mkdir -p "$(dirname "$path")"
  for try in 1 2 3 4 5; do
    wget -q -c -O "$path" "$BASE/$path" && echo "$sum  $path" | sha256sum -c --status && return 0
    rm -f "$path"; sleep $((try * 5))
  done
  echo "FAILED $path" >&2
}
export -f fetch_one
export BASE
awk '{print $1, $2}' SHA256SUMS.cohorts.txt | xargs -P "$JOBS" -n 2 bash -c 'fetch_one "$0" "$1"'

# 4. 최종 검증
sha256sum -c --quiet SHA256SUMS.cohorts.txt && echo "검증 완료: 모든 파일 일치"
for c in $COHORTS; do
  echo "$c: hea=$(find training/$c -name '*.hea' | wc -l) mat=$(find training/$c -name '*.mat' | wc -l)"
done
