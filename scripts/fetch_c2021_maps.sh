#!/usr/bin/env bash
# Challenge 2021 평가 저장소의 SNOMED CT 코드표 두 개를 고정 커밋에서 받아 SHA-256으로 검증한다.
# 공개 저장소에는 이 두 파일을 싣지 않는다(SNOMED CT 용어의 재배포 조건 미확인, 2026-10-31 결정).
# c2021/mapping_superclass_v1.csv(우리 사상표)를 만들 때 참고한 표이며, 학습 캐시 생성에는 필요 없다.
#
#   bash fetch_c2021_maps.sh [OUT]
set -euo pipefail

OUT=${1:-c2021}
COMMIT=e3a9eb494181cf074648cf964a0be3841da27fc3     # physionetchallenges/evaluation-2021, BSD-2-Clause
BASE=https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/$COMMIT
mkdir -p "$OUT"
cd "$OUT"
for f in dx_mapping_scored.csv dx_mapping_unscored.csv; do
  curl -sfL -o "$f" "$BASE/$f"
done
sha256sum -c - <<'SUMS'
fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0  dx_mapping_scored.csv
ce53c9e35406e922f1634a38bb13c31e8cf7603291641a16c201d8e679f4fe94  dx_mapping_unscored.csv
SUMS
