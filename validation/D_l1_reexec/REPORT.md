# 실험 D — 유형 B 정규 학습 단위의 L1 독립 재실행 보고

- 실행: 2026-11-10, 저장소 HEAD `2629d2f`(태그 `jbhi-v69` + 검토용 PDF 커밋), 사전 등록 `PREREG.md`(SHA-256 `12fcc0da…`, `PREREG.sha256`에 기록 시각 11:33:16). 실패한 1차 실행은 `../DEVIATIONS.md` D절.
- 입력(읽기 전용): 원장 이벤트 캐시(`../C_independent_audit/results/raw/merged/`, 유형 B 컨트랙트 5개), 체크포인트 `~/data/runs/final_e3b/B_s{1..5}/orig/ckpt_r{0..50}.pt`, 데이터 분할 캐시 `~/data/cache/ptbxl_k3`. 실험 기록(`results/final/e3/B_seed*.json`의 `training_L1`, 실행 폴더 `events.jsonl`)은 비교 대상으로만 썼다.
- 환경: `tools/toolchain.json`의 고정 이미지(`nvcr.io/nvidia/pytorch@sha256:3becd068…`), `--gpus all --ipc=host`, 결정적 GPU 모드(`T.configure("gpu_det")`), torch 2.14.0a0+4fdf77b940.nv26.08. `fl/` 핵심 코드는 유형 B 실행 시점(커밋 f9d1154)과 동일(`git diff --stat f9d1154 HEAD -- fl/{train,run,audit,evidence,model,consent}.py` 차이 없음).

## 1. 결론

**H-D1 성립.** 원장의 추첨 seed와 커밋만으로 고른 표본 학습 단위 **206개 전부**가 (1) 저장소에서 재구성한 입력 집합의 루트 = 원장 `rootS`, (2) `ckpt_r{r-1}`의 모델 해시 = 원장 `hStart`, (3) 재실행 출력의 해시 = 원장 `hDelta`를 **비트 단위로** 만족했다(불일치 0). 실험 기록 `training_L1.fail = 0`(시드별 36/41/43/44/42)이 실험 코드의 감사자 기록이 아닌 독립 재실행으로 확인됐다.

이 결과가 뜻하는 것과 뜻하지 않는 것:
- 원고의 "no false positives in 1,083 honest re-executions" 가운데 유형 B 206건은 이제 독립 근거가 있다. 유형 A 877건(그리고 주입 단위 22건의 불일치)은 실행 폴더가 없어 여전히 실험 코드의 기록만 있다.
- 재실행은 같은 이미지·장치·결정적 설정에서만 의미가 있다(E0, A6). 다른 환경에서의 재현은 보이지 않는다.
- 검증기를 만든 주체가 원 실험과 같은 연구 환경에 있으므로 제3자 재현은 아니다. 독립인 것은 입력(원장 추첨·커밋, 체크포인트)과 절차(실험의 감사 기록을 쓰지 않음)다.

## 2. 결과

| 시드 | 컨트랙트 | 표본(원장 추첨, p = 0.3) | 실험 기록 `sampled` | 통과 | 불일치 | 정규화 통계 일치 |
|---|---|---|---|---|---|---|
| 1 | 0x9DA2FAF3BADb71ddb7cebdAB5B220968008984a2 | 36 | 36 | 36 | 0 | 예 |
| 2 | 0xEad35e48dDD2855511Cbc440eD3E59Ca316ef64E | 41 | 41 | 41 | 0 | 예 |
| 3 | 0xB2EB7208261F6253AE5F5B90eA828f8B9D141144 | 43 | 43 | 43 | 0 | 예 |
| 4 | 0xBF24f942a26BfF38dC15F4d6F88a4da230e50272 | 44 | 44 | 44 | 0 | 예 |
| 5 | 0x8BEb95D2707d5BC60FF87240C00F90e0187F41f1 | 42 | 42 | 42 | 0 | 예 |
| 합 | | **206** | 206 | **206** | **0** | |

- 표본 추첨: 시드마다 `AuditSampled` 50개(라운드마다 하나), 입력·갱신 커밋 150개(r ≤ 50; r = 1000+의 T9a 위조 라운드는 제외). Algorithm S3 2행(`SHA-256("0x<seed>|<k>")/2^256 < 0.3`)으로 고른 표본 수가 실험 기록과 시드마다 같다.
- 원장 커밋과 실행 폴더 `events.jsonl`의 같은 (k, r) 커밋(rootS, hStart, hDelta)은 206/206 일치(기록의 일관성; 입력으로 쓰지 않음).
- 정규화 통계(평균 −0.000826, 표준편차 0.232101)를 1라운드 입력 범위에서 다시 계산한 값이 `NormStatsCommitted`와 1e-12 안에서 일치.
- 재실행 시간: 단위당 중앙값 0.51 s, 최대 1.07 s, 합계 115 s(한 로컬 에폭 34–60 스텝). 재시작 0회.

## 3. 원고와의 관계

- Extended R11 "Not independently verifiable"의 "the outcomes of the \Lone{} re-executions (type A: 877 …; type B: 206 units) and therefore the zero false positives"에서 유형 B 206건을 빼고, 독립 재실행 결과를 적을 수 있다.
- 보충 S2 "7, among them the outcomes of the \Lone{} re-executions and hence the zero false positives, rest only on the experiment's own records"는 유형 A에 한정하는 표현으로 바꿀 수 있다(보충 5쪽 유지 확인 필요).
- 본문 IV-C의 "no false positives in 1,083 honest re-executions"는 그대로 참이다. 본문은 꽉 차 있어 수정하지 않는다.
- 두 문장의 수정은 저자 승인(2026-11-10 "둘 다 적용") 뒤 `jbhi-v70`에 반영했다(보충 5쪽 유지).

## 4. 실행 기록

| 단계 | 스크립트 | 로그 | 결과 |
|---|---|---|---|
| D1 표본·커밋 추출(호스트, 표준 라이브러리 + `c_lib`) | `code/d1_units.py` | `logs/d1_units_20261110T113316.log`(1차, k 정수 → `events.jsonl` 일치 0), `logs/d1_units_20261110T113455.log`(2차) | `results/d1_units.json` |
| D2 재실행(컨테이너 GPU) | `code/d2_reexec.py` | `logs/d2_reexec_20261110T113346.log`(1차 실패: `KeyError: 2`), `logs/d2_reexec_20261110T113455.log`(2차, 완료) | `results/d2_reexec.json` |

재현: 저장소 루트에서

    python3 validation/D_l1_reexec/code/d1_units.py
    IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
    docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
      -v "$PWD":/w:ro -v ~/data/cache/ptbxl_k3:/data:ro -v ~/data/runs/final_e3b:/runs:ro -v "$PWD/validation/D_l1_reexec/results":/out -w /w "$IMG" \
      python /w/validation/D_l1_reexec/code/d2_reexec.py --units /w/validation/D_l1_reexec/results/d1_units.json --out /out/d2_reexec.json

D1은 실험 C가 받아 둔 원장 이벤트 캐시(저장소에 넣지 않은 51 MB)를 읽는다; 캐시가 없으면 `validation/C_independent_audit/code/c1_fetch_chain.py`로 먼저 받는다. D2는 유형 B 실행 폴더(8.7 GB, 저장소 밖)와 PTB-XL 캐시가 필요하다.
