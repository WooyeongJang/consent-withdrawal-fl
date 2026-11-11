# 실험 D 사전 등록 — 유형 B 정규 학습 단위의 L1 독립 재실행

- 작성: 2026-11-10, 실행 전. 이 파일의 SHA-256과 작성 시각은 `PREREG.sha256`에 남긴다. 기준 저장소 HEAD `2629d2f`(태그 `jbhi-v69` + 검토용 PDF 커밋).
- 사용자 지시(2026-11-10): "유형 B L1 206건 재실행도 해줘". 실험 C 보고 4절과 Extended R11이 "독립 검증 불가"로 적은 항목 가운데, 체크포인트가 남아 있어 재실행이 가능한 유일한 것이다.
- 공통 원칙은 `../PREREG.md`와 같다(기존 코드·결과 읽기만, 새 산출물은 이 폴더에, 실패·이탈은 `../DEVIATIONS.md` D절, 불리해도 그대로 보고, 원고는 승인 전 수정하지 않음).

## 목적

원고(VI/IV-C절 "no false positives in 1,083 honest re-executions", Extended R11 "the outcomes of the L1 re-executions … rest only on the experiment's own records")가 밝힌 대로, L1 재실행 결과의 근거는 실험 코드의 감사자가 쓴 기록뿐이다. 유형 B(E3-B, 시드 1–5)의 정규 학습 단위는 체크포인트(`~/data/runs/final_e3b/B_s{1..5}/orig/ckpt_r{0..50}.pt`)가 보존되어 있으므로, **원장에 기록된 표본 추첨과 커밋만을 입력으로** 같은 단위를 다시 실행해 커밋된 갱신 해시와 비트 단위로 비교한다. 유형 A(877건)는 실행 폴더가 없어 할 수 없다.

## 입력 (읽기 전용)

1. 원장 이벤트(실험 C가 받아 둔 캐시 `../C_independent_audit/results/raw/merged/<address>.json`; 다시 받으려면 `c1_fetch_chain.py`): 유형 B 컨트랙트 5개(`chain_index.json`의 type B, which contract)에서 `AuditSampled`(라운드별 추첨 seed), `RoundInputCommitted`(k, r, rootS, n, sigma), `UpdateCommitted`(k, r, hDelta, hStart), r ≤ 50.
2. 체크포인트 `ckpt_r{r-1}.pt`(출발 상태)와 데이터 분할 캐시 `~/data/cache/ptbxl_k3`.
3. 실험 코드 `fl/audit.py reexecute`(알고리즘 S3)와 `fl/train.py`: 유형 B 실행 시점의 코드(커밋 f9d1154, 태그 `final-rerun-v1.2`)와 현재 HEAD의 `fl/train.py`·`run.py`·`audit.py`·`evidence.py`·`model.py`·`consent.py`가 같음을 `git diff --stat f9d1154 HEAD`로 확인했다(차이 없음).
4. 환경: `tools/toolchain.json`의 고정 이미지(`nvcr.io/nvidia/pytorch@sha256:3becd068…`), `--gpus all --ipc=host`, `T.configure("gpu_det")`(스레드 8) — `scripts/run_final.sh`의 e3b 호출과 같은 옵션.

실험 코드가 쓴 기록 가운데 쓰지 **않는** 것: `results/final/e3/B_seed*.json`의 `training_L1`(비교 대상), 실행 폴더 `events.jsonl`의 커밋 기록(원장 값과의 일치만 확인하고 입력으로 쓰지 않음), 감사자의 `VerificationRecorded`.

## 절차

- **D1(호스트, 표준 라이브러리 + `c_lib`)**: 시드마다 r ≤ 50의 입력·갱신 커밋 150개를 원장에서 모으고, `AuditSampled` seed로 Algorithm S3 2행(`SHA-256("0x<seed>|<k>")/2^256 < 0.3`)을 적용해 표본 단위를 고른다. 출력 `results/d1_units.json`: 시드별 표본 단위 목록 {k, r, root_S, sigma, n, h_start, h_delta}와 수. 확인: 시드별 표본 수가 `B_seed{s}.json`의 `training_L1.sampled`(합 206)와 같은지(다르면 그대로 보고하고 원장 값으로 진행), 원장 커밋이 `events.jsonl`의 같은 (k, r) 커밋과 같은지(일치 수 기록).
- **D2(컨테이너 GPU)**: 시드마다 `T.Data("/data")`, `Store`, 철회 정책(E3-B와 같은 규칙: 학습 환자 5%, rng 20261025, 라운드 45 효력)을 만들고 정규화 통계를 1라운드 입력 범위에서 다시 계산한다(`events.jsonl`의 `NormStatsCommitted` 값과 일치 여부 기록). 표본 단위마다 `fl.audit.reexecute`에 원장 커밋으로 만든 commit dict를 넘겨 (1) 입력 재구성 루트 = rootS, (2) `ckpt_r{r-1}` 해시 = hStart, (3) 재실행 출력 해시 = hDelta를 검사한다. 결과 `results/d2_reexec.json`(단위별 결과·사유·시간), 로그 `logs/d2_*.log`(첫머리에 시각·git HEAD·이미지 digest·스크립트 SHA-256).
- 재실행은 단위마다 독립이며 순서를 바꾸어도 같아야 한다. 중단되면 끝난 단위까지 저장하고 이어서 돌린다(재시작 횟수 기록).

## 가설과 판정 기준

- **H-D1**: 표본 단위 206개(원장 추첨 기준) 전부가 (1)–(3)을 통과한다 → 실험 기록의 `training_L1.fail = 0`이 독립 재실행으로 확인된다. 하나라도 불일치면 그 단위(시드, k, r)와 사유(입력 루트/출발 상태/출력)를 그대로 보고하고, 원고의 "no false positives in 1,083"은 유형 B 부분에 대해 정정 대상이 된다.
- 부수 기록: 단위별 재실행 시간, 시드별 표본 수와 JSON 기록의 일치, 원장 커밋과 `events.jsonl`의 일치 수, 정규화 통계 일치.
- 이 실험은 **유형 B 206건만** 독립 확인한다. 유형 A 877건과 "오탐 0"의 나머지 부분은 여전히 실험 코드의 기록만 있다. 결과가 좋아도 원고 서술은 "유형 B 206건은 독립 재실행으로 확인, 유형 A 877건은 기록만"으로 한정한다.

## 원고 반영(실행 뒤, 승인 필요)

- Extended R11 "Not independently verifiable" 문장: L1 재실행 결과 중 유형 B 206건을 뺀다. 보충 S2의 "7, among them the outcomes of the L1 re-executions and hence the zero false positives" → 유형 A에 한정하는 표현으로(보충 5쪽 유지 확인). 본문은 꽉 차 있어 바꾸지 않는다(VI/IV-C의 1,083 서술은 그대로 참).
