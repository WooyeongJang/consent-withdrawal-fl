# 실험 E — 유형 A 학습 궤적의 결정적 재생성과 L1 독립 재실행 보고

- 실행: 2026-11-10 11:53–12:33, 저장소 HEAD `4e23cc6`(태그 `jbhi-v70` + 검토용 PDF 커밋), 사전 등록 `PREREG.md`(SHA-256 `a384b279…`, `PREREG.sha256`에 기록 시각 11:53:09). 계획 이탈·실패 없음(`../DEVIATIONS.md` E절).
- 입력(읽기 전용): 원장 이벤트 캐시(`../C_independent_audit/results/raw/merged/`, 유형 A 컨트랙트 20개), 실험 C가 원장·데이터 분할만으로 재식별한 주입 위치(`../C_independent_audit/results/typeA_verdicts.json`), 데이터 분할 캐시 `~/data/cache/ptbxl_k3`. 실험 기록(`results/final/e3/A_seed*_p0.3.json`의 판정 필드, 감사자의 `VerificationRecorded`)은 비교 대상으로만.
- 환경: 고정 이미지(`nvcr.io/nvidia/pytorch@sha256:3becd068…`), `--gpus all --ipc=host`, `T.configure("gpu_det")`, torch 2.14.0a0+4fdf77b940.nv26.08, `fl/` 코드 해시 `f15b…`. `fl/` 핵심 모듈은 유형 A 실행 시점(커밋 b0b0e87)과 차이 없음(`git diff --stat b0b0e87 HEAD -- fl/{train,run,audit,evidence,model,consent}.py` 출력 없음).

## 1. 결론

사전 등록한 네 가설이 모두 성립했다.

| 가설 | 기준 | 결과 |
|---|---|---|
| H-E0 주입 배정 재현 | 시드 20개의 배정(`random.Random(seed)` 규칙)이 원장 기반 주입 위치와 일치 | **20/20** |
| H-E1 궤적 결속 | 시작 모델 = `RoundOpened.startModel`, 라운드별 집계 해시 = `RoundAggregated.hOut`, 거절 단위 집합 = `UpdateRejected` | 시작 **20/20**, 집계 해시 **1,000/1,000**, 거절 집합 **20/20**(시드마다 2개: T2a·T3) |
| H-E2 정직 표본 | 표본 중 주입 단위가 아닌 단위 전부 통과 | **877/877** |
| H-E3 주입 표본 | 표본 중 주입 단위 전부 불일치 | **22/22** — T2a 5·T3 4·T10 9는 "입력 재구성 실패"(커밋 루트 ≠ 정책대로 재구성한 입력), T2b 4는 "출력 불일치"(커밋은 정직, 계산이 다름) |

뜻하는 것:
- 원고 IV-C의 "no false positives in 1,083 honest re-executions"가 **전부 독립 재실행으로 확인**됐다: 유형 A 877(이 실험) + 유형 B 206(실험 D). Extended R11의 "독립 검증 불가" 7행 중 L1 재실행 행이 지워진다(남는 6행: T2b 16회의 실제 사용, T5의 L1·L2, T6 95건, T9b, T9c, B6 열).
- T2b 표본 수 4는 원고의 T2b 탐지 4/20과 같은 수이고, 네 건 모두 재실행이 출력 불일치를 냈다 — "T2b는 L1 표본 추첨으로만 잡힌다"는 서술의 독립 확인이다.
- 재생성이 원장과 1,000/1,000 일치한 것은 E0의 재현성 결론(같은 이미지·장치·결정적 설정에서 비트 동일)이 20개 50라운드 학습 전체에서 성립함을 뜻한다. 다른 환경에서의 재현은 보이지 않는다.

뜻하지 않는 것:
- 제3자 재현이 아니다(같은 연구 환경). 독립인 것은 입력(원장 추첨·커밋·집계 해시)과 절차(실험의 감사 기록을 쓰지 않음)다.
- 주입 배정은 실험 코드의 규칙으로 다시 만들었다(실험 C가 원장·데이터만으로 재식별한 위치와 20/20 일치하므로, 배정 규칙은 독립 검사를 통과한 입력이다).
- 주입 단위 22개의 "불일치"는 L1 재실행이 그 단위를 잡는다는 뜻이지, 표본 밖의 주입(T2b 16회)이 잡힌다는 뜻이 아니다.

## 2. 결과

| 시드 | 표본 | 정직 통과 | 주입 불일치 | 학습(s) |
|---|---|---|---|---|
| 1 | 38 | 37/37 | 1/1 (T10) | 87 |
| 2 | 59 | 57/57 | 2/2 (T10, T2b) | 86 |
| 3 | 36 | 36/36 | 0/0 | 87 |
| 4 | 51 | 50/50 | 1/1 (T10) | 87 |
| 5 | 50 | 49/49 | 1/1 (T2a) | 87 |
| 6 | 52 | 50/50 | 2/2 (T3, T2a) | 87 |
| 7 | 41 | 40/40 | 1/1 (T10) | 87 |
| 8 | 54 | 53/53 | 1/1 (T10) | 86 |
| 9 | 43 | 43/43 | 0/0 | 86 |
| 10 | 42 | 40/40 | 2/2 (T10, T2b) | 87 |
| 11 | 48 | 47/47 | 1/1 (T10) | 87 |
| 12 | 38 | 38/38 | 0/0 | 87 |
| 13 | 37 | 35/35 | 2/2 (T3, T2a) | 87 |
| 14 | 51 | 50/50 | 1/1 (T10) | 87 |
| 15 | 33 | 33/33 | 0/0 | 87 |
| 16 | 41 | 40/40 | 1/1 (T2b) | 87 |
| 17 | 52 | 50/50 | 2/2 (T10, T2a) | 87 |
| 18 | 43 | 42/42 | 1/1 (T3) | 87 |
| 19 | 39 | 39/39 | 0/0 | 88 |
| 20 | 51 | 48/48 | 3/3 (T2b, T3, T2a) | 87 |
| 합 | **899** | **877/877** | **22/22** | 1,738 |

- 모든 시드에서 시작 모델·집계 해시 50/50·거절 단위 2개가 원장과 일치. 표본 수는 시드마다 `A_seed{s}_p0.3.json`의 `l1_sampled_units`와 같다(합 899).
- 재실행 시간 단위당 중앙값 0.51 s(최대 0.78 s), 합계 511 s; 학습 1,738 s(시드당 86–88 s). 전체 약 40분, 재시작 0회. 재생성 체크포인트 1.9 GB(`~/data/runs/validation/E/A_s{1..20}/`, 저장소 밖).

## 3. 원고와의 관계 (승인 뒤 반영)

- Extended R11 "Not independently verifiable": L1 재실행 항목을 지우고, 유형 A 877·유형 B 206 전부가 원장 추첨·커밋에서 독립 재실행으로 확인됐음을 적는다(궤적 1,000/1,000 포함).
- 보충 S2: "7, among them the outcomes of the L1 re-executions in the 20 type-A runs (…)"를 "6: …(여섯 항목 열거)"로 — 11-10 구조 보고의 S-4 문장이 이 결과로 확정된다.
- 본문 IV-C "no false positives in 1,083 honest re-executions"는 그대로 참이며, 여유(27 pt)가 허락하면 "(all re-executed independently afterwards)"를 붙일 수 있다 — 사본 컴파일 필요.

## 4. 실행 기록

| 단계 | 스크립트 | 로그 | 결과 |
|---|---|---|---|
| E0 원장 수집(호스트) | `code/e0_chain.py` | `logs/e0_chain_20261110T115309.log` | `results/e0_chain.json` |
| E1 재생성·대조·재실행(컨테이너 GPU) | `code/e1_regen_reexec.py` | `logs/e1_regen_reexec_20261110T115320.log` | `results/e1_results.json` |

재현: 저장소 루트에서

    python3 validation/E_typeA_regen/code/e0_chain.py
    IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$IMG" --ipc=host \
      -v "$PWD":/w:ro -v ~/data/cache/ptbxl_k3:/data:ro -v ~/data/runs/validation/E:/runs -v "$PWD/validation/E_typeA_regen/results":/out -w /w "$IMG" \
      python /w/validation/E_typeA_regen/code/e1_regen_reexec.py --chain /w/validation/E_typeA_regen/results/e0_chain.json --out /out/e1_results.json

E0는 실험 C의 원장 이벤트 캐시(저장소 밖 51 MB; `c1_fetch_chain.py`로 받음)와 `typeA_verdicts.json`을 읽는다. E1은 PTB-XL 캐시와 GPU가 필요하다(약 40분).
