# 실험 E 사전 등록 — 유형 A 학습 궤적의 결정적 재생성과 L1 독립 재실행

- 작성: 2026-11-10, 실행 전. SHA-256과 시각은 `PREREG.sha256`. 기준 저장소 HEAD `4e23cc6`(태그 `jbhi-v70` + 검토용 PDF 커밋).
- 사용자 지시(2026-11-10): "실험 E로 사전 등록해서 돌려줘" — 실험 D 뒤 "유형 A 877건은 실행 폴더가 없다"는 설명에 대한 후속. 유형 A(E3-A, 시드 1–20)의 체크포인트는 `scripts/run_e3a.sh` 14행이 시드마다 지웠으므로(`rm -rf "$RUNS/A_s${s}"`, 디스크 절약), 재실행하려면 궤적을 먼저 다시 만들어야 한다.
- 공통 원칙은 `../PREREG.md`와 같다. **이 실험은 새 FL 학습을 포함하지만 새 결과를 만들지 않는다**: 이미 원장에 기록된 20개 학습의 라운드별 모델을 결정적으로 다시 만들고, 라운드마다 원장의 집계 해시와 대조하며, 그 위에서 표본 단위를 재실행한다. 원고의 수치는 바뀔 수 없고(불일치가 나오면 그대로 보고), 원고는 승인 전 수정하지 않는다.

## 목적

Extended R11 "Not independently verifiable"에 남은 "유형 A의 L1 재실행 결과(정직 877 통과, 주입 22 불일치)와 그에 따른 오탐 0"의 유일한 근거는 실험 코드의 감사자가 원장에 쓴 `VerificationRecorded`다. 학습은 결정적(E0: 같은 시드·같은 이미지에서 해시 10/10)이고, 라운드별 집계 모델 해시 `hOut`이 원장 `RoundAggregated`에, 시작 모델이 `RoundOpened`에 있으므로, 궤적을 다시 만들어 원장과 대조하면 출발 상태가 원장에 결속된 상태에서 표본 단위를 재실행할 수 있다.

## 입력 (읽기 전용)

1. 원장 이벤트 캐시(`../C_independent_audit/results/raw/merged/`): 유형 A 컨트랙트 20개(`chain_index.json`의 type A, which contract)에서 `RoundOpened`(r = 1의 startModel), `RoundAggregated`(r = 1..50의 hOut), `AuditSampled`, `RoundInputCommitted`, `UpdateCommitted`, `UpdateRejected`(r ≤ 50).
2. 실험 C가 원장과 데이터 분할만으로 재식별한 주입 위치(`../C_independent_audit/results/typeA_verdicts.json`의 `runs[].found`: T1 pid, T2a/T2b/T3/T10의 (k, r, pid)) — 재생성에 쓰는 주입 배정(아래)과 대조하는 **독립 기준**.
3. 데이터 분할 캐시 `~/data/cache/ptbxl_k3`, 고정 이미지(`tools/toolchain.json`), `--gpus all --ipc=host`, `T.configure("gpu_det")`. `fl/` 핵심 코드는 유형 A 실행 시점(커밋 b0b0e87, 태그 `final-rerun-v1`과 `fl/` 내용 동일)과 비교해 학습·증거·감사 모듈이 같음을 `git diff --stat` 로 확인하고 로그에 적는다.

실험 기록 가운데 쓰지 **않는** 것: `results/final/e3/A_seed*_p0.3.json`의 판정 필드와 `l1_sampled_units`(비교 대상), 감사자의 `VerificationRecorded`.

## 절차

- **E0(호스트, `c_lib`)**: 시드마다 위 1·2를 모아 `results/e0_chain.json`에 쓴다: startModel, hOut[1..50], AuditSampled seed로 고른 표본 단위(p = 0.3)와 그 커밋, UpdateRejected 단위, 주입 위치. 확인: 시드별 표본 수가 `A_seed{s}_p0.3.json`의 `l1_sampled_units`(합 899)와 같은지.
- **E1(컨테이너 GPU, 시드마다, 재시작 가능)**:
  1. 주입 배정을 `fl/e3.py` `run_type_a` 54–64행과 같은 규칙(`random.Random(seed)`: 라운드 후보 7개 중 5개, 기관 순열, 기관의 학습 환자 중 미사용 환자 1명)으로 다시 만들고, **E0의 원장 기반 주입 위치와 같은지 확인한다**(T1은 pid, 나머지는 (k, r, pid)). 다르면 그 시드는 멈추고 보고한다.
  2. `fl.run.run_fl`을 원장 없이(`ledger=None`) 같은 정책(`Policy(withdraw={pid: r})`, T1 제외)·같은 `tamper`로 50라운드 실행해 `~/data/runs/validation/E/A_s{seed}/ckpt_r{0..50}.pt`를 만든다. 수락·거절 판정은 `run_fl`이 원장 유무와 무관하게 같은 비소속 검사로 내린다(`fl/run.py` 179–188행).
  3. **궤적 대조**: `ckpt_r0` 해시 = startModel, `round_hashes[r]` = hOut[r](r = 1..50). 거절 단위 집합 = 원장 `UpdateRejected` 집합.
  4. **재실행**: E0의 표본 단위마다 `fl.audit.reexecute`(원장 커밋으로 만든 commit dict, 재생성한 체크포인트, 재생성 실행의 정규화 통계·prog)를 적용한다. 단위가 주입 단위인지(E0의 주입 위치)와 함께 결과·사유를 기록한다.
  5. 결과 `results/e1_results.json`, 로그 `logs/e1_*.log`(시각·git HEAD·이미지 digest·스크립트 SHA-256·`fl/` 코드 해시).

## 가설과 판정 기준

- **H-E0** 주입 배정 재현: 20개 시드 모두 원장 기반 주입 위치와 일치. (불일치면 그 시드의 2–4단계를 하지 않는다.)
- **H-E1** 궤적 결속: 시작 모델 20/20, 집계 해시 1,000/1,000(20 × 50)이 원장과 비트 단위 일치; 거절 단위 집합 일치. 하나라도 다르면 그 시드의 표본 재실행은 "출발 상태가 원장에 결속되지 않음"으로 표시한다.
- **H-E2** 정직 표본: 표본 중 주입 단위가 아닌 단위(기대 877) 전부 통과.
- **H-E3** 주입 표본: 표본 중 주입 단위(기대 22) 전부 불일치 — T2a·T3·T10은 입력 재구성 루트 ≠ rootS, T2b는 출력 해시 ≠ hDelta.
- 결과가 어느 쪽이든 그대로 보고한다. H-E2가 깨지면 원고의 "no false positives in 1,083"이 정정 대상이 된다. H-E3가 깨지면(주입 단위가 통과) 탐지 수 4/20 등이 정정 대상이 된다.

## 비용과 산출물

- GPU 약 20 × 1.5분 + 재실행 899 × 0.5 s ≈ 40분. 디스크 약 2 GB(`~/data/runs/validation/E/`, 저장소 밖). 저장소에는 코드·로그·결과 JSON·보고만 둔다.

## 원고 반영(실행 뒤, 승인 필요)

- Extended R11 "Not independently verifiable"에서 L1 재실행 항목을 지우고 결과를 적는다; 보충 S2의 "7"과 그 괄호를 고친다(5쪽 유지 확인). 본문 IV-C의 1,083 서술은 불변.
