# 실험 C — 독립 감사 재현 보고

- 실행: 2026-11-09, 저장소 HEAD `4649db5`, 사전 등록 `validation/PREREG.md`(SHA-256 `79af83c0…`)의 "실험 C".
- 새 검증기는 `fl/audit.py`, `fl/e3.py`, `fl/e3b.py`, `fl/l3.py`, `scripts/summarize_*.py`, `scripts/l3_*.py`, `scripts/recount_l3_deploy.py`, `reports/audit/*.py`를 import하거나 복사하지 않았다. 표준 라이브러리만 쓴다(체크포인트 해시만 고정된 NGC 컨테이너의 torch).
- 원장에는 읽기 호출만 했다(거래 없음). 학습과 재실행은 하지 않았다.

## 1. 결론

비교한 101개 행 가운데 **일치 91, 불일치 3, 독립 검증 불가 7**이다(행 단위 전체 표: `results/comparison.md`).

- C3의 탐지 수(Table III), T2b의 4/20, T11 표(보충 Table S7), 양성 대조 규칙 비교(Table S9), 53/269개 필드는 모두 원고 값과 같게 다시 도출됐다.
- 불일치 3행의 원인은 두 가지이고, 둘 다 수치가 아니라 서술·출처 표기의 문제다(3절).
- 독립 검증 불가 7행은 근거가 보존되지 않았거나, 실험 코드의 감사자가 쓴 기록이 유일한 근거이거나, 측정이 아닌 값이다(4절).

원고 서술보다 강하게 확인된 점이 하나 있다. 유형 A의 주입 위치는 실험 코드의 기록 없이 원장과 데이터 분할만으로 다시 식별됐다(T1·T2a·T3·T10·T2b 각 20/20, 3,000개 단위 중 정직하지 않은 커밋 60개가 전부 주입 단위). 사전 등록에서는 주입 위치를 "실험 코드의 기록이 유일한 근거"로 적었는데, T2b의 실제 사용 사실만 그렇다.

## 2. 비교표(요약)

근거 표기: chain = 원장 이벤트와 최신 상태 조회, data = 기관 분할 캐시에서 다시 계산한 머클 루트, ckpt = 저장된 체크포인트의 해시, stats = 실행 로그에 저장된 수치에서 다시 계산, record = 실험 코드의 감사자가 원장에 쓴 검증 기록(독립 재실행 아님).

| 항목 | 주장(원고 위치) | 원고 값 | 독립 도출값 | 근거 | 판정 |
|---|---|---|---|---|---|
| C-1 | T1 탐지(Table III) | 20/20 | 20/20 | chain+data | 일치 |
| C-1 | T2a 탐지 | 20/20 | 20/20 | chain+data | 일치 |
| C-1 | T3 탐지 | 20/20 | 20/20 | chain+data | 일치 |
| C-1 | T10 탐지 | 20/20 | 20/20 | chain+data | 일치 |
| C-1 | T2b 탐지(B4, B6) | 4/20, 4/20 | 4/20, 4/20 | chain+record | 일치 |
| C-1 | T2b 주입 사실(표본 밖 16회) | 주입 20회 | 확인 불가 | 없음 | 독립 검증 불가 |
| C-1 | T4 탐지, 서로 다른 모델 수 | 100/100, 5 | 100/100, 5 | chain+ckpt | 일치 |
| C-1 | T5 탐지, 서로 다른 모델 수 | 100/100, 25 | 100/100, 25 | record+stats | 일치 |
| C-1 | T5는 L1·L2 통과, L3로만 탐지 | 100/100 | 기록상 100/100 | record | 독립 검증 불가 |
| C-1 | T6 탐지 — 폴더가 남은 5건 | 5/5 | 5/5 | chain+ckpt+events.jsonl | 일치 |
| C-1 | T6 탐지 — 나머지 95건 | 95/95 | 기록상 95/95 | record | 독립 검증 불가 |
| C-1 | T7 탐지, 서로 다른 모델 수 | 100/100, 84 | 100/100, 84 | chain+ckpt | 일치 |
| C-1 | T8 탐지, 서로 다른 모델 수 | 100/100, 80 | 100/100, 80 | chain+ckpt | 일치 |
| C-1 | T9a 탐지 | 35/35 | 35/35 | chain | 일치 |
| C-1 | T9b, T9c 탐지 | 35/35, 30/30 | 구성상 | 구성 | 독립 검증 불가(2행) |
| C-1 | B6 열의 0과 T9 인정 | 0, 35·35·30 | 정의에 따른 값 | 정의 | 독립 검증 불가 |
| C-1 | 완료 기록 515건, 주입 작업 배포 가능 0건 | 515, 0/510 | 515(두 컨트랙트), 0/510 | chain | 일치 |
| C-1 | 정직한 재실행 1,083건 | 1,083 | 877 + 206 = 1,083 | chain | 일치 |
| C-1 | 그 재실행의 오탐 0건 | 0 | 기록상 0 | record | 독립 검증 불가 |
| C-2 | T2b 표본 포함(p = 0.3), Wilson CI | 4/20, 0.08–0.42 | 4/20, 0.08–0.42 | chain | 일치 |
| C-2 | T2b의 p별 탐지 수(0.1/0.3/0.5/1) | 1/4/11/20 | 1/4/11/20 | chain | 일치 |
| C-2 | T10의 L1 표본 탐지 수, 자기 검사 없는 탐지율 | 4/9/11/20, 0.45 | 4/9/11/20, 0.45 | chain | 일치 |
| C-2 | 수락 검사+자기 검사 제거 시 T2a, T3 | 0.25, 0.20 | 0.25, 0.20 | chain+record | 일치 |
| C-3 | 보충 Table S7의 26개 칸(L3 단독, 최종 ε, ε_r, 제출 배치, B4) | 표 값 | 전부 같음 | stats, ckpt, chain | 일치(26행) |
| C-3 | P1–U0 상대 거리(시드 1), 최종 ε | 0.646, 0.65 | 0.646, 0.65 | ckpt, stats | 일치 |
| C-4 | L1: 원장의 라운드 모델 해시 = 저장된 체크포인트 | — | 250/250 | chain+ckpt | 일치 |
| C-4 | L1: U0로 선언한 작업 415건의 판정 | — | 원장 기록과 415/415 같음 | chain+ckpt | 일치 |
| C-4 | L2: 정규 라운드 입력 루트, 정직한 U0의 입력 루트 | — | 750/750, 5/5 | chain+data | 일치 |
| C-4 | L3: Table S10 첫 행(U0/U1/U2/U3/U4/U4s) | 145/77/66/68/3/11 | 145/77/66/68/3/11 | stats | 일치 |
| C-4 | L3: 실행 로그의 분류 필드 1,350개 | 로그와 같아야 함 | 1,345개 같음 | stats | 불일치 |
| C-4 | L4: 원 모델 철회, 정직한 U0 배포 | — | 5/5 | chain+ckpt | 일치 |
| C-5 | Table S9 위쪽: 대조 없음 / 실행 당시 / 원고 규칙 | 60·53·59·57·58 / 11·11·11·8·10 / 3·3·3·0·2 | 같음 | stats | 일치 |
| C-5 | Table S9 아래쪽: 네 시나리오의 전·후 | 표 값 | 같음 | stats | 일치(5행) |
| C-5 | 실행 당시 코드는 P1 하한 > 0만 봄, 변경일 11-01, 결과 커밋 10-30 | 서술 | 같음 | git | 일치 |
| C-5 | 예로 든 P1의 CI 두 개 | (0.006, 0.033), (0.043, 0.087) | 같음 | stats | 일치 |
| C-6 | 53/269, 방법별 12/9/11/9/12, 네 시나리오, base 0 | 서술 | 같음 | stats | 일치 |
| C-6 | 보조 로그: 차례별 4/16, 나머지 0/45 | 서술 | 4/16, 0/45 | stats | 일치 |
| C-7 | C3 원장 실행의 작업은 모두 정확 방법으로 선언(S7절, Table S8) | 515개 모두 exact | exact 415, approximate 100 | chain | 불일치 |
| C-7 | Table S8: 실행 로그와 다른 곳은 53개 필드 | 53개 | 53개 + U0 구간 95개(분류 변경 5개) | stats | 불일치 |
| C-7 | Table S8: 본문 C2의 57→0, 58→2, 53–60→3 | 서술 | 같음 | stats | 일치 |

## 3. 불일치와 원인

**(1) "C3 원장 실행의 처리 작업은 모두 재학습(정확)으로 선언됐다"는 서술이 원장과 다르다.**
- 원고: 보충 S7절 "every remediation job there declared retraining (exact)", Table S8 마지막 행 "all jobs exact".
- 원장: 515개 작업 중 T5의 100개는 `U4-FedOSD(…)`(kind = 2, 근사)로 생성됐고 L3 기록이 배포 조건이었다. 나머지 415개가 U0(kind = 1)이다.
- 원인: T5는 설계상 "불완전한 FedOSD를 완료로 선언"하는 주입이라 근사 방법으로 등록된다. 서술이 T5를 빠뜨렸다.
- 영향: 그 문장의 결론(양성 대조 규칙 변경이 C3 원장 실행에 영향이 없다)은 유지된다. T5의 L3 기록은 100건 모두 불통과이고(저장 통계로도 exceeds 100/100), 대조는 통과만 바꾼다. 문장만 고치면 된다(예: "every job except T5 declared retraining; T5 declared FedOSD (approximate) and never passed").

**(2) 실행 로그의 U0 L3 필드는 자기 참조를 빼지 않은 값이고, Table S8은 이 차이를 적지 않았다.** 불일치 2행(C-4의 로그 필드, C-7의 Table S8)이 같은 원인이다.
- 실행 당시 코드(`final-rerun-v1`)는 U0 시드 1–5를 자기 자신이 든 참조 5개와 비교했고 `self_ref` 필드도 쓰지 않았다. 그래서 실행 로그의 U0 구간 95개가 원고 방식(자기 참조 제외)과 다르고, 분류가 5개 달라진다(E2-F65 s3, E2-mid s1, E2-rate01-early s3, E2-rate01-mid s4, E2-rate01 s4: 로그는 통과 1·판정 불가 4, 다시 계산하면 5개 모두 기준 초과).
- 원고 수치는 맞다. 자기 참조를 빼고 다시 계산하면 Table S10 첫 행(U0 145/190)이 그대로 나오고, Table S10 설명도 "U0 is tested against the other reference models"라고 적는다.
- 빠진 것은 출처 표기다. Table S8의 "Differs from run logs" 열은 Tables S5–S6에 대해 "53 fields in four scenarios"만 적는다. U0 행도 실행 로그와 다르다는 점(구간 95개, 분류 5개)을 한 구절 더하면 된다.
- 실험 A의 A0도 같은 사실을 따로 확인했다(`validation/DEVIATIONS.md` A절).

## 4. 독립 검증이 안 된 부분과 이유

| 대상 | 이유 |
|---|---|
| T2b의 주입 사실(표본에 들지 않은 16회) | 커밋이 정직하므로 철회 기록을 실제로 썼다는 사실은 원장에 없다. 유형 A의 원시 실행 폴더(`~/data/runs/final_e3`)는 비어 있어 재실행으로도 확인할 수 없다 |
| L1 재실행의 결과(유형 A 899건: 통과 877, 불일치 22 / 유형 B 학습 단위 206건) | 표본에 든 단위는 원장 seed로 다시 계산했지만(독립), 통과·불일치 결과는 실험 코드의 감사자가 쓴 원장 기록(유형 A)이나 로그 필드(유형 B)뿐이다. 다시 확인하려면 학습 단위를 재실행해야 한다. 유형 B는 체크포인트가 남아 있어 GPU로 가능하고, 유형 A는 불가능하다 |
| T5의 L1·L2 | FedOSD 산출물과 입력 커밋이 보존되지 않았다. L3는 저장된 AUC·90% CI로만 다시 분류했다(부트스트랩 CI도 저장되지 않아 표준오차는 다시 계산하지 못함) |
| T6의 95건 | 주입마다 같은 폴더(`T6`, `T6_audit`)를 덮어써 시드별 마지막 주입만 남았다 |
| T9b, T9c | 구성상 탐지다(원고도 그렇게 표시). 측정할 것이 없다 |
| B6 열의 0과 T9 인정 | 측정이 아니라 B6의 정의에서 나오는 값이다. 원고의 B6 정의는 Table III 설명 한 문장뿐이라, "커밋된 입력으로 재실행한다", "처리 작업은 재실행하지 않는다"는 규칙을 원고만 보고는 다시 구현할 수 없다 |
| T1의 "환자가 보관한 서명 요청" | 서명 계정 주소만 로그에 있고 서명 원문은 보존되지 않았다. 원장 쪽 사실(그 정보주체의 철회 요청 없음, 계정 거래 수 0)만 확인했다 |
| T11a 20건, T11b 15건의 모델 | 보존되지 않아 L3와 거리는 저장된 수치로만 다시 셌다. L1 판정(hOut ≠ U0 재실행 출력)은 원장의 hOut으로 독립 확인했다 |
| T11c·d의 라운드 편차 | 재실행 폴더(`T11_roundcheck`)가 덮어써져 저장된 편차 값으로만 다시 셌다 |
| T11f1 10건의 거부된 거래 | 실패한 `createJob` 거래의 해시가 기록에 없다. 승인 목록(MethodApproved)에 그 (방법, prog)가 없고 `methodKind` = 0이라는 것으로 대신 확인했다 |
| 작업 시점의 `deployable` 상태 | 공개 RPC가 과거 상태 조회를 거부한다(`missing trie node`). 이벤트를 컨트랙트 조건에 넣어 재연했다 |
| U0의 "재실행" | 저장된 두 실행(U0, U0_audit)이 51개 체크포인트 모두 같다는 것만 확인했다. 두 실행 모두 원 실험이 만든 것이고, 새 재실행은 하지 않았다(시드 5개 약 7–10분 GPU) |

검증기를 만든 주체가 원 실험과 같은 연구 환경에 있으므로 제3자 재현은 아니다.

## 5. 덧붙이는 관찰

- **코드 판 결속.** 원장의 설정 커밋 `H(config, 기준 시드, 시드 규칙, prog, 표준화 범위 루트)`를 git 이력에서 다시 만든 prog로 맞춰 보니 25회 모두 맞았다. 유형 A 20회는 커밋 `b0b0e87`(`fl/` 내용이 태그 `final-rerun-v1`과 같음), 유형 B 5회는 `f9d1154`(태그 `final-rerun-v1.2`, `fl/e3b.py`의 import 수정)의 코드다. 설정 커밋은 25회 모두 첫 라운드와 첫 철회보다 앞에 있다.
- **배포된 코드.** 25개 ConsentLedger와 25개 CompletionRecord의 런타임 코드는 각각 서로 같고, 현재 소스를 다시 컴파일한 것과 메타데이터 꼬리(53바이트)를 빼면 같다.
- **수락 버전 기록.** 유형 A 20회의 UpdateAccepted 2,960건 모두에서 `withdrawalVersion`이 그 로그보다 앞선 효력 철회 수와 같다(v3.1이 포함 시점의 버전을 기록한다는 서술의 확인).
- **T8과 공개된 v3.1 공백.** 단말에 전달된 100개 모델 가운데 58개는 지금 원장에서 `deployable()`이 true를 돌려준다(정직한 작업이 배포되어 미해소 철회 수가 0이 된 뒤 정규 라운드 모델이 다시 허용됨). T8의 탐지는 `deployed`와의 비교로 성립하며(100/100), `deployable`만 보면 성립하지 않는다. Extended R2가 공개한 "scope after resolution" 공백과 같은 현상이고 v3.2가 고친 항목이다. 원고와 모순되지 않지만, T8의 탐지 근거를 "`deployed`와의 대조"로 적는 편이 정확하다.
- **결과 요약 파일의 이름.** `results/final/E3.json`의 `A_l1_sampled_honest_units` = 899는 주입 단위 22개가 든 전체 표본 수다. 정직한 단위는 877이다. 원고의 1,083(= 877 + 206)은 맞다.

## 6. 재현 절차

`validation/C_independent_audit/code/`에서 실행한다. 입력: `results/final/e3/*.json`(주소와 비교용), `~/data/ledger/state/deployments.jsonl`(배포 블록), `~/data/cache/ptbxl_k3/meta.csv`, `~/data/runs/final_e3b/B_s1..5/`, `results/final/e2*/`, `results/late/late_methods.json`, `results/t11/*.json`, git 이력.

    cd validation/C_independent_audit/code
    python3 c1_fetch_chain.py          # 50개 컨트랙트의 이벤트 33,609 + 4,320건을 읽어 results/raw/에 저장(약 1.5분, 다시 실행하면 캐시 사용)
    DIG=$(python3 -c "import json;print(json.load(open('../../../tools/toolchain.json'))['training_image']['repo_digest'])")
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e IMAGE_DIGEST="$DIG" \
      -v ~/data/runs/final_e3b:/runs:ro -v "$PWD":/code:ro -v "$PWD/../results":/out "$DIG" \
      python /code/c2_hash_ckpts.py    # 체크포인트 해시와 거리(CPU, 약 1분) → results/ckpt_hashes.json
    python3 c3_verify_typeA.py         # 유형 A(약 20초) → results/typeA_verdicts.json
    python3 c3_verify_typeB.py         # 유형 B(첫 실행 약 1분, 읽기 호출 135건) → results/typeB_verdicts.json
    python3 c4_verify_l3.py            # L3 규칙·배포 필드(수 초) → results/l3_rules.json
    python3 c5_compare.py              # 비교표 → results/comparison.json, results/comparison.md

- 실행 기록: `logs/`(시각, git HEAD, 스크립트 SHA-256, 환경). 실패한 실행과 계획과 달라진 점은 `validation/DEVIATIONS.md`의 C절.
- 원장 응답 원문은 `results/raw/`(51 MB)에 있다. 저장소에 올릴지는 정하지 않았다.

## 7. 환경 의존성

- Kairos 공개 RPC(`https://public-en-kairos.node.kaia.io`): 과거 로그 조회는 된다(5,000블록 단위로 읽음). 과거 상태 조회는 안 된다. 노드가 로그를 정리하면 C-1·C-2·C-3은 `results/raw/`의 캐시로만 재현된다.
- 최신 상태 조회(T8, T9a, T11f1)는 실험 뒤 컨트랙트 상태가 바뀌지 않았다는 전제에 선다. 역할 키를 가진 쪽만 상태를 바꿀 수 있다.
- `tools/bin/solc` 0.8.38(이벤트 ABI), NGC PyTorch 26.08 이미지(digest 고정, 체크포인트 읽기), Python 3.12 표준 라이브러리.
- `~/data/runs/final_e3b/`(8.7 GB)와 `~/data/cache/ptbxl_k3/meta.csv`는 저장소 밖에 있다. 공개 저장소만으로는 C-1의 chain+data·chain+ckpt 행을 재현할 수 없다(PTB-XL에서 캐시를 다시 만들고 실행 폴더를 받아야 함).
- prog 재구성은 git 이력(2026-10-24–10-31의 `fl/` 커밋)에 의존한다.

## 8. 정의의 출처

- 원고에서: 탐지 근거(Table III), Algorithm S3의 표본 규칙, Algorithm S5의 L3 분류와 양성 대조, 배포 조건(본문 III-D), 결속값 B의 필드.
- 컨트랙트 소스에서: 이벤트 스키마, `deployable`·`aggregate`·`createJob`의 조건.
- 코드를 읽고 옮긴 것(원고에 바이트 수준 정의가 없음): 해시 `H`의 구분자, rid, 정렬 머클 트리의 접두사(`fl/evidence.py`), 모델 해시(`fl/train.py`), 상대 거리(`fl/t11.py`), prog의 구성(`fl/run.py`), 표본 추출의 문자열 부호화(`fl/e3.py`의 한 줄).
- se_a는 로그에 저장된 95% 부트스트랩 CI의 폭 / (2·1.96)으로 두었다.
