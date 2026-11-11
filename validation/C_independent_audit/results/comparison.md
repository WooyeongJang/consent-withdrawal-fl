| 항목 | 원고 위치 | 주장 | 원고 값 | 독립 도출값 | 근거 | 판정 | 비고 |
|---|---|---|---|---|---|---|---|
| C-1 | 06_c3.tex Table III | T1 B4 탐지 | 20/20 | 20/20 | chain+data | 일치 | 정책 등록 5건 중 철회 요청이 원장에 없는 정보주체가 실행마다 1명, 그 환자가 로그의 T1 대상과 20/20 일치. 서명 요청 계정의 거래 수 0(20/20). 환자가 서명 요청을 보관한다는 전제는 구성상(서명 원문은 보존되지 않음) |
| C-1 | 06_c3.tex Table III | T2a B4 탐지 | 20/20 | 20/20 | chain+data | 일치 | 철회자 기록이 든 루트가 커밋되고(다시 계산한 루트와 일치) 그 단위가 UpdateRejected |
| C-1 | 06_c3.tex Table III | T3 B4 탐지 | 20/20 | 20/20 | chain+data | 일치 | σ = 0(낡은 스냅샷)이고 철회자 기록이 든 루트, UpdateRejected |
| C-1 | 06_c3.tex Table III | T10 B4 탐지 | 20/20 | 20/20 | chain+data | 일치 | 철회자 기록이 든 루트가 수락됨. 환자 rid의 소속 증명이 원장 루트로 검증되어 비소속 증명이 불가능(자기 검사 실패) 20/20 |
| C-1 | 06_c3.tex Table III | T2b B4 탐지 | 4/20 | 4/20 | chain+record | 일치 | 표본 포함 여부는 원장 seed로 다시 계산(독립). 불일치 결과 자체는 감사자가 쓴 L1 기록(passed=false, 사유 '출력 불일치') |
| C-1 | 06_c3.tex Table III | T2b 주입 사실(표본에 들지 않은 16회) | (주입 20회) | 원장으로 확인 불가 | none | 독립 검증 불가 | 커밋이 정직하므로 철회 기록을 실제로 썼다는 사실은 원장에 없다. 원시 실행 폴더가 삭제되어 재실행으로도 확인할 수 없다 |
| C-1 | 06_c3.tex Table III | T2b B6 탐지 | 4/20 | 4/20 | chain+record | 일치 |  |
| C-1 | 06_c3.tex | 12 of 13 유형을 모든 주입에서 탐지 | 12/13 | 12/13 | mixed | 일치 | T9b·T9c는 구성상 탐지로 계산에 넣음. T5·T6은 아래 행의 한계가 붙는다 |
| C-1 | 06_c3.tex | 주입 위치(유형·기관·라운드·환자)가 원장에서 식별되는가 | (원고 수치 없음) | T1 20/20, T2a 20/20, T2b 20/20, T3 20/20, T10 20/20 | chain+data | 일치 | 3,000개 단위 중 정직한 루트·σ=1·수락이 아닌 단위는 60개이고 전부 T2a·T3·T10. T2b는 남은 철회로만 식별(실제 사용 사실은 원장에 없음) |
| C-1 | 06_c3.tex | 정직한 학습 단위 재실행 1,083건 | 1083 | 1083 | chain | 일치 | 유형 A 877(표본 899 − 주입 단위 22) + 유형 B 206. 원장 seed로 다시 계산 |
| C-1 | 06_c3.tex | 정직한 재실행의 오탐 0건 | 0 | 유형 A 0 (기록 877건 모두 passed) | record | 독립 검증 불가 | 유형 A는 감사자가 쓴 원장 기록(877건 passed), 유형 B의 206건은 원장에 없고 실행 로그 필드뿐. 재실행에는 학습이 필요(유형 A는 원시 폴더 삭제) |
| C-1 | 06_c3.tex | 완료 기록이 515개 작업 전부에 기록됨 | 515 | 515 (JobCompleted), 515 (CompletionRecorded) | chain | 일치 |  |
| C-1 | 06_c3.tex | 주입 작업 중 배포 가능해진 것 없음 | 0/510 | 0/510 | chain | 일치 | 이벤트 순서대로 컨트랙트의 deployable 조건을 재연. 정직한 작업은 5/5 배포 가능 |
| C-1 | 06_c3.tex Table III | T4 B4 탐지 | 100/100 | 100/100 | chain+ckpt | 일치 | hOut = 원 학습 최종 모델(계보 중복 100/100) ≠ 저장된 U0 재실행 출력 → L1 불일치 100/100 |
| C-1 | 06_c3.tex Table III caption | T4 서로 다른 주입 모델 수 | 5 | 5 | chain | 일치 |  |
| C-1 | 06_c3.tex Table III | T5 B4 탐지 | 100/100 | 100/100 | record+stats | 일치 | 원장 기록: L1·L2 passed 100/100, L3 failed 100/100. 저장된 AUC·CI로 다시 분류하면 exceeds 100/100. 모델·부트스트랩 CI는 보존되지 않음 |
| C-1 | 06_c3.tex Table III caption | T5 서로 다른 주입 모델 수 | 25 | 25 | chain | 일치 |  |
| C-1 | 06_c3.tex Table III | T6 B4 탐지 — 폴더가 남은 5건 | 5/5 | 5/5 | chain+ckpt+events.jsonl | 일치 | 입력 루트가 기대 집합과 다름(불일치 단위 [150, 150, 150, 150, 150]), 재실행 해시는 같음(5/5) → L1 통과·L2 불일치 |
| C-1 | 06_c3.tex Table III | T6 B4 탐지 — 나머지 95건 | 95/95 | 원장 기록상 L2 failed 95/95 | record | 독립 검증 불가 | 주입마다 같은 폴더를 덮어써 시드별 마지막 주입의 증거만 남았다. 95건은 감사자가 쓴 원장 기록(L2 failed)이 유일한 근거 |
| C-1 | 06_c3.tex Table III caption | T6 서로 다른 주입 모델 수 | 100 | 100 | chain | 일치 |  |
| C-1 | 06_c3.tex Table III | T7 B4 탐지 | 100/100 | 100/100 | chain+ckpt | 일치 | hOut = 과거 라운드 모델(계보 중복 100/100) ≠ U0 재실행 출력 → L1 불일치 100/100 |
| C-1 | 06_c3.tex Table III caption | T7 서로 다른 주입 모델 수 | 84 | 84 | chain | 일치 |  |
| C-1 | 06_c3.tex Table III | T8 B4 탐지 | 100/100 | 100/100 | chain+ckpt | 일치 | 전달된 모델(로그 기록)이 원장의 deployed(정직한 U0, 5/5)와 다름. 최신 상태에서 deployable()이 false인 것은 42/100 — 아래 관찰 참조 |
| C-1 | 06_c3.tex Table III caption | T8 서로 다른 주입 모델 수 | 80 | 80 | log(전달 해시)+chain(계보) | 일치 |  |
| C-1 | 06_c3.tex Table III | T9a B4 탐지 | 35/35 | 35/35 | chain | 일치 | 라운드 ≥ 1000: 입력 3, 수락 2, RoundAggregated 없음(35/35). 지금 집계 주체 주소로 aggregate를 호출하면 'decisions'로 되돌려짐(35/35) |
| C-1 | 06_c3.tex Table III | T9b B4 탐지 | 35/35 | 35/35 (구성상) | construction | 독립 검증 불가 | 위조 커밋 ≠ 원장 루트는 정의상 참. 측정이 아님(원고도 구성상으로 표시) |
| C-1 | 06_c3.tex Table III | T9c B4 탐지 | 30/30 | 30/30 (구성상) | construction | 독립 검증 불가 | 원장 불변성에 기대는 구성상 탐지(원고도 구성상으로 표시) |
| C-1 | 06_c3.tex Table III | B6: T1·T2a·T3·T10·T4–T8 = 0, T9 = B4와 같음 | 0 / 35·35·30 | (정의에 따른 값) | definition | 독립 검증 불가 | 측정이 아니라 B6의 정의에서 나오는 값. 원고의 B6 정의(커밋된 입력으로 재실행, 작업은 재실행하지 않음, T9는 모두 인정)는 표 설명 한 문장뿐이라 다시 구현할 만큼 구체적이지 않음 |
| C-1 | 06_c3.tex | T5는 L1·L2를 100건 모두 통과하고 L3로만 탐지 | 100/100 | L1 100, L2 100, L3 failed 100 | record | 독립 검증 불가 | 원장의 검증 기록으로는 일치. L1·L2는 감사자의 기록뿐(FedOSD 산출물 미보존), L3는 저장 통계에서 exceeds 100/100 |
| C-2 | 06_c3.tex | T2b 표본 포함(p = 0.3) | 4/20 | 4/20 | chain | 일치 | Algorithm S3 2행을 원장 seed로 다시 계산 |
| C-2 | 06_c3.tex | T2b Wilson 95% CI | 0.08–0.42 | 0.08–0.42 | computed | 일치 |  |
| C-2 | results/final/E3.json p_grid (그림 E의 자료) | T2b 탐지 수, p = 0.1/0.3/0.5/1.0 | 1/4/11/20 | 1/4/11/20 | chain | 일치 |  |
| C-2 | results/final/E3.json p_grid | T10의 L1 표본 탐지 수, p = 0.1/0.3/0.5/1.0 | 4/9/11/20 | 4/9/11/20 | chain | 일치 |  |
| C-2 | 06_c3.tex | 자기 검사 없이 T10 탐지율 | 0.45 | 0.45 | chain+record | 일치 |  |
| C-2 | extended Table (tab:e3a) | 수락 검사+자기 검사 제거: T2a 0.25, T3 0.20 | 0.25 / 0.20 | 0.25 / 0.20 | chain+record | 일치 |  |
| C-2 | supplement Alg. S3 | 원장의 L1 기록 집합 = 다시 계산한 표본(p = 0.3) | (원고 수치 없음) | 20/20 실행, 기록 899건 전부 결속값 B(k,r) 또는 H(Δ)에 대응 | chain+data | 일치 | 결속값의 prog는 git 이력의 fl/*.py·이미지 digest·torch 판·스레드 수에서 다시 만들었고, 원장의 설정 커밋으로 확인(유형 A 20/20, 유형 B 5/5) |
| C-3 | supplement Table S7 | T11a L3 단독 | 0/20 | 0/20 | stats | 일치 | 저장된 90% CI에서 다시 분류 |
| C-3 | supplement Table S7 | T11a 최종 ε(0.65) | 16/20 | 16/20 | ckpt/stats | 일치 | 거리: 체크포인트에서 0/20, 나머지는 로그 수치 |
| C-3 | supplement Table S7 | T11a B4 | 20/20 | 20/20 | chain+ckpt | 일치 | hOut ≠ 저장된 U0 재실행 출력 → L1 불일치 |
| C-3 | supplement Table S7 | T11b L3 단독 | 0/20 | 0/20 | stats | 일치 | 저장된 90% CI에서 다시 분류 |
| C-3 | supplement Table S7 | T11b 최종 ε(0.65) | 0/20 | 0/20 | ckpt/stats | 일치 | 거리: 체크포인트에서 5/20, 나머지는 로그 수치 |
| C-3 | supplement Table S7 | T11b ε_r ≤ 1e-3 | 20/20 | 20/20 | stats | 일치 | 저장된 라운드 편차(b는 최종 거리)에서 다시 셈 |
| C-3 | supplement Table S7 | T11b ε_r = 1e-2 | 7/20 | 7/20 | stats | 일치 |  |
| C-3 | supplement Table S7 | T11b B4 | 20/20 | 20/20 | chain+ckpt | 일치 | hOut ≠ 저장된 U0 재실행 출력 → L1 불일치 |
| C-3 | supplement Table S7 | T11c L3 단독 | 20/20 | 20/20 | stats | 일치 | 저장된 90% CI에서 다시 분류 |
| C-3 | supplement Table S7 | T11c 최종 ε(0.65) | 16/20 | 16/20 | ckpt/stats | 일치 | 거리: 체크포인트에서 20/20, 나머지는 로그 수치 |
| C-3 | supplement Table S7 | T11c ε_r ≤ 1e-3 | 20/20 | 20/20 | stats | 일치 | 저장된 라운드 편차(b는 최종 거리)에서 다시 셈 |
| C-3 | supplement Table S7 | T11c ε_r = 1e-2 | 20/20 | 20/20 | stats | 일치 |  |
| C-3 | supplement Table S7 | T11c 제출 배치 재현 | 20/20 | 20/20 | stats | 일치 | 저장된 라운드 편차에서 다시 셈(재실행 폴더는 덮어써짐) |
| C-3 | supplement Table S7 | T11c B4 | 20/20 | 20/20 | chain+ckpt | 일치 | hOut ≠ 저장된 U0 재실행 출력 → L1 불일치 |
| C-3 | supplement Table S7 | T11d L3 단독 | 4/20 | 4/20 | stats | 일치 | 저장된 90% CI에서 다시 분류 |
| C-3 | supplement Table S7 | T11d 최종 ε(0.65) | 4/20 | 4/20 | ckpt/stats | 일치 | 거리: 체크포인트에서 20/20, 나머지는 로그 수치 |
| C-3 | supplement Table S7 | T11d ε_r ≤ 1e-3 | 20/20 | 20/20 | stats | 일치 | 저장된 라운드 편차(b는 최종 거리)에서 다시 셈 |
| C-3 | supplement Table S7 | T11d ε_r = 1e-2 | 20/20 | 20/20 | stats | 일치 |  |
| C-3 | supplement Table S7 | T11d 제출 배치 재현 | 0/20 | 0/20 | stats | 일치 | 저장된 라운드 편차에서 다시 셈(재실행 폴더는 덮어써짐) |
| C-3 | supplement Table S7 | T11d B4 | 20/20 | 20/20 | chain+ckpt | 일치 | hOut ≠ 저장된 U0 재실행 출력 → L1 불일치 |
| C-3 | supplement Table S7 | T11e L3 단독 | 0/20 | 0/20 | stats | 일치 | 저장된 90% CI에서 다시 분류 |
| C-3 | supplement Table S7 | T11e 최종 ε(0.65) | 6/20 | 6/20 | ckpt/stats | 일치 | 거리: 체크포인트에서 20/20, 나머지는 로그 수치 |
| C-3 | supplement Table S7 | T11e B4 | 20/20 | 20/20 | chain+ckpt | 일치 | hOut ≠ 저장된 U0 재실행 출력 → L1 불일치 |
| C-3 | supplement Table S7 | T11f L3 단독 | 20/20 | 20/20 | stats | 일치 | 저장된 90% CI에서 다시 분류 |
| C-3 | supplement Table S7 | T11f 최종 ε(0.65) | 16/20 | 16/20 | ckpt/stats | 일치 | 거리: 체크포인트에서 20/20, 나머지는 로그 수치 |
| C-3 | supplement Table S7 | T11f B4 | 20/20 | 20/20 | chain+ckpt | 일치 | hOut ≠ 저장된 U0 재실행 출력 → L1 불일치; f1 10건은 승인 목록에 없는 (방법, prog)이고 methodKind = 0(작업 생성 이벤트 없음). 실패한 거래 해시는 기록에 없음 |
| C-3 | supplement Table S7 caption | 시드 1의 P1–U0 상대 거리 | 0.646 | 0.646 | ckpt | 일치 |  |
| C-3 | supplement Table S7 caption | 최종 ε | 0.65 | 0.65 | stats(results/t11/t11_eps.json) | 일치 | 보정 실행의 산출물은 다시 계산하지 않음 |
| C-3 | 06_c3.tex | 전체 감사자가 T11 120건 모두 탐지 | 120/120 | 120/120 | chain+ckpt | 일치 |  |
| C-4 | (계보) | L1 근거: 원장의 라운드 모델 해시 = 저장된 체크포인트 해시 | (원고 수치 없음) | 250/250 | chain+ckpt | 일치 |  |
| C-4 | (재실행) | L1 근거: U0와 U0_audit(두 번의 저장된 실행)이 비트 단위로 같음 | (원고 수치 없음) | 최종 5/5, 51개 체크포인트 전부 5/5 | ckpt | 일치 | 새 재실행은 하지 않았다(학습 금지 조건). 두 실행 모두 원 실험이 만든 것 |
| C-4 | (L1 기록) | U0로 선언한 작업 415건의 L1: hOut = U0 재실행 출력인가 | (원고 수치 없음) | 독립 도출 = 원장 기록 320/320 | chain+ckpt | 일치 |  |
| C-4 | (L2) | 정규 라운드 입력 루트 = 다시 계산한 루트(유형 B) | (원고 수치 없음) | 750/750 | chain+data | 일치 |  |
| C-4 | (L2) | 정직한 U0 작업의 입력 루트 = 기대 집합의 루트 | (원고 수치 없음) | 5/5 (150단위씩) | events.jsonl+data | 일치 |  |
| C-4 | (L3) | E2 실행 로그 1,350개 L3 필드: 저장된 통계에서 다시 분류(실행 당시 규칙) | (로그와 일치해야 함) | 1345/1350 | stats | 불일치 | 다른 5건은 모두 U0(시드 1–5): 실행 로그는 자기 자신을 참조에 넣어 계산했고 원고는 자기 참조를 뺀 값으로 다시 계산해 보고한다. 원고 수치(U0 145/190)와는 일치 |
| C-4 | supplement Table S10 첫 행 | L3 통과 수(190개 실행): U0/U1/U2/U3/U4/U4s | 145/77/66/68/3/11 | 145/77/66/68/3/11 | stats | 일치 |  |
| C-4 | (L4) | 배포 이벤트: 원 최종 모델 철회, 정직한 U0 배포 | (원고 수치 없음) | 철회 5/5, 배포 = U0 5/5, 최신 deployed() = U0 5/5 | chain+ckpt | 일치 |  |
| C-5 | supplement Table S9 | 초기·중기 80개 실행의 근사 방법 통과 수(no_control) | 60/53/59/57/58 | 60/53/59/57/58 | stats | 일치 |  |
| C-5 | supplement Table S9 | 초기·중기 80개 실행의 근사 방법 통과 수(runtime) | 11/11/11/8/10 | 11/11/11/8/10 | stats | 일치 |  |
| C-5 | supplement Table S9 | 초기·중기 80개 실행의 근사 방법 통과 수(written) | 3/3/3/0/2 | 3/3/3/0/2 | stats | 일치 |  |
| C-5 | supplement Table S9 | 두 규칙의 통과 수가 다른 시나리오 | 20% mid, 10% mid, institution exit, three waves | E2-exit2, E2-rate10-mid, E2-rate20-mid, E2-seq3 | stats | 일치 |  |
| C-5 | supplement Table S9 | E2-rate20-mid: 실행 당시 → 원고 규칙 | 5/5/5/5/5 → 0/0/0/0/0 | 5/5/5/5/5 → 0/0/0/0/0 | stats | 일치 |  |
| C-5 | supplement Table S9 | E2-rate10-mid: 실행 당시 → 원고 규칙 | 4/4/4/3/4 → 1/1/1/0/1 | 4/4/4/3/4 → 1/1/1/0/1 | stats | 일치 |  |
| C-5 | supplement Table S9 | E2-exit2: 실행 당시 → 원고 규칙 | 9/0/4/3/9 → 6/0/2/2/6 | 9/0/4/3/9 → 6/0/2/2/6 | stats | 일치 |  |
| C-5 | supplement Table S9 | E2-seq3: 실행 당시 → 원고 규칙 | 7/7/7/1/3 → 6/6/6/1/2 | 7/7/7/1/3 → 6/6/6/1/2 | stats | 일치 |  |
| C-5 | supplement S7 | 실행 당시 코드(tag final-rerun-v1)는 P1의 하한 > 0만 봄 | l > 0 | return bool(p1_l3) and p1_l3["ci90"][0] > 0 | git | 일치 |  |
| C-5 | supplement S7 | 원고 규칙이 코드에 들어간 날 | 2026-11-01 | 2026-11-01 | git | 일치 |  |
| C-5 | supplement S7 | 결과 커밋 날짜 | 2026-10-30 | 2026-10-30 | git | 일치 |  |
| C-5 | supplement S7 | 실행 로그의 deploy 필드는 실행 당시 규칙을 따름 | (서술) | 940/940 필드가 실행 당시 규칙과 일치 | stats | 일치 |  |
| C-5 | supplement S7 | 20% mid 시드 1의 P1 CI | (0.006, 0.033) | (0.006, 0.033) | stats | 일치 |  |
| C-5 | supplement S7 | base 시드 1의 P1 CI | (0.043, 0.087) | (0.043, 0.087) | stats | 일치 |  |
| C-5 | supplement S7 | 80개 실행에서 근사 방법 통과 최대치: 실행 당시 11, 원고 규칙 3 | 11 / 3 | 11 / 3 | stats | 일치 |  |
| C-5 | supplement S7 | base 시나리오와 재학습(정확)은 두 규칙에서 같음 | 같음 | 같음 | stats | 일치 |  |
| C-6 | supplement S7 | L3 통과로 배포 가능이라 기록된 필드 수 | 269 | 269 | stats | 일치 |  |
| C-6 | supplement S7 | 원고 규칙에서 판정 불가로 바뀌는 필드 수 | 53 | 53 | stats | 일치 |  |
| C-6 | supplement S7 | 방법별 U1/U2/U3/U4/U4s | 12/9/11/9/12 | 12/9/11/9/12 | stats | 일치 |  |
| C-6 | supplement S7 | 바뀐 필드가 속한 시나리오 | 네 시나리오 | E2-exit2 9, E2-rate10-mid 15, E2-rate20-mid 25, E2-seq3 4 | stats | 일치 |  |
| C-6 | supplement S7 | 네 시나리오는 P1이 TOST를 통과하면서 하한 > 0인 실행이 있는 곳 | 네 시나리오 | E2-exit2, E2-rate10-mid, E2-rate20-mid, E2-seq3 | stats | 일치 |  |
| C-6 | supplement S7 | base 시나리오의 바뀐 필드 | 0 | 0 | stats | 일치 |  |
| C-6 | supplement S7 | 차례별(per-wave) 보조 로그 | 4 of 16 | 4 of 16 | stats | 일치 |  |
| C-6 | supplement S7 | K=5, lead I, Challenge 2021 보조 로그 | 0 of 45 | 0 of 45 | stats | 일치 |  |
| C-7 | supplement S7 본문, Table S8 마지막 행 | C3 원장 실행의 처리 작업은 모두 재학습(정확)으로 선언됨 | 515개 모두 exact | exact 415, approximate 100 | chain | 불일치 | T5 작업 100개는 원장에 U4-FedOSD(근사, kind = 2)로 생성되어 L3 기록이 배포 조건이었다. 결론(양성 대조 규칙 변경의 영향 없음)은 유지된다: T5의 L3는 100건 모두 failed(exceeds)이고 대조는 통과만 바꾼다 |
| C-7 | supplement Table S8 | Table S5–S6이 실행 로그와 다른 곳: 네 시나리오의 53개 필드 | 53개 필드 | deploy 필드 53개 + U0의 L3 구간 95개 필드(분류가 바뀌는 것 5개) | stats | 불일치 | 실행 당시 코드는 U0(시드 1–5)를 자기 자신이 든 참조 5개와 비교했고 self_ref 필드도 쓰지 않았다. 원고는 자기 참조를 뺀 값으로 다시 계산해 보고하며(Table S10 설명에 명시), 수치는 맞다. Table S8의 '실행 로그와 다른 점' 열에는 이 차이가 없다 |
| C-7 | supplement Table S8 | 본문 C2의 통과 수 57→0, 58→2, 53–60→3 | U4 57→0, U4s 58→2, U1·U2·U3 60·53·59→3 | U4 57→0, U4s 58→2, U1·U2·U3 60·53·59→3 | stats | 일치 |  |
| C-7 | supplement S7 | C3 원장 실행은 규칙 변경의 영향을 받지 않음(T5는 exceeds) | 영향 없음 | T5의 L3 기록 failed 100/100, 저장 통계로 exceeds 100/100 | record+stats | 일치 |  |
