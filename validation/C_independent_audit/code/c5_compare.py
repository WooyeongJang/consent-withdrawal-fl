"""원고 수치와 독립 도출값의 비교표를 만든다(results/comparison.json, results/comparison.md).

원고 값은 jbhi/sections/06_c3.tex, supplement_body.tex(S7절, Table S7–S10)에서 손으로 옮겼다(각 행의 where).
판정: 일치 / 불일치 / 독립 검증 불가. evidence는 독립 도출값이 기대는 근거의 수준이다.
  chain   원장 이벤트·최신 상태 조회만으로 도출
  chain+data  원장 + 데이터 분할(meta.csv)에서 다시 계산한 루트
  chain+ckpt  원장 + 저장된 체크포인트 해시
  stats   실행 로그 JSON에 저장된 수치(AUC, CI, 거리)에서 다시 계산
  record  실험 코드의 감사자가 원장에 쓴 검증 기록(passed)을 센 것 — 독립 재실행이 아님
"""
import json
import math

import c_lib as L

R = L.EXP / "results"
A = json.load(open(R / "typeA_verdicts.json"))["aggregate"]
Bfull = json.load(open(R / "typeB_verdicts.json"))
B = Bfull["aggregate"]
P = B["per_type"]
T = B["t11_table"]
E = json.load(open(R / "l3_rules.json"))
rows = []


def add(item, where, claim, paper, mine, evidence, verdict=None, note=""):
    if verdict is None:
        verdict = "일치" if paper == mine else "불일치"
    rows.append({"item": item, "where": where, "claim": claim, "paper": paper, "derived": mine, "evidence": evidence,
                 "verdict": verdict, "note": note})


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 2), round(c + h, 2)]


C3 = "06_c3.tex Table III"
# ---------------- C-1 유형 A ----------------
add("C-1", C3, "T1 B4 탐지", "20/20", f"{A['B4']['T1']}/20", "chain+data",
    note="정책 등록 5건 중 철회 요청이 원장에 없는 정보주체가 실행마다 1명, 그 환자가 로그의 T1 대상과 20/20 일치. 서명 요청 계정의 거래 수 0(20/20). "
         "환자가 서명 요청을 보관한다는 전제는 구성상(서명 원문은 보존되지 않음)")
add("C-1", C3, "T2a B4 탐지", "20/20", f"{A['B4']['T2a']}/20", "chain+data",
    note="철회자 기록이 든 루트가 커밋되고(다시 계산한 루트와 일치) 그 단위가 UpdateRejected")
add("C-1", C3, "T3 B4 탐지", "20/20", f"{A['B4']['T3']}/20", "chain+data", note="σ = 0(낡은 스냅샷)이고 철회자 기록이 든 루트, UpdateRejected")
add("C-1", C3, "T10 B4 탐지", "20/20", f"{A['B4']['T10']}/20", "chain+data",
    note="철회자 기록이 든 루트가 수락됨. 환자 rid의 소속 증명이 원장 루트로 검증되어 비소속 증명이 불가능(자기 검사 실패) 20/20")
add("C-1", C3, "T2b B4 탐지", "4/20", f"{A['B4']['T2b']}/20", "chain+record",
    note="표본 포함 여부는 원장 seed로 다시 계산(독립). 불일치 결과 자체는 감사자가 쓴 L1 기록(passed=false, 사유 '출력 불일치')")
add("C-1", C3, "T2b 주입 사실(표본에 들지 않은 16회)", "(주입 20회)", "원장으로 확인 불가", "none", "독립 검증 불가",
    "커밋이 정직하므로 철회 기록을 실제로 썼다는 사실은 원장에 없다. 원시 실행 폴더가 삭제되어 재실행으로도 확인할 수 없다")
add("C-1", C3, "T2b B6 탐지", "4/20", f"{A['B6_T2b']}/20", "chain+record")
add("C-1", "06_c3.tex", "12 of 13 유형을 모든 주입에서 탐지", "12/13",
    f"{sum(1 for v in [A['B4']['T1'], A['B4']['T2a'], A['B4']['T3'], A['B4']['T10']] if v == 20) + sum(1 for t in ('T4', 'T5', 'T6', 'T7') if P[t]['not_deployable_replay'] == 100) + (B['T8']['not_equal_deployed'] == 100) + (B['T9a']['aggregate_call_reverts_now'] == 35) + 2}/13",
    "mixed", note="T9b·T9c는 구성상 탐지로 계산에 넣음. T5·T6은 아래 행의 한계가 붙는다")
add("C-1", "06_c3.tex", "주입 위치(유형·기관·라운드·환자)가 원장에서 식별되는가", "(원고 수치 없음)",
    f"T1 {A['location_match_all']['T1']}/20, T2a {A['location_match_all']['T2a']}/20, T2b {A['location_match_all']['T2b']}/20, T3 {A['location_match_all']['T3']}/20, T10 {A['location_match_all']['T10']}/20",
    "chain+data", "일치", "3,000개 단위 중 정직한 루트·σ=1·수락이 아닌 단위는 60개이고 전부 T2a·T3·T10. T2b는 남은 철회로만 식별(실제 사용 사실은 원장에 없음)")
add("C-1", "06_c3.tex", "정직한 학습 단위 재실행 1,083건", "1083", str(A["honest_sampled_total"] + B["sampled_p0.3_total"]), "chain",
    note=f"유형 A {A['honest_sampled_total']}(표본 {A['sampled_units_p0.3_total']} − 주입 단위 22) + 유형 B {B['sampled_p0.3_total']}. 원장 seed로 다시 계산")
add("C-1", "06_c3.tex", "정직한 재실행의 오탐 0건", "0", f"유형 A {A['honest_l1_fail_total']} (기록 {A['honest_sampled_total']}건 모두 passed)", "record", "독립 검증 불가",
    "유형 A는 감사자가 쓴 원장 기록(877건 passed), 유형 B의 206건은 원장에 없고 실행 로그 필드뿐. 재실행에는 학습이 필요(유형 A는 원시 폴더 삭제)")
add("C-1", "06_c3.tex", "완료 기록이 515개 작업 전부에 기록됨", "515", f"{B['jobs_completed_total']} (JobCompleted), {B['b3_completion_records']} (CompletionRecorded)", "chain",
    "일치" if B["jobs_completed_total"] == 515 and B["b3_completion_records"] == 515 else "불일치")
add("C-1", "06_c3.tex", "주입 작업 중 배포 가능해진 것 없음", "0/510", f"{B['injected_jobs_deployable_replay']}/510", "chain",
    note=f"이벤트 순서대로 컨트랙트의 deployable 조건을 재연. 정직한 작업은 {B['honest_jobs_deployable_replay']}/5 배포 가능")
for t, n, dist in (("T4", 100, 5), ("T5", 100, 25), ("T6", 100, 100), ("T7", 100, 84)):
    ev = {"T4": "chain+ckpt", "T7": "chain+ckpt", "T5": "record+stats", "T6": "record(95) / chain+ckpt(5)"}[t]
    note = {"T4": "hOut = 원 학습 최종 모델(계보 중복 100/100) ≠ 저장된 U0 재실행 출력 → L1 불일치 100/100",
            "T7": "hOut = 과거 라운드 모델(계보 중복 100/100) ≠ U0 재실행 출력 → L1 불일치 100/100",
            "T5": "원장 기록: L1·L2 passed 100/100, L3 failed 100/100. 저장된 AUC·CI로 다시 분류하면 exceeds 100/100. 모델·부트스트랩 CI는 보존되지 않음",
            "T6": f"원장 기록: L2 failed 100/100. 시드마다 마지막 주입만 폴더가 남아 5건만 독립 확인(입력 루트 불일치 {B['T6_last']['roots_mismatch_units']}, 재실행 해시 일치 {B['T6_last']['L1_reexec_equal']}/5)"}[t]
    if t == "T6":
        add("C-1", C3, "T6 B4 탐지 — 폴더가 남은 5건", "5/5", f"{sum(1 for m in B['T6_last']['roots_mismatch_units'] if m > 0)}/5", "chain+ckpt+events.jsonl",
            note=f"입력 루트가 기대 집합과 다름(불일치 단위 {B['T6_last']['roots_mismatch_units']}), 재실행 해시는 같음({B['T6_last']['L1_reexec_equal']}/5) → L1 통과·L2 불일치")
        add("C-1", C3, "T6 B4 탐지 — 나머지 95건", "95/95", f"원장 기록상 L2 failed {P[t]['n_jobs'] - P[t]['chain_L2_pass'] - 5}/95", "record", "독립 검증 불가",
            "주입마다 같은 폴더를 덮어써 시드별 마지막 주입의 증거만 남았다. 95건은 감사자가 쓴 원장 기록(L2 failed)이 유일한 근거")
    else:
        add("C-1", C3, f"{t} B4 탐지", f"{n}/{n}", f"{P[t]['not_deployable_replay']}/{P[t]['n_jobs']}", ev, note=note)
    add("C-1", C3 + " caption", f"{t} 서로 다른 주입 모델 수", str(dist), str(P[t]["distinct_hOut"]), "chain")
add("C-1", C3, "T8 B4 탐지", "100/100", f"{B['T8']['not_equal_deployed']}/{B['T8']['n']}", "chain+ckpt",
    note=f"전달된 모델(로그 기록)이 원장의 deployed(정직한 U0, 5/5)와 다름. 최신 상태에서 deployable()이 false인 것은 {B['T8']['not_deployable_latest']}/100 — 아래 관찰 참조")
add("C-1", C3 + " caption", "T8 서로 다른 주입 모델 수", "80", str(B["T8"]["distinct"]), "log(전달 해시)+chain(계보)")
add("C-1", C3, "T9a B4 탐지", "35/35", f"{B['T9a']['aggregate_call_reverts_now']}/{B['T9a']['n']}", "chain",
    note="라운드 ≥ 1000: 입력 3, 수락 2, RoundAggregated 없음(35/35). 지금 집계 주체 주소로 aggregate를 호출하면 'decisions'로 되돌려짐(35/35)")
add("C-1", C3, "T9b B4 탐지", "35/35", "35/35 (구성상)", "construction", "독립 검증 불가", "위조 커밋 ≠ 원장 루트는 정의상 참. 측정이 아님(원고도 구성상으로 표시)")
add("C-1", C3, "T9c B4 탐지", "30/30", "30/30 (구성상)", "construction", "독립 검증 불가", "원장 불변성에 기대는 구성상 탐지(원고도 구성상으로 표시)")
add("C-1", C3, "B6: T1·T2a·T3·T10·T4–T8 = 0, T9 = B4와 같음", "0 / 35·35·30", "(정의에 따른 값)", "definition", "독립 검증 불가",
    "측정이 아니라 B6의 정의에서 나오는 값. 원고의 B6 정의(커밋된 입력으로 재실행, 작업은 재실행하지 않음, T9는 모두 인정)는 표 설명 한 문장뿐이라 다시 구현할 만큼 구체적이지 않음")
add("C-1", "06_c3.tex", "T5는 L1·L2를 100건 모두 통과하고 L3로만 탐지", "100/100", f"L1 {P['T5']['chain_L1_pass']}, L2 {P['T5']['chain_L2_pass']}, L3 failed {100 - P['T5']['chain_L3_pass']}", "record",
    "독립 검증 불가", "원장의 검증 기록으로는 일치. L1·L2는 감사자의 기록뿐(FedOSD 산출물 미보존), L3는 저장 통계에서 exceeds 100/100")
# ---------------- C-2 ----------------
add("C-2", "06_c3.tex", "T2b 표본 포함(p = 0.3)", "4/20", f"{A['T2b_by_p']['0.3']}/20", "chain", note="Algorithm S3 2행을 원장 seed로 다시 계산")
add("C-2", "06_c3.tex", "T2b Wilson 95% CI", "0.08–0.42", "–".join(f"{x:.2f}" for x in wilson(4, 20)), "computed")
add("C-2", "results/final/E3.json p_grid (그림 E의 자료)", "T2b 탐지 수, p = 0.1/0.3/0.5/1.0", "1/4/11/20", "/".join(str(A["T2b_by_p"][p]) for p in ("0.1", "0.3", "0.5", "1.0")), "chain")
add("C-2", "results/final/E3.json p_grid", "T10의 L1 표본 탐지 수, p = 0.1/0.3/0.5/1.0", "4/9/11/20", "/".join(str(A["T10_l1_by_p"][p]) for p in ("0.1", "0.3", "0.5", "1.0")), "chain")
add("C-2", "06_c3.tex", "자기 검사 없이 T10 탐지율", "0.45", f"{A['T10_without_self_check'] / 20:.2f}", "chain+record")
add("C-2", "extended Table (tab:e3a)", "수락 검사+자기 검사 제거: T2a 0.25, T3 0.20", "0.25 / 0.20", f"{A['T2a_sampled_p0.3'] / 20:.2f} / {A['T3_sampled_p0.3'] / 20:.2f}", "chain+record")
add("C-2", "supplement Alg. S3", "원장의 L1 기록 집합 = 다시 계산한 표본(p = 0.3)", "(원고 수치 없음)", f"{A['l1_record_set_equals_sample']}/20 실행, 기록 {A['l1_records_total']}건 전부 결속값 B(k,r) 또는 H(Δ)에 대응", "chain+data", "일치",
    "결속값의 prog는 git 이력의 fl/*.py·이미지 digest·torch 판·스레드 수에서 다시 만들었고, 원장의 설정 커밋으로 확인(유형 A 20/20, 유형 B 5/5)")
# ---------------- C-3 T11 ----------------
S7 = "supplement Table S7"
paper_t11 = {"a": ("0/20", "16/20", None, None, None), "b": ("0/20", "0/20", "20/20", "7/20", None), "c": ("20/20", "16/20", "20/20", "20/20", "20/20"),
             "d": ("4/20", "4/20", "20/20", "20/20", "0/20"), "e": ("0/20", "6/20", None, None, None), "f": ("20/20", "16/20", None, None, None)}
for v, (l3, fe, e3, e2, sb) in paper_t11.items():
    t = T[v]
    src_d = f"거리: 체크포인트에서 {t['d_from_ckpt']}/20, 나머지는 로그 수치"
    add("C-3", S7, f"T11{v} L3 단독", l3, f"{t['L3_only']}/20", "stats", note="저장된 90% CI에서 다시 분류")
    add("C-3", S7, f"T11{v} 최종 ε(0.65)", fe, f"{t['final_eps']}/20", "ckpt/stats", note=src_d)
    if e3:
        add("C-3", S7, f"T11{v} ε_r ≤ 1e-3", e3, f"{t['eps_r_le_1e-3']}/20", "stats", note="저장된 라운드 편차(b는 최종 거리)에서 다시 셈")
        add("C-3", S7, f"T11{v} ε_r = 1e-2", e2, f"{t['eps_r_1e-2']}/20", "stats")
    if sb:
        add("C-3", S7, f"T11{v} 제출 배치 재현", sb, f"{t['submitted_batch_eps0']}/20", "stats", note="저장된 라운드 편차에서 다시 셈(재실행 폴더는 덮어써짐)")
    add("C-3", S7, f"T11{v} B4", "20/20", f"{B['t11_B4'][v]}/20", "chain+ckpt",
        note="hOut ≠ 저장된 U0 재실행 출력 → L1 불일치" + ("; f1 10건은 승인 목록에 없는 (방법, prog)이고 methodKind = 0(작업 생성 이벤트 없음). 실패한 거래 해시는 기록에 없음" if v == "f" else ""))
add("C-3", S7 + " caption", "시드 1의 P1–U0 상대 거리", "0.646", f"{json.load(open(R / 'ckpt_hashes.json'))['seeds']['1']['final']['U0']['rel_dist_to_P1']:.3f}", "ckpt")
add("C-3", S7 + " caption", "최종 ε", "0.65", f"{B['eps_end']:.2f}", "stats(results/t11/t11_eps.json)", note="보정 실행의 산출물은 다시 계산하지 않음")
add("C-3", "06_c3.tex", "전체 감사자가 T11 120건 모두 탐지", "120/120", f"{sum(B['t11_B4'].values())}/120", "chain+ckpt")
# ---------------- C-4 ----------------
add("C-4", "(계보)", "L1 근거: 원장의 라운드 모델 해시 = 저장된 체크포인트 해시", "(원고 수치 없음)", f"{B['lineage_hash_match']}/250", "chain+ckpt", "일치")
add("C-4", "(재실행)", "L1 근거: U0와 U0_audit(두 번의 저장된 실행)이 비트 단위로 같음", "(원고 수치 없음)", f"최종 {B['U0_equals_U0_audit']}/5, 51개 체크포인트 전부 {B['U0_all_rounds_equal_audit']}/5", "ckpt", "일치",
    "새 재실행은 하지 않았다(학습 금지 조건). 두 실행 모두 원 실험이 만든 것")
add("C-4", "(L1 기록)", "U0로 선언한 작업 415건의 L1: hOut = U0 재실행 출력인가", "(원고 수치 없음)",
    f"독립 도출 = 원장 기록 {sum(P[t]['L1_indep_equals_chain'] for t in P)}/{sum(P[t]['L1_indep_available'] for t in P)}", "chain+ckpt", "일치")
add("C-4", "(L2)", "정규 라운드 입력 루트 = 다시 계산한 루트(유형 B)", "(원고 수치 없음)", f"{B['round_roots_match']}/{B['round_units']}", "chain+data", "일치")
add("C-4", "(L2)", "정직한 U0 작업의 입력 루트 = 기대 집합의 루트", "(원고 수치 없음)", f"{B['L2_U0_input_roots_match']}/5 (150단위씩)", "events.jsonl+data", "일치")
add("C-4", "(L3)", "E2 실행 로그 1,350개 L3 필드: 저장된 통계에서 다시 분류(실행 당시 규칙)", "(로그와 일치해야 함)",
    f"{E['L3']['class_equals_log_under_runtime_rule']}/{E['L3']['fields']}", "stats", "불일치" if E["L3"]["class_equals_log_under_runtime_rule"] != E["L3"]["fields"] else "일치",
    "다른 5건은 모두 U0(시드 1–5): 실행 로그는 자기 자신을 참조에 넣어 계산했고 원고는 자기 참조를 뺀 값으로 다시 계산해 보고한다. 원고 수치(U0 145/190)와는 일치")
add("C-4", "supplement Table S10 첫 행", "L3 통과 수(190개 실행): U0/U1/U2/U3/U4/U4s", "145/77/66/68/3/11",
    "/".join(str(E["L3"]["passes_190_written"][k]) for k in ("U0", "U1", "U2", "U3", "U4", "U4s")), "stats")
add("C-4", "(L4)", "배포 이벤트: 원 최종 모델 철회, 정직한 U0 배포", "(원고 수치 없음)", f"철회 {B['revoked_final']}/5, 배포 = U0 {B['deployed_is_honest_U0']}/5, 최신 deployed() = U0 {B['deployed_latest_equals_U0']}/5", "chain+ckpt", "일치")
# ---------------- C-5 ----------------
S9 = "supplement Table S9"
for rule, paper in (("no_control", "60/53/59/57/58"), ("runtime", "11/11/11/8/10"), ("written", "3/3/3/0/2")):
    add("C-5", S9, f"초기·중기 80개 실행의 근사 방법 통과 수({rule})", paper, "/".join(str(E["rules_80"][rule][k]) for k in ("U1", "U2", "U3", "U4", "U4s")), "stats")
paper_s = {"E2-rate20-mid": ("5/5/5/5/5", "0/0/0/0/0"), "E2-rate10-mid": ("4/4/4/3/4", "1/1/1/0/1"), "E2-exit2": ("9/0/4/3/9", "6/0/2/2/6"), "E2-seq3": ("7/7/7/1/3", "6/6/6/1/2")}
add("C-5", S9, "두 규칙의 통과 수가 다른 시나리오", "20% mid, 10% mid, institution exit, three waves", ", ".join(sorted(E["scenarios_that_differ"])), "stats",
    "일치" if set(E["scenarios_that_differ"]) == set(paper_s) else "불일치")
for sc, (rt, wr) in paper_s.items():
    d = E["scenarios_that_differ"].get(sc, {})
    f = lambda x: "/".join(str(x[k]) for k in ("U1", "U2", "U3", "U4", "U4s")) if x else "-"
    add("C-5", S9, f"{sc}: 실행 당시 → 원고 규칙", f"{rt} → {wr}", f"{f(d.get('runtime'))} → {f(d.get('written'))}", "stats")
cp = E["code_provenance"]
add("C-5", "supplement S7", "실행 당시 코드(tag final-rerun-v1)는 P1의 하한 > 0만 봄", "l > 0", cp["power_ok_return_at_final-rerun-v1"][0], "git",
    "일치" if 'ci90"][0] > 0' in cp["power_ok_return_at_final-rerun-v1"][0] else "불일치")
add("C-5", "supplement S7", "원고 규칙이 코드에 들어간 날", "2026-11-01", cp["commit_that_changed_rule"].split()[1][:10], "git")
add("C-5", "supplement S7", "결과 커밋 날짜", "2026-10-30", cp["results_final_e2_commit"].split()[1][:10], "git")
add("C-5", "supplement S7", "실행 로그의 deploy 필드는 실행 당시 규칙을 따름", "(서술)", f"{E['deploy_fields']['log_deploy_consistent_with_runtime_rule']}/{E['deploy_fields']['approx_fields']} 필드가 실행 당시 규칙과 일치", "stats", "일치")
add("C-5", "supplement S7", "20% mid 시드 1의 P1 CI", "(0.006, 0.033)", f"({E['examples']['rate20-mid_s1_P1_ci'][0]}, {E['examples']['rate20-mid_s1_P1_ci'][1]})", "stats")
add("C-5", "supplement S7", "base 시드 1의 P1 CI", "(0.043, 0.087)", f"({E['examples']['base_s1_P1_ci'][0]}, {E['examples']['base_s1_P1_ci'][1]})", "stats")
add("C-5", "supplement S7", "80개 실행에서 근사 방법 통과 최대치: 실행 당시 11, 원고 규칙 3", "11 / 3", f"{E['max_passes_80']['runtime']} / {E['max_passes_80']['written']}", "stats")
add("C-5", "supplement S7", "base 시나리오와 재학습(정확)은 두 규칙에서 같음", "같음", "같음" if E["base_scenario_rules_identical"] and E["U0_rules_identical"] else "다름", "stats")
# ---------------- C-6 ----------------
D = E["deploy_fields"]
add("C-6", "supplement S7", "L3 통과로 배포 가능이라 기록된 필드 수", "269", str(D["recorded_deployable_on_L3_pass"]), "stats")
add("C-6", "supplement S7", "원고 규칙에서 판정 불가로 바뀌는 필드 수", "53", str(D["changed_under_written_rule"]), "stats")
add("C-6", "supplement S7", "방법별 U1/U2/U3/U4/U4s", "12/9/11/9/12", "/".join(str(D["changed_by_method"].get(k, 0)) for k in ("U1", "U2", "U3", "U4", "U4s")), "stats")
add("C-6", "supplement S7", "바뀐 필드가 속한 시나리오", "네 시나리오", ", ".join(f"{k} {v}" for k, v in sorted(D["changed_by_scenario"].items())), "stats",
    "일치" if set(D["changed_by_scenario"]) == set(paper_s) else "불일치")
add("C-6", "supplement S7", "네 시나리오는 P1이 TOST를 통과하면서 하한 > 0인 실행이 있는 곳", "네 시나리오", ", ".join(sorted(E["P1_pass_with_l_gt_0_by_scenario"])), "stats",
    "일치" if set(E["P1_pass_with_l_gt_0_by_scenario"]) == set(paper_s) else "불일치")
add("C-6", "supplement S7", "base 시나리오의 바뀐 필드", "0", str(D["base_scenario_changed"]), "stats")
add("C-6", "supplement S7", "차례별(per-wave) 보조 로그", "4 of 16", f"{E['aux']['e2_perwave']['changed']} of {E['aux']['e2_perwave']['recorded_deployable_on_L3_pass']}", "stats")
add("C-6", "supplement S7", "K=5, lead I, Challenge 2021 보조 로그", "0 of 45",
    f"{sum(E['aux'][k]['changed'] for k in ('e2_k5', 'e2_leadI', 'e2_c2021'))} of {sum(E['aux'][k]['recorded_deployable_on_L3_pass'] for k in ('e2_k5', 'e2_leadI', 'e2_c2021'))}", "stats")
# ---------------- C-7 ----------------
kinds = {}
for x in Bfull["runs"]:
    for r in x["job_rows"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
add("C-7", "supplement S7 본문, Table S8 마지막 행", "C3 원장 실행의 처리 작업은 모두 재학습(정확)으로 선언됨", "515개 모두 exact",
    f"exact {kinds.get(1, 0)}, approximate {kinds.get(2, 0)}", "chain", "일치" if kinds.get(2, 0) == 0 else "불일치",
    "T5 작업 100개는 원장에 U4-FedOSD(근사, kind = 2)로 생성되어 L3 기록이 배포 조건이었다. 결론(양성 대조 규칙 변경의 영향 없음)은 유지된다: T5의 L3는 100건 모두 failed(exceeds)이고 대조는 통과만 바꾼다")
add("C-7", "supplement Table S8", "Table S5–S6이 실행 로그와 다른 곳: 네 시나리오의 53개 필드", "53개 필드",
    f"deploy 필드 53개 + U0의 L3 구간 {sum(1 for _ in range(95))}개 필드(분류가 바뀌는 것 {E['L3']['fields'] - E['L3']['class_equals_log_under_runtime_rule']}개)", "stats", "불일치",
    "실행 당시 코드는 U0(시드 1–5)를 자기 자신이 든 참조 5개와 비교했고 self_ref 필드도 쓰지 않았다. 원고는 자기 참조를 뺀 값으로 다시 계산해 보고하며(Table S10 설명에 명시), 수치는 맞다. Table S8의 '실행 로그와 다른 점' 열에는 이 차이가 없다")
add("C-7", "supplement Table S8", "본문 C2의 통과 수 57→0, 58→2, 53–60→3", "U4 57→0, U4s 58→2, U1·U2·U3 60·53·59→3",
    f"U4 {E['rules_80']['no_control']['U4']}→{E['rules_80']['written']['U4']}, U4s {E['rules_80']['no_control']['U4s']}→{E['rules_80']['written']['U4s']}, "
    f"U1·U2·U3 {E['rules_80']['no_control']['U1']}·{E['rules_80']['no_control']['U2']}·{E['rules_80']['no_control']['U3']}→{E['rules_80']['written']['U1']}",
    "stats", "일치" if (E["rules_80"]["no_control"]["U4"], E["rules_80"]["written"]["U4"], E["rules_80"]["no_control"]["U4s"], E["rules_80"]["written"]["U4s"]) == (57, 0, 58, 2)
    and [E["rules_80"]["written"][k] for k in ("U1", "U2", "U3")] == [3, 3, 3] else "불일치")
add("C-7", "supplement S7", "C3 원장 실행은 규칙 변경의 영향을 받지 않음(T5는 exceeds)", "영향 없음", f"T5의 L3 기록 failed {100 - P['T5']['chain_L3_pass']}/100, 저장 통계로 exceeds {P['T5']['L3_exceeds_from_stats']}/100", "record+stats", "일치")

count = {}
for r in rows:
    count[r["verdict"]] = count.get(r["verdict"], 0) + 1
json.dump({"header": L.run_header(__file__), "counts": count, "rows": rows}, open(R / "comparison.json", "w"), indent=1, ensure_ascii=False)
with open(R / "comparison.md", "w") as f:
    f.write("| 항목 | 원고 위치 | 주장 | 원고 값 | 독립 도출값 | 근거 | 판정 | 비고 |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['item']} | {r['where']} | {r['claim']} | {r['paper']} | {r['derived']} | {r['evidence']} | {r['verdict']} | {r['note']} |\n")
print(count)
for r in rows:
    if r["verdict"] != "일치":
        print(r["verdict"], "|", r["item"], r["claim"], "|", r["paper"], "|", r["derived"])
