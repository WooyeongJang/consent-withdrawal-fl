"""실험 A 보고서용 표(마크다운)를 결과 JSON에서 만든다. 수치를 손으로 옮기지 않기 위한 스크립트."""
import json, math, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
o = json.load(open(os.path.join(R, "a1_observed.json"))); s = json.load(open(os.path.join(R, "a1_simulated.json"))); a2 = json.load(open(os.path.join(R, "a2_summary.json")))
thr = 0.05 + 3 * math.sqrt(0.05 * 0.95 / s["meta"]["reps"])
order = ["E2-rate01", "E2-rate01-mid", "E2-rate01-early", "E2-reconsent", "E2-base", "E2-mid", "E2-early", "E2-inst2", "E2-latejoin", "E2-seq3", "E2-F65", "E2-diagMI",
         "E2-rate10", "E2-rate10-mid", "E2-rate10-early", "E2-rate20", "E2-rate20-mid", "E2-rate20-early", "E2-exit2"]
f = lambda x, n=3: "–" if x is None else f"{x:.{n}f}"
print(f"<!-- false-equivalence threshold 0.05 + 3 MC SE = {thr:.4f} -->\n")
print("### 표 A1. 분산 성분(관측)과 동등성 검정의 작동 특성(모의, 참조 5개)\n")
print("| 시나리오 | \\|X\\| | s_ref (90% 구간) | se_boot | P(통과 \\| Δ=0) M1 / M2 / M3 | P(통과 \\| Δ=0.025) M2 | P(통과 \\| Δ=0.05) M1 / M2 / M3 | 통과 불가 실행(U0) | 관측 U0 통과 |")
print("|---|---|---|---|---|---|---|---|---|")
n_ok = n_fe = n_fe3 = 0; fe_m2 = []; fe_m3 = []
for sc in order:
    ob = o["scenarios"][sc]; m = s["scenarios"][sc]["models"]
    g = lambda mod, k: m[mod]["5"][k]
    p0 = g("M2", "power_at_0"); fe = g("M2", "false_equiv_at_0.05"); fe3 = g("M3", "false_equiv_at_0.05")
    n_ok += p0 >= 0.80; n_fe += fe > thr; n_fe3 += fe3 > thr
    if ob["methods"]["U0"]["pass_arith_impossible"] < 10: fe_m2.append(fe); fe_m3.append(fe3)
    print(f"| {sc} | {ob['nX']} | {ob['s_ref_10']:.4f} ({ob['s_ref_10_ci90'][0]:.4f}–{ob['s_ref_10_ci90'][1]:.4f}) | {ob['methods']['U0']['se_boot_mean']:.4f} | "
          f"{f(g('M1','power_at_0'))} / {f(p0)} / {f(g('M3','power_at_0'))} | {f(g('M2','power_at_0.025'))} | "
          f"{f(g('M1','false_equiv_at_0.05'))} / {f(fe)}{'*' if fe > thr else ''} / {f(fe3)}{'*' if fe3 > thr else ''} | {ob['methods']['U0']['pass_arith_impossible']}/10 | {ob['methods']['U0']['class'].get('pass',0)}/10 |")
print(f"\n검정력 충분(M2, Δ=0에서 ≥ 0.80): {n_ok}/19. 명목 수준 초과(*, > {thr:.4f}): M2 {n_fe}/19, M3 {n_fe3}/19. 통과가 가능한 16개 시나리오의 경계 통과 확률 범위: M2 {min(fe_m2):.3f}–{max(fe_m2):.3f}, M3 {min(fe_m3):.3f}–{max(fe_m3):.3f}.\n")
print("### 표 A2. 참조 수에 따른 변화(모의, M2): P(통과 | Δ=0) / P(통과 | Δ=0.05) / 독립 반복의 분류 일치 확률(Δ=0)\n")
print("| 시나리오 | n=2 | n=3 | n=5 | n=10 | n=20 |\n|---|---|---|---|---|---|")
for sc in order:
    m = s["scenarios"][sc]["models"]["M2"]
    print(f"| {sc} | " + " | ".join(f"{m[n]['power_at_0']:.2f} / {m[n]['false_equiv_at_0.05']:.3f} / {m[n]['replicate_agreement_at_0']:.2f}" for n in ("2", "3", "5", "10", "20")) + " |")
print("\n### 표 A3. 양성 대조의 검출 한계(모의, 참조 5개)와 관측\n")
print("| 시나리오 | 50% 검출 잔여 | 80% 검출 잔여 | 90% 검출 잔여 | 관측 P1 평균 차이 | 그 차이에서 예측 검출 확률 | 관측 검출(기준 초과) | 관측 P1 통과 |\n|---|---|---|---|---|---|---|---|")
for sc in order:
    pc = s["scenarios"][sc]["positive_control"]["5"]; p1 = o["scenarios"][sc]["methods"]["P1"]
    print(f"| {sc} | {f(pc['MDR50'])} | {f(pc['MDR80'])} | {f(pc['MDR90'])} | {p1['diff_mean']:+.4f} | {pc['detect_at_observed_P1_diff']:.2f} | {p1['class'].get('exceeds',0)}/10 | {p1['class'].get('pass',0)}/10 |")
print("\n### 표 A4. 모형 적합: U0(Δ=0이 구성상 참)의 통과 수, 관측 대 예측\n")
c = s["calibration_U0_total"]
print(f"합계: 관측 {c['observed_pass_total']}/190, 예측 평균 {c['plug_in']['mean']:.1f} (95% 예측 구간 {c['plug_in']['pi95'][0]:.0f}–{c['plug_in']['pi95'][1]:.0f}); σ_ref 불확실성 포함 {c['posterior']['mean']:.1f} ({c['posterior']['pi95'][0]:.0f}–{c['posterior']['pi95'][1]:.0f}).\n")
ins = sum(v["observed_in_pi95_plug_in"] for v in s["calibration_U0_shared_refs"].values())
print(f"시나리오별: 19개 중 {ins}개에서 관측이 95% 예측 구간 안.\n")
print("### 표 A5. 참조 부분집합 전수 열거(관측): 보고된 분류가 유지되는 비율\n")
print("| 방법 | n=2 | n=3 | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 | n=5에서 90% 이상 유지된 실행 |\n|---|---|---|---|---|---|---|---|---|---|")
for m_, v in o["subset_enumeration_overall"].items():
    st = o["run_level_stability_n5"][m_]
    print(f"| {m_} | " + " | ".join(f"{v[str(n)]['agree_with_reported']:.3f}" for n in range(2, 10)) + f" | {st['stable_runs']}/{st['runs']} |")
print("\n보고된 분류별(n=5, 90% 이상 유지된 실행 / 해당 실행):")
for m_, st in o["run_level_stability_n5"].items():
    print(f"- {m_}: " + ", ".join(f"{k} {v['stable']}/{v['n']}" for k, v in st["by_reported"].items()))
print("\n### 표 A6. A2 재평가: 기록 표집·그림자 표집 변동(참조 5개 평균)\n")
print("| 시나리오 | σ_rec(대응 부트스트랩) | 저장된 se_boot | σ_rec,d | 모델 간 상관 | 규칙의 se | 실제 SD(시드) | 실제 SD(시드+기록) | 규칙 se / 실제 SD | 그림자 잭나이프 SE(AUC) | (차이 d) |\n|---|---|---|---|---|---|---|---|---|---|---|")
rat = []
for sc in order:
    v = a2["scenarios"][sc]; r_ = v["se_rule_mean"] / v["sd_actual_seed_plus_records"]; rat.append(r_)
    print(f"| {sc} | {v['sigma_rec_mean']:.4f} | {o['scenarios'][sc]['methods']['U0']['se_boot_mean']:.4f} | {v['sigma_rec_d_mean']:.4f} | {v['corr_between_models_mean']:.2f} | {v['se_rule_mean']:.4f} | {v['sd_actual_seed_only']:.4f} | {v['sd_actual_seed_plus_records']:.4f} | {r_:.2f} | {v['shadow_jackknife_se_auc_mean']:.4f} | {v['shadow_jackknife_se_d_mean']:.4f} |")
print(f"\n규칙 se / 실제 SD 범위: {min(rat):.2f}–{max(rat):.2f}. 재현: {json.dumps({k: v for k, v in a2['reproduction'].items() if k != 'records_IN_or_OUT_in_all_shadows'})}")
print("\n정규성(표준화한 U0 AUC 190개):", o["u0_normality_pooled"], "\n외삽:", o["extrapolation_min_n"])
sr = s["shared_ref_counts"]
print("\n### 표 A7. 참조 공유의 영향(모의, M2, 참조 5개, 시드 10개): 통과 수의 표준편차(이항 대비)와 급내 상관\n")
print("| 시나리오 | Δ | P(통과) | P(0/10) | P(10/10) | SD(통과 수) | 이항 SD | 급내 상관 |\n|---|---|---|---|---|---|---|---|")
for sc in ("E2-base", "E2-mid", "E2-diagMI", "E2-rate10", "E2-rate20"):
    for d in ("0.0", "0.025", "0.05"):
        x = sr[sc][d]; print(f"| {sc} | {d} | {x['p_pass']:.3f} | {x['P_0_of_10']:.3f} | {x['P_10_of_10']:.3f} | {x['sd_count']:.2f} | {x['binomial_sd_count']:.2f} | {f(x['icc'])} |")
