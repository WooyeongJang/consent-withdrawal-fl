"""실험 B 보고서용 표(마크다운)를 results/b_summary.json에서 만든다."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
b = json.load(open(os.path.join(R, "b_summary.json")))
L, T = b["local"], b["testnet"]
pc = lambda k, n: f"{k}/{n}" if n else "–"
if L:
    print("### 표 B1. 로컬 EVM(순서 통제): 조건별 결과 (칸마다 50회; 같은 기관 / 다른 기관)\n")
    print("| 조건 | 버전 | 원장 순서 | 수락 | 낡은 수락 | 부적절한 수락 | 되돌림 | 불필요한 되돌림 | D-pos 표시 | D-time 표시 |\n|---|---|---|---|---|---|---|---|---|---|")
    name = {"L1": "L1 철회 확정 뒤 새로 조회", "L2": "L2 수락 확정 뒤 철회", "L3": "L3 거의 동시 제출", "L4": "L4 제출 순서 ≠ 포함 순서", "L5": "L5 같은 블록", "L6": "L6 지연된 조회"}
    for c in ("L1", "L2", "L3", "L4", "L5", "L6"):
        for v in ("v3.1", "v3.2"):
            s, o = L["cells"][f"{v}|{c}|same"], L["cells"][f"{v}|{c}|other"]
            cls = {}
            for x in (s, o):
                for k, n in x["ledger_class"].items():
                    cls[k] = cls.get(k, 0) + n
            f = lambda key: f"{s[key]} / {o[key]}"
            print(f"| {name[c]} | {v} | " + ", ".join(f"{k.replace(' block','')} {n}" for k, n in sorted(cls.items())) + f" | {f('accepted')} | {f('stale_acceptance')} | {f('improper_acceptance')} | {f('reverted')} | {f('unnecessary_revert')} | {f('flag_pos')} | {f('flag_time')} |")
    print(f"\n재현: {L['runs']}회 실행의 결과 벡터 해시 일치 = {L['reproduced']} (전체 기록 일치 = {L['full_record_identical']}).\n")
def hrow(v, h, label):
    return (f"| {label} | {h['n']} | {h['W_reflected']}/{h['n']} | {h['stale_read_with_W_first']} | {h['stale_read_accepted']} | {h['stale_read_reverted']} | {h['stale_acceptance']} "
            f"({h['stale_acceptance_same_inst']} / {h['stale_acceptance_other_inst']}) | {h['improper_acceptance']} | {h['reverted']} | {h['unnecessary_revert']} | "
            f"{h['version_recorded_consistent']}/{h['accepted_with_det']} | {h['Dpos_TP']}/{h['improper_acceptance']}, 오탐 {h['Dpos_FP']} | {h['Dtime_TP']}/{h['improper_acceptance']}, 오탐 {h['Dtime_FP']} |")
print("### 표 B2. 버전별 합계\n")
print("| 실행 | 시행 | 철회 반영 | 낡은 조회 + W 먼저 | 그중 수락 | 그중 되돌림 | 낡은 수락(같은 / 다른 기관) | 부적절한 수락 | 되돌림 | 불필요한 되돌림 | 기록된 버전 = 포함 시점 버전 | D-pos 탐지 | D-time 탐지 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for v in ("v3.1", "v3.2"):
    if L: print(hrow(v, L["by_version"][v], f"로컬 {v}"))
for v in ("v3.1", "v3.2"):
    if T.get(v): print(hrow(v, T[v], f"Kairos {v}"))
for v in ("v3.1", "v3.2"):
    t = T.get(v)
    if not t: continue
    print(f"\n### 표 B3-{v}. Kairos {v}: 관측된 원장 순서별 결과 (같은 기관 / 다른 기관)\n")
    print(f"컨트랙트 `{t['deploy']['address']}` (블록 {t['deploy']['block']}, 배포 코드 = 로컬 컴파일: {t['deploy']['code_matches_local_compile']}), {t['started']} – {t['finished']}, 계획 {t['planned']}, 실행 {t['run']}, 제외 {len(t['excluded'])}, 분석 {t['analysed']}.\n")
    print("| 원장 순서 | 시행 | 새 조회로 거절 | 수락 | 낡은 수락 | 부적절한 수락 | 되돌림 |\n|---|---|---|---|---|---|---|")
    for c in ("W<D other block", "W<D same block", "D<W same block", "D<W other block"):
        s, o = t["by_class_relation"][f"{c}|same"], t["by_class_relation"][f"{c}|other"]
        f = lambda key: f"{s[key]} / {o[key]}"
        print(f"| {c} | {f('n')} | {f('rejected_by_fresh_read')} | {f('accepted')} | {f('stale_acceptance')} | {f('improper')} | {f('reverted')} |")
    ci = lambda x: f"{x[0]:.3f}–{x[1]:.3f}"
    print(f"\n- 낡은 수락 {t['stale_acceptance']}/{t['analysed']} (Wilson 95% CI {ci(t['stale_acceptance_ci95'])}), 부적절한 수락 {t['improper_acceptance']}/{t['analysed']} ({ci(t['improper_acceptance_ci95'])}).")
    print(f"- 되돌림 사유: {t['revert_reasons']}; 새로 읽고도 되돌려진 수락: {t['reverted_fresh_read']}; 후속 결정: {t['followup']}; 되돌려진 시행의 평균 제출 횟수: {t['attempts_mean_when_reverted']}.")
    print(f"- 지연(제출→영수증): W 중앙값 {t['latency_W_s']['median']:.2f} s (95% {t['latency_W_s']['p95']:.2f}, 최대 {t['latency_W_s']['max']:.2f}), D 중앙값 {t['latency_D_s']['median']:.2f} s (95% {t['latency_D_s']['p95']:.2f}, 최대 {t['latency_D_s']['max']:.2f}); 블록 수 W {t['latency_blocks_W']}, D {t['latency_blocks_D']}.")
    print(f"- 제출 순서와 포함 순서가 뒤바뀐 시행: {t['order_inverted']}/{t['analysed']}. W 영수증 직후 조회 {t['probes']}건 중 낡은 머리 블록 {t['probe_stale_head']}건, 오류 {t['probe_errors']}건. 서로 다른 블록인데 시각이 같은 경우 {t['different_blocks_same_timestamp']}건.")
    print(f"- 같은 블록·D 먼저·같은 기관 {t['D_first_same_block_same_inst']}건 중 D-time이 위반으로 표시한 것 {t['D_first_same_block_same_inst_flag_time']}건(D-pos 0건). D-time 오탐의 원장 순서: {t['Dtime_FP_by_class']}.")
    print(f"- 가스(중앙값): W {t['gas']['W_median']:.0f}, D 수락 {t['gas']['D_accept_median']:.0f}, D 되돌림 {t['gas']['D_reverted_median']:.0f}, 준비 4건 {t['gas']['setup_median']:.0f}. 수수료 합계 {t['fee_kaia']:.4f} KAIA(잔액 차이 {t['payer_balance_diff']:.4f}).")
if L:
    print("\n### 표 B4. 로컬 EVM 가스와 재제출(2차)\n")
    print("| 버전 | W | D 수락 | D 거절 | D 되돌림 | 후속 결정 |\n|---|---|---|---|---|---|")
    g = lambda x: "–" if not x else f"{x['median']:.0f} ({x['min']}–{x['max']}, n={x['n']})"
    for v in ("v3.1", "v3.2"):
        x = L["gas"][v]; print(f"| {v} | {g(x['W'])} | {g(x['D_accept_ok'])} | {g(x['D_reject_ok'])} | {g(x['D_reverted'])} | {g(x['followup'])} |")
    print("\n| 무관한 철회 도착 확률 q(블록당) | v3.1 평균 제출 횟수 | v3.2 평균 제출 횟수(최대, 상한 20 도달) | 기대값 1/(1−q) |\n|---|---|---|---|")
    for q in ("0.1", "0.5", "0.9"):
        a, c = L["liveness"].get(f"v3.1|q={q}"), L["liveness"].get(f"v3.2|q={q}")
        if a and c: print(f"| {q} | {a['mean_attempts']:.2f} | {c['mean_attempts']:.2f} ({c['max']}, {c['capped_at_20']}) | {c['expected_1_over_1_minus_q']:.2f} |")
    for v in ("v3.1", "v3.2"):
        h = L["by_version"][v]
        print(f"\n로컬 {v}: 되돌림 사유 {h['revert_reasons']}, 후속 결정 {h['followup']}, D 먼저 {h['D_first']}건 중 수락 {h['D_first_accepted']}·D-pos 표시 {h['D_first_flag_pos']}, 같은 블록·D 먼저·같은 기관 {h['D_first_same_block_same_inst']}건 중 D-time 표시 {h['D_first_same_block_same_inst_flag_time']}건, D-time 오탐 분포 {h['Dtime_FP_by_class']}.")
