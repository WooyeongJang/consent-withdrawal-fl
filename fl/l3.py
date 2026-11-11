"""L3 판정: U0 모델 분포와의 동등성 검정(TOST) (3장 개정 제안 3번, 2026-10-25 결정).

대상 모델의 잔존 영향 지표(LiRA offline AUC)와 U0 모델들(시드별)의 평균 차이 d에 대해
H0: |d| ≥ δ 를 두 단측 검정으로 기각하면 "통과(동등)". 90% 신뢰구간(α=0.05 TOST)이 (−δ, δ) 안에 있는지로 판정한다.
d의 표준오차는 대상 AUC의 부트스트랩 표준오차와 U0 평균의 표준오차(시드 간 SD/√n)를 합친다.
δ = 0.05(2026-10-25 사용자 확정, reports/2026-10-25_3장_개정제안.md 22번).

판정(알고리즘 5의 세 결과, 3장 반영 44번 (a), 2026-10-27):
  통과        90% CI ⊂ (−δ, δ)
  기준 초과    CI 하한 > 0 (U0보다 유의하게 높음 = 잔존)
  판정 불가    그 외. 사유: 과잉 제거(CI 상한 < 0, U0보다 유의하게 낮음) / 정밀도 부족(CI가 0을 포함)
              / 검정력 부족(33번 (a)): 같은 X·Z에서 원 모델(P1, 영향 제거 전)이 기준 초과로 검출되지 않으면
                그 시험은 잔존 영향을 잡을 힘이 없으므로, 근사 방법의 "통과"를 판정 불가로 바꾼다(apply_power)
배포 규칙(deploy): 정확 방법(처음부터 재학습, 동등 조건을 충족한 체크포인트 재수행)은 L1·L2로 완료하고 L3는 참고값이다.
근사 방법은 L3 통과일 때만 배포 허용, 기준 초과·판정 불가면 미완료 → 정확 방법으로 전환한다.
"""

import math
import statistics

DELTA = 0.05
Z90 = 1.6448536269514722


def tost(auc, ci95, u0_aucs, delta=DELTA):
    se_t = (ci95[1] - ci95[0]) / (2 * 1.959963984540054)
    mu, sd = statistics.mean(u0_aucs), statistics.stdev(u0_aucs)
    se = math.sqrt(se_t ** 2 + sd ** 2 / len(u0_aucs))
    d = auc - mu
    lo, hi = d - Z90 * se, d + Z90 * se
    ok = lo > -delta and hi < delta
    cls, why = ("통과", None) if ok else ("기준 초과", "잔존") if lo > 0 else ("판정 불가", "과잉 제거" if hi < 0 else "정밀도 부족")
    return {"result": "통과" if ok else "불통과", "class": cls, "reason": why, "diff": d, "ci90": [lo, hi],
            "delta": delta, "u0_mean": mu, "u0_sd": sd}


def power_ok(p1_l3):
    """양성 대조: 원 모델 P1이 같은 시험에서 *기준 초과*(TOST 불통과이면서 CI 하한 > 0)로 검출되면 검정력이 있다.
    2026-11-01 수정(코드 검토 C1): 종전에는 하한 > 0만 보아 P1이 통과여도 검정력을 인정했다. 최종 재실행의 집계
    (scripts/summarize_e2.py)는 이미 이 규칙으로 다시 판정했으므로 보고된 표는 바뀌지 않는다."""
    return bool(p1_l3) and p1_l3["class"] == "기준 초과"


def apply_power(l3, p1_l3, kind):
    """근사 방법의 통과를 양성 대조가 실패한 경우 판정 불가(검정력 부족)로 바꾼다. 정확 방법은 L3가 참고값이라 그대로 둔다."""
    if l3 is None:
        return l3
    l3 = {**l3, "power_ok": power_ok(p1_l3)}
    if kind == "approx" and l3["class"] == "통과" and not l3["power_ok"]:
        l3.update({"result": "불통과", "class": "판정 불가", "reason": "검정력 부족"})   # result도 함께 갱신(C3)
    return l3


def deploy(kind, l3):
    """처리 작업 결과 모델의 배포 전이(44번 (a)). kind: exact | approx. L1·L2 통과를 전제로 한다."""
    if kind == "exact":
        return "배포 허용(L1·L2)"
    return "배포 허용(L3 통과)" if l3 and l3["class"] == "통과" else "미완료 → 정확 방법 전환"
