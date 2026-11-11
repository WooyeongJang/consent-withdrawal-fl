"""통계 분석 도구 (3장 v1.6 8.5).

- 연속 지표: 평균과 95% CI(t 분포)
- 조건 간 비교(같은 시드 대응): 대응 t 검정(Shapiro–Wilk로 정규성이 기각되면 윌콕슨 부호순위) + 효과크기
  (대응 t: Cohen's d_z, 윌콕슨: 순위 이연 상관 r_rb)
- 탐지율·오탐률: 윌슨 CI
- AUROC·멤버십 추론 AUC: 부트스트랩 CI(fl.exp_lira.evaluate와 같은 방식)
- 다중 비교: Holm 보정
- 표본크기: 대응 비교의 정규 근사 n = ((z_{1−α′/2} + z_{1−β}) σ_d / Δ)², 그리고 윌콕슨 정확 검정이 α′에 도달할 수 있는 최소 n(2/2ⁿ ≤ α′)
"""

import math

import numpy as np
from scipy import stats as st


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    m = float(x.mean())
    if len(x) < 2:
        return {"mean": m, "ci": [None, None], "n": len(x)}
    h = float(st.t.ppf((1 + conf) / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return {"mean": m, "ci": [m - h, m + h], "sd": float(x.std(ddof=1)), "n": len(x)}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def paired(a, b):
    """a − b의 대응 비교. 반환: 검정 종류, p값, 효과크기, 차이의 평균·CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    out = {"diff": mean_ci(d), "n": len(d)}
    if len(d) < 3 or np.allclose(d, d[0]):
        out.update({"test": None, "p": None})
        return out
    normal = st.shapiro(d).pvalue > 0.05
    if normal:
        r = st.ttest_rel(a, b)
        out.update({"test": "paired_t", "p": float(r.pvalue), "effect": {"d_z": float(d.mean() / d.std(ddof=1))}})
    else:
        r = st.wilcoxon(a, b)
        n = np.count_nonzero(d)
        ranks = st.rankdata(np.abs(d[d != 0]))
        rpos, rneg = ranks[d[d != 0] > 0].sum(), ranks[d[d != 0] < 0].sum()
        out.update({"test": "wilcoxon", "p": float(r.pvalue), "effect": {"r_rb": float((rpos - rneg) / (n * (n + 1) / 2))}})
    out["shapiro_normal"] = bool(normal)
    return out


def holm(pvals):
    """Holm 보정된 p값(입력 순서 유지). None은 그대로 둔다."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    adj, run = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - rank) * pvals[i]))
        adj[i] = run
    return adj


def sample_size(sd_diff, delta, alpha=0.05, power=0.8, m=1):
    """대응 비교 m개에 Bonferroni 수준 α/m(Holm의 가장 엄격한 단계)을 적용할 때 필요한 n."""
    a = alpha / m
    z = st.norm.ppf(1 - a / 2) + st.norm.ppf(power)
    n_t = math.ceil((z * sd_diff / delta) ** 2)
    n_w = next(n for n in range(2, 64) if 2 / 2 ** n <= a)     # 윌콕슨 정확 검정의 최소 도달 p ≤ α′
    return {"n_normal": max(n_t, 2), "n_wilcoxon_min": n_w, "alpha_per_test": a, "z": z,
            "inputs": {"sd_diff": sd_diff, "delta": delta, "alpha": alpha, "power": power, "m": m}}
