"""IEEE 원고 결과 그림(Fig. 4 E1, Fig. 5 E2, Fig. 6 E4, Fig. A1 E3 p 격자)을 만든다.

python3 scripts/make_figures.py            # 예비 결과(results/)
python3 scripts/make_figures.py --final    # 최종 재실행(results/final/)
출력: access/figures/fig_{e1,e2,e4,a1_p}.pdf. 색은 dataviz 기본 범주 팔레트 1–7번(밝은 면, 검증 통과),
대비가 3:1 아래인 색은 표식 모양과 범례로 보완한다.
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL_W = 3.5      # IEEE 단 너비(인치)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02})


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return xs.mean(), 0.0
    from scipy import stats
    return xs.mean(), stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / math.sqrt(len(xs))


# ---------------------------------------------------------------- Fig. 4 (E1)
E1_COND = [("during", {}, "during training"), ("aftersend", {}, "after sending"),
           ("aggdelay", {"agg_delay": 0.0}, "agg. delay 0"), ("aggdelay", {"agg_delay": 30.0}, "agg. delay 0.5L"),
           ("aggdelay", {"agg_delay": 120.0}, "agg. delay 2L"), ("version", {"sync_q": 0.3}, "stale replica"),
           ("offline", {"offline_m": 3}, "offline 3 rounds")]


def fig_e1(path, out):
    runs = json.load(open(path))["runs"]
    sel = lambda sc, pa, ru: [r for r in runs if r["scenario"] == sc and r["params"] == pa and r["rule"] == ru]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.65), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.1},
                               sharey=True)
    y = np.arange(len(E1_COND))[::-1]
    zero = lambda xs: all(v == 0 for v in xs)
    # (a) 위반: 수락 시점에 이미 효력이 있던 철회자 기록이 수락된 갱신에 든 건수(실행당 평균, 95% CI)
    m, e = zip(*[ci95([r["viol_acc"] for r in sel(sc, pa, "sel")]) for sc, pa, _ in E1_COND])
    a.barh(y, m, height=0.6, xerr=e, color=SERIES[0], error_kw={"lw": 0.6, "ecolor": INK2})
    for yi, mi in zip(y, m):
        a.text(mi + 8, yi, f"{mi:.0f}", va="center", ha="left", fontsize=6, color=INK2)
    others = [ru for ru in ("acc", "strict") if zero([r["viol_acc"] for sc, pa, _ in E1_COND for r in sel(sc, pa, ru)])]
    a.set_yticks(y, [c[2] for c in E1_COND])
    a.set_xlim(0, max(m) * 1.22)
    a.set_xlabel("violations per run")
    a.grid(axis="y", visible=False)
    a.set_title("(a) violations: sel" + (" (%s: 0)" % ", ".join(others) if others else ""), loc="left", color=INK)
    # (b) 잘못된 거절 ÷ 전체 거절(시드 합), strict
    share = []
    for sc, pa, _ in E1_COND:
        rs = sel(sc, pa, "strict")
        w, c = sum(r["rej_wrong"] for r in rs), sum(r["rej_correct"] for r in rs)
        share.append(w / (w + c) if w + c else 0.0)
    # acc의 잘못된 거절(구성상 0)도 막대로 보인다(JBHI 심사 Minor 4, 2026-11-01; 같은 E1 집계)
    acc_share = []
    for sc, pa, _ in E1_COND:
        rs = sel(sc, pa, "acc")
        w, c = sum(r["rej_wrong"] for r in rs), sum(r["rej_correct"] for r in rs)
        acc_share.append(w / (w + c) if w + c else 0.0)
    b.barh(y + 0.17, share, height=0.32, color=SERIES[2], label="strict")
    for yi, s_ in zip(y, share):
        b.text(s_ + 0.01, yi + 0.17, f"{s_:.2f}", va="center", ha="left", fontsize=6, color=INK2)
    b.barh(y - 0.17, acc_share, height=0.32, color=SERIES[1], label="acc")
    for yi, s_ in zip(y, acc_share):
        b.text(s_ + 0.01, yi - 0.17, f"{s_:.2f}" if s_ else "0", va="center", ha="left", fontsize=6, color=INK2)
    b.set_xlim(0, 0.72)
    b.set_xlabel("wrong / all rejections")
    b.grid(axis="y", visible=False)
    b.legend(loc="upper right", fontsize=5.5, handlelength=0.8, labelspacing=0.2, borderaxespad=0.2)
    b.set_title("(b) wrong rejections: strict, acc", loc="left", color=INK)
    b.tick_params(axis="y", length=0)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5 (E2)
E2_METHODS = [("P1", "P1 blocking only"), ("U1", "U1 replay"), ("U2", "U2 FedEraser"), ("U3", "U3 SISA"),
              ("U4", "U4 FedOSD"), ("U4s", "U4s FedOSD-large")]


def is_late(sc):
    """scripts/l3_sensitivity.py late()와 같은 기준: 시나리오 정의의 철회 시점(재동의 시나리오는 중기 철회)."""
    from fl.consent import SPECS
    s = SPECS.get(sc, {})
    return s.get("timing") == "late" or "waves" in s


def l3_mark(cls):
    if not cls:
        return "inconclusive"
    top = max(cls, key=cls.get)
    return {"pass": "pass", "residual": "exceeds", "overremoval": "over-removal"}.get(top, "inconclusive")


MARK = {"pass": dict(marker="o", fill=True), "exceeds": dict(marker="X", fill=True), "over-removal": dict(marker="v", fill=True),
        "inconclusive": dict(marker="o", fill=False)}


def fig_e2(path, out):
    d = json.load(open(path))
    scen = sorted(d)
    groups = [("(a) late", [s for s in scen if is_late(s)]), ("(b) early or mid", [s for s in scen if not is_late(s)])]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, ss) in zip(axes, groups):
        ax.axvspan(-0.05, 0.05, color=GRID, alpha=0.6, lw=0)
        ax.axvline(0, color=AXIS, lw=0.6)
        ax.axhline(0, color=AXIS, lw=0.6)
        for i, (m, _) in enumerate(E2_METHODS):
            for s in ss:
                cmp = d[s]["compare_vs_U0"]
                if f"{m}:lira" not in cmp:
                    continue
                x, yv = cmp[f"{m}:lira"]["diff"]["mean"], cmp[f"{m}:test"]["diff"]["mean"]
                k = MARK[l3_mark(d[s]["methods"][m].get("L3_class"))]
                ax.scatter(x, yv, s=16, marker=k["marker"], linewidths=0.8, zorder=3,
                           facecolors=SERIES[i] if k["fill"] else "white", edgecolors=SERIES[i])
        ax.set_title(f"{title} ({len(ss)} scenarios)", loc="left", color=INK)
        ax.set_xlabel(r"$\Delta$ LiRA AUC vs. U0")
    axes[0].set_ylabel(r"$\Delta$ test AUROC vs. U0")
    hm = [plt.Line2D([], [], ls="", marker="s", ms=4.5, color=SERIES[i], label=n) for i, (_, n) in enumerate(E2_METHODS)]
    hc = [plt.Line2D([], [], ls="", marker=MARK[c]["marker"], ms=4.5, mec=INK2, mfc=INK2 if MARK[c]["fill"] else "white",
                     label=f"L3 {c}") for c in ("pass", "exceeds", "over-removal", "inconclusive")]
    fig.legend(handles=hm + hc, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.47), columnspacing=0.9,
               handletextpad=0.2)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6 (E4)
GROUPS = [("per subject", ["registerPolicy"]), ("per withdrawal", ["requestWithdrawal", "declareFirstUse"]),
          ("per institution-round", ["commitInput", "commitUpdate", "decide"]),
          ("per round", ["openRound", "aggregate", "sampleAudit"])]


E4_LABELS = {"per subject": "per subject", "per withdrawal": "per withdrawal",
             "per institution-round": "per institution-round", "per round": "per round", "atitle": "(a) measured ($R$=10, 100 subj.)",
             "subjects": "subjects"}
E4_LABELS_JBHI = {**E4_LABELS, "per subject": "per patient", "per institution-round": "per data-center round",
                  "atitle": "(a) measured ($R$=10)", "subjects": "patients"}


def fig_e4(e4dir, out, labels=E4_LABELS):
    runs = {K: json.load(open(os.path.join(e4dir, f"E4_K{K}.json"))) for K in (3, 5, 10)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.9), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.42})
    xs = np.arange(3)
    left = np.zeros(3)
    for j, (g, fns) in enumerate(GROUPS):
        v = np.array([sum(runs[K]["fn"][f]["fee_kaia"] for f in fns if f in runs[K]["fn"]) for K in (3, 5, 10)])
        a.bar(xs, v, bottom=left, width=0.62, color=SERIES[j], edgecolor="white", linewidth=0.8, label=labels[g])
        left += v
    for x, K in zip(xs, (3, 5, 10)):
        a.text(x, left[x] + 0.015, f"{runs[K]['fee_kaia']:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    a.set_xticks(xs, ["$K$=3", "5", "10"])
    a.set_ylabel("fee per run (KAIA)")
    a.set_ylim(0, max(left) * 1.18)
    a.grid(axis="x", visible=False)
    a.set_title(labels["atitle"], loc="left", color=INK)
    # (b) 외삽: 사건별 단가(K=3 측정) × 사건 수, R=50
    r3 = runs[3]
    unit = {g: sum(r3["fn"][f]["fee_kaia"] for f in fns) / {"per subject": r3["subjects"],
                                                          "per withdrawal": r3["withdrawals"],
                                                          "per institution-round": r3["K"] * r3["rounds"],
                                                          "per round": r3["rounds"]}[g] for g, fns in GROUPS}
    N = np.logspace(2, 5, 50)
    for j, w in enumerate((0.01, 0.05, 0.10, 0.20)):
        fee = N * unit["per subject"] + N * w * unit["per withdrawal"] + 50 * (3 * unit["per institution-round"] +
                                                                             unit["per round"])
        b.plot(N, fee, color=SERIES[j], label=f"{int(w * 100)}% withdraw")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(labels["subjects"]); b.set_ylabel("fee (KAIA), $K$=3, $R$=50")
    b.legend(loc="upper left", handlelength=1.2, borderaxespad=0.1)
    b.set_title("(b) extrapolated", loc="left", color=INK)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.0)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. A1 (E3 p 격자)
def fig_a1(path, out):
    g = json.load(open(path))["p_grid"]
    fig, ax = plt.subplots(figsize=(COL_W * 0.75, 1.75))
    pp = np.linspace(0, 1, 101)
    ax.plot(pp, pp, color=MUTED, lw=0.9, ls="--", label=r"$1-(1-p)^m$, $m=1$")
    for j, (t, mk) in enumerate((("T2b", "o"), ("T10", "s"))):
        if t not in g:
            continue
        ps = sorted(g[t], key=float)
        x = np.array([float(p) for p in ps]) + (j - 0.5) * 0.012
        r = np.array([g[t][p]["rate"] for p in ps])
        lo = np.array([g[t][p]["wilson95"][0] for p in ps]); hi = np.array([g[t][p]["wilson95"][1] for p in ps])
        ax.errorbar(x, r, yerr=[r - lo, hi - r], fmt=mk, ms=4, color=SERIES[j], ecolor=SERIES[j], elinewidth=0.7,
                    capsize=1.5, label=f"{t} ($n$={g[t][ps[0]]['n']})")
    ax.set_xlabel(r"\Lone{} sampling rate $p$".replace(r"\Lone{}", "L1"))
    ax.set_ylabel("detection rate")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", handlelength=1.4)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="results/final/ 사용")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--jbhi", action="store_true", help="JBHI 원고 용어(patient, data center)로 그림 E4의 범례를 쓴다")
    a = ap.parse_args()
    if a.final:
        src = dict(e1="results/final/e1/E1.json", e2="results/final/E2_e2.json", e3="results/final/E3.json",
                   e4="results/final/e4")
    else:
        src = dict(e1="results/e1/E1.json", e2="results/E2.json", e3="results/E3.json", e4="results/e4_solo")
    os.makedirs(a.out, exist_ok=True)
    fig_e1(src["e1"], os.path.join(a.out, "fig_e1.pdf"))
    fig_e2(src["e2"], os.path.join(a.out, "fig_e2.pdf"))
    fig_e4(src["e4"], os.path.join(a.out, "fig_e4.pdf"), E4_LABELS_JBHI if a.jbhi else E4_LABELS)
    fig_a1(src["e3"], os.path.join(a.out, "fig_a1_p.pdf"))
    print("sources:", json.dumps(src, ensure_ascii=False))


if __name__ == "__main__":
    main()
