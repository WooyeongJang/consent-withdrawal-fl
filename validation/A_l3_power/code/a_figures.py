"""실험 A 그림. 기존 원고 그림(scripts/make_figures.py)과 같은 글꼴·색 규칙을 쓴다.

fig_l3_power.pdf  (a) 후기 철회 1/5/10/20%에서 참 차이 Δ별 통과·기준 초과 확률(모의, M2, 참조 5개; M1·M3의 통과 곡선은 가는 선)
                  (b) 양성 대조 검출 곡선(모의)과 시나리오 19개의 관측 빈도(P1, 시드 10개)
모의값은 선, 관측값은 표식으로 구분한다.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # 철회 규모 1/5/10/20%(순서형, 한 색상)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02})

sim = json.load(open(os.path.join(RES, "a1_simulated.json")))["scenarios"]
obs = json.load(open(os.path.join(RES, "a1_observed.json")))["scenarios"]
MAIN = [("E2-rate01", "1%"), ("E2-base", "5%"), ("E2-rate10", "10%"), ("E2-rate20", "20%")]

fig = plt.figure(figsize=(7.16, 2.05))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.22, 1.55], wspace=0.16)
for i, (sc, lab) in enumerate(MAIN):
    ax = fig.add_subplot(gs[0, i])
    m = sim[sc]["models"]
    d = m["M2"]["5"]["curve"]["delta"]
    ax.axvspan(-0.05, 0.05, color="#f1f0ea", lw=0, zorder=0)
    ax.axhline(0.05, color=MUTED, lw=0.5, ls=(0, (1, 2)))
    ax.plot(d, m["M1"]["5"]["curve"]["pass"], color=BLUE, lw=0.7, ls=(0, (4, 2)), label="pass, M1")
    if "M3" in m:
        ax.plot(d, m["M3"]["5"]["curve"]["pass"], color=BLUE, lw=0.7, ls=(0, (1, 1.5)), label="pass, M3")
    ax.plot(d, m["M2"]["5"]["curve"]["pass"], color=BLUE, label="pass, M2")
    ax.plot(d, m["M2"]["5"]["curve"]["exceeds"], color=ORANGE, label="exceeds, M2")
    ax.set_xlim(-0.10, 0.15); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15]); ax.set_xticklabels(["", "$-$.05", "0", ".05", ".10", ""])
    ax.set_title(f"{lab} ($|X|$={obs[sc]['nX']})", pad=3)
    if i == 0:
        ax.set_ylabel("probability of outcome")
        ax.text(0.0, 0.2, "pass\nimpossible", ha="center", va="center", fontsize=6.3, color=INK2)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.0), handlelength=1.5, borderpad=0, labelspacing=0.2, handletextpad=0.4, fontsize=6.2)
    else:
        ax.set_yticklabels([])
    if i == 1:
        ax.set_xlabel(r"true AUC difference $\Delta$ from retraining" + " " * 28, labelpad=2)
fig.text(0.012, 0.99, "(a)", fontsize=8, va="top")

ax = fig.add_subplot(gs[0, 5])
for (sc, lab), col in zip(MAIN, RAMP):
    pc = sim[sc]["positive_control"]["5"]["curve"]
    ax.plot(pc["delta"], pc["exceeds"], color=col, label=lab)
MK = {"late": "o", "mid": "s", "early": "^"}
for sc, o in obs.items():
    p1 = o["methods"]["P1"]
    timing = "early" if sc.endswith("early") else "mid" if sc.endswith("mid") else "late"
    nx = o["nX"]
    col = RAMP[0] if nx < 300 else RAMP[1] if nx < 1200 else RAMP[2] if nx < 2500 else RAMP[3]
    ax.plot(p1["diff_mean"], p1["class"].get("exceeds", 0) / 10, MK[timing], ms=3.6, mfc="white", mec=col, mew=0.9, zorder=3)
for t, mk in MK.items():
    ax.plot([], [], mk, ms=3.6, mfc="white", mec=INK2, mew=0.9, label=f"observed, {t}")
ax.axvline(0.05, color=MUTED, lw=0.5, ls=(0, (1, 2)))
ax.set_xlim(-0.02, 0.15); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(r"residual of the blocking-only model (AUC)", labelpad=2)
ax.set_ylabel("P(control detects)", labelpad=2)
ax.legend(loc="lower right", handlelength=1.4, borderpad=0, labelspacing=0.22, ncol=1)
fig.text(0.70, 0.99, "(b)", fontsize=8, va="top")
out = os.path.join(RES, "fig_l3_power.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=220)
print(out)
