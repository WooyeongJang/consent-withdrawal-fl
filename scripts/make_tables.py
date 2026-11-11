"""부록 결과 표를 최종 재실행 결과에서 만든다(LaTeX 조각 → access/tables/).

python3 scripts/make_tables.py            # results/final/
python3 scripts/make_tables.py --out jbhi/tables --aux   # + JBHI 보충 문서 보조 표
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

SIM_ORDER = [("during", {}, "during training"), ("aftersend", {}, "after sending"),
             ("aggdelay", {"agg_delay": 0.0}, "aggregation delay $0$"),
             ("aggdelay", {"agg_delay": 30.0}, "aggregation delay $0.5L$"),
             ("aggdelay", {"agg_delay": 120.0}, "aggregation delay $2L$"),
             ("version", {"sync_q": 0.3}, "stale policy replica"),
             ("offline", {"offline_m": 3}, "offline and reconnecting")]
CHAIN_LABEL = {"during": "during training", "aftersend": "after sending", "version": "stale policy replica",
               "offline": "offline and reconnecting (2 rounds)"}


def _r1(x):                                                 # 소수 첫째 자리 사사오입(117.85 → 117.9, 부동소수점 표기와 무관)
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def e1_sim(path):
    runs = json.load(open(path))["runs"]
    g = defaultdict(list)
    for r in runs:
        g[(r["scenario"], json.dumps(r["params"], sort_keys=True), r["rule"])].append(r)

    def m(sc, p, rule, f):
        v = g[(sc, json.dumps(p, sort_keys=True), rule)]
        return st.mean(r[f] for r in v), len(v)

    rows = []
    for sc, p, label in SIM_ORDER:
        sel, n = m(sc, p, "sel", "viol_acc")
        cells = [_r1(sel)]
        for rule in ("acc", "strict"):
            viol, _ = m(sc, p, rule, "viol_acc")
            wrong, _ = m(sc, p, rule, "rej_wrong")
            steps, _ = m(sc, p, rule, "recompute_steps")
            cells += [f"{viol:.0f}", _r1(wrong), f"{steps:,.0f}"]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return rf"""\begin{{table}}[t]
\centering
\caption{{E1 simulation, mean per run over {n} seeds (148 withdrawals, 50 rounds). Viol.: violations; wrong: wrong rejections; steps: optimizer steps recomputed after rejections. Selection-time decisions (sel) never reject.}}
\label{{tab:a-e1}}
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}l r r r r r r r@{{}}}}
\toprule
& sel & \multicolumn{{3}}{{c}}{{acc}} & \multicolumn{{3}}{{c}}{{strict}} \\
\cmidrule(lr){{2-2}}\cmidrule(lr){{3-5}}\cmidrule(l){{6-8}}
Condition & viol. & viol. & wrong & steps & viol. & wrong & steps \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def e1_chain(path):
    runs = json.load(open(path))
    rows, note = [], ""
    for r in runs:
        label = CHAIN_LABEL.get(r["scenario"]) or f"aggregation delay {r['params']['D']:.0f}\\,s"
        lat = r["inclusion_latency_s"]
        # 철회 수는 철회한 환자 수로 적는다. 같은 철회를 여러 번 제출한 실행(2026-11-01 코드 검토 A3: during 조건이
        # 데이터센터마다 한 번씩 제출)은 환자 수에 별표를 달고 캡션 각주에서 제출 횟수를 밝힌다.
        pids = {e["pid"] for e in r.get("withdrawal_events", [])}
        wd = str(r["withdrawals"])
        if pids and len(pids) < r["withdrawals"]:
            wd = f"{len(pids)}$^{{\\ast}}$"
            times = {2: "twice", 3: "three times"}.get(r["withdrawals"] // len(pids), f"{r['withdrawals'] // len(pids)} times")
            note = (f" $^{{\\ast}}$The harness submitted each of these {len(pids)} withdrawals {times} ({r['withdrawals']} transactions, "
                    "one per data center); v3.1 accepts such replays, which v3.2 rejects (Section~\\ref{app:v32}).")
        rows.append(f"{label} & {wd} & {r['updates']} & {r['accepted']} & {r['rejections_correct']} & "
                    f"{r['rejections_wrong']} & {r['violations']} & {lat['mean']:.2f} ({lat['min']:.2f}--{lat['max']:.2f}) \\\\")
    rounds = {r["rounds"] for r in runs}
    return rf"""\begin{{table}}[t]
\centering
\caption{{E1 on Kairos under the acceptance-time check (acc), one run per condition ({'/'.join(map(str, sorted(rounds)))} rounds, real local training, decisions judged from block timestamps). Accepted: accepted updates; rej.: correct rejections; wrong: wrong rejections; viol.: violations; inclusion: transaction inclusion latency, mean (range) in seconds.{note}}}
\label{{tab:a-e1chain}}
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}l r r r r r r l@{{}}}}
\toprule
Condition & withdr. & updates & accepted & rej. & wrong & viol. & inclusion (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


E2_LABEL = {"E2-base": r"5\%, late (base)", "E2-rate01": r"1\%, late", "E2-rate10": r"10\%, late",
            "E2-rate20": r"20\%, late", "E2-early": r"5\%, early", "E2-mid": r"5\%, mid",
            "E2-rate01-early": r"1\%, early", "E2-rate01-mid": r"1\%, mid", "E2-rate10-early": r"10\%, early",
            "E2-rate10-mid": r"10\%, mid", "E2-rate20-early": r"20\%, early", "E2-rate20-mid": r"20\%, mid",
            "E2-diagMI": "MI patients", "E2-F65": r"women aged $\ge65$", "E2-inst2": "one institution",
            "E2-seq3": "three waves", "E2-reconsent": "re-consent", "E2-exit2": "institution exit",
            "E2-latejoin": "late-joining cohort"}
E2_DELTA = ["P1", "U1", "U2", "U3", "U4", "U4s"]
E2_L3 = ["P1", "U0", "U1", "U2", "U3", "U4", "U4s"]


def e2_tables(path):
    d = json.load(open(path))
    seeds = sorted({len(v["seeds"]) for v in d.values()})
    rows_d, rows_c = [], []
    for sc, label in E2_LABEL.items():
        if sc not in d:
            continue
        ms = d[sc]["methods"]
        u0 = ms.get("U0")
        cells = []
        for m in E2_DELTA:
            if m in ms and u0:
                cells += [f"{1e3 * (ms[m]['test']['mean'] - u0['test']['mean']):+.1f}",
                          f"{1e2 * (ms[m]['lira']['mean'] - u0['lira']['mean']):+.1f}"]
            else:
                cells += ["--", "--"]
        rows_d.append(f"{label} & " + " & ".join(cells) + r" \\")
        cells = []
        for m in E2_L3:
            c = ms.get(m, {}).get("L3_class")
            cells.append(f"{c['pass']}/{c['residual']}/{c['inconclusive'] + c['lowpower']}/{c['overremoval']}" if c else "--")
        rows_c.append(f"{label} & " + " & ".join(cells) + r" \\")
    n = "/".join(map(str, seeds))
    head_d = " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in E2_DELTA)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(E2_DELTA)))
    return rf"""\begin{{table*}}[t]
\centering
\caption{{E2 per scenario, mean over {n} seeds: test macro AUROC difference from U0 ($\times10^{{-3}}$) and LiRA AUC difference from U0 ($\times10^{{-2}}$).}}
\label{{tab:a-e2}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{@{{}}l {'r ' * (2 * len(E2_DELTA))}@{{}}}}
\toprule
& {head_d} \\
{rules}
Scenario & {' & '.join(['AUROC & LiRA'] * len(E2_DELTA))} \\
\midrule
""" + "\n".join(rows_d) + rf"""
\bottomrule
\end{{tabular}}}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\caption{{E2 per scenario: \Lthree{{}} outcomes as pass / exceeds / inconclusive / over-removal, counted over seeds. Inconclusive includes a failed positive control.}}
\label{{tab:a-e2l3}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{@{{}}l {'c ' * len(E2_L3)}@{{}}}}
\toprule
Scenario & {' & '.join(E2_L3)} \\
\midrule
""" + "\n".join(rows_c) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


L3_RULES = [("main", r"$\delta_{\mathrm{L3}}=0.05$, $z$ (main)"), ("delta0.025", r"$\delta_{\mathrm{L3}}=0.025$, $z$"),
            ("t", r"$\delta_{\mathrm{L3}}=0.05$, $t$"), ("delta0.025_t", r"$\delta_{\mathrm{L3}}=0.025$, $t$")]
L3_METHODS = ["U0", "U1", "U2", "U3", "U4", "U4s"]


def l3_sens(path):
    d = json.load(open(path))
    rows = []
    for g, gl in (("all", "all"), ("late", "late")):
        for rule, label in L3_RULES:
            r = d["rules"][rule][g]
            rows.append(f"{label} & {gl} & " + " & ".join(f"{r[m]['k']}/{r[m]['n']}" for m in L3_METHODS) + r" \\")
        if g == "all":
            rows.append(r"\midrule")
    tpr = d["tpr_at_fpr1_late"]
    tline = ", ".join(f"{m} {tpr[m]['mean']:.3f}" for m in ["P1"] + L3_METHODS if m in tpr)
    return rf"""\begin{{table}}[h]
\centering
\caption{{\Lthree{{}} sensitivity: runs passed under each margin and CI quantile ($t$: $n_{{\mathrm{{ref}}}}-1$ degrees of freedom), with the positive control re-applied under the same rule. U0 is tested against the other references. Late: late-withdrawal scenarios. LiRA TPR at 1\% FPR (late, mean): {tline}.}}
\label{{tab:a-l3sens}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}l l {'c ' * len(L3_METHODS)}@{{}}}}
\toprule
Rule & Runs & {' & '.join(L3_METHODS)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


# ---- 보조 결과 표(JBHI 보충 문서, 2026-10-31) --------------------------------
AUX = [("E2_e2_k5.json", "E2-K5-base", r"$K=5$, 5\%, late", ""),
       ("E2_e2_leadI.json", "E2-base", r"lead~I, 5\%, late", ""),
       ("E2_e2_leadI.json", "E2-mid", r"lead~I, 5\%, mid", ""),
       ("E2_e2_c2021.json", "C2021-base", r"Challenge 2021, late", ""),
       ("E2_e2_c2021.json", "C2021-mid", r"Challenge 2021, mid", ""),
       ("E2_e2_perwave.json", "E2-seq3-perwave", "three waves, per wave", "-seq")]


def _l3(c):
    return f"{c['pass']}/{c['residual']}/{c['inconclusive'] + c['lowpower']}/{c['overremoval']}" if c else "--"


def aux_tables(root):
    rows_d, rows_c, ns = [], [], set()
    for fn, sc, label, suf in AUX:
        d = json.load(open(os.path.join(root, fn)))[sc]
        ms, u0 = d["methods"], d["methods"]["U0" + suf]
        ns.add(len(d["seeds"]))
        cells = []
        for m in E2_DELTA:
            mm = ms.get(m if m in ("P1",) else m + suf)
            cells += ([f"{1e3 * (mm['test']['mean'] - u0['test']['mean']):+.1f}",
                       f"{1e2 * (mm['lira']['mean'] - u0['lira']['mean']):+.1f}"] if mm else ["--", "--"])
        rows_d.append(f"{label} & {u0['test']['mean']:.4f} & " + " & ".join(cells) + r" \\")
        rows_c.append(f"{label} & " + " & ".join(_l3(ms.get(m if m == 'P1' else m + suf, {}).get("L3_class"))
                                                 for m in E2_L3) + r" \\")
    n = "/".join(map(str, sorted(ns)))
    head_d = " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in E2_DELTA)
    rules = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(E2_DELTA)))
    return rf"""\begin{{table*}}[t]
\centering
\caption{{E2 auxiliary conditions, mean over {n} seeds: U0 test macro AUROC, and differences from U0 in test macro AUROC ($\times10^{{-3}}$) and LiRA AUC ($\times10^{{-2}}$). Challenge 2021: $K=4$ cohorts, record-level withdrawal. Per wave: sequential withdrawals (rounds 15, 30, 45) processed after each wave; methods and U0 are the per-wave (``-seq'') versions.}}
\label{{tab:a-aux}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{@{{}}l r {'r ' * (2 * len(E2_DELTA))}@{{}}}}
\toprule
& U0 & {head_d} \\
{rules}
Condition & AUROC & {' & '.join(['AUROC & LiRA'] * len(E2_DELTA))} \\
\midrule
""" + "\n".join(rows_d) + rf"""
\bottomrule
\end{{tabular}}}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\caption{{E2 auxiliary conditions: \Lthree{{}} outcomes as pass / exceeds / inconclusive / over-removal, counted over seeds. Inconclusive includes a failed positive control.}}
\label{{tab:a-auxl3}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\adjustbox{{max width=\textwidth}}{{%
\begin{{tabular}}{{@{{}}l {'c ' * len(E2_L3)}@{{}}}}
\toprule
Condition & {' & '.join(E2_L3)} \\
\midrule
""" + "\n".join(rows_c) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""


SUB_METHODS = ["P1", "U0", "U1", "U2", "U3", "U4", "U4s"]


def subgroup_table(root):
    b = json.load(open(os.path.join(root, "E2_e2.json")))["E2-base"]
    k5 = json.load(open(os.path.join(root, "E2_e2_k5.json")))["E2-K5-base"]
    cols = [(b, "F65"), (b, "inst0"), (b, "inst1"), (b, "inst2"), (k5, "inst3"), (k5, "inst4")]
    rows = []
    for m in SUB_METHODS:
        cells = []
        for d, g in cols:
            v = d["methods"].get(m, {}).get("subgroup", {}).get(g)
            cells.append(f"{v['mean']:.4f}" if v else "--")
        rows.append(f"{m} & " + " & ".join(cells) + r" \\")
    nb, nk = len(b["seeds"]), len(k5["seeds"])

    def _sub_n(pattern):                                     # 하위 집단 평가 기록 수(시드와 무관), 원 결과 첫 파일에서
        import glob

        def walk(x):
            if isinstance(x, dict):
                if "subgroup" in x:
                    return {g: v["n"] for g, v in x["subgroup"].items()}
                x = list(x.values())
            for v in x if isinstance(x, list) else []:
                r = walk(v)
                if r:
                    return r
        return walk(json.load(open(sorted(glob.glob(os.path.join(root, pattern)))[0])))
    nb_n, nk_n = _sub_n("e2/E2-base_s*.json"), _sub_n("e2_k5/E2-K5-base_s*.json")
    ncells = [nb_n[g] if d is b else nk_n[g] for d, g in cols]
    rows.append(r"\midrule")
    rows.append("Test records & " + " & ".join(f"{n:,}" for n in ncells) + r" \\")
    return rf"""\begin{{table}}[h]
\centering
\caption{{E2 subgroup test macro AUROC, base scenario (5\%, late). $K=3$: women aged $\ge65$ (F65) and the test records of each data center (DC), mean over {nb} seeds. $K=5$: the two smallest data centers, mean over {nk} seeds.}}
\label{{tab:a-sub}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}l c c c c c c@{{}}}}
\toprule
& \multicolumn{{4}}{{c}}{{$K=3$}} & \multicolumn{{2}}{{c}}{{$K=5$}} \\
\cmidrule(lr){{2-5}}\cmidrule(l){{6-7}}
Method & F65 & DC1 & DC2 & DC3 & DC4 & DC5 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


OSD = [("U4", "conditional", "all labels"), ("U4-pa", "always", "all labels"), ("U4-up", "conditional", "positive labels"),
       ("U4-off", "always", "positive labels"), ("U4s", "conditional", "all labels"), ("U4s-off", "always", "positive labels")]


def osd_table(root):
    d = json.load(open(os.path.join(root, "E2_e2_osdcheck.json")))["E2-base-osdcheck"]
    ms, u0 = d["methods"], d["methods"]["U0"]
    rows = []
    for m, pp, uc in OSD:
        x = ms[m]
        rows.append(f"{m} & {pp} & {uc} & {1e3 * (x['test']['mean'] - u0['test']['mean']):+.1f} & {x['lira']['mean']:.3f} & "
                    f"{1e2 * (x['lira']['mean'] - u0['lira']['mean']):+.1f} & {_l3(x['L3_class'])} \\\\")
    return rf"""\begin{{table}}[h]
\centering
\caption{{FedOSD variants, base scenario (5\%, late; {len(d['seeds'])} seeds). Post-projection: conditional (ours) or always (official code); unlearning loss over all labels (ours) or positive labels only (official multi-label form). U4: 5 steps at 0.005; U4s: 20 steps at 0.025. $\Delta$: difference from U0 (AUROC $\times10^{{-3}}$, LiRA $\times10^{{-2}}$); \Lthree{{}}: pass / exceeds / inconclusive / over-removal.}}
\label{{tab:a-osd}}
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}l l l r r r c@{{}}}}
\toprule
Variant & Post-proj. & Loss & $\Delta$AUROC & LiRA & $\Delta$LiRA & \Lthree{{}} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def micro_table(root):
    d = json.load(open(os.path.join(root, "e4/E4_micro.json")))
    rows = [f"{r['n']:,} & {1e3 * r['build_s']:.2f} & {1e6 * r['prove_s']:.1f} & {1e6 * r['verify_s']:.1f} & {r['proof_bytes_json']:,.0f} \\\\"
            for r in d["merkle"]]
    return rf"""\begin{{table}}[h]
\centering
\caption{{E4 micro-benchmarks: sorted-Merkle tree build, non-membership proof generation and verification, and proof size (JSON). Hashing the model ({d['param_bytes'] / 1e6:.1f}\,MB of parameters) took {1e3 * d['model_hash_s']:.1f}\,ms and computing a binding value {1e6 * d['binding_value_s']:.1f}\,$\mu$s.}}
\label{{tab:a-micro}}
\footnotesize
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}r r r r r@{{}}}}
\toprule
Records & Build (ms) & Prove ($\mu$s) & Verify ($\mu$s) & Proof (B) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def latency_table(root):
    import glob
    runs = {}
    for f in glob.glob(os.path.join(root, "dist/lat_*/aggregator.json")):
        r = json.load(open(f))
        runs[(r["K"], r["delay_ms"], r["rate_mbps"])] = r
    rates, delays, rows, notes = [0.0, 100.0, 10.0], [0.0, 50.0, 200.0], [], []
    for K in sorted({k for k, _, _ in runs}):
        hashes = {runs[(K, dl, rt)]["final_hash"] for dl in delays for rt in rates}
        r0 = runs[(K, 0.0, 0.0)]
        dc = r0["comm"]["0"]["by_path"]["/round"]
        notes.append(f"$K={K}$: {len(hashes)} distinct final hash{'es' if len(hashes) > 1 else ''} over the nine settings, "
                     rf"{(dc['sent'] + dc['recv']) / r0['rounds'] / 1e6:.1f}\,MB per round with data center 1")
        for dl in delays:
            cells = [f"{st.mean(x['wall_s'] for x in runs[(K, dl, rt)]['per_round']):.2f}" for rt in rates]
            rows.append(f"{K} & {dl:.0f} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows.pop()
    nr = {r["rounds"] for r in runs.values()}
    return rf"""\begin{{table}}[h]
\centering
\caption{{E4 latency and bandwidth sensitivity of the multi-container deployment: mean wall-clock seconds per round ($R={'/'.join(map(str, sorted(nr)))}$). Delay is an artificial one-way delay applied to each request and each response; bandwidth is an artificial per-link rate. {'; '.join(notes)}.}}
\label{{tab:a-lat}}
\footnotesize
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}r r r r r@{{}}}}
\toprule
& & \multicolumn{{3}}{{c}}{{Bandwidth}} \\
\cmidrule(l){{3-5}}
$K$ & Delay (ms) & unlimited & 100\,Mb/s & 10\,Mb/s \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def l1cost_table(root):
    ps = ["0.1", "0.3", "0.5", "1.0"]
    rows = []
    for K in (3, 5, 10):
        d = json.load(open(os.path.join(root, f"e4/E4_K{K}.json")))
        rows.append(f"{K} & {d['l1_unit_mean_s']:.2f} & {d['time']['per_round_train_s']:.2f} & "
                    + " & ".join(f"{d['l1_cost_by_p'][p]:.2f}" for p in ps) + r" \\")
    det = " & ".join(f"{1 - (1 - float(p)) ** 3:.2f}" for p in ps)
    return rf"""\begin{{table}}[h]
\centering
\caption{{\Lone{{}} sampling cost (E4, $R=10$): mean re-execution time per unit, training time per round, and expected re-execution time per round, $pK\times$ unit time, for sampling rate $p$. Detection probability for $m$ falsified units is $1-(1-p)^m$; the last row gives it for $m=3$ (for $m=1$ it equals $p$).}}
\label{{tab:a-l1cost}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\adjustbox{{max width=\columnwidth}}{{%
\begin{{tabular}}{{@{{}}r r r r r r r@{{}}}}
\toprule
& Unit & Train/round & \multicolumn{{4}}{{c}}{{Re-execution per round (s) at $p$}} \\
\cmidrule(l){{4-7}}
$K$ & (s) & (s) & 0.1 & 0.3 & 0.5 & 1.0 \\
\midrule
""" + "\n".join(rows) + rf"""
\midrule
\multicolumn{{3}}{{@{{}}l}}{{detection, $m=3$}} & {det} \\
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/final")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--aux", action="store_true", help="보조 결과 표도 만든다(JBHI 보충 문서, Access 부록)")
    ap.add_argument("--term", choices=["data center", "institution"], default="data center",
                    help="보조 표의 기관 명칭(JBHI: data center, Access: institution)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = [("a_e1.tex", e1_sim(os.path.join(a.root, "e1/E1.json"))),
           ("a_e1chain.tex", e1_chain(os.path.join(a.root, "e1_chain/E1_chain.json")))]
    e2 = os.path.join(a.root, "E2_e2.json")                 # 집계 단계(run_final.sh summary) 이후에만 있다
    if os.path.exists(e2):
        out.append(("a_e2.tex", e2_tables(e2)))
    else:
        print("skip E2 tables:", e2, "not found")
    sens = os.path.join(a.root, "L3_sensitivity.json")      # scripts/l3_sensitivity.py results/final/e2 results/final/L3_sensitivity.json
    if os.path.exists(sens):
        out.append(("a_l3sens.tex", l3_sens(sens)))
    else:
        print("skip L3 sensitivity table:", sens, "not found")
    if a.aux:                                               # JBHI 보충 문서용 보조 표
        out += [("a_aux.tex", aux_tables(a.root)), ("a_sub.tex", subgroup_table(a.root)),
                ("a_osd.tex", osd_table(a.root)), ("a_micro.tex", micro_table(a.root)),
                ("a_lat.tex", latency_table(a.root)), ("a_l1cost.tex", l1cost_table(a.root))]
        if a.term == "institution":                         # Access판 용어: 기관 번호 열 DCk → Ik
            aux = {"a_aux.tex", "a_sub.tex", "a_osd.tex", "a_micro.tex", "a_lat.tex", "a_l1cost.tex"}
            out = [(n, t.replace("data centers", "institutions").replace("data center", "institution")
                     .replace("(DC)", "(I)").replace(" & DC", " & I")) if n in aux else (n, t) for n, t in out]
    for name, text in out:
        open(os.path.join(a.out, name), "w").write(text)
        print("wrote", os.path.join(a.out, name))


if __name__ == "__main__":
    main()
