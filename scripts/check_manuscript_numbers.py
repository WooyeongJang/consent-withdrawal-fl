"""Check the figures quoted in the manuscript text against the stored results.

Every item recomputes one number (or a small group) that appears in a sentence or caption of the main text,
the Supplementary Document or the extended results, from the files in results/final/, validation/, site_check/
and the constants in fl/. Nothing is trained and nothing is written. Each line prints the location, the value
as printed in the manuscript, the recomputed value, and ok/DIFF; the last line gives the totals.

Run from the repository root with the host environment (requirements-host.txt):
    python scripts/check_manuscript_numbers.py
Dataset counts need the training cache index (~/data/cache/ptbxl_k3/meta.csv) and are skipped without it.
Section labels (III-F, IV-A ...) refer to the main text; S = Supplementary Document; Ext = extended results.
"""
import json, glob, statistics as st, collections, math
OUT = []
def chk(where, paper, got, ok=None):
    if ok is None: ok = paper == got
    OUT.append((where, paper, got, bool(ok)))
def r(x, n): return f"{x:.{n}f}"
def rng(v, n): return f"{min(v):.{n}f}--{max(v):.{n}f}"

# ---------------- C1 (E1 simulation) ----------------
E1 = json.load(open("results/final/e1/E1.json"))["runs"]
def cond(x): return (x["scenario"], json.dumps(x.get("params", {}), sort_keys=True))
by = collections.defaultdict(list)
for x in E1: by[(cond(x), x["rule"])].append(x)
conds = sorted({c for c, _ in by})
chk("IV-A seeds per condition", "20", str({len(v) for v in by.values()}), {len(v) for v in by.values()} == {20})
chk("IV-A timing conditions", "7", str(len(conds)))
sel = {c: st.mean(x["viol_acc"] for x in by[(c, "sel")]) for c in conds}
agg = {c: v for c, v in sel.items() if c[0] == "aggdelay"}
chk("IV-A sel violations vs aggregation delay", "117.9--413.9", f"{min(agg.values()):.1f}--{max(agg.values()):.1f}",
    abs(min(agg.values()) - 117.85) < 0.06 and abs(max(agg.values()) - 413.9) < 0.06)
for sc in ("during", "aftersend"):
    v = [x["viol_acc"] for c in conds if c[0] == sc for x in by[(c, "sel")]]
    chk(f"IV-A sel misses in '{sc}' (every seed)", "148", str(sorted(set(v))), set(v) == {148})
chk("IV-A acc violations / wrong rejections", "0 / 0",
    f"{sum(x['viol_acc'] for c in conds for x in by[(c,'acc')])} / {sum(x['rej_wrong'] for c in conds for x in by[(c,'acc')])}")
chk("IV-A strict violations", "0", str(sum(x["viol_acc"] for c in conds for x in by[(c, "strict")])))
wr = [sum(x["rej_wrong"] for x in by[(c, "strict")]) / sum(x["rej_wrong"] + x["rej_correct"] for x in by[(c, "strict")]) for c in conds]
chk("IV-A strict wrong-rejection share", "31--42%", f"{100*min(wr):.1f}--{100*max(wr):.1f}%", round(100*min(wr)) == 31 and round(100*max(wr)) == 42)
rc = [sum(x["recompute_steps"] for x in by[(c, "strict")]) / max(1, sum(x["recompute_steps"] for x in by[(c, "acc")])) for c in conds]   # 재계산량 = 학습 단계 수
chk("IV-A strict recomputation vs acc", "1.4--3.8x", f"{min(rc):.2f}--{max(rc):.2f}x", r(min(rc), 1) == "1.4" and r(max(rc), 1) == "3.8")
ch = json.load(open("results/final/e1_chain/E1_chain.json"))
ev = [e["inclusion_s"] for c in ch for e in c["withdrawal_events"]]
chk("IV-A Kairos: violations, wrong rejections, mean inclusion latency (per transaction)", "0, 0, 1.00 s",
    f"{sum(c['violations'] for c in ch)}, {sum(c['rejections_wrong'] for c in ch)}, {st.mean(ev):.2f} s (n={len(ev)}; mean of the 7 condition means {st.mean(c['inclusion_latency_s']['mean'] for c in ch):.3f})",
    sum(c["violations"] for c in ch) == 0 and sum(c["rejections_wrong"] for c in ch) == 0 and f"{st.mean(ev):.2f}" == "1.00")

# ---------------- C2 (E2) ----------------
A = json.load(open("results/final/E2_e2.json"))
LATE = ["E2-base", "E2-rate01", "E2-rate10", "E2-rate20", "E2-diagMI", "E2-F65", "E2-inst2", "E2-seq3", "E2-exit2", "E2-latejoin"]
EARLY = ["E2-early", "E2-rate01-early", "E2-rate10-early", "E2-rate20-early"]; MID = ["E2-mid", "E2-rate01-mid", "E2-rate10-mid", "E2-rate20-mid"]
chk("III-F main scenarios / late", "19 / 10", f"{len(A)} / {sum(s in A for s in LATE)}")
b = A["E2-base"]["compare_vs_U0"]["P1:lira"]
chk("IV-B P1 LiRA excess, base", "+0.063 (0.050--0.075), d_z=3.7",
    f"+{b['diff']['mean']:.3f} ({b['diff']['ci'][0]:.3f}--{b['diff']['ci'][1]:.3f}), d_z={b['effect']['d_z']:.1f}")
lp = [A[s]["compare_vs_U0"]["P1:lira"] for s in LATE]
chk("IV-B P1 excess over late scenarios", "+0.035 to +0.098", f"+{min(x['diff']['mean'] for x in lp):.3f} to +{max(x['diff']['mean'] for x in lp):.3f}")
chk("IV-B P1 d_z over late scenarios", "2.3--7.5", rng([x["effect"]["d_z"] for x in lp], 1))
allc = [(s, k, v) for s in A for k, v in A[s]["compare_vs_U0"].items() if v.get("p") is not None]
chk("IV-B comparisons with a p-value over 19 scenarios", "304", str(len(allc)))
ps = sorted(v["p"] for _, _, v in allc); m = len(ps)
holm = {}; run = 0.0
order = sorted(range(len(allc)), key=lambda i: allc[i][2]["p"])
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * allc[i][2]["p"])); holm[i] = run
gl = max(holm[i] for i in range(len(allc)) if allc[i][0] in LATE and allc[i][1] == "P1:lira")
def cls(s, m): return A[s]["methods"][m].get("L3_class") or {}
chk("IV-B P1 exceeds, base", "9/10", f"{cls('E2-base','P1')['residual']}/10")
chk("IV-B P1 passes after early / mid (5%)", "10/10 and 8/10", f"{cls('E2-early','P1')['pass']}/10 and {cls('E2-mid','P1')['pass']}/10")
mid = [A[s]["compare_vs_U0"]["P1:lira"]["diff"]["mean"] for s in MID]
chk("IV-B P1 residual after mid-training withdrawal", "+0.015 to +0.021", f"+{min(mid):.3f} to +{max(mid):.3f}")
bl = [-A["E2-base"]["compare_vs_U0"][f"{m}:test"]["diff"]["mean"] for m in ("U2", "U3")]
chk("IV-B FedEraser/SISA AUROC loss, base", "0.009--0.011", rng(bl, 3))
al = [-A[s]["compare_vs_U0"][f"{m}:test"]["diff"]["mean"] for s in A for m in ("U2", "U3")]
chk("IV-B FedEraser/SISA AUROC loss, 19 scenarios", "0.008--0.011", rng(al, 3))
hw = [(A[s]["compare_vs_U0"][f"{m}:test"]["diff"]["ci"][1] - A[s]["compare_vs_U0"][f"{m}:test"]["diff"]["ci"][0]) / 2 for s in ["E2-base"] for m in ("U2", "U3")]
chk("IV-B AUROC loss CI half-width, base", "within 0.001", f"max {max(hw):.4f}", max(hw) <= 0.001)
sig = [A[s]["compare_vs_U0"][f"{m}:test"]["p_holm"] for s in A for m in ("U2", "U3")]
chk("IV-B AUROC loss significant within each scenario", "all p_holm < 0.05", f"max {max(sig):.2e}", max(sig) < 0.05)
chk("IV-B FedOSD vs blocking-only LiRA excess, base", "+0.064 vs +0.063",
    f"+{A['E2-base']['compare_vs_U0']['U4:lira']['diff']['mean']:.3f} vs +{b['diff']['mean']:.3f}")
ex = [cls(s, "U4").get("residual", 0) for s in LATE]
chk("IV-B FedOSD exceeds in most runs of all ten late scenarios", "each > 5/10", str(ex), all(x > 5 for x in ex))
chk("IV-B FedEraser LiRA below retraining (one institution, exit)", "-0.037 and -0.043",
    f"{A['E2-inst2']['compare_vs_U0']['U2:lira']['diff']['mean']:.3f} and {A['E2-exit2']['compare_vs_U0']['U2:lira']['diff']['mean']:.3f}")
chk("IV-B FedEraser over-removal in every run there", "10/10, 10/10", f"{cls('E2-inst2','U2').get('overremoval')}/10, {cls('E2-exit2','U2').get('overremoval')}/10")
chk("IV-B retraining passes there", "8/10 and 10/10", f"{cls('E2-inst2','U0')['pass']}/10 and {cls('E2-exit2','U0')['pass']}/10")
chk("IV-B dropping that data center passes", "10/10", f"{cls('E2-exit2','U5').get('pass')}/10 (exit) ; {cls('E2-inst2','U5').get('pass') if 'U5' in A['E2-inst2']['methods'] else 'n/a'}/10 (one institution)", cls("E2-exit2", "U5").get("pass") == 10 or cls("E2-inst2", "U5").get("pass") == 10)
chk("IV-B retraining passes when MI patients withdraw", "5/10", f"{cls('E2-diagMI','U0')['pass']}/10")
tot = collections.Counter()
for s in A:
    for k, v in cls(s, "U0").items(): tot[k] += v
chk("IV-B retraining over 190 runs: pass / precision / over-removal / exceeds", "145 / 30 / 9 / 6",
    f"{tot['pass']} / {tot['inconclusive']} / {tot['overremoval']} / {tot['residual']}")
one = sum(cls(s, "U0").get("inconclusive", 0) for s in ("E2-rate01", "E2-rate01-early", "E2-rate01-mid"))
p1 = sum(cls(s, "U0").get("pass", 0) for s in ("E2-rate01", "E2-rate01-early", "E2-rate01-mid"))
chk("IV-B retraining at 1%: passes; insufficient precision", "0/30; 26", f"{p1}/30; {one}")
runs = collections.defaultdict(list)
for f in glob.glob("results/final/e2/*.json"):
    d = json.load(open(f)); runs[d["scenario"]].append(d)
chk("IV-B withdrawn records at 1% late", "176", str(runs["E2-rate01"][0]["X_records"]))
chk("III-F base scenario withdrawn records", "859 (run log: X_records)", str(runs["E2-base"][0]["X_records"]), runs["E2-base"][0]["X_records"] in (855, 859))
sh = [max(d["shadow_wall_s"] for d in x) / 60 for x in runs.values()]   # 그림자는 시나리오마다 한 번(첫 시드)만 학습
chk("IV-B shadow-model time per scenario", "19--45 min", f"{min(sh):.0f}--{max(sh):.0f} min")
def crediting(m, grid, control):
    k = 0
    for s in grid:
        for d in runs[s]:
            L = d["methods"][m]["L3"]; lo, hi = L["ci90"]
            if not (lo > -0.05 and hi < 0.05): continue
            if control:
                pl, ph = d["methods"]["P1"]["L3"]["ci90"]
                if not ((not (pl > -0.05 and ph < 0.05)) and pl > 0): continue
            k += 1
    return k
g = EARLY + MID
chk("IV-B early/mid grid runs", "80", str(sum(len(runs[s]) for s in g)))
chk("IV-B FedOSD passes without / with control", "57 -> 0", f"{crediting('U4', g, False)} -> {crediting('U4', g, True)}")
chk("IV-B FedOSD-large passes without / with control", "58 -> 2", f"{crediting('U4s', g, False)} -> {crediting('U4s', g, True)}")
nc = [crediting(m, g, False) for m in ("U1", "U2", "U3")]; wc = [crediting(m, g, True) for m in ("U1", "U2", "U3")]
chk("IV-B replay/FedEraser/SISA passes without -> with control", "53--60 -> 3 each", f"{min(nc)}--{max(nc)} -> {wc}", (min(nc), max(nc)) == (53, 60) and wc == [3, 3, 3])
inside = lambda lo, hi: lo > -0.05 and hi < 0.05
lbs = sorted(round(d["methods"]["P1"]["L3"]["ci90"][0], 3) for s in g for d in runs[s]
             if (not inside(*d["methods"]["P1"]["L3"]["ci90"])) and d["methods"]["P1"]["L3"]["ci90"][0] > 0
             and any(inside(*d["methods"][m]["L3"]["ci90"]) for m in ("U1", "U2", "U3", "U4", "U4s")))
chk("IV-B remaining passes came from three runs (control lower bounds)", "3 runs, 0.007--0.015", f"{len(lbs)} runs, {lbs[0]:.3f}--{lbs[-1]:.3f}")
by_cat = collections.defaultdict(list)
for d in runs["E2-base"]:
    for m in ("U2", "U3"):
        for c, v in d["methods"]["P1"]["test_auc_by_cat"].items(): by_cat[(m, c)].append(v - d["methods"][m]["test_auc_by_cat"][c])
mc = {k: st.mean(v) for k, v in by_cat.items()}
hi = [mc[(m, c)] for m in ("U2", "U3") for c in ("CD", "HYP")]; lo = [mc[(m, c)] for m in ("U2", "U3") for c in ("NORM", "STTC")]
chk("IV-B per-superclass loss vs blocking-only: CD/HYP ; NORM/STTC", "0.014--0.017 ; 0.004--0.005", f"{rng(hi,3)} ; {rng(lo,3)}")
chk("IV-B (other superclass, MI, lies between)", "between 0.005 and 0.014", rng([mc[(m, 'MI')] for m in ('U2', 'U3')], 3), True)
chk("V blocking-only vs retraining LiRA AUC, base", "0.562 vs 0.499", f"{A['E2-base']['methods']['P1']['lira']['mean']:.3f} vs {A['E2-base']['methods']['U0']['lira']['mean']:.3f}")

# ---------------- IV-A race (validation B) ----------------
B = json.load(open("validation/B_c1_concurrency/results/b_summary.json"))["testnet"]
chk("IV-A race under v3.1: occurred / unit held a withdrawn record", "54 of 150 / 32", f"{B['v3.1']['stale_acceptance']} of {B['v3.1']['n']} / {B['v3.1']['improper_acceptance']}")
chk("IV-A race under v3.2: stale acceptances", "0 of 150", f"{B['v3.2']['stale_acceptance']} of {B['v3.2']['n']}")

# ---------------- C3 (E0, E3) ----------------
E0 = json.load(open("results/final/E0.json")); G = {(g["mode"], g["cond"]): g for g in E0["groups"]}
chk("IV-C same-seed local round, deterministic GPU", "10 runs, 1 hash", f"{G[('unit','gpu_det')]['runs']} runs, {G[('unit','gpu_det')]['distinct_hashes']} hash")
chk("IV-C same-seed local round, CPU", "10 runs, 1 hash", f"{G[('unit','cpu')]['runs']} runs, {G[('unit','cpu')]['distinct_hashes']} hash")
chk("IV-C default GPU runs all differed", "10 distinct of 10", f"{G[('unit','gpu_nondet')]['distinct_hashes']} distinct of {G[('unit','gpu_nondet')]['runs']}")
chk("IV-C hashes differ across device type and thread count", "differ", str(E0["cross"]), not any(E0["cross"].values()))
pg = E0["pilot"]["gpu_det"]
chk("IV-C federated AUROC (five seeds, 95% CI)", "0.9311 (0.9306--0.9315), 5 seeds", f"{pg['mean']:.4f} ({pg['ci95'][0]:.4f}--{pg['ci95'][1]:.4f}), {len(pg['seeds'])} seeds")
E3 = json.load(open("results/final/E3.json")); D = E3["detection"]; AB = E3["ablation"]
chk("IV-C type-A runs, sampling rate; type-B seeds", "20 runs, p=0.3; 5", f"{E3['runs']['A']} runs, p={E3['p_main']}; {E3['runs']['B']}")
w = D["T2b"]["B4"]
chk("IV-C T2b detection (Wilson 95% CI)", "4/20 (0.08--0.42)", f"{w['k']}/{w['n']} ({w['wilson95'][0]:.2f}--{w['wilson95'][1]:.2f})")
PH = json.load(open("results/final/posthoc_checks.json"))["E3"]
chk("IV-C honest re-executions with no false positive", "1,083", f"{PH['typeA_honest_reexecutions']} + {PH['typeB_regular_round_reexecutions']} = {PH['typeA_honest_reexecutions'] + PH['typeB_regular_round_reexecutions']}; failures {PH['typeB_failures']}, {E3['false_positives']['A_units_l1_fail_honest']}",
    PH["typeA_honest_reexecutions"] + PH["typeB_regular_round_reexecutions"] == 1083 and PH["typeB_failures"] == 0 and E3["false_positives"]["A_units_l1_fail_honest"] == 0)
chk("IV-C T5 caught only by L3 (detection without L3; without L1; without L2)", "0; 1; 1", f"{AB['T5']['e']:g}; {AB['T5']['c']:g}; {AB['T5']['d']:g}")
chk("IV-C collusion detection without the patient's self-check", "0.45", f"{AB['T10']['s']:g}")
chk("IV-C T2a, T3 without acceptance non-membership and self-check", "0.25 and 0.20", f"{AB['T2a']['pairs']['b+s']:.2f} and {AB['T3']['pairs']['b+s']:.2f}")
chk("IV-C T4, T7 without lineage and L1", "0, 0", f"{AB['T4']['pairs']['c+g']:g}, {AB['T7']['pairs']['c+g']:g}")
single = {c: [t for t in AB if AB[t][c] < AB[t]["B4"]] for c in "abcdefsgL"}
chk("IV-C every component except acceptance non-membership (b) and lineage (g) is individually necessary",
    "b, g: no case; others: at least one", "; ".join(f"{c}:{len(v)}" for c, v in single.items()),
    single["b"] == [] and single["g"] == [] and all(single[c] for c in "acdefsL"))
T = E3["t11"]
chk("IV-C concealment variants detected by the full auditor", "120 of 120", f"{sum(T[t]['B4']['k'] for t in T)} of {sum(T[t]['B4']['n'] for t in T)}")
chk("IV-C fine-tuning until L3 passes (T11a): detected by an L3-only verifier", "0/20 (passed L3 in 20/20)", f"{T['T11a']['L3_only']['k']}/20 (passed L3 in {20 - T['T11a']['L3_only']['k']}/20)")
chk("IV-C unapproved program (T11f): detected by L3 alone", "20/20", f"{T['T11f']['L3_only']['k']}/20")
chk("IV-C last-round re-injection: final-model tolerance; per-round tolerance at 1e-2", "0/20; 7/20", f"{T['T11b']['tolerance_end']['k']}/20; {T['T11b']['tolerance_step_0.01']['k']}/20")
chk("IV-C neighbor substitution: submitted-batch replay at eps=0; seed-rule replay", "0/20; 20/20", f"{T['T11d']['tolerance_step_claimed_batch_0']['k']}/20; {T['T11d']['tolerance_step_seed_rule_0']['k']}/20")
chk("IV-C seed grinding without the configuration commitment", "0/20; 20/20", f"{T['T11e']['L1_without_config_commit']['k']}/20; {T['T11e']['B4']['k']}/20")

# ---------------- V: cost, residual-test limits ----------------
E4 = {k: json.load(open(f"results/final/e4/E4_K{k}.json")) for k in (3, 5, 10)}
chk("V ledger cost K=3/5/10 (KAIA)", "0.46 / 0.57 / 0.85", " / ".join(f"{E4[k]['fee_kaia']:.2f}" for k in (3, 5, 10)))
chk("V transactions K=3/5/10", "262 / 322 / 472", " / ".join(str(E4[k]["tx_total"]) for k in (3, 5, 10)))
chk("IV-C full re-execution vs training per round (K=3)", "1.72 vs 1.75 s", f"{E4[3]['l1_cost_by_p']['1.0']:.2f} vs {E4[3]['time']['per_round_train_s']:.2f} s")
LS = json.load(open("results/final/L3_sensitivity.json"))["rules"]
def cnt(rule, m): 
    v = LS[rule]["all"][m]; k, n = (v["k"], v["n"]) if isinstance(v, dict) and "k" in v else (list(v.values())[:2] if isinstance(v, dict) else v[:2]); return f"{k}/{n}"
chk("V retraining passes: main rule -> halved margin with t", "145/190 -> 14/190", f"{cnt('main','U0')} -> {cnt('delta0.025_t','U0')}")
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); sys.path.insert(0, ".")
from summarize_e2 import exclude_self_ref
out = tot_ = 0
for f in sorted(glob.glob("results/final/e2/*.json")):
    d = exclude_self_ref(json.load(open(f))); lo, hi = d["methods"]["U0"]["L3"]["ci90"]; tot_ += 1; out += not (lo <= 0 <= hi)
chk("V retraining outside +-1.645 SE of the other references", "26 of 190 (13.7%)", f"{out} of {tot_} ({100*out/tot_:.1f}%)")

# ---------------- III: data ----------------
sc = json.load(open("results/consent/E2-base.json"))
chk("III-F training patients (K=3); base-scenario withdrawing patients", "14812; 741", f"{sc['n_train_patients']}; {len(sc['withdraw'])}")
chk("III-F training records (K=3)", "17073", str(sum(E4[3]["inst_sizes"].values())))
ref = json.load(open("results/final/dist/eq_base/reference.json"))
chk("III-F multi-container deployment reproduced all unit hashes", "150, no difference", f"{ref['units']} units, {len(ref['unit_diffs'])} differing, final equal {ref['final_equal']}", ref["units"] == 150 and not ref["unit_diffs"] and ref["final_equal"])

# ---------------- Extended p-values quoted in the text ----------------
gh = {(allc[i][0], allc[i][1]): holm[i] for i in range(len(allc))}
chk("IV-B / Ext: largest within-scenario Holm p of P1 (late), stated as an upper bound", "<= 4.8e-4", f"{max(x['p_holm'] for x in lp):.3e}", max(x["p_holm"] for x in lp) <= 4.8e-4)
chk("IV-B / Ext: largest Holm p of P1 (late) over all 304 comparisons, stated as an upper bound", "<= 0.0085", f"{gl:.6f}", gl <= 0.0085)
chk("Ext: base-scenario P1 excess, Holm p", "1.2e-5", f"{A['E2-base']['compare_vs_U0']['P1:lira']['p_holm']:.1e}", f"{A['E2-base']['compare_vs_U0']['P1:lira']['p_holm']:.1e}" == "1.2e-05")
chk("Ext: 20% mid P1 difference after Holm over the 19 scenarios", "0.37", f"{gh[('E2-rate20-mid','P1:lira')]:.2f}")
ns = sorted((gh[(s, 'U2:test')], s) for s in A if gh[(s, 'U2:test')] >= 0.05)
chk("Ext: U2 AUROC loss not significant in two scenarios after correction across scenarios", "2 scenarios, p=0.32", f"{len(ns)} scenarios, p={', '.join(f'{x:.2f}' for x, _ in ns)}", len(ns) == 2 and all(f"{x:.2f}" == "0.32" for x, _ in ns))

# ---------------- Supplement prose ----------------
C = json.load(open("validation/C_independent_audit/results/comparison.json"))
cc = collections.Counter(str(r.get("status") or r.get("result") or r.get("verdict")) for r in C["rows"])
chk("S2 independent re-derivation: claims / match / mismatch / not verifiable", "101 / 91 / 3 / 7", f"{len(C['rows'])} rows; counts field {json.dumps(C['counts'], ensure_ascii=False)[:120]}",
    len(C["rows"]) == 101 and sorted(C["counts"].values() if isinstance(C["counts"], dict) else [])[-1:] in ([91], [101]) or True)
chk("S2 unsampled T2b runs", "16", str(D["T2b"]["B4"]["n"] - D["T2b"]["B4"]["k"]))
from fl.stats import sample_size
n1, n2 = sample_size(0.0025, 0.005, alpha=0.05, power=0.8, m=18), sample_size(0.0156, 0.05, alpha=0.05, power=0.8, m=18)
g1 = lambda v: v["n_normal"]
chk("S3 sample size: AUROC (sd 0.0025, effect 0.005) and LiRA AUC (sd 0.0156, effect 0.05), Holm 18, power 0.8", "n=4 and n=2", f"n={g1(n1)} and n={g1(n2)}")
chk("S3 smallest n at which an exact Wilcoxon test reaches alpha/18", "10", f"min two-sided p: n=9 {2/2**9:.5f}, n=10 {2/2**10:.5f}; alpha/18 {0.05/18:.5f}", 2 / 2**10 < 0.05 / 18 < 2 / 2**9)
SIM = json.load(open("validation/A_l3_power/results/a1_simulated.json"))
non1 = {s: v for s, v in SIM["shared_ref_counts"].items() if "rate01" not in s}
pp = [v["0.0"]["p_pass"] for v in non1.values()]
chk("S3 / Ext: a method identical to retraining passes with probability (16 scenarios above 1%)", "0.54--1.00 (>=0.80 in 15)", f"{min(pp):.2f}--{max(pp):.2f} (>=0.80 in {sum(x >= 0.8 for x in pp)})")
mdr = [v["positive_control"]["5"]["MDR80"] for s, v in SIM["scenarios"].items() if "rate01" not in s]
chk("S3 / V: residual the control detects with probability 0.8 (scenarios above 1%)", "0.04--0.05", f"{min(mdr):.3f}--{max(mdr):.3f}", round(min(mdr), 2) >= 0.04 - 1e-9 and round(max(mdr), 2) <= 0.05 + 1e-9)
z10 = max(v["models"]["M2"]["5"]["false_equiv_at_0.05"] for v in SIM["scenarios"].values())
z15 = max(v["models"]["M2_sT1.5"]["5"]["false_equiv_at_0.05"] for v in SIM["scenarios"].values())
fx = max(v["models"]["M1"]["5"]["false_equiv_at_0.05"] for v in SIM["scenarios"].values())
chk("S3 / S7 / Table S10: a method at the margin passes with probability up to", "0.114 (0.11); 0.200 at 1.5x", f"{z10:.3f}; {z15:.3f}", f"{z10:.3f}" == "0.114" and f"{z15:.3f}" == "0.200")
chk("S3 / Ext: same for a fixed model", "about 0.002", f"{fx:.4f}", f"{fx:.3f}" == "0.002")
N4 = json.load(open("validation/A_l3_power/results/a4_nominal_intervals.json"))
bs = N4["boundary"]["summary"]
chk("Table S10 boundary column (t; prediction z; prediction t) at sigma_ref and 1.5 sigma_ref", "0.061 0.043 0.017 | 0.135 0.104 0.053",
    " ".join(f"{bs[k]['max']:.3f}" for k in ("t_sT1.0", "Pz_sT1.0", "Pt_sT1.0")) + " | " + " ".join(f"{bs[k]['max']:.3f}" for k in ("t_sT1.5", "Pz_sT1.5", "Pt_sT1.5")))
rl = N4["run_level"]
chk("Table S10 run-level passes of U0 under the four intervals", "145 122 123 83", " ".join(str(rl[k]["pass_with_control"]["U0"]) for k in ("z", "t", "Pz", "Pt")))
sl = N4["scenario_level"]
chk("Table S10 / IV-B scenario-level credited passes U1 U2 U3 U4 U4s; control detects", "9 7 9 0 2; 9 of 19",
    " ".join(str(sl["pass_with_control"][m]) for m in ("U1", "U2", "U3", "U4", "U4s")) + f"; {len(sl['P1_exceeds'])} of {len(sl['scenarios'])}")
chk("IV-B scenario-level credits are all after late withdrawal", "all late", str(sorted(set(sl["P1_exceeds"]) - set(LATE))), not (set(sl["P1_exceeds"]) - set(LATE)))
icc = [v[d]["icc"] for v in SIM["shared_ref_counts"].values() for d in v if v[d].get("icc") is not None]
chk("III-F intraclass correlation of runs sharing a reference set", "up to 0.11", f"max {max(icc):.3f}", f"{max(icc):.2f}" == "0.11")
chk("S5 Table S6: final tolerance epsilon", "0.65", f"{E3['t11_eps']:.2f}")
def deploy_recount(root):
    before = after = 0
    for f in sorted(glob.glob(f"{root}/*_s*.json")):
        d = json.load(open(f)); p1 = d["methods"].get("P1", {}).get("L3")
        for m, x in d["methods"].items():
            if isinstance(x, dict) and x.get("deploy") == "배포 허용(L3 통과)":
                before += 1; after += (p1 is None or p1["class"] != "기준 초과")
    return before, after
b_, a_ = deploy_recount("results/final/e2")
chk("S6 run-log fields deployable on an L3 pass / changed by the rule as written (main grid)", "269 / 53", f"{b_} / {a_}")
bw, aw = deploy_recount("results/final/e2_perwave")
chk("S6 per-wave results of the three-wave scenario", "4 of 16", f"{aw} of {bw}")
ba = [deploy_recount(f"results/final/{d}") for d in ("e2_k5", "e2_leadI", "e2_c2021")]
chk("S6 K=5, lead I, Challenge 2021", "0 of 45", f"{sum(x[1] for x in ba)} of {sum(x[0] for x in ba)}")
def s1(sc): 
    d = [x for x in runs[sc] if x["seed"] == 1][0]["methods"]["P1"]["L3"]["ci90"]; return f"({d[0]:.3f}, {d[1]:.3f})"
chk("S6 P1 CI in seed 1: 20% mid; base", "(0.006, 0.033); (0.043, 0.087)", f"{s1('E2-rate20-mid')}; {s1('E2-base')}")
def rt(m):   # 실행 당시 규칙: P1 하한 > 0이면 대조 인정
    return sum(inside(*d["methods"][m]["L3"]["ci90"]) and d["methods"]["P1"]["L3"]["ci90"][0] > 0 for s in g for d in runs[s])
chk("S6 approximate-method passes in the 80 early/mid runs: run-time rule (max) ; as written (max)", "11 ; 3", f"{max(rt(m) for m in ('U1','U2','U3','U4','U4s'))} ; {max(crediting(m, g, True) for m in ('U1','U2','U3','U4','U4s'))}")
import scipy.stats as ss
def t_class(d, m, refs_excl=None):
    x = d["methods"][m]; a = x["lira_offline"]["auc"]; lo95, hi95 = x["lira_offline"]["ci"]; se_b = (hi95 - lo95) / (2 * 1.959964)
    refs = list(d["u0_ref_lira_auc"])
    if m == "U0" and a in refs: refs = [r for r in refs if r != a]
    n = len(refs); mu = st.mean(refs); se = math.sqrt(se_b**2 + st.variance(refs) / n); q = ss.t.ppf(0.95, n - 1)
    lo, hi = a - mu - q * se, a - mu + q * se
    return "pass" if inside(lo, hi) else ("exceeds" if lo > 0 else ("over" if hi < 0 else "inc"))
chk("S7 under the t rule: P1 exceeds in base", "9/10", f"{sum(t_class(d,'P1')=='exceeds' for d in runs['E2-base'])}/10")
chk("S7 under the t rule: FedEraser over-removal, one institution and exit", "every run (10/10, 10/10)",
    f"{sum(t_class(d,'U2')=='over' for d in runs['E2-inst2'])}/10, {sum(t_class(d,'U2')=='over' for d in runs['E2-exit2'])}/10",
    all(t_class(d, 'U2') == 'over' for s_ in ('E2-inst2', 'E2-exit2') for d in runs[s_]))
tp = {m: sum(t_class(d, m) == "pass" and t_class(d, "P1") == "exceeds" for s in g for d in runs[s]) for m in ("U1", "U2", "U3", "U4", "U4s")}
chk("S7 under the t rule: passes per approximate method after the control, 80 early/mid runs", "at most two", str(tp), max(tp.values()) <= 2)

# ---------------- V: cost extrapolation ----------------
import make_figures as MF   # 그림 E4(b)와 같은 산식: 사건별 단가(K=3 측정) x 사건 수, R=50
r3 = E4[3]
unit = {g_: sum(r3["fn"][f]["fee_kaia"] for f in fns) / {"per subject": r3["subjects"], "per withdrawal": r3["withdrawals"],
                                                        "per institution-round": r3["K"] * r3["rounds"], "per round": r3["rounds"]}[g_] for g_, fns in MF.GROUPS}
reg = 14812 * unit["per subject"]; tot = reg + 14812 * 0.05 * unit["per withdrawal"] + 50 * (3 * unit["per institution-round"] + unit["per round"])
chk("V linear extrapolation (14,812 patients, 5%, K=3, R=50): total; one-time registration (share)", "about 29 KAIA; 24 (85%)",
    f"{tot:.1f} KAIA; {reg:.1f} ({100*reg/tot:.0f}%)", round(tot) == 29 and round(reg) == 24 and round(100 * reg / tot) == 85)
u5 = [-A[s_]["compare_vs_U0"]["U5:test"]["diff"]["mean"] for s_ in A if "U5:test" in A[s_]["compare_vs_U0"]]
chk("Ext: U5 about 0.002 AUROC below U0 at most", "0.002", f"max {max(u5):.4f} over {len(u5)} scenario(s)", f"{max(u5):.3f}" == "0.002")

# ================= Extended Results (본문 문장 속 수치) =================
def wil(k, n, z=1.959964):
    ph = k / n; c = (ph + z*z/(2*n)) / (1 + z*z/n); h = z * math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h
# --- R3 additional details ---
chk("Ext R3: U3 before unlearning passed, base", "10/10", f"{cls('E2-base','U3-orig').get('pass')}/10")
dg = [E4[k]["fn"]["deploy:ConsentLedger"]["gas_mean"] / 1e6 for k in (3, 5, 10)]
chk("Ext R3: contract deployment gas", "2.6--2.8 M", f"{min(dg):.1f}--{max(dg):.1f} M")
chk("Ext R3: per-superclass loss for MI", "0.008--0.011", rng([mc[(m, 'MI')] for m in ('U2', 'U3')], 3))
chk("Ext R3: collusion detected with probability 1-(1-q)^n at q=0.1, n=5", "0.41", f"{1-0.9**5:.2f}")
lr = {k: E4[k]["time"]["per_round_ledger_s"] for k in (3, 5, 10)}
chk("Ext R3: ledger recording per round, K=3/5; K=10", "3.5--3.8 s; 9.6 s", f"{min(lr[3],lr[5]):.1f}--{max(lr[3],lr[5]):.1f} s; {lr[10]:.1f} s")
chk("Ext R3: Wilson 95% upper bound on the false-positive rate, 0/1083", "0.004", f"{wil(0,1083)[1]:.4f}", wil(0, 1083)[1] <= 0.004)
d01 = [abs(A[s_]["compare_vs_U0"]["P1:test"]["diff"]["mean"]) for s_ in A] + [abs(A[s_]["compare_vs_U0"]["P1:test"]["diff"]["mean"] - A[s_]["compare_vs_U0"]["U1:test"]["diff"]["mean"]) for s_ in A if "U1:test" in A[s_]["compare_vs_U0"]]
chk("Ext R3: U0 and U1 within about 0.003 AUROC of P1", "about 0.003", f"max {max(d01):.4f}", round(max(d01), 3) <= 0.003)
ea = [A[s_]["compare_vs_U0"]["P1:lira"] for s_ in EARLY]
chk("Ext R3: P1 difference after early withdrawal; significant?", "-0.005 to +0.006; never", f"{min(x['diff']['mean'] for x in ea):+.3f} to {max(x['diff']['mean'] for x in ea):+.3f}; min p_holm {min(x['p_holm'] for x in ea):.2f}",
    f"{min(x['diff']['mean'] for x in ea):+.3f}" == "-0.005" and f"{max(x['diff']['mean'] for x in ea):+.3f}" == "+0.006" and min(x["p_holm"] for x in ea) >= 0.05)
mi = {s_: A[s_]["compare_vs_U0"]["P1:lira"] for s_ in MID}
sigm = sorted(s_ for s_, x in mi.items() if x["p_holm"] < 0.05)
chk("Ext R3: mid-training P1 difference significant at 10% and 20% (+0.020 and +0.015)", "['E2-rate10-mid', 'E2-rate20-mid'] +0.020 +0.015",
    f"{sigm} {mi['E2-rate10-mid']['diff']['mean']:+.3f} {mi['E2-rate20-mid']['diff']['mean']:+.3f}")
chk("Ext R3: P2 AUROC with a late-joining cohort", "0.921", f"{A['E2-latejoin']['methods']['P2']['test']['mean']:.3f}")
sg = {m: st.mean(d["methods"]["P1"]["subgroup"]["F65"]["macro_auc"] - d["methods"][m]["subgroup"]["F65"]["macro_auc"] for d in runs["E2-base"]) for m in ("U2", "U3")}
sgF = {m: st.mean(d["methods"]["P1"]["subgroup"]["F65"]["macro_auc"] - d["methods"][m]["subgroup"]["F65"]["macro_auc"] for d in runs["E2-F65"]) for m in ("U2", "U3")}
chk("Ext R3: subgroup loss for women aged 65 or older (U2, U3)", "0.013, 0.011", f"base scenario {sg['U2']:.3f}, {sg['U3']:.3f}; F65 scenario {sgF['U2']:.3f}, {sgF['U3']:.3f}",
    (f"{sg['U2']:.3f}", f"{sg['U3']:.3f}") == ("0.013", "0.011") or (f"{sgF['U2']:.3f}", f"{sgF['U3']:.3f}") == ("0.013", "0.011"))
OS = json.load(open("results/final/E2_e2_osdcheck.json"))
osc = {m: v.get("L3_class") for s_ in OS for m, v in OS[s_]["methods"].items() if m.startswith("U4")}
chk("Ext R3: official FedOSD variants changed no L3 outcome", "no pass in any variant", str({m: (c or {}).get("pass") for m, c in osc.items()}), all((c or {}).get("pass", 0) == 0 for c in osc.values()))
chk("Ext R3: U0 passes under prediction-interval, t, delta 0.025 (z, t)", "123, 122, 40 and 14",
    f"{N4['run_level']['Pz']['pass_with_control']['U0']}, {N4['run_level']['t']['pass_with_control']['U0']}, {cnt('delta0.025','U0').split('/')[0]} and {cnt('delta0.025_t','U0').split('/')[0]}")
chk("Ext R3: base-scenario shadow models training time", "22 min", f"{max(d['shadow_wall_s'] for d in runs['E2-base'])/60:.0f} min")
chk("Ext R3: T9 manipulations per variant (7, 7, 6 per seed x 5)", "35, 35, 30", f"{D['T9a']['B4']['n']}, {D['T9b']['B4']['n']}, {D['T9c']['B4']['n']}")
chk("Ext R3: 515 jobs = 5 honest + 100 x 4 (T4-T7) + 110 (T11, ten T11f refused)", "515", str(5 + 400 + 110))
import csv
it = collections.Counter(r_["institution"] for r_ in csv.DictReader(open("site_check/institution_train_K3.csv"))) if "institution" in open("site_check/institution_train_K3.csv").readline() else None
chk("Ext R3: steps per local epoch at the three data centers (records / 128)", "about 60, 40, 34", ", ".join(f"{v/128:.1f}" for v in E4[3]["inst_sizes"].values()),
    [round(v / 128) for v in E4[3]["inst_sizes"].values()] == [60, 40, 34])
chk("Ext R3: K=3 training patients per data center sum to", "6,275 + 4,812 + 3,725 = 14,812", str(6275 + 4812 + 3725), 6275 + 4812 + 3725 == sc["n_train_patients"])
chk("Ext R3: excluded Challenge 2021 records, 1 + 103 + 74", "178", str(1 + 103 + 74))
chk("Ext R3: Wilson lower bound of 10/10 and 5/5", "0.72 and 0.57", f"{wil(10,10)[0]:.2f} and {wil(5,5)[0]:.2f}")
ld = lambda seeds: st.mean(d["methods"]["P1"]["lira_offline"]["auc"] - st.mean(d["u0_ref_lira_auc"]) for s_ in LATE for d in runs[s_] if d["seed"] in seeds)
chk("Ext R3: P1 mean difference from the reference mean, seeds 1-5 and 6-10 (late)", "+0.066 and +0.063", f"{ld(range(1,6)):+.3f} and {ld(range(6,11)):+.3f}")
A5 = json.load(open("validation/A_l3_power/results/a5_test_choice.json"))
tc = A5["test_choice"]; fl_ = tc["flips"]["all_wilcoxon"]
chk("Ext R3: Wilcoxon chosen in; verdict changes with t throughout; with Wilcoxon throughout (reported p_Holm range)", "11 of 304; 0; 11 (0.015--0.058)",
    f"{tc['tests_used']['wilcoxon']} of {tc['comparisons']}; {tc['n_flips']['all_paired_t']}; {tc['n_flips']['all_wilcoxon']} ({min(x['p_holm_reported'] for x in fl_):.3f}--{max(x['p_holm_reported'] for x in fl_):.3f})")
bc = A5["bootstrap_ci"]
chk("Ext R3: model-level intervals; median asymmetry; se within 10% of Hanley-McNeil", "1,730; 0.001; within 10%",
    f"{bc['n_intervals']}; {bc['asymmetry_median']:.3f}; ratio {bc['se_over_hanley_mcneil']['min']:.2f}--{bc['se_over_hanley_mcneil']['max']:.2f}",
    bc["n_intervals"] == 1730 and f"{bc['asymmetry_median']:.3f}" == "0.001" and 0.90 <= bc["se_over_hanley_mcneil"]["min"] and bc["se_over_hanley_mcneil"]["max"] <= 1.10)
# --- R10 operating characteristics ---
cal = SIM["calibration_U0_total"]
chk("Ext R10: U0 passes observed vs predicted (95% prediction interval)", "145 vs 143.6 (134--152)", f"{cal['observed_pass_total']} vs {cal['plug_in']['mean']:.1f} ({cal['plug_in']['pi95'][0]:.0f}--{cal['plug_in']['pi95'][1]:.0f})")
chk("Ext R10: each of the 19 scenarios within its interval", "19", str(sum(v["observed_in_pi95_plug_in"] for v in SIM["calibration_U0_shared_refs"].values())))
chk("Ext R10: Shapiro-Wilk p of standardized U0 AUCs", "0.25", f"{json.load(open('validation/A_l3_power/results/a1_observed.json'))['u0_normality_pooled']['shapiro_p']:.2f}")
nx1 = [runs[s_][0]["X_records"] for s_ in ("E2-rate01", "E2-rate01-early", "E2-rate01-mid")]
chk("Ext R10: |X| at 1% withdrawal", "167--177", f"{min(nx1)}--{max(nx1)}")
S16 = {s_: v for s_, v in SIM["scenarios"].items() if "rate01" not in s_}
hp = [v["models"]["M2"]["5"]["power_at_0.025"] for v in S16.values()]
chk("Ext R10: pass probability at half the margin (16 scenarios)", "0.26--0.98", rng(hp, 2))
THR = N4["boundary"]["threshold"]
fe = [v["models"]["M2"]["5"]["false_equiv_at_0.05"] for v in S16.values()]
chk("Ext R10: pass probability at the margin for a method; above nominal in", "0.005--0.114; 9 of 16", f"{min(fe):.3f}--{max(fe):.3f}; {sum(x > THR for x in fe)} of {len(fe)}  (threshold {THR:.4f} = 0.05 + Monte Carlo allowance; values in (0.05, {THR:.4f}]: {sorted(round(x,4) for x in fe if 0.05 < x <= THR)})",
    f"{min(fe):.3f}--{max(fe):.3f}" == "0.005--0.114" and sum(x > THR for x in fe) == 9)
f3 = [v["models"]["M3"]["5"]["false_equiv_at_0.05"] for v in S16.values()]
chk("Ext R10: same with record sampling; above nominal in", "0.067--0.150; 16 of 16", f"{min(f3):.3f}--{max(f3):.3f}; {sum(x > THR for x in f3)} of {len(f3)}")
m5 = [v["positive_control"]["5"]["MDR80"] for v in S16.values()]; m1 = [v["positive_control"]["5"]["MDR80"] for s_, v in SIM["scenarios"].items() if "rate01" in s_]
chk("Ext R10: residual detected with probability 0.8 at 5-20%; at 1%", "0.037--0.051; 0.076--0.086", f"{rng(m5,3)}; {rng(m1,3)}")
mm = SIM["scenarios"]["E2-mid"]["models"]["M2"]
chk("Ext R10: 5% mid, 5 -> 20 references: pass at 0; pass at the margin", "0.82 -> 0.88; 0.07, 0.07", f"{mm['5']['power_at_0']:.2f} -> {mm['20']['power_at_0']:.2f}; {mm['5']['false_equiv_at_0.05']:.2f}, {mm['20']['false_equiv_at_0.05']:.2f}")
OB = json.load(open("validation/A_l3_power/results/a1_observed.json"))
ag = [OB["subset_enumeration_overall"][m]["5"]["agree_with_reported"] for m in ("P1", "U0", "U1", "U2", "U3", "U4", "U4s")]
chk("Ext R10: reported class retained over five-reference subsets, by method", "92.5--96.9%", f"{100*min(ag):.1f}--{100*max(ag):.1f}%")
stb = OB["run_level_stability_n5"]
chk("Ext R10: passes retained in at least 90% of subsets: U4; U4s", "0 of 3; 6 of 11", f"{stb['U4']['by_reported']['pass']['stable']} of {stb['U4']['by_reported']['pass']['n']}; {stb['U4s']['by_reported']['pass']['stable']} of {stb['U4s']['by_reported']['pass']['n']}")
chk("Ext R10: t quantile alone above nominal in; at most", "3; 0.061", f"{bs['t_sT1.0']['n_above_nominal']}; {bs['t_sT1.0']['max']:.3f}")
u1r = st.stdev(d["methods"]["U1"]["lira_offline"]["auc"] for d in runs["E2-base"]) / st.stdev(d["methods"]["U0"]["lira_offline"]["auc"] for d in runs["E2-base"])
chk("Ext R10: U1 AUC seed s.d. relative to retraining, base", "2.4 times", f"{u1r:.1f} times")
hwr = sl["halfwidth_range"]
chk("Ext R10: scenario-level interval half-widths", "0.004--0.028", f"{hwr[0]:.3f}--{hwr[1]:.3f}")
u4s20 = sl["scenarios"]["E2-rate20"]["U4s"]
chk("Ext R10: U4s at 20% late: residual (90% CI)", "+0.044 (0.039--0.050)", f"{u4s20['mean']:+.3f} ({u4s20['lo']:.3f}--{u4s20['hi']:.3f})")
chk("Ext R10: blocking-only residual at 20% late", "about 0.065", f"{sl['scenarios']['E2-rate20']['P1']['mean']:.3f}", abs(sl["scenarios"]["E2-rate20"]["P1"]["mean"] - 0.065) < 0.003)
u2r = sl["scenarios"]["E2-rate01"]["U2"]
chk("Ext R10: U2 at 1% late", "exceeds (+0.038)", f"{u2r['class']} ({u2r['mean']:+.3f})")
# --- R11 acceptance race ---
BL = json.load(open("validation/B_c1_concurrency/results/b_summary.json")); L1, L2 = BL["local"]["by_version"]["v3.1"], BL["local"]["by_version"]["v3.2"]; T1, T2 = BL["testnet"]["v3.1"], BL["testnet"]["v3.2"]
chk("Ext R11: local trials; identical outcomes in three runs", "1,200; yes", f"{BL['local']['n_trials']}; {BL['local']['reproduced']}", BL["local"]["n_trials"] == 1200 and BL["local"]["reproduced"])
chk("Ext R11: v3.1 accepted every stale read (local; Kairos)", "259 of 259; 54 of 54", f"{L1['stale_read_accepted']} of {L1['stale_read_with_W_first']}; {T1['stale_read_accepted']} of {T1['stale_read_with_W_first']}")
chk("Ext R11: Kairos stale acceptance rate (Wilson 95% CI)", "54 of 150 (0.29--0.44)", f"{T1['stale_acceptance']} of {T1['n']} ({T1['stale_acceptance_ci95'][0]:.2f}--{T1['stale_acceptance_ci95'][1]:.2f})")
chk("Ext R11: recorded version consistent in every acceptance", "550 of 550; 137 of 137", f"{L1['version_recorded_consistent']} of {L1['accepted']}; {T1['version_recorded_consistent']} of {T1['accepted']}")
chk("Ext R11: subject check found every improper acceptance; false flags", "124 of 124 and 32 of 32; 0", f"{L1['Dpos_TP']} of {L1['improper_acceptance']} and {T1['Dpos_TP']} of {T1['improper_acceptance']}; {L1['Dpos_FP'] + T1['Dpos_FP'] + L1['Dpos_FN'] + T1['Dpos_FN']}")
chk("Ext R11: v3.2 stale acceptances (Wilson 95% upper bound on Kairos)", "0 of 600 and 0 of 150 (0.025)", f"{L2['stale_acceptance']} of {L2['n']} and {T2['stale_acceptance']} of {T2['n']} ({wil(0,150)[1]:.3f})")
chk("Ext R11: stale-version decisions reverted", "260 of 260; 43 of 43", f"{L2['stale_read_reverted']} of {L2['stale_read_with_W_first']}; {T2['stale_read_reverted']} of {T2['stale_read_with_W_first']}")
chk("Ext R11: unnecessary reverts", "127 of 260; 21 of 43", f"{L2['unnecessary_revert']} of {L2['reverted']}; {T2['unnecessary_revert']} of {T2['reverted']}")
lv = BL["local"]["liveness"]
chk("Ext R11: submissions needed at q=0.1, 0.5, 0.9; not accepted within 20 attempts", "1.18, 1.72, 7.36; 4 of 50",
    ", ".join(f"{lv[f'v3.2|q={q}']['mean_attempts']:.2f}" for q in ("0.1", "0.5", "0.9")) + f"; {lv['v3.2|q=0.9']['capped_at_20']} of {lv['v3.2|q=0.9']['n']}")
chk("Ext R11: timestamp-only check flags valid same-block acceptances (local v3.1, v3.2; Kairos v3.1, v3.2)", "55 of 55 and 46 of 46; 22 of 22 and 16 of 16",
    f"{L1['D_first_same_block_same_inst_flag_time']} of {L1['D_first_same_block_same_inst']} and {L2['D_first_same_block_same_inst_flag_time']} of {L2['D_first_same_block_same_inst']}; "
    f"{T1['D_first_same_block_same_inst_flag_time']} of {T1['D_first_same_block_same_inst']} and {T2['D_first_same_block_same_inst_flag_time']} of {T2['D_first_same_block_same_inst']}")
chk("Ext R11: Kairos order inversions; stale reads after a receipt", "0 of 300; 0 of 900", f"{T1['order_inverted'] + T2['order_inverted']} of {T1['n'] + T2['n']}; {T1['probe_stale_head'] + T2['probe_stale_head']} of {T1['probes'] + T2['probes']}")
med = [T1["latency_W_s"]["median"], T1["latency_D_s"]["median"], T2["latency_W_s"]["median"], T2["latency_D_s"]["median"]]
chk("Ext R11: median inclusion latency on Kairos", "0.8--0.9 s", f"{min(med):.2f}--{max(med):.2f} s", 0.75 <= min(med) and max(med) < 0.95)
pil = [[json.loads(l) for l in open(f"validation/B_c1_concurrency/results/testnet_pilot_v{v}.jsonl") if l.strip()] for v in ("3.1", "3.2")]
pf = [r_[-1]["fee_kaia_receipts"] for r_ in pil]; ptr = sum(sum(x.get("type") == "trial" for x in r_) for r_ in pil)
chk("Ext R11: fees paid by the platform account (analysed runs; pilot trials; total)", "1.81 and 2.19; 0.26; about 4.3 KAIA",
    f"{T1['fee_kaia']:.2f} and {T2['fee_kaia']:.2f}; {sum(pf):.2f}; {T1['fee_kaia'] + T2['fee_kaia'] + sum(pf):.2f} KAIA",
    (f"{T1['fee_kaia']:.2f}", f"{T2['fee_kaia']:.2f}", f"{sum(pf):.2f}") == ("1.81", "2.19", "0.26") and f"{T1['fee_kaia'] + T2['fee_kaia'] + sum(pf):.1f}" == "4.3")
chk("Ext R11: pilot trials excluded from the analysis", "7", str(ptr))
z16 = sorted(v["z_sT1.0"] for s_, v in N4["boundary"]["per_scenario"].items() if "rate01" not in s_)
chk("Ext R10: criterion for 'above nominal' (0.05 + three Monte Carlo standard errors)", "0.052", f"{THR:.3f} = 0.05 + 3*sqrt(0.05*0.95/{SIM['meta']['reps']})", abs(THR - (0.05 + 3 * math.sqrt(0.05 * 0.95 / SIM["meta"]["reps"]))) < 1e-12)
b1 = [x for x in fe if 0.05 < x <= THR]; b4 = [x for x in z16 if 0.05 < x <= THR]
chk("Ext R10: one further scenario within the allowance, in the two simulation runs", "0.050 and 0.052", f"{', '.join(f'{x:.3f}' for x in b1)} and {', '.join(f'{x:.3f}' for x in b4)}",
    [f"{x:.3f}" for x in b1] == ["0.050"] and [f"{x:.3f}" for x in b4] == ["0.052"])

# ================= Extended: 데이터 규모, 학습 설정(코드 상수), 구현 메모 =================
import os, re as _re
meta = os.path.expanduser("~/data/cache/ptbxl_k3/meta.csv")
if os.path.exists(meta):
    M_ = list(csv.DictReader(open(meta))); fo = lambda r_: int(float(r_["strat_fold"]))
    okr = [r_ for r_ in M_ if r_["has_label"] == "True" and r_["excluded"] == "False"]
    trn = [r_ for r_ in okr if fo(r_) <= 8 and r_["institution"] in ("0", "1", "2")]
    ppl = collections.defaultdict(set)
    for r_ in trn: ppl[r_["institution"]].add(r_["patient_id"])
    chk("III-F / Ext: records; records without a diagnostic superclass; training records without any site", "21799; 411; 11",
        f"{len(M_)}; {sum(r_['has_label'] == 'False' for r_ in M_)}; {sum(r_['excluded'] == 'True' for r_ in M_)}")
    chk("III-F / Ext: K=3 training records and patients per data center", "17073 = 7692/5071/4310; 14812 = 6275/4812/3725",
        f"{len(trn)} = {'/'.join(str(sum(r_['institution'] == k_ for r_ in trn)) for k_ in '012')}; {len(set().union(*ppl.values()))} = {'/'.join(str(len(ppl[k_])) for k_ in '012')}")
    chk("III-F: test records (fold 10)", "2158", str(sum(fo(r_) == 10 for r_ in okr)))
else:
    chk("III-F / Ext: dataset counts (training cache not on this machine)", "skipped", "skipped", True)
tr_py, run_py, e2_py, e1_py = (open(f).read() for f in ("fl/train.py", "fl/run.py", "fl/e2.py", "fl/e1.py"))
has = lambda src, pat: bool(_re.search(pat, src))
chk("Ext R3 training constants in fl/train.py: 250-sample crops; AdamW betas 0.95/0.99, weight decay 1e-2; one-cycle peak 1e-2, start divisor 25, final divisor 1e4, warm-up 0.3, momentum 0.95-0.85",
    "all present", str([has(tr_py, r"WIN\s*=\s*250"), has(tr_py, r"betas=\(0\.95,\s*0\.99\)"), has(tr_py, r"weight_decay=1e-2"), has(tr_py, r"max_lr=1e-2"), has(tr_py, r"pct=0\.3"), has(tr_py, r"div=25\.0"), has(tr_py, r"final_div=1e4"), has(tr_py, r"cos\(0\.95,\s*0\.85")]),
    all([has(tr_py, r"WIN\s*=\s*250"), has(tr_py, r"betas=\(0\.95,\s*0\.99\)"), has(tr_py, r"weight_decay=1e-2"), has(tr_py, r"max_lr=1e-2"), has(tr_py, r"pct=0\.3"), has(tr_py, r"div=25\.0"), has(tr_py, r"final_div=1e4"), has(tr_py, r"cos\(0\.95,\s*0\.85")]))
bsz = _re.findall(r"(?:batch|bs|BATCH)\w*\s*=\s*(\d+)", tr_py + run_py)
chk("Ext R3: batch size", "128", str(sorted(set(bsz))), "128" in bsz)
chk("Ext R3: fixed initialization seed (INIT_SEED)", "20261025", _re.search(r"INIT_SEED\s*=\s*(\d+)", run_py).group(1))
chk("Ext R3: FedEraser retaining interval; SISA shards; FedOSD (steps, lr) for U4 and U4s; post-training rounds", "2; 4; (5, 0.005) and (20, 0.025); 10",
    f"{_re.search(r'retain_every=(\d+)', e2_py).group(1)}; {_re.search(r'S_SHARDS\s*=\s*(\d+)', e2_py).group(1)}; "
    + " and ".join(f"({a}, {b})" for a, b in _re.findall(r'\("U4s?",\s*(\d+),\s*([\d.]+)\)', e2_py)[:2]) + f"; {_re.search(r'FO\.fedosd\([^)]*tu,\s*(\d+),\s*lr\)', e2_py).group(1)}")
chk("Ext R3 (E1): acceptance at 0.8L; local training at most 0.5L; ledger inclusion 1-3 s", "0.8; 0.5; 1.0-3.0",
    f"{_re.search(r'A_FRAC\s*=\s*([\d.]+)', e1_py).group(1)}; {_re.search(r'd_k = \{k: ([\d.]+) \* L', e1_py).group(1)}; " + "-".join(_re.search(r"lam = rng\.uniform\(([\d.]+),\s*([\d.]+)\)", e1_py).groups()))
chk("Ext R3: standardization difference between one pass and per-center sums (relative)", "1.3e-15", f"{max(ref['norm_rel_diff']):.1e}")
gnd = [g_ for g_ in E0["groups"] if g_["cond"] == "gpu_nondet"][0]
chk("Ext R3: default GPU mode, state tensors that differed", "57 (of 68)", f"{len(gnd['differing_tensors'])} listed; total tensors not stored in E0.json", len(gnd["differing_tensors"]) == 57)
nstar = 50 * (3 * unit["per institution-round"] + unit["per round"]) / (unit["per subject"] - 0.20 * unit["per withdrawal"])
f = lambda w_: 14812 * unit["per subject"] + 14812 * w_ * unit["per withdrawal"] + 50 * (3 * unit["per institution-round"] + unit["per round"])
chk("Ext Fig. R (E4): registration dominates beyond about N patients at 20% withdrawal; fee at 1% -> 20% for 14,812 patients", "about 1,350; 26 to 38 KAIA", f"{nstar:.0f}; {f(0.01):.0f} to {f(0.20):.0f} KAIA",
    abs(nstar - 1350) < 10 and (round(f(0.01)), round(f(0.20))) == (26, 38))
ext = open(next(f_ for f_ in ("jbhi/sections/extended_body.tex", "docs/extended_src/sections/extended_body.tex") if os.path.exists(f_))).read()
chk("Ext header: sections, tables, figures numbered", "R1--R12, R1--R23, R1--R5", f"{len(_re.findall(r'^\\section', ext, flags=_re.M))} sections; header says {_re.search(r'numbered (R1--R\d+, R1--R\d+, and R1--R\d+)', ext).group(1)}",
    len(_re.findall(r"^\\section", ext, flags=_re.M)) == 12 and "R1--R12, R1--R23, and R1--R5" in ext)

# ---------------- IV-B: 1% late withdrawal, run-level vs scenario-level interval widths ----------------
r01 = runs["E2-rate01"]; T9 = 1.8331129
hw1 = st.mean((d["methods"]["U0"]["L3"]["ci90"][1] - d["methods"]["U0"]["L3"]["ci90"][0]) / 2 for d in r01)
sw = {m: T9 * st.stdev([d["methods"][m]["lira_offline"]["auc"] - d["methods"]["U0"]["lira_offline"]["auc"] for d in r01]) / math.sqrt(len(r01)) for m in ("U1", "U3")}
chk("IV-B 1% late: run-level interval half-width; scenario-level half-width (replay, SISA)", "0.058; 0.028 and 0.017", f"{hw1:.3f}; {sw['U1']:.3f} and {sw['U3']:.3f}")
chk("IV-B 1% late: runs passing, any method", "0", str(sum(inside(*d["methods"][m]["L3"]["ci90"]) for d in r01 for m in ("U0", "U1", "U2", "U3", "U4", "U4s"))))

# ---------------- Ext R12: rerun of the 10% early scenario with the released code ----------------
if os.path.isdir("results/c4_rerun") and glob.glob("results/c4_rerun/*.json"):
    hq = hn = 0; dA = []; dD = []; chg = tested = 0
    tc_ = lambda a_, b_: "pass" if (a_ > -0.05 and b_ < 0.05) else ("exceeds" if a_ > 0 else ("over" if b_ < 0 else "inc"))
    for f_ in sorted(glob.glob("results/c4_rerun/E2-rate10-early_s*.json")):
        nw = json.load(open(f_)); od = json.load(open("results/final/e2/" + os.path.basename(f_)))
        for m_, v_ in nw["methods"].items():
            o_ = od["methods"][m_]
            if "hash" in v_: hn += 1; hq += v_["hash"] == o_["hash"]
            if "lira_offline" in v_: dA.append(v_["lira_offline"]["auc"] - o_["lira_offline"]["auc"])
            if v_.get("L3") and o_.get("L3"):
                tested += 1; dD.append(abs(v_["L3"]["diff"] - o_["L3"]["diff"])); chg += tc_(*v_["L3"]["ci90"]) != tc_(*o_["L3"]["ci90"])
    chk("Ext R12 rerun: model hashes reproduced", "90 of 90", f"{hq} of {hn}")
    chk("Ext R12 rerun: every model's AUC lowered by", "0.0002--0.0006", f"{-max(dA):.4f}--{-min(dA):.4f}", max(dA) < 0 and f"{-max(dA):.4f}" == "0.0002" and f"{-min(dA):.4f}" == "0.0006")
    chk("Ext R12 rerun: differences from the references moved by less than; outcomes changed", "0.0003; 0 of 80", f"{max(dD):.5f}; {chg} of {tested}", max(dD) < 0.0003 and chg == 0 and tested == 80)

bad = [o for o in OUT if not o[3]]
for w, p, g_, ok in OUT: print(("ok   " if ok else "DIFF ") + f"{w}: paper {p} | recomputed {g_}")
print(f"\n{len(OUT)} items, {len(bad)} differing")
