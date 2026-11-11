"""Compare the rerun of the 10% early-withdrawal scenario (results/c4_rerun/) with the stored results (results/final/e2/).

In that scenario one withdrawn record was a member in all 16 shadow models. The run-time code gave it an undefined
LiRA score that ranked first; the released code guards against this. The rerun repeats seeds 1-10 with the released
code, reusing the stored shadow and reference models. Equal model hashes show that training is bit-identical, so the
remaining differences come from scoring alone.
    python scripts/compare_c4_rerun.py
"""
import json, glob, os, statistics as st
SC = "E2-rate10-early"
rows, hash_ok, n = {}, 0, 0
for f in sorted(glob.glob(f"results/c4_rerun/{SC}_s*.json")):
    new = json.load(open(f)); old = json.load(open(f"results/final/e2/{os.path.basename(f)}"))
    for m, v in new["methods"].items():
        o = old["methods"][m]
        if "hash" in v:
            n += 1; hash_ok += v["hash"] == o["hash"]
        if "lira_offline" not in v: continue
        r = rows.setdefault(m, {"dauc": [], "ddiff": [], "cls": 0, "runs": 0})
        r["runs"] += 1
        r["dauc"].append(v["lira_offline"]["auc"] - o["lira_offline"]["auc"])
        if v.get("L3") and o.get("L3"):
            r["ddiff"].append(v["L3"]["diff"] - o["L3"]["diff"])
            lo, hi = v["L3"]["ci90"]; lo0, hi0 = o["L3"]["ci90"]
            inside = lambda a, b: a > -0.05 and b < 0.05
            r["cls"] += (inside(lo, hi) != inside(lo0, hi0)) or ((lo > 0) != (lo0 > 0))
    print(os.path.basename(f), "u0 refs max |Δ|: %.6f" % max(abs(a - b) for a, b in zip(new["u0_ref_lira_auc"], old["u0_ref_lira_auc"])))
print(f"model hashes equal: {hash_ok}/{n}")
for m, r in rows.items():
    d = r["dauc"]; dd = r["ddiff"] or [0.0]
    print(f"{m:8s} runs={r['runs']:2d}  ΔAUC mean {st.mean(d):+.6f} (min {min(d):+.6f}, max {max(d):+.6f})  Δ(diff vs U0) max |.| {max(abs(x) for x in dd):.6f}  interval-side changes {r['cls']}")
