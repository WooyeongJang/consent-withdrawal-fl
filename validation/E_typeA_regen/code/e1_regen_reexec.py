"""E1: 유형 A 학습 궤적을 원장 없이 결정적으로 다시 만들어 원장 해시와 대조하고, 표본 단위를 재실행한다(컨테이너 GPU).

컨테이너 안에서(저장소 /w 읽기 전용, 데이터 /data, 재생성 폴더 /runs, 출력 /out):
  python /w/validation/E_typeA_regen/code/e1_regen_reexec.py --chain /w/validation/E_typeA_regen/results/e0_chain.json --out /out/e1_results.json
시드 단위로 재시작 가능(끝난 시드는 건너뜀). 주입 배정은 fl/e3.py run_type_a 54–64행의 규칙을 그대로 쓴다.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, "/w")
from fl import audit as AU  # noqa: E402
from fl import evidence as EV  # noqa: E402
from fl import train as T  # noqa: E402
from fl.run import INIT_SEED, INSTS, Policy, Store, code_hash, run_fl  # noqa: E402

ATTACKS_A = ["T1", "T2a", "T2b", "T3", "T10"]
SPEC = {"T2a": {"mode": "included"}, "T3": {"mode": "stale"}, "T2b": {"mode": "hidden"}, "T10": {"mode": "included", "collude": True}}


def assignment(data, store, seed, R=50):
    """fl/e3.py run_type_a 52–64행과 같은 규칙."""
    rng = random.Random(seed)
    cands = sorted({max(2, round(R * f)) for f in (0.16, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88)})
    slots = rng.sample(cands, len(ATTACKS_A))
    insts = [INSTS[i % len(INSTS)] for i in rng.sample(range(len(ATTACKS_A)), len(ATTACKS_A))]
    used_pids, inst_list = set(), []
    for atk, r, k in zip(ATTACKS_A, slots, insts):
        pids = sorted({int(data.pid[i]) for i in data.train_idx[k]} - used_pids)
        pid = rng.choice(pids); used_pids.add(pid)
        inst_list.append({"attack": atk, "k": k, "r": r, "pid": pid})
    return inst_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--runs", default="/runs")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", "/w", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"# E1 regen+reexec; start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; git HEAD {head}; image {os.environ.get('IMAGE_DIGEST')}; "
          f"script sha256 {hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16]}; fl code hash {code_hash()[:16]}; torch {torch.__version__}", flush=True)
    device, max_pool = T.configure("gpu_det")
    data = T.Data(a.cache)
    store = Store(data)
    chain = json.load(open(a.chain))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"restarts": 0, "seeds": {}}
    if os.path.exists(a.out):
        res["restarts"] += 1
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else sorted(int(s) for s in chain["seeds"])
    for s in seeds:
        if str(s) in res["seeds"] and res["seeds"][str(s)].get("done"):
            continue
        info = chain["seeds"][str(s)]
        t0 = time.time()
        inst_list = assignment(data, store, s, a.rounds)
        mine = {x["attack"]: (x["k"], x["r"], x["pid"]) for x in inst_list}
        inj = info["injections"]
        match = all(mine[t] == (inj[t][0], inj[t][1], inj[t][2]) for t in ("T2a", "T2b", "T3", "T10")) and mine["T1"][2] == inj["T1"]
        rec = {"contract": info["contract"], "assignment": mine, "assignment_matches_chain": match}
        print(f"seed {s:2d}: assignment {mine} matches chain {match}", flush=True)
        if not match:
            rec.update(done=True, skipped="assignment mismatch")
            res["seeds"][str(s)] = rec
            json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
            continue
        policy = Policy(withdraw={x["pid"]: x["r"] for x in inst_list if x["attack"] != "T1"})
        tamper = {(x["k"], x["r"]): {**SPEC[x["attack"]], "pids": [x["pid"]]} for x in inst_list if x["attack"] in SPEC}
        run_dir = os.path.join(a.runs, f"A_s{s}")
        run = run_fl(data, store, policy, base_seed=s, device=device, max_pool=max_pool, cond="gpu_det", rounds=a.rounds,
                     run_dir=run_dir, label=f"E1-A-s{s}", tamper=tamper, log_every=0)
        t_train = time.time() - t0
        # 궤적 대조
        m0 = T.new_model(max_pool, device, INIT_SEED); m0.load_state_dict(torch.load(os.path.join(run_dir, "ckpt_r0.pt")))
        start_ok = T.model_hash(m0)[0] == info["start_model"]
        hout_ok = sum(1 for r in range(1, a.rounds + 1) if run["round_hashes"][r] == info["h_out"][str(r)])
        first_bad = next((r for r in range(1, a.rounds + 1) if run["round_hashes"][r] != info["h_out"][str(r)]), None)
        rejected = sorted((c["C"]["k"], c["C"]["r"]) for c in run["commits"] if c["status"] != "accepted")
        rej_ok = rejected == [tuple(x) for x in info["rejected"]]
        rec.update(start_model_matches=start_ok, h_out_matches=hout_ok, first_mismatch_round=first_bad, rejected=rejected,
                   rejected_matches_chain=rej_ok, train_seconds=round(t_train, 1), norm=run["norm"])
        print(f"seed {s:2d}: start {start_ok}, hOut {hout_ok}/{a.rounds}, rejected match {rej_ok} ({len(rejected)}), train {t_train:.0f}s", flush=True)
        # 표본 재실행
        inj_units = {(inj[t][0], inj[t][1]): t for t in ("T2a", "T2b", "T3", "T10")}
        units = {}
        for u in info["sampled"]:
            commit = {"C": {"k": u["k"], "r": u["r"], "root_S": u["root_S"], "sigma": u["sigma"]}, "U": {"h_start": u["h_start"], "h_delta": u["h_delta"]}}
            t1 = time.time()
            v = AU.reexecute(data, store, policy, run_dir, commit, s, device, max_pool, a.rounds, run["norm"], run["prog"])
            units[f"{u['k']}:{u['r']}"] = {"result": v["result"], "reason": v.get("reason", ""), "injected": inj_units.get((u["k"], u["r"])),
                                           "seconds": round(time.time() - t1, 2)}
        honest = [x for x in units.values() if not x["injected"]]
        injected = [x for x in units.values() if x["injected"]]
        rec.update(units=units, n_sampled=len(units), honest_pass=sum(x["result"] == "통과" for x in honest), honest_n=len(honest),
                   injected_mismatch=sum(x["result"] != "통과" for x in injected), injected_n=len(injected),
                   injected_detail={x["injected"]: (x["result"], x["reason"]) for x in injected}, done=True)
        print(f"seed {s:2d}: sampled {len(units)}, honest pass {rec['honest_pass']}/{rec['honest_n']}, injected mismatch {rec['injected_mismatch']}/{rec['injected_n']} {rec['injected_detail']}", flush=True)
        res["seeds"][str(s)] = rec
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    done = [v for v in res["seeds"].values() if v.get("done") and "units" in v]
    res["summary"] = {"seeds_done": len(done), "assignment_matches": sum(v["assignment_matches_chain"] for v in res["seeds"].values()),
                      "start_model_matches": sum(v.get("start_model_matches", False) for v in done),
                      "h_out_matches": sum(v.get("h_out_matches", 0) for v in done), "h_out_total": 50 * len(done),
                      "rejected_matches": sum(v.get("rejected_matches_chain", False) for v in done),
                      "sampled": sum(v["n_sampled"] for v in done), "honest_pass": sum(v["honest_pass"] for v in done), "honest_n": sum(v["honest_n"] for v in done),
                      "injected_mismatch": sum(v["injected_mismatch"] for v in done), "injected_n": sum(v["injected_n"] for v in done),
                      "train_seconds": round(sum(v["train_seconds"] for v in done), 1)}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("summary", res["summary"], flush=True)


if __name__ == "__main__":
    main()
