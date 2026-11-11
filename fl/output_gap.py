"""잔존 영향의 출력 기반 지표 (3장 v1.6 8.4절 E2 '재학습 대비 출력 불일치율과 출력 분포 차이').

각 모델의 X(철회 기록)·Z(비구성원) 예측 확률을 U0 모델들(시드별)과 비교한다.
  gap(m, S) = U0 각 모델과의 평균 절대 확률 차이의 평균
  기준선 = U0 모델들끼리의 같은 값(시드 변동). gap이 기준선보다 X에서만 크면 X에 대한 잔존 영향의 신호다.
"""
import argparse, itertools, json, os
import numpy as np, torch
from fl import train as T
from fl.run import INIT_SEED

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="/data"); ap.add_argument("--runs", required=True)
    ap.add_argument("--l3", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    device, mp = T.configure("gpu_det"); data = T.Data(a.cache)
    res = json.load(open(a.l3))
    from fl.run import INSTS
    rng = np.random.default_rng(20261025)
    pids = sorted({int(data.pid[i]) for k in INSTS for i in data.train_idx[k]})
    cs = set(int(p) for p in rng.choice(pids, int(len(pids) * res["cohort"]["frac"]), replace=False))
    X = np.sort(np.array([i for k in INSTS for i in data.train_idx[k] if int(data.pid[i]) in cs]))
    Z = np.sort(np.random.default_rng(1).choice(data.val_idx, min(len(X), len(data.val_idx)), replace=False))
    assert len(X) == res["cohort"]["records"]
    def load(path, norm):
        m = T.new_model(mp, device, INIT_SEED); m.load_state_dict(torch.load(path)); return m, norm
    on = res["methods"]["P1"]  # 표준화 통계는 runs의 events에서 읽는다
    def norm_of(run):
        for line in open(os.path.join(a.runs, run, "events.jsonl")):
            e = json.loads(line)
            if e["event"] == "NormStatsCommitted": return (e["mean"], e["std"])
    n_orig = norm_of("orig")
    models = {"P1": load(os.path.join(a.runs, "orig", "ckpt_r50.pt"), n_orig),
              "U1": load(os.path.join(a.runs, "U1", "ckpt_r50.pt"), n_orig)}
    u0 = {s: load(os.path.join(a.runs, f"U0_s{s}", "ckpt_r50.pt"), norm_of(f"U0_s{s}")) for s in range(1, 6)}
    pred = lambda mn, S: T.predict(mn[0], data, S, device, norm=mn[1])
    PU = {s: {"X": pred(m, X), "Z": pred(m, Z)} for s, m in u0.items()}
    out = {"baseline_U0_pairs": {}}
    for S in ("X", "Z"):
        out["baseline_U0_pairs"][S] = float(np.mean([np.abs(PU[i][S] - PU[j][S]).mean() for i, j in itertools.combinations(PU, 2)]))
    for name, m in models.items():
        P = {"X": pred(m, X), "Z": pred(m, Z)}
        out[name] = {S: float(np.mean([np.abs(P[S] - PU[s][S]).mean() for s in PU])) for S in ("X", "Z")}
        # U0 대비 판정 불일치율(임계 0.5)
        out[name]["disagree_X"] = float(np.mean([((P["X"] > .5) != (PU[s]["X"] > .5)).mean() for s in PU]))
    # 기준: U0 하나를 빼고 나머지와 비교(leave-one-out)
    loo = {S: [float(np.mean([np.abs(PU[i][S] - PU[j][S]).mean() for j in PU if j != i])) for i in PU] for S in ("X", "Z")}
    out["U0_loo"] = {S: {"mean": float(np.mean(v)), "max": float(np.max(v))} for S, v in loo.items()}
    out["U0_loo_disagree_X"] = float(np.mean([np.mean([((PU[i]["X"] > .5) != (PU[j]["X"] > .5)).mean() for j in PU if j != i]) for i in PU]))
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(os.path.join(a.out, "output_gap.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
main()
