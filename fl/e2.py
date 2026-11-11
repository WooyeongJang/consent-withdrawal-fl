"""E2 영향 관리와 잔존 영향(RQ2) — 시나리오 하나 × 시드 하나 (3장 v1.6 8.4, 8.3 U0–U5·P1·P2).

시나리오는 fl.consent가 만든 동의 사건(results/consent/<이름>.json)이다. 시드 s에서:
  P1  참여 차단만(원 학습: 철회 효력 라운드부터 입력에서 제외)          P2  계보 관리만(r₀ 이전 버전으로 회귀; r₀=1이면 초기 모델)
  (r₀ = 철회자의 참여 시작 라운드 중 최소. 시나리오에 enroll이 없으면 1 — 후기 합류 집단 시나리오는 25, 40번)
  U0  처음부터 재학습(기준; 참조 시드 1..nref는 시나리오마다 한 번 학습해 공유)
  U1  r₀ 체크포인트 재수행(원 학습의 표준화 통계 유지)                  U2  FedEraser(Δt=2, 보정 비율 0.5)
  U3  SISA(기관 내부 S조각, 철회자가 속한 조각만 재학습; 원 학습도 SISA 구조)
  U4  FedOSD(기본 T_u=5·lr 0.005, 강한 설정 T_u=20·lr 0.025)              U5  철회자가 속한 기관 전체 제외(모든 기관에 철회자가 있으면 해당 없음)
지표(8.4 표): 진단 성능(평가 집합 매크로·범주별 AUROC, F1[검증 집합에서 범주별 최적 임계값]),
하위 집단(여성 65세 이상, 기관별, 범주별), 잔존 영향(LiRA offline AUC와 U0 참조 대비 TOST, 출력 차이와 불일치율),
처리 비용(실행 시간, 모델 전송 횟수·바이트, 보관 상태 크기).
처리 작업의 배포 전이(44번 (a)): 방법마다 kind(exact | approx)와 deploy를 기록한다(fl.l3.deploy). U1은 r₀ 이전 표준화
통계 범위에 철회 기록이 없을 때만(r₀ > 1) 동등 조건을 충족해 exact다. U3(SISA)은 기준(U0)과 학습 구조가 달라 approx로 둔다.

  python -m fl.e2 --scenario results/consent/E2-base.json --seed 1 --out /out --runs /runs --ptbxl /ptbxl
"""

import argparse
import csv
import glob
import json
import os
import shutil
import time

import numpy as np
import torch

from fl.consent import demographics
from fl import fedosd as FO
from fl import sisa
from fl import train as T
from fl import unlearn2 as U2
from fl.exp_lira import evaluate, scores, stat
from fl.l3 import apply_power, deploy, tost
from fl.run import INIT_SEED, INSTS, Policy, Store, run_fl, state_cpu

S_SHARDS = 4
EXACT = {"U0", "U0-seq", "U5"}                       # 처음부터의 재학습 계열. U1·U1-seq는 r₀ > 1일 때 추가


def shadow_fl(data, base_by_k, members, member_w, seed, device, max_pool, rounds, norm, base_e=None, member_e=None):
    """그림자 모델: 구성원 기록은 자기 참여 시작 라운드 e부터 철회 라운드 w 전까지만 입력에 들어간다
    (exp_lira.shadow_fl의 일반화). 비구성원 기본 집합도 참여 시작 라운드 e부터 들어간다. e가 없으면 모두 1."""
    g = T.new_model(max_pool, device, T.derive_seed(seed, "init", 0))
    for r in range(1, rounds + 1):
        states, ws = [], []
        for k in INSTS:
            on = member_w > r if member_e is None else (member_e <= r) & (member_w > r)
            add = members[(data.inst[members] == k) & on]
            base = base_by_k[k] if base_e is None else base_by_k[k][base_e[k] <= r]
            S = np.sort(np.concatenate([base, add]))
            if len(S) == 0:
                continue
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(g.state_dict())
            m, _ = T.local_train(m, data, k, r, seed, device, rounds, idx=S, norm=norm)
            states.append({n: v.detach().clone() for n, v in m.state_dict().items()}); ws.append(len(S))
        g.load_state_dict(T.fedavg(states, ws))
    return g


def best_thresholds(Y, P):
    th = []
    for j in range(Y.shape[1]):
        cand = np.unique(np.round(P[:, j], 3))
        f1 = [f1_score(Y[:, j], P[:, j] >= c) for c in cand]
        th.append(float(cand[int(np.argmax(f1))]))
    return np.array(th)


def f1_score(y, yhat):
    tp = float(np.sum((y == 1) & yhat)); fp = float(np.sum((y == 0) & yhat)); fn = float(np.sum((y == 1) & ~yhat))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    for a in ("--scenario", "--out", "--runs", "--ptbxl"):
        ap.add_argument(a, required=True)
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--nref", type=int, default=5)
    ap.add_argument("--shadows", type=int, default=16)
    ap.add_argument("--keep", action="store_true", help="체크포인트를 지우지 않는다")
    ap.add_argument("--sequential", action="store_true", help="순차 철회를 차례별로 처리(39번). 결과 이름에 -perwave")
    ap.add_argument("--osd-variants", action="store_true",
                    help="FedOSD 공식 코드 차이 점검(46번 (a)): U0·U4·U4s와 변형(사후 사영 all, UCE pos)만. 결과 이름에 -osdcheck")
    args = ap.parse_args()
    sc = json.load(open(args.scenario))
    name, s, R = sc["name"], args.seed, args.rounds
    device, max_pool = T.configure("gpu_det")
    data = T.Data(args.cache)
    store = Store(data)
    root = os.path.join(args.runs, name)
    os.makedirs(root, exist_ok=True)

    withdraw = {int(p): w for p, w in sc["withdraw"].items()}
    reconsent = {int(p): c for p, c in sc["reconsent"].items()}
    enroll = {int(p): e for p, e in sc.get("enroll", {}).items()}   # 참여 시작 라운드(없으면 1)
    policy = Policy(enroll=enroll, withdraw=withdraw, reconsent=reconsent)
    permanent = sorted(p for p in withdraw if p not in reconsent)
    pol_wo = Policy(enroll={**enroll, **{p: 10 ** 9 for p in permanent}})   # 현재 동의 상태 기준의 재학습 정책(재동의자는 유효)
    pw = set(permanent)
    X = np.sort(np.array([int(i) for k in INSTS for i in data.train_idx[k] if int(data.pid[i]) in pw]))
    X_w = np.array([withdraw[int(data.pid[i])] for i in X])
    demo = {}
    with open(demographics(args.cache, args.ptbxl)) as f:                  # 보조 데이터는 캐시 meta.csv의 age·sex
        for r in csv.DictReader(f):
            demo[int(r["ecg_id"])] = (float(r["age"]) if r["age"] else -1, int(float(r["sex"])) if r["sex"] else -1)
    # 비구성원 집합 Z: 철회 레코드와 같은 분포(8.4) — 시나리오의 집중 조건을 검증 집합(fold 9)에 같게 적용한다.
    # 무작위 분포 시나리오는 검증 집합 전체에서 뽑는다(기존과 같은 난수·같은 결과).
    dist, val = sc["spec"].get("dist", {"type": "random"}), data.val_idx
    if "exit_inst" in sc["spec"]:
        elig = val[data.inst[val] == sc["spec"]["exit_inst"]]
    elif dist["type"] == "diag":
        elig = val[data.Y[val][:, T.CATS.index(dist["cat"])] == 1]
    elif dist["type"] == "agesex":
        elig = val[np.array([demo[int(data.ecg[i])][1] == dist["sex"] and demo[int(data.ecg[i])][0] >= dist["age_min"] for i in val])]
    elif dist["type"] == "inst":
        elig = val[data.inst[val] == dist["inst"]]
    else:
        elig = val
    Z = np.sort(np.random.default_rng(1).choice(elig, min(len(X), len(elig)), replace=False))
    Zk = np.array([z for z in Z if data.inst[z] in INSTS])
    pool = np.r_[X, Zk]
    nX = len(X)
    r0 = min((enroll.get(p, 1) for p in permanent), default=1)  # 철회자의 첫 사용 라운드 중 최소(3장 4.3)
    exact = EXACT | ({"U1", "U1-seq"} if r0 > 1 else set())   # r₀ > 1이면 1라운드 표준화 범위에 철회 기록이 없다

    # 평가용 하위 집단: 평가 집합(fold 10)의 여성 65세 이상, 기관별
    tst = data.test_idx
    groups = {"F65": tst[np.array([demo[int(data.ecg[i])][1] == 1 and demo[int(data.ecg[i])][0] >= 65 for i in tst])]}
    groups.update({f"inst{k}": tst[data.inst[tst] == k] for k in INSTS})

    param_bytes = None

    def evaluate_model(model, nm, refs_X=None, refs_Z=None, self_ref=None):
        Pt = T.predict(model, data, tst, device, norm=nm)
        Pv = T.predict(model, data, data.val_idx, device, norm=nm)
        th = best_thresholds(data.Y[data.val_idx], Pv)
        macro, per = T.macro_auc(data.Y[tst], Pt)
        f1 = [f1_score(data.Y[tst][:, j], Pt[:, j] >= th[j]) for j in range(Pt.shape[1])]
        sub = {}
        for gname, gi in groups.items():
            pos = np.isin(tst, gi)
            Yg, Pg = data.Y[tst][pos], Pt[pos]
            ok = [j for j in range(Yg.shape[1]) if 0 < Yg[:, j].sum() < len(Yg)]
            sub[gname] = {"n": int(pos.sum()), "macro_auc": float(np.mean([T.auc(Yg[:, j], Pg[:, j]) for j in ok]))}
        out = {"test_macro_auc": macro, "test_auc_by_cat": dict(zip(T.CATS, per)), "test_f1_by_cat": dict(zip(T.CATS, f1)),
               "test_macro_f1": float(np.mean(f1)), "subgroup": sub}
        _, off = scores(stat(model, data, pool, device, nm), phis, inmask)
        e = evaluate(off, nX)
        out["lira_offline"] = e
        PX = T.predict(model, data, X, device, norm=nm)
        PZ = T.predict(model, data, Zk, device, norm=nm)
        out["_PX"], out["_PZ"] = PX, PZ
        if refs_X is not None:
            use = [j for j in range(len(refs_X)) if j != self_ref]
            out["gap_X"] = float(np.mean([np.abs(PX - refs_X[j]).mean() for j in use]))
            out["gap_Z"] = float(np.mean([np.abs(PZ - refs_Z[j]).mean() for j in use]))
            out["disagree_X"] = float(np.mean([((PX >= 0.5) != (refs_X[j] >= 0.5)).mean() for j in use]))
            out["disagree_Z"] = float(np.mean([((PZ >= 0.5) != (refs_Z[j] >= 0.5)).mean() for j in use]))
        return out

    # ── 그림자 모델(시나리오마다 한 번) ──
    base_by_k = {k: np.array([i for i in data.train_idx[k] if int(data.pid[i]) not in set(withdraw)], dtype=np.int64)
                 for k in INSTS}
    rng_w = np.random.default_rng(7)
    iz = rng_w.choice(len(X_w), len(Zk))                       # rng_w.choice(X_w, n)과 같은 난수열
    pool_w = np.r_[X_w, X_w[iz]]                              # 비구성원 기록에도 철회 라운드를 같은 분포로 배정
    X_e = np.array([enroll.get(int(data.pid[i]), 1) for i in X], dtype=np.int64)
    pool_e = np.r_[X_e, X_e[iz]] if enroll else None           # 참여 시작 라운드도 같은 쌍으로 배정
    base_e = ({k: np.array([enroll.get(int(data.pid[i]), 1) for i in base_by_k[k]], dtype=np.int64) for k in INSTS}
              if enroll else None)
    sh_file = os.path.join(root, "shadows.npz")
    norm_sh = None
    t0 = time.time()
    # 그림자의 표준화 통계: 원 학습과 같은 범위(1라운드 입력 집합)
    scope = np.concatenate([store.input_set(policy, k, 1) for k in INSTS])
    norm_sh = T.norm_stats(data, scope)
    if os.path.exists(sh_file):
        z = np.load(sh_file); phis, inmask = z["phis"], z["inmask"]
    else:
        phis, inmask = np.zeros((args.shadows, len(pool))), np.zeros((args.shadows, len(pool)), bool)
        for i in range(args.shadows):
            mask = np.zeros(len(pool), bool)
            mask[np.random.default_rng(1000 + i).choice(len(pool), len(pool) // 2, replace=False)] = True
            g = shadow_fl(data, base_by_k, pool[mask], pool_w[mask], 1000 + i, device, max_pool, R, norm_sh,
                          base_e, None if pool_e is None else pool_e[mask])
            phis[i], inmask[i] = stat(g, data, pool, device, norm_sh), mask
            print(f"[{name}] shadow {i + 1}/{args.shadows}", flush=True)
        np.savez(sh_file, phis=phis, inmask=inmask)
    t_shadow = time.time() - t0
    # 공격의 검정력(33번 (a)): 그림자 하나를 대상으로 두고 나머지로 공격한 leave-one-out AUC(online·offline)
    loo = {"online": [], "offline": []}
    for i in range(len(phis)):
        keep = np.arange(len(phis)) != i
        on, off = scores(phis[i], phis[keep], inmask[keep])
        loo["online"].append(T.auc(inmask[i].astype(float), on)); loo["offline"].append(T.auc(inmask[i].astype(float), off))
    shadow_loo = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in loo.items()}

    # ── U0 참조(시나리오마다 nref개, 공유) ──
    refs = []
    for j in range(1, args.nref + 1):
        d = os.path.join(root, f"U0_s{j}")
        f = os.path.join(d, f"ckpt_r{R}.pt")
        t0 = time.time()
        if not os.path.exists(f):
            o = run_fl(data, store, pol_wo, base_seed=j, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                       run_dir=d, label=f"U0_s{j}", log_every=0)
            for c in glob.glob(os.path.join(d, "ckpt_r*.pt")):
                if not c.endswith(f"ckpt_r{R}.pt"):
                    os.remove(c)
            json.dump({"wall_s": time.time() - t0, "commits": len(o["commits"])}, open(os.path.join(d, "cost.json"), "w"))
        ev = next(json.loads(l) for l in open(os.path.join(d, "events.jsonl")) if "NormStatsCommitted" in l)
        m = T.new_model(max_pool, device, INIT_SEED); m.load_state_dict(torch.load(f))
        refs.append((m, (ev["mean"], ev["std"]), json.load(open(os.path.join(d, "cost.json")))))
    param_bytes = sum(v.numel() * v.element_size() for v in refs[0][0].state_dict().values())
    refs_X = [T.predict(m, data, X, device, norm=nm) for m, nm, _ in refs]
    refs_Z = [T.predict(m, data, Zk, device, norm=nm) for m, nm, _ in refs]
    ref_lira = [scores(stat(m, data, pool, device, nm), phis, inmask)[1] for m, nm, _ in refs]
    u0_aucs = [evaluate(o, nX)["auc"] for o in ref_lira]

    res = {"scenario": name, "scenario_hash": sc["hash"], "seed": s, "Z_matched": dist["type"] != "random" or "exit_inst" in sc["spec"], "rounds": R, "X_records": int(nX), "Z_records": int(len(Zk)),
           "u0_ref_lira_auc": u0_aucs, "shadow_wall_s": t_shadow, "shadow_loo_auc": shadow_loo, "param_bytes": int(param_bytes), "methods": {}}

    def record(label, model, nm, cost, self_ref=None):
        t0 = time.time()
        out = evaluate_model(model, nm, refs_X, refs_Z, self_ref)
        out.pop("_PX"); out.pop("_PZ")
        out["L3"] = tost(out["lira_offline"]["auc"], out["lira_offline"]["ci"], u0_aucs) if label not in ("P2",) else None
        if label not in ("P1", "P2", "U3-orig"):                # 처리 작업만 배포 전이(44번 (a))
            out["kind"] = "exact" if label in exact else "approx"
            out["L3"] = apply_power(out["L3"], res["methods"]["P1"]["L3"], out["kind"])   # 양성 대조(33번 (a))
            out["deploy"] = deploy(out["kind"], out["L3"])
        out["cost"] = cost
        out["self_ref"] = self_ref                                 # 참조 집합 안의 자기 자신 인덱스(없으면 None), 집계에서 제외(C2)
        out["hash"] = T.model_hash(model)[0]
        res["methods"][label] = out
        print(f"[{name} s{s}] {label:8s} test {out['test_macro_auc']:.4f}  LiRA {out['lira_offline']['auc']:.4f}  "
              f"gapX {out['gap_X']:.4f}/Z {out['gap_Z']:.4f}  L3 {out['L3']['result'] if out['L3'] else '-'}  "
              f"({cost.get('wall_s', 0):.0f}s)", flush=True)

    def transfers(commits):
        return {"model_transfers": 2 * len(commits), "bytes": 2 * len(commits) * param_bytes}

    def perwave():
        """순차 철회 차례별 처리(3장 반영 39번): 세 요청을 차례대로 받아 앞 처리 결과 위에 다시 처리한다.
        요청 i는 그 시점의 동의 상태(차례 1..i 철회자 제외)를 쓴다. 정확한 방법(U0·U1·U3)은 마지막 결과가
        누적 1회 처리와 같아야 하며(해시로 확인), 근사 방법(U2·U4)은 누적 오차가 생길 수 있다."""
        waves = sorted({withdraw[p] for p in permanent})
        pols, cum = [], []
        for w in waves:
            cum += [p for p in permanent if withdraw[p] == w]
            pols.append(Policy(enroll={**enroll, **{p: 10 ** 9 for p in cum}}))
        Xw = [np.sort(np.array([i for i in X if withdraw[int(data.pid[i])] == w], dtype=np.int64)) for w in waves]
        ck0 = os.path.join(od, f"ckpt_r{r0 - 1}.pt")
        for lab in ("U0", "U1"):
            t0, n_c = time.time(), 0
            for i, pol in enumerate(pols):
                kw = {} if lab == "U0" else {"start_state": torch.load(ck0), "start_round": r0, "norm": orig["norm"]}
                o = run_fl(data, store, pol, base_seed=s, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                           run_dir=os.path.join(root, f"{lab}seq{i}_s{s}"), label=f"{lab}-seq{i}", log_every=0, **kw)
                n_c += len(o["commits"])
            record(f"{lab}-seq", o["model"], nm if lab == "U1" else (o["norm"]["mean"], o["norm"]["std"]),
                   {"wall_s": time.time() - t0, **transfers(range(n_c)), "requests": len(pols)},
                   self_ref=s - 1 if lab == "U0" and s <= args.nref else None)   # U0-seq는 같은 시드 참조와 같은 모델
        t0, src, n_c = time.time(), od, 0
        for i, pol in enumerate(pols):
            d = os.path.join(root, f"U2seq{i}_s{s}")
            o = U2.federaser(data, store, pol, src, s, device, max_pool, R, orig["norm"], save_dir=d)
            src, n_c = d, n_c + len(o["commits"])
        record("U2-seq", o["model"], nm, {"wall_s": time.time() - t0, **transfers(range(n_c)), "requests": len(pols)})
        t0 = time.time()
        _, st, _ = sisa.train(data, store, policy, S_SHARDS, s, device, max_pool, R, orig["norm"],
                              os.path.join(root, f"sisa_orig_s{s}"))
        t0, n_c, hits = time.time(), 0, []
        for i, pol in enumerate(pols):
            hit = sorted({sisa.shard_of(int(data.pid[j]), S_SHARDS) for j in Xw[i]})
            sis, st, runs = sisa.train(data, store, pol, S_SHARDS, s, device, max_pool, R, orig["norm"],
                                       os.path.join(root, f"sisa_seq{i}_s{s}"), only=set(hit), prev=st)
            n_c += sum(len(o["commits"]) for o in runs.values()); hits.append(len(hit))
        record("U3-seq", sis, nm, {"wall_s": time.time() - t0, **transfers(range(n_c)), "requests": len(pols),
                                   "shards_retrained": hits, "shards": S_SHARDS})
        for lab, tu, lr in (("U4", 5, 0.005), ("U4s", 20, 0.025)):
            t0, start, n_c = time.time(), torch.load(os.path.join(od, f"ckpt_r{R}.pt")), 0
            for i, pol in enumerate(pols):
                o = FO.fedosd(data, store, pol, Xw[i], start, s, device, max_pool, R, orig["norm"], tu, 10, lr)
                start, n_c = state_cpu(o["model"]), n_c + len(o["commits"])
            record(f"{lab}-seq", o["model"], nm, {"wall_s": time.time() - t0, **transfers(range(n_c)),
                                                  "requests": len(pols), "Tu": tu, "lr": lr})

    # ── 원 학습(P1) ──
    od = os.path.join(root, f"orig_s{s}")
    t0 = time.time()
    orig = run_fl(data, store, policy, base_seed=s, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                  run_dir=od, label="orig", log_every=0, retain_every=2)
    t_orig = time.time() - t0
    nm = (orig["norm"]["mean"], orig["norm"]["std"])
    res["orig_wall_s"] = t_orig
    record("P1", orig["model"], nm, {"wall_s": 0.0, "model_transfers": 0, "bytes": 0})
    # P2: r₀ 이전 버전으로 회귀(r₀ = 1이면 초기 모델)
    m = T.new_model(max_pool, device, INIT_SEED); m.load_state_dict(torch.load(os.path.join(od, f"ckpt_r{r0 - 1}.pt")))
    record("P2", m, nm, {"wall_s": 0.0, "model_transfers": 0, "bytes": 0, "rollback_to": r0 - 1})

    if args.sequential:
        perwave()
    elif args.osd_variants:
        m, nmu, c = refs[s - 1] if s <= args.nref else (None, None, None)
        if m is not None:
            record("U0", m, nmu, {"wall_s": c["wall_s"], **transfers(range(c["commits"]))}, self_ref=s - 1)
        start = torch.load(os.path.join(od, f"ckpt_r{R}.pt"))
        for lab, tu, lr, pp, uc in (("U4", 5, 0.005, "conflict", "all"), ("U4-pa", 5, 0.005, "all", "all"),
                                    ("U4-up", 5, 0.005, "conflict", "pos"), ("U4-off", 5, 0.005, "all", "pos"),
                                    ("U4s", 20, 0.025, "conflict", "all"), ("U4s-off", 20, 0.025, "all", "pos")):
            t0 = time.time()
            o = FO.fedosd(data, store, pol_wo, X, start, s, device, max_pool, R, orig["norm"], tu, 10, lr, post_proj=pp, uce=uc)
            record(lab, o["model"], nm, {"wall_s": time.time() - t0, **transfers(o["commits"]), "Tu": tu, "lr": lr,
                                         "post_proj": pp, "uce": uc})
    else:
        # U0(이 시드)
        if s <= args.nref:
            m, nmu, c = refs[s - 1]
            record("U0", m, nmu, {"wall_s": c["wall_s"], **transfers(range(c["commits"]))}, self_ref=s - 1)
        else:
            t0 = time.time()
            o = run_fl(data, store, pol_wo, base_seed=s, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                       run_dir=os.path.join(root, f"U0x_s{s}"), label="U0", log_every=0)
            record("U0", o["model"], (o["norm"]["mean"], o["norm"]["std"]), {"wall_s": time.time() - t0, **transfers(o["commits"])})
        # U1: r₀ 체크포인트에서 같은 표준화 통계로 재수행
        t0 = time.time()
        o = run_fl(data, store, pol_wo, base_seed=s, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                   start_state=torch.load(os.path.join(od, f"ckpt_r{r0 - 1}.pt")), start_round=r0, norm=orig["norm"],
                   run_dir=os.path.join(root, f"U1_s{s}"), label="U1", log_every=0)
        record("U1", o["model"], nm, {"wall_s": time.time() - t0, **transfers(o["commits"]), "storage_bytes": param_bytes})
        # U2: FedEraser
        t0 = time.time()
        o = U2.federaser(data, store, pol_wo, od, s, device, max_pool, R, orig["norm"])
        upd = sum(os.path.getsize(f) for f in glob.glob(os.path.join(od, "upd_r*_k*.pt")))
        ck = sum(os.path.getsize(os.path.join(od, f"ckpt_r{t - 1}.pt")) for t in range(1, R + 1, 2))
        record("U2", o["model"], nm, {"wall_s": time.time() - t0, **transfers(o["commits"]), "storage_bytes": upd + ck})
        # U3: SISA(원 학습도 SISA 구조로 따로 학습)
        t0 = time.time()
        sis_orig, states, _ = sisa.train(data, store, policy, S_SHARDS, s, device, max_pool, R, orig["norm"],
                                         os.path.join(root, f"sisa_orig_s{s}"))
        t_sis = time.time() - t0
        record("U3-orig", sis_orig, nm, {"wall_s": 0.0, "model_transfers": 0, "bytes": 0, "train_wall_s": t_sis})
        hit = sorted({sisa.shard_of(p, S_SHARDS) for p in withdraw})
        t0 = time.time()
        sis_u, _, runs = sisa.train(data, store, pol_wo, S_SHARDS, s, device, max_pool, R, orig["norm"],
                                    os.path.join(root, f"sisa_u_s{s}"), only=set(hit), prev=states)
        record("U3", sis_u, nm, {"wall_s": time.time() - t0, **transfers([c for o in runs.values() for c in o["commits"]]),
                                 "shards_retrained": len(hit), "shards": S_SHARDS, "storage_bytes": S_SHARDS * param_bytes})
        # U4: FedOSD(기본·강한 설정)
        start = torch.load(os.path.join(od, f"ckpt_r{R}.pt"))
        for lab, tu, lr in (("U4", 5, 0.005), ("U4s", 20, 0.025)):
            t0 = time.time()
            o = FO.fedosd(data, store, pol_wo, X, start, s, device, max_pool, R, orig["norm"], tu, 10, lr)
            record(lab, o["model"], nm, {"wall_s": time.time() - t0, **transfers(o["commits"]), "Tu": tu, "lr": lr})
        # U5: 철회자가 속한 기관 전체 제외
        hit_inst = sorted({str(data.inst[i]) for i in X})
        if len(hit_inst) < len(INSTS):
            ex = Policy(enroll={int(data.pid[i]): 10 ** 9 for k in hit_inst for i in data.train_idx[k]})
            t0 = time.time()
            o = run_fl(data, store, ex, base_seed=s, device=device, max_pool=max_pool, cond="gpu_det", rounds=R,
                       run_dir=os.path.join(root, f"U5_s{s}"), label="U5", log_every=0)
            record("U5", o["model"], (o["norm"]["mean"], o["norm"]["std"]),
                   {"wall_s": time.time() - t0, **transfers(o["commits"]), "excluded_inst": hit_inst})
        else:
            res["methods"]["U5"] = {"not_applicable": f"철회자가 모든 기관에 있음({hit_inst})"}

    os.makedirs(args.out, exist_ok=True)
    if args.sequential:
        res["scenario"] = name = name + "-perwave"
    elif args.osd_variants:
        res["scenario"] = name = name + "-osdcheck"
    with open(os.path.join(args.out, f"{name}_s{s}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    if not args.keep:
        for d in glob.glob(os.path.join(root, f"*_s{s}")):
            if not os.path.basename(d).startswith("U0_s"):
                shutil.rmtree(d, ignore_errors=True)
    print(f"[{name} s{s}] done", flush=True)


if __name__ == "__main__":
    main()
