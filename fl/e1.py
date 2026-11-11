"""E1 정책 집행 정합성(RQ1) — 이산 사건 시뮬레이션 (3장 v1.6 8.4 E1, 알고리즘 2).

판정 기준 시점을 비교한다.
  sel    데이터 선택 시점 판정(B1: 중앙 동의 DB를 선택 때 조회, 수락 때는 검사하지 않음)
  acc    집계 수락 시점 판정(B4: 커밋된 입력 집합에 수락 시점까지 효력이 생긴 철회 레코드가 없음을 비소속 증명으로 확인, 불통과면 거절·재계산)
  strict 집계 수락 시점 + 정책 버전 엄격 비교(커밋의 정책 버전 σ가 현재 버전보다 낮으면 무조건 거절) — 잘못된 거절의 비교 조건

시간 모델(라운드 길이 L초): 라운드 r의 기관 k는 T_r에서 입력을 선택하고(t_sel), 학습 시간 d_k 뒤에 갱신을 보내며(t_send),
집계 주체는 T_r + a·L(+집계 지연)에서 수락 판정을 한다(t_acc). 철회 요청은 원장 포함 지연 λ 뒤 효력(δ=0)을 얻는다.
로컬 학습은 실행하지 않고 단계 수(⌈n/128⌉)로 비용을 센다. 입력 집합·레코드 식별자·비소속 증명은 실제 PTB-XL 기관 분할과
fl.evidence의 정렬 머클 트리를 쓴다.

시나리오: during(선택 후·전송 전 효력), aftersend(전송 후·수락 전 효력), aggdelay(효력 시각 균일, 집계 지연 D=0·0.5L·2L; D=0이 기준),
          version(기관 정책 사본의 동기화 지연), offline(기관 오프라인 후 재접속 시 과거 선택으로 만든 갱신 제출).

  python -m fl.e1 --cache ~/data/cache/ptbxl_k3 --out results/e1 --seeds 20
"""

import argparse
import json
import math
import os

import numpy as np

from fl import evidence as EV
from fl import train as T
from fl.run import INSTS, Store

L = 60.0          # 라운드 길이(초)
A_FRAC = 0.8      # 수락 판정 시점(라운드 시작 후 a·L)
BS = 128


def simulate(data, store, scen, rule, seed, R=50, n_w=148, agg_delay=0.0, sync_q=0.3, offline_m=3):
    rng = np.random.default_rng([seed, 11])
    pids_k = {k: sorted({int(data.pid[i]) for i in data.train_idx[k]}) for k in INSTS}
    recs = {k: {} for k in INSTS}
    for k in INSTS:
        for i in data.train_idx[k]:
            recs[k].setdefault(int(data.pid[i]), []).append(int(i))
    n_k = {k: len(data.train_idx[k]) for k in INSTS}
    d_k = {k: 0.5 * L * n_k[k] / max(n_k.values()) for k in INSTS}       # 학습 시간(기록 수에 비례)
    t_round = lambda r: (r - 1) * L
    t_acc0 = lambda r: t_round(r) + A_FRAC * L + agg_delay

    # 철회 사건: 시나리오가 정한 창 안에서 효력 시각을 뽑는다(라운드 5–45)
    all_p = [(k, p) for k in INSTS for p in pids_k[k]]
    pick = rng.choice(len(all_p), n_w, replace=False)
    W = []
    for j in pick:
        k, p = all_p[j]
        r = int(rng.integers(5, 46))
        if scen == "during":
            t = t_round(r) + rng.uniform(0, d_k[k])
        elif scen == "aftersend":
            t = t_round(r) + rng.uniform(d_k[k], A_FRAC * L)
        else:
            t = t_round(r) + rng.uniform(0, L)
        lam = rng.uniform(1.0, 3.0)                                        # 원장 포함 지연(Kaia 1초 블록, 제출~영수증)
        W.append({"k": k, "pid": p, "t_req": t - lam, "t_eff": t})
    W.sort(key=lambda w: w["t_eff"])
    eff_by_k = {k: [w for w in W if w["k"] == k] for k in INSTS}

    def withdrawn_at(k, t):
        return {w["pid"] for w in eff_by_k[k] if w["t_eff"] <= t}

    def version_at(t):
        return 1 + sum(w["t_eff"] <= t for w in W)

    # 오프라인 기관: 한 기관이 라운드 a..a+m-1 동안 빠졌다가 a+m에 a에서 선택한 자료로 만든 갱신을 낸다
    off = None
    if scen == "offline":
        off = {"k": INSTS[int(rng.integers(0, len(INSTS)))], "a": int(rng.integers(5, 45 - offline_m))}   # 2026-11-01(C6): K 하드코딩 제거
    sync_t = {k: 0.0 for k in INSTS}                                       # version: 기관 정책 사본의 마지막 동기화 시각

    out = {"updates": 0, "accepted": 0, "viol_acc": 0, "viol_sel": 0, "rej_correct": 0, "rej_wrong": 0,
           "recompute_updates": 0, "recompute_steps": 0, "extra_wait_s": 0.0}
    last_inclusion = {}                                                    # pid -> 마지막으로 수락된 갱신의 수락 시각
    round_acc = {}                                                         # 라운드별 실제 집계 시각(재계산을 기다린 뒤)
    for r in range(1, R + 1):
        t_acc = t_acc0(r)
        round_acc[r] = t_acc
        for k in INSTS:
            if off and off["k"] == k and off["a"] <= r < off["a"] + offline_m:
                continue
            # 선택 시점과 선택에 쓴 철회 목록
            if off and off["k"] == k and r == off["a"] + offline_m:
                t_sel = t_round(off["a"])                                  # 재접속: 오프라인 직전 선택으로 만든 갱신
            else:
                t_sel = t_round(r)
            if scen == "version":
                if rng.random() > sync_q:
                    sync_t[k] = t_sel
                t_view = sync_t[k]                                         # 동기화하지 못하면 이전 사본
            else:
                t_view = t_sel
            excl = withdrawn_at(k, t_view)
            used = [p for p in pids_k[k] if p not in excl]
            sigma = version_at(t_view)
            out["updates"] += 1
            acc_t = t_acc
            while True:
                bad = withdrawn_at(k, acc_t) & set(used)                  # 수락 시점에 효력이 있는 철회자의 기록 포함 여부
                q = [store.rid[i] for p in withdrawn_at(k, acc_t) for i in recs[k][p]]
                if rule == "sel" or not q and rule == "acc":
                    reject = False
                else:
                    S_rids = sorted(store.rid[i] for p in used for i in recs[k][p])
                    tree = EV.SortedMerkle(S_rids)
                    member = any(not EV.verify_non_membership(tree.root, tree.non_membership(x)) for x in q)
                    reject = member or (rule == "strict" and sigma < version_at(acc_t))
                if not reject:
                    break
                out["rej_correct" if bad else "rej_wrong"] += 1
                # 재계산: 수락 시점의 정책으로 다시 선택하고 학습한다(집계는 그만큼 기다린다)
                excl = withdrawn_at(k, acc_t)
                used = [p for p in pids_k[k] if p not in excl]
                sigma = version_at(acc_t)
                n = sum(len(recs[k][p]) for p in used)
                out["recompute_updates"] += 1
                out["recompute_steps"] += math.ceil(n / BS)
                out["extra_wait_s"] += d_k[k]
                acc_t += d_k[k]
            bad = withdrawn_at(k, acc_t) & set(used)
            out["accepted"] += 1
            out["viol_acc"] += len(bad)                                    # 수락 시점에 효력이 있던 철회자의 기록 사용(정보주체 단위)
            out["viol_sel"] += len(withdrawn_at(k, t_sel) & set(used))     # 선택 시점에 이미 효력이 있던 철회자의 기록 사용
            round_acc[r] = max(round_acc[r], acc_t)
            for p in used:
                last_inclusion[p] = r
    # 효력 반영 지연: 효력 시각부터, 그 정보주체의 기록을 쓴 마지막 수락 갱신 이후 첫 수락 시점까지
    # 효력 반영 시각 = 효력 이후, 그 정보주체의 기록이 들어간 갱신이 더는 수락되지 않게 된 첫 집계 시각
    lat = []
    for w in W:
        r_last = last_inclusion.get(w["pid"], 0)
        cands = [r for r in range(1, R + 1) if round_acc[r] >= w["t_eff"] and r > r_last]
        if cands:
            lat.append(round_acc[cands[0]] - w["t_eff"])
    out["latency_s"] = {"mean": float(np.mean(lat)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None,
                        "n": len(lat)}
    out["withdrawals"] = len(W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data")
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    data = T.Data(a.cache)
    store = Store(data)
    conds = [("during", {}), ("aftersend", {}), ("aggdelay", {"agg_delay": 0.0}), ("aggdelay", {"agg_delay": 0.5 * L}),
             ("aggdelay", {"agg_delay": 2.0 * L}),
             ("version", {"sync_q": 0.3}), ("offline", {"offline_m": 3})]
    res = []
    for scen, kw in conds:
        for rule in ("sel", "acc", "strict"):
            for s in range(1, a.seeds + 1):
                o = simulate(data, store, "uniform" if scen == "aggdelay" else scen, rule, s, **kw)
                res.append({"scenario": scen, "params": kw, "rule": rule, "seed": s, **o})
            sub = [x for x in res if x["scenario"] == scen and x["params"] == kw and x["rule"] == rule]
            print(scen, kw, rule, {k: sum(x[k] for x in sub) for k in ("updates", "viol_acc", "viol_sel", "rej_correct",
                                                                          "rej_wrong", "recompute_steps")},
                  "latency", round(float(np.mean([x["latency_s"]["mean"] for x in sub])), 1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"L": L, "A_FRAC": A_FRAC, "runs": res}, open(os.path.join(a.out, "E1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
