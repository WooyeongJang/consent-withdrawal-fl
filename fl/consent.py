"""동의 사건 생성기 (3장 v1.6 8.1 '동의 사건').

변수: 처리 수준(level 1–3), 철회 비율(학습 환자의 1·5·10·20%), 철회 시점(초기·중기·후기 라운드),
철회 분포(무작위, 진단 범주 집중, 연령·성별 집중, 기관 집중), 순차 철회(여러 차례)와 재동의, 기관 탈퇴,
참여 시작 라운드(후기 합류 집단, 40번).
같은 명세와 시드면 같은 사건 목록이 나온다(결정적). 사건 목록의 해시를 결과와 함께 기록한다.

  python -m fl.consent --spec E2-base --cache ~/data/cache/ptbxl_k3 --ptbxl ~/data/physionet.org/files/ptb-xl/1.0.3

시점: R=50 기준 초기 10, 중기 25, 후기 45. 철회 효력 라운드 w부터 그 정보주체의 기록은 입력 집합에서 빠진다.
PTB-XL의 sex는 0=남성, 1=여성이며 age 300은 89세 초과 익명화 값이다(Wagner et al., 2020).
"""

import argparse
import csv
import hashlib
import json
import os

import numpy as np

INSTS = tuple(str(i) for i in range(int(os.environ.get("FL_K", "3"))))   # fl.train.inst_keys와 같음
CATS = ("NORM", "MI", "STTC", "CD", "HYP")
TIMING = {"early": 10, "mid": 25, "late": 45}

# E2 시나리오: 기준(5%, 후기, 무작위, 단일, 처리 수준 3)에서 한 변수씩 바꾼다
SPECS = {
    "E2-base": {"rate": 0.05, "timing": "late", "dist": {"type": "random"}},
    "E2-rate01": {"rate": 0.01, "timing": "late", "dist": {"type": "random"}},
    "E2-rate10": {"rate": 0.10, "timing": "late", "dist": {"type": "random"}},
    "E2-rate20": {"rate": 0.20, "timing": "late", "dist": {"type": "random"}},
    # 비율 × 시점 부분 요인(3장 반영 38번 (b)): 1·10·20% × 초기·중기(5%와 후기는 위 시나리오)
    **{f"E2-rate{int(r * 100):02d}-{t}": {"rate": r, "timing": t, "dist": {"type": "random"}}
       for r in (0.01, 0.10, 0.20) for t in ("early", "mid")},
    "E2-early": {"rate": 0.05, "timing": "early", "dist": {"type": "random"}},
    "E2-mid": {"rate": 0.05, "timing": "mid", "dist": {"type": "random"}},
    "E2-diagMI": {"rate": 0.05, "timing": "late", "dist": {"type": "diag", "cat": "MI"}},
    "E2-F65": {"rate": 0.05, "timing": "late", "dist": {"type": "agesex", "sex": 1, "age_min": 65}},
    "E2-inst2": {"rate": 0.05, "timing": "late", "dist": {"type": "inst", "inst": "2"}},
    "E2-seq3": {"rate": 0.05, "waves": [15, 30, 45], "dist": {"type": "random"}},
    "E2-reconsent": {"rate": 0.05, "timing": "mid", "dist": {"type": "random"}, "reconsent": {"frac": 0.5, "round": 40}},
    "E2-exit2": {"exit_inst": "2", "timing": "late"},
    # 후기 합류 집단(3장 반영 40번 (c)): 학습 환자 20%가 25라운드에 합류하고, 철회자(전체의 5%)는 이 집단에서 나온다.
    # 철회자의 r₀ = 25이므로 P2는 M₂₄로 회귀하고, 표준화 통계(1라운드 입력 집합)에 철회 기록이 없어 U1이 동등 조건을 충족한다.
    "E2-latejoin": {"rate": 0.05, "timing": "late", "dist": {"type": "random"}, "join": {"frac": 0.2, "round": 25}},
    # 기관 이질성(K=5, 8.1): FL_K=5와 ptbxl_k5 캐시로 생성·실행한다
    "E2-K5-base": {"rate": 0.05, "timing": "late", "dist": {"type": "random"}, "K": 5},
    # 보조 데이터 Challenge 2021(8.1, 코호트 = 기관 4곳, 철회 단위 = 기록): FL_K=4와 c2021 캐시로 생성·실행한다
    "C2021-base": {"rate": 0.05, "timing": "late", "dist": {"type": "random"}, "K": 4, "data": "c2021"},
    "C2021-mid": {"rate": 0.05, "timing": "mid", "dist": {"type": "random"}, "K": 4, "data": "c2021"},
}


def demographics(cache, ptbxl):
    """나이·성별 표: 캐시의 meta.csv에 age·sex 열이 있으면(보조 데이터) 그것을, 없으면 PTB-XL 표를 쓴다."""
    meta = os.path.join(cache, "meta.csv")
    with open(meta) as f:
        has = "age" in next(csv.reader(f))
    return meta if has else os.path.join(ptbxl, "ptbxl_database.csv")


def patients(cache, ptbxl):
    """학습 집합(fold 1–8, 사용 가능, 기관 배정) 환자의 속성: 기관, 진단 범주 보유 여부, 첫 기록의 나이·성별."""
    rec = {}
    with open(os.path.join(cache, "meta.csv")) as f:
        for r in csv.DictReader(f):
            if r["has_label"] == "True" and r["excluded"] == "False" and int(r["strat_fold"]) <= 8 and r["institution"] in INSTS:
                p = int(float(r["patient_id"]))
                d = rec.setdefault(p, {"inst": r["institution"], "cats": set(), "ecg": []})
                d["cats"] |= {c for c in CATS if r[c] == "1"}
                d["ecg"].append(int(r["ecg_id"]))
    first = {}
    with open(demographics(cache, ptbxl)) as f:
        for r in csv.DictReader(f):
            p = int(float(r["patient_id"]))
            if p in rec and (p not in first or int(r["ecg_id"]) < first[p][0]):
                first[p] = (int(r["ecg_id"]), float(r["age"]) if r["age"] else None, int(float(r["sex"])) if r["sex"] else -1)
    for p, d in rec.items():
        d["age"], d["sex"] = first[p][1], first[p][2]
    return rec


def eligible(P, dist):
    t = dist["type"]
    if t == "random":
        return sorted(P)
    if t == "diag":
        return sorted(p for p, d in P.items() if dist["cat"] in d["cats"])
    if t == "agesex":
        return sorted(p for p, d in P.items() if d["sex"] == dist["sex"] and d["age"] is not None and d["age"] >= dist["age_min"])
    if t == "inst":
        return sorted(p for p, d in P.items() if d["inst"] == dist["inst"])
    raise ValueError(t)


def generate(name, P, seed=20261026):
    spec = SPECS[name]
    rng = np.random.default_rng([seed, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])
    withdraw, reconsent, enroll = {}, {}, {}
    if "join" in spec:                                       # 합류 집단을 먼저 뽑고 철회자는 그 안에서 뽑는다
        jc = sorted(int(p) for p in rng.choice(sorted(P), int(round(len(P) * spec["join"]["frac"])), replace=False))
        enroll = {p: spec["join"]["round"] for p in jc}
    if "exit_inst" in spec:
        w = TIMING[spec["timing"]]
        withdraw = {p: w for p, d in P.items() if d["inst"] == spec["exit_inst"]}
    else:
        n = int(round(len(P) * spec["rate"]))
        pool = sorted(enroll) if enroll else eligible(P, spec["dist"])
        if len(pool) < n:
            raise ValueError(f"{name}: 대상 {len(pool)}명 < 필요 {n}명")
        cohort = sorted(int(p) for p in rng.choice(pool, n, replace=False))
        waves = spec.get("waves") or [TIMING[spec["timing"]]]
        for i, p in enumerate(rng.permutation(cohort)):          # 차례마다 같은 수
            withdraw[int(p)] = waves[i * len(waves) // n]
        if "reconsent" in spec:
            back = rng.choice(cohort, int(len(cohort) * spec["reconsent"]["frac"]), replace=False)
            reconsent = {int(p): spec["reconsent"]["round"] for p in sorted(back)}
    out = {"name": name, "spec": spec, "seed": seed, "n_train_patients": len(P),
           "withdraw": {str(p): w for p, w in sorted(withdraw.items())},
           "reconsent": {str(p): c for p, c in sorted(reconsent.items())},
           "level": {str(p): 3 for p in sorted(withdraw)},
           "summary": summarize(P, withdraw)}
    keys = ("spec", "seed", "withdraw", "reconsent", "level")
    if enroll:                                               # 합류 라운드가 없는 시나리오의 해시는 그대로
        out["enroll"] = {str(p): e for p, e in sorted(enroll.items())}
        out["summary"]["joined"] = len(enroll)
        keys += ("enroll",)
    out["hash"] = hashlib.sha256(json.dumps({k: out[k] for k in keys}, sort_keys=True).encode()).hexdigest()
    return out


def summarize(P, withdraw):
    c = list(withdraw)
    return {"patients": len(c), "records": sum(len(P[p]["ecg"]) for p in c),
            "by_inst": {k: sum(P[p]["inst"] == k for p in c) for k in INSTS},
            "by_cat": {k: sum(k in P[p]["cats"] for p in c) for k in CATS},
            "female": sum(P[p]["sex"] == 1 for p in c),
            "age65": sum((P[p]["age"] or 0) >= 65 for p in c),
            "waves": {str(w): sum(v == w for v in withdraw.values()) for w in sorted(set(withdraw.values()))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", default=list(SPECS))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--out", default="results/consent")
    a = ap.parse_args()
    P = patients(a.cache, a.ptbxl)
    os.makedirs(a.out, exist_ok=True)
    for name in a.spec:
        s = generate(name, P)
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(s, f, indent=1, ensure_ascii=False)
        print(name, s["hash"][:12], json.dumps(s["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
