"""PTB-XL site(기관) 분할 점검과 배정표 생성 (3장 v1.6 8.1절 "기관 분할").

규칙
  - 환자 -> site: 기록이 가장 많은 site, 동률이면 번호가 작은 site. 그 환자의 모든 기록이 그 site에 속한다.
    site 값이 없는 기록도 같은 환자의 다른 기록으로 site가 정해지면 그 site를 따른다.
  - site -> 기관: site를 (학습 환자 수 내림차순, site 번호 오름차순)으로 정렬하고, 그때까지 배정된
    학습 환자 수가 가장 적은 기관(동률이면 번호가 작은 기관)에 차례로 배정한다.
    학습 환자는 strat_fold 1-8의 환자다.
  - 학습 환자가 없는 site는 기관에 배정하지 않고 institution을 "unassigned"로 표시한다.
  - 환자의 어느 기록에도 site가 없으면 그 환자의 학습 기록(strat_fold 1-8)을 제외하고
    excluded_records.csv에 남긴다. 검증·평가 기록(strat_fold 9-10)은 유지한다.
  - K마다 기관별 학습 집합의 환자 수, 기록 수, 다섯 진단 상위 범주별 양성 기록 수를 산출하고
    모든 기관에 다섯 범주가 모두 포함되는지 판정한다.
"""

import argparse
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
TRAIN_FOLDS = range(1, 9)
EVAL_FOLDS = (9, 10)
UNASSIGNED = "unassigned"


def load(root):
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    db["patient_id"] = db["patient_id"].astype(int)
    db["site"] = db["site"].astype("Int64")

    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp.loc[scp["diagnostic"] == 1, "diagnostic_class"].to_dict()
    # Wagner et al. 예제 코드와 같이 likelihood와 무관하게 진단 코드의 상위 범주를 모은다.
    db["superclass"] = db["scp_codes"].apply(
        lambda codes: sorted({diag[c] for c in codes if c in diag})
    )
    for c in SUPERCLASSES:
        db[c] = db["superclass"].apply(lambda s, c=c: c in s).astype(int)
    db["NONE"] = (db["superclass"].str.len() == 0).astype(int)
    db["train"] = db["strat_fold"].isin(TRAIN_FOLDS)
    db["eval"] = db["strat_fold"].isin(EVAL_FOLDS).astype(int)
    return db


def js_distance(p, q):
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(np.sqrt((kl(p, m) + kl(q, m)) / 2))


def site_stats(db, overall):
    """원래 site 기준 기록·환자 수, 평가 fold 비율, 상위 범주 유병률, 전체 대비 JS 거리."""
    cols = SUPERCLASSES + ["NONE"]
    g = db.dropna(subset=["site"]).groupby("site")
    st = pd.DataFrame({
        "records": g.size(),
        "patients": g["patient_id"].nunique(),
        "eval_ratio": g["eval"].mean().round(4),
    }).join(g[cols].mean().round(4))
    st["js_vs_all"] = [round(js_distance(g[cols].sum().loc[k].values, overall), 4) for k in st.index]
    st.index = st.index.astype(int)
    return st.sort_values(["records", "patients"], ascending=False)


def assign_patients(db):
    """환자 -> site. 기록이 가장 많은 site, 동률이면 번호가 작은 site."""
    agg = (
        db.dropna(subset=["site"])
        .groupby(["patient_id", "site"]).size().rename("n").reset_index()
        .sort_values(["patient_id", "n", "site"], ascending=[True, False, True])
    )
    chosen = agg.drop_duplicates("patient_id").set_index("patient_id")["site"].astype(int)
    n_sites = agg.groupby("patient_id")["site"].nunique()
    return chosen, n_sites


def group_sites(site_patients, k):
    """site -> 기관(0..k-1). 학습 환자 수 내림차순(동률 site 번호 오름차순)으로, 학습 환자 수가 가장 적은 기관에."""
    order = sorted(site_patients.items(), key=lambda x: (-x[1], x[0]))
    load_ = [0] * k
    mapping = {}
    for site, n in order:
        inst = min(range(k), key=lambda i: (load_[i], i))
        mapping[site] = inst
        load_[inst] += n
    return mapping


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--out", type=Path, default=Path("site_check"))
    ap.add_argument("--top", type=int, default=15, help="summary에 표로 넣을 상위 site 수")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    db = load(args.root.expanduser())
    cols = SUPERCLASSES + ["NONE"]
    overall = db[cols].sum().values
    lines = []
    w = lines.append

    w("# PTB-XL 기관 분할 점검 (3장 v1.6 8.1절)")
    w(f"root: {args.root}")
    w(f"기록 {len(db)}건, 환자 {db['patient_id'].nunique()}명, "
      f"학습(fold 1-8) {int(db['train'].sum())}건, 검증·평가(fold 9-10) {int(db['eval'].sum())}건")
    w("")

    # 1. site 분포
    st = site_stats(db, overall)
    st.to_csv(args.out / "site_stats.csv")
    w("## 1. site 수와 site별 기록·환자 수 (원래 site 기준)")
    w(f"site 수 {len(st)}, site 결측 기록 {int(db['site'].isna().sum())}건")
    for t in (1, 10, 50, 100, 500, 1000):
        w(f"  기록 ≥{t:>4}건 site: {(st['records'] >= t).sum()}")
    w(f"  상위 3개 site 기록 점유율 {st['records'].iloc[:3].sum() / st['records'].sum():.4f}")
    w("")
    w(f"### 기록 수 상위 {args.top}개 site")
    w(st.head(args.top).to_string())
    w(f"JS 거리(전체 대비) 기록 가중 평균 {np.average(st['js_vs_all'], weights=st['records']):.4f}, "
      f"최대 {st['js_vs_all'].max():.4f}")
    w("")

    # 2. 환자 -> site 배정
    chosen, n_sites = assign_patients(db)
    db["assigned_site"] = db["patient_id"].map(chosen).astype("Int64")
    multi = n_sites[n_sites > 1]
    multi_df = (
        db[db["patient_id"].isin(multi.index)]
        .groupby(["patient_id", "site"], dropna=False).size().rename("records").reset_index()
    )
    multi_df["assigned_site"] = multi_df["patient_id"].map(chosen)
    multi_df.to_csv(args.out / "multi_site_patients.csv", index=False)
    sited = db.dropna(subset=["site"])
    moved = int((sited["site"] != sited["assigned_site"]).sum())
    w("## 2. 여러 site에 기록이 있는 환자")
    w(f"환자 {len(multi)}명, 관련 기록 {int(multi_df['records'].sum())}건. "
      f"배정 규칙(기록 최다 site, 동률이면 작은 site 번호)으로 원래 site와 다른 site에 속하게 된 기록 {moved}건")
    w("")

    # 3. site 결측 기록
    missing = db["site"].isna()
    known = db["assigned_site"].notna()
    inherited = db[missing & known]
    excluded = db.loc[missing & ~known & db["train"], ["patient_id", "strat_fold"]].sort_index()
    excluded["reason"] = "site_missing_all_records"
    excluded.to_csv(args.out / "excluded_records.csv")
    w("## 3. site 결측 기록")
    w(f"site 결측 {int(missing.sum())}건 중 같은 환자의 다른 기록으로 site가 정해진 기록 {len(inherited)}건 → 그 site를 따름")
    for i, r in inherited.iterrows():
        w(f"  ecg_id {i} (patient {r['patient_id']}, fold {r['strat_fold']}) → site {r['assigned_site']}")
    w(f"환자의 어느 기록에도 site가 없는 학습 기록(fold 1-8) {len(excluded)}건 → 학습 집합에서 제외 (excluded_records.csv)")
    w(excluded.to_string())
    kept = db[missing & ~known & ~db["train"]]
    w(f"환자의 어느 기록에도 site가 없는 검증·평가 기록(fold 9-10) {len(kept)}건 → 유지(기관 없음, patient_map에 없음): "
      f"ecg_id {', '.join(map(str, kept.index))}")
    w("")

    # 4. K개 기관
    # 학습 집합 = fold 1-8에서 제외 기록을 뺀 것. 기관은 환자의 배정 site로 정한다.
    train = db[db["train"] & known]
    site_patients = chosen[chosen.index.isin(train["patient_id"])].value_counts().to_dict()
    unassigned = sorted(set(chosen) - set(site_patients))
    w("## 미배정 site (학습 환자 없음 → 기관에 배정하지 않고 institution = unassigned)")
    for s in unassigned:
        sub = db[db["assigned_site"] == s]
        w(f"  site {s}: 환자 {sub['patient_id'].nunique()}명, 기록 {len(sub)}건, "
          f"fold {sorted(sub['strat_fold'].unique().tolist())}")
    w("")
    hashes = [f"{sha256(args.out / 'excluded_records.csv')}  excluded_records.csv"]
    verdicts = []
    w("## 4. K개 기관: 학습 집합(fold 1-8, 제외 기록 뺌)의 기관별 환자·기록 수와 범주별 양성 기록 수")
    for k in args.k:
        if k > len(site_patients):
            w(f"K={k}: 학습 환자가 있는 site {len(site_patients)}개보다 많아 건너뜀")
            continue
        mapping = group_sites(site_patients, k)
        full = {s: mapping.get(s, UNASSIGNED) for s in sorted(set(chosen))}
        inst_map = pd.Series(full, name="institution").rename_axis("site").reset_index()
        pat_map = chosen.rename("site").rename_axis("patient_id").sort_index().reset_index()
        pat_map.insert(1, "strat_fold", pat_map["patient_id"].map(db.groupby("patient_id")["strat_fold"].first()))
        pat_map["institution"] = pat_map["site"].map(full)
        p1 = args.out / f"institution_map_K{k}.csv"
        p2 = args.out / f"patient_map_K{k}.csv"
        inst_map.to_csv(p1, index=False)
        pat_map.to_csv(p2, index=False)
        hashes += [f"{sha256(p1)}  {p1.name}", f"{sha256(p2)}  {p2.name}"]

        t = train.assign(institution=train["assigned_site"].astype(int).map(mapping))
        g = t.groupby("institution")
        tab = pd.DataFrame({
            "n_sites": pd.Series(mapping).value_counts(),
            "train_patients": g["patient_id"].nunique(),
            "train_records": g.size(),
        }).join(g[SUPERCLASSES].sum()).sort_index()
        body = tab.copy()
        tab.loc["합계"] = tab.sum()
        tab.to_csv(args.out / f"institution_train_K{k}.csv")
        ok = bool((body[SUPERCLASSES] > 0).all().all())
        ratio = body["train_patients"].max() / body["train_patients"].min()
        verdicts.append((k, ok, int(body[SUPERCLASSES].min().min()), ratio))
        w(f"### K={k}")
        w(tab.to_string())
        w(f"최대/최소 기관 학습 환자 비 {ratio:.2f}")
        w(f"모든 기관의 학습 집합에 다섯 범주가 모두 포함: {'예' if ok else '아니오'} "
          f"(기관·범주별 최소 양성 기록 수 {verdicts[-1][2]})")
        w("")

    w("## 5. K별 판정")
    for k, ok, mn, ratio in verdicts:
        w(f"K={k}: {'충족' if ok else '불충족'} (최소 양성 기록 수 {mn}, 최대/최소 학습 환자 비 {ratio:.2f})")
    w("")

    (args.out / "hashes.txt").write_text("\n".join(hashes) + "\n")
    w("## 해시 (SHA-256)")
    lines += hashes
    text = "\n".join(lines) + "\n"
    (args.out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
