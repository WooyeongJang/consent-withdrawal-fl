"""L3 t 분위수 재분석(2026-10-31): 저장된 E2 결과로 시나리오·방법별 L3 분류를 z(주 규칙)와 t(자유도 = 참조 수 - 1)로 비교한다. 새 학습 없음.

분류: pass / exc(기준 초과) / over(과잉 제거) / impr(정밀도 부족) / power(근사 방법의 통과가 양성 대조 실패로 판정 불가).
.venv/bin/python scripts/l3_t_breakdown.py results/final/L3_t_breakdown.json [results/final/e2]
"""
import glob, json, math, statistics, sys, os
from collections import defaultdict, Counter
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fl.consent import SPECS
APPROX=["U1","U2","U3","U4","U4s"]
def cls(auc,ci,refs,q,delta=0.05):
    se_t=(ci[1]-ci[0])/(2*1.959963984540054); mu,sd=statistics.mean(refs),statistics.stdev(refs)
    se=math.sqrt(se_t**2+sd**2/len(refs)); k=stats.norm.ppf(0.95) if q=="z" else stats.t.ppf(0.95,len(refs)-1)
    d=auc-mu; lo,hi=d-k*se,d+k*se
    if lo>-delta and hi<delta: return "pass"
    if lo>0: return "exc"
    return "over" if hi<0 else "impr"
R={q:defaultdict(Counter) for q in "zt"}
E2DIR=sys.argv[2] if len(sys.argv)>2 else os.path.join(os.path.dirname(__file__), "..", "results/final/e2")
for f in sorted(glob.glob(os.path.join(E2DIR, "*_s*.json"))):
    run=json.load(open(f)); refs=run["u0_ref_lira_auc"]; ms=run["methods"]; sc=run["scenario"]
    for q in "zt":
        p1=ms["P1"]["lira_offline"]; c1=cls(p1["auc"],p1["ci"],refs,q); R[q][(sc,"P1")][c1]+=1
        for m in APPROX+["U0"]:
            x=ms.get(m)
            if not(x and "lira_offline" in x): continue
            lo=x["lira_offline"]; rr=refs
            if m=="U0" and lo["auc"] in refs:
                j=refs.index(lo["auc"]); rr=refs[:j]+refs[j+1:]
            c=cls(lo["auc"],lo["ci"],rr,q)
            exact=m=="U0" or (m=="U1" and "latejoin" in sc)   # summarize_e2.py와 같은 정확 방법 판정
            if not exact and c=="pass" and c1!="exc": c="power"
            R[q][(sc,m)][c]+=1
json.dump({q:{f"{k[0]}|{k[1]}":dict(v) for k,v in R[q].items()} for q in R},open(sys.argv[1],"w"),indent=1)
