"""재현성 기록(3장 8.6) 생성: 논문에 명시할 커밋 ID, 도구 버전, 데이터·분할·사상표·동의 사건 해시, 컨트랙트 아티팩트와
실험별 Kairos 배포 주소를 results/·site_check/·tools/에서 모아 reports/common/재현성_기록.md로 쓴다.

  .venv/bin/python scripts/repro_manifest.py [reports/common/재현성_기록.md]

최종 재실행 배포의 주요 거래 ID는 reports/common/재현성_기록_주요거래.csv로 쓴다. 데이터 원본은 재배포하지 않으며 버전·경로·해시만 기록한다.
"""

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.expanduser("~/data/ledger")
EXPLORER = "https://kairos.kaiascan.io/address/"
TXS = os.environ.get("REPRO_TXS", "reports/common/재현성_기록_주요거래.csv")   # 공개본: REPRO_TXS=key_transactions.csv


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def final_runs():
    """최종 재실행 결과에서 원장을 쓴 실행(contract 필드가 있는 객체)을 (파일, 실행 이름, 객체)로 낸다. E1 Kairos는 시나리오 목록이다."""
    for f in sorted(glob.glob("results/final/e1_chain/*.json") + glob.glob("results/final/e3/*.json")
                    + glob.glob("results/final/e4/E4_K*.json")):
        r = json.load(open(f))
        for x in (r if isinstance(r, list) else [r]):
            if isinstance(x, dict) and "contract" in x:
                yield f, (f"{x['scenario']} s{x.get('seed', '')}" if "scenario" in x else ""), x


def main(out="reports/common/재현성_기록.md"):
    os.chdir(ROOT)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tag = subprocess.run(["git", "rev-parse", "final-rerun-v1^{commit}"], capture_output=True, text=True).stdout.strip() or "—"
    tc = json.load(open("tools/toolchain.json"))
    L = ["# 재현성 기록 (3장 8.6)", "",
         f"- 생성: `scripts/repro_manifest.py`, 저장소 커밋 `{head}`(생성 시점).",
         f"- 최종 재실행 코드: 태그 `final-rerun-v1` = 커밋 `{tag}`. 결과는 `results/final/`.",
         "- 데이터는 재배포하지 않는다. 버전·출처·해시만 적는다.", "", "## 1. 실행 환경", "",
         "| 항목 | 값 |", "|---|---|"]
    img = tc.get("training_image", {})
    for k in ("image", "repo_digest", "arch", "python", "torch", "cuda", "cudnn"):
        if k in img:
            L.append(f"| 학습 이미지 {k} | `{img[k]}` |")
    for k, v in tc.items():
        if k != "training_image" and not isinstance(v, (dict, list)) and not k.startswith("_"):
            L.append(f"| {k} | `{v}` |")
    L += ["", "## 2. 데이터·분할·사상표", "",
          "- PTB-XL v1.0.3(PhysioNet, S3 미러 `s3://physionet-open`, 배포 SHA256 87,203개 일치)",
          "- Challenge 2021 v1.0.3(PhysioNet, 코호트 ningbo·georgia·chapman_shaoxing·cpsc_2018·cpsc_2018_extra)", "",
          "| 파일 | SHA-256 |", "|---|---|"]
    for line in open("site_check/hashes.txt"):
        if line.strip():
            h, f = line.split()[:2]
            L.append(f"| `site_check/{f}` | `{h}` |")
    for f in sorted(glob.glob("c2021/mapping_superclass_v*.csv")):
        L.append(f"| `{f}` | `{sha(f)}` |")
    L += ["", "### E3-B 보정 입력(예비 실행에서 생성, 최종 재실행에 그대로 사용 — 계획 0-7)", "", "| 파일 | SHA-256 |", "|---|---|"]
    for f in (os.path.expanduser("~/data/runs/lira/shadows_late.npz"), "results/late/late_methods.json", "results/t11/t11_eps.json"):
        if os.path.exists(f):
            L.append(f"| `{f.replace(os.path.expanduser('~'), '~')}` | `{sha(f)}` |")
    L += ["", "## 3. 동의 사건(시나리오)", "", "| 시나리오 | 사건 목록 해시 | 철회 정보주체 | 기록 |", "|---|---|---:|---:|"]
    for f in sorted(glob.glob("results/consent/*.json")):
        c = json.load(open(f))
        L.append(f"| {c['name']} | `{c['hash']}` | {c['summary']['patients']} | {c['summary']['records']} |")
    L += ["", "## 4. 컨트랙트", "", "| 아티팩트 | 판 | SHA-256(아티팩트 JSON) | 배포 바이트코드 SHA-256 |", "|---|---|---|---|"]
    arts = sorted(glob.glob(f"{ART}/artifact_history/*.json")) + sorted(glob.glob(f"{ART}/artifact/*.json"))
    for f in arts:
        a = json.load(open(f))
        L.append(f"| `{os.path.relpath(f, ART)}` | {a.get('version', '—')} | `{sha(f)[:16]}…` | "
                 f"`{hashlib.sha256(bytes.fromhex(a['bin'].removeprefix('0x'))).hexdigest()[:16]}…` |")
    L += ["", f"- 소스: `contracts/ConsentLedger.sol`(`{sha('contracts/ConsentLedger.sol')[:16]}…`), "
          f"`contracts/CompletionRecord.sol`(`{sha('contracts/CompletionRecord.sol')[:16]}…`)",
          "", "## 5. 최종 재실행의 Kairos 배포(체인 ID 1001)", "",
          f"주요 거래 ID(배포, 학습 설정 커밋, 처리 작업 생성·완료, 검증 기록, 배포 전이 등)는 `{TXS}`에 있다.", "",
          "| 결과 | ConsentLedger | B3 CompletionRecord | 주요 거래 |", "|---|---|---|---:|"]
    txs = []
    for f, label, r in final_runs():
        b3 = r.get("b3_contract")
        kt = r.get("key_txs") or []
        txs += [(f, label, t["fn"], t["hash"], t.get("block") or "") for t in kt]
        L.append(f"| `{f}` {label} | [`{r['contract']}`]({EXPLORER}{r['contract']}) | "
                 + (f"[`{b3}`]({EXPLORER}{b3})" if b3 else "—") + f" | {len(kt)} |")
    os.makedirs(os.path.dirname(TXS) or ".", exist_ok=True)
    with open(TXS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["result", "run", "function", "tx_hash", "block"])
        w.writerows(txs)
    L += ["", "### 참고: 예비 실행·시연 배포(논문 수치에 쓰지 않음)", "", "| 결과 파일 | ConsentLedger | B3 CompletionRecord |", "|---|---|---|"]
    for f in sorted(glob.glob("results/e3/*.json") + glob.glob("results/e4*/E4_K*.json") + glob.glob("results/onchain*/*.json")):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(r, dict) or "contract" not in r:
            continue
        b3 = r.get("b3_contract")
        L.append(f"| `{f}` | [`{r['contract']}`]({EXPLORER}{r['contract']}) | " + (f"[`{b3}`]({EXPLORER}{b3})" if b3 else "—") + " |")
    L += ["", "- 역할 계정의 개인 키는 공개하지 않는다(`~/.config/dissertation/kairos/exp/`, 권한 600)."]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out, len(L), "lines")


if __name__ == "__main__":
    main(*sys.argv[1:2])
