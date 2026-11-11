"""컨트랙트 아티팩트 생성: contracts/*.sol을 tools/bin/solc(0.8.38, prague, optimize)로 컴파일해 원장 서비스가 읽는
<출력 디렉터리>/<이름>.json({abi, bin, solc, evm, optimize[, version]})으로 쓴다. 컴파일 옵션은 tests/test_contract.py와 같다.

  python3 scripts/build_artifacts.py <출력 디렉터리> [--check <기존 아티팩트 디렉터리>]

--check를 주면 새로 만든 bin이 기존 아티팩트(실험에 쓴 것)와 바이트 단위로 같은지 확인한다.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = {"ConsentLedger": "v3.1 (training config commit before rounds/withdrawals; (method, prog) approval, 2026-10-27)",
             "CompletionRecord": None}


def compile_one(name):
    solc = str(ROOT / "tools/bin/solc")
    out = subprocess.run([solc, "--evm-version", "prague", "--optimize", "--combined-json", "abi,bin",
                          str(ROOT / f"contracts/{name}.sol")], check=True, capture_output=True, text=True).stdout
    c = next(v for k, v in json.loads(out)["contracts"].items() if k.endswith(f":{name}"))
    ver = subprocess.run([solc, "--version"], check=True, capture_output=True, text=True).stdout.split("Version: ")[1].strip()
    art = {"abi": c["abi"] if isinstance(c["abi"], list) else json.loads(c["abi"]), "bin": c["bin"],
           "solc": ver, "evm": "prague", "optimize": True}
    if CONTRACTS[name]:
        art["version"] = CONTRACTS[name]
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--check", help="실험에 쓴 아티팩트 디렉터리(bin 바이트 비교)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    for name in CONTRACTS:
        art = compile_one(name)
        json.dump(art, open(os.path.join(a.out, f"{name}.json"), "w"))
        msg = f"{name}: {len(art['bin']) // 2} bytes"
        if a.check:
            old = json.load(open(os.path.join(a.check, f"{name}.json")))
            same = old["bin"].removeprefix("0x") == art["bin"] and old["abi"] == art["abi"]   # 옛 아티팩트 일부는 bin에 0x 접두사가 있다
            bad += not same
            msg += ", identical to experiment artifact" if same else ", DIFFERS from experiment artifact"
        print(msg)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
