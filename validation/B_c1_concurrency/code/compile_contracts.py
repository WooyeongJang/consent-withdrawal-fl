"""컨트랙트를 원본 그대로 컴파일해 산출물과 해시를 기록한다(소스 수정 없음).

  python validation/B_c1_concurrency/code/compile_contracts.py
"""
import hashlib
import json
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "artifacts")
SRC = {"v3.1": ("contracts/ConsentLedger.sol", "ConsentLedger"), "v3.2": ("contracts/ConsentLedger_v3_2.sol", "ConsentLedgerV32")}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    solc = os.path.join(ROOT, "tools/bin/solc")
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "solc_sha256": sha(open(solc, "rb").read()),
            "solc_version": subprocess.run([solc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1],
            "flags": "--evm-version prague --optimize", "contracts": {}}
    for ver, (path, name) in SRC.items():
        out = subprocess.run([solc, "--evm-version", "prague", "--optimize", "--combined-json", "abi,bin,bin-runtime",
                              os.path.join(ROOT, path)], check=True, capture_output=True, text=True).stdout
        c = next(v for k, v in json.loads(out)["contracts"].items() if k.endswith(":" + name))
        abi = c["abi"] if isinstance(c["abi"], list) else json.loads(c["abi"])
        json.dump({"abi": abi, "bin": c["bin"], "bin_runtime": c["bin-runtime"]}, open(os.path.join(OUT, f"{ver}.json"), "w"))
        info["contracts"][ver] = {"source": path, "source_sha256": sha(open(os.path.join(ROOT, path), "rb").read()),
                                  "bin_sha256": sha(bytes.fromhex(c["bin"])), "runtime_sha256": sha(bytes.fromhex(c["bin-runtime"])),
                                  "runtime_bytes": len(c["bin-runtime"]) // 2}
    # 실험(E1–E4)에 쓴 v3.1 산출물과 같은 바이트코드인지 확인
    exp = os.path.expanduser("~/data/ledger/artifact/ConsentLedger.json")
    if os.path.exists(exp):
        e = json.load(open(exp))
        info["contracts"]["v3.1"]["same_bin_as_experiment_artifact"] = e["bin"].removeprefix("0x") == json.load(open(os.path.join(OUT, "v3.1.json")))["bin"]
        info["contracts"]["v3.1"]["experiment_artifact_bin_sha256"] = sha(bytes.fromhex(e["bin"].removeprefix("0x")))
    json.dump(info, open(os.path.join(OUT, "compile_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
