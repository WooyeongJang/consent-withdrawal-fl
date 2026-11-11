#!/usr/bin/env bash
# 고정된 solc(공식 ARM64 빌드)를 tools/bin/solc로 받고 SHA256을 검증한다.
# 버전과 체크섬은 tools/toolchain.json과 같아야 한다.
set -euo pipefail
cd "$(dirname "$0")"
VERSION=0.8.38
FILE=solc-linux-arm64-v0.8.38+commit.1835d247
SHA256=7a986c8f85945fec
mkdir -p bin
curl -sSf -o bin/solc.tmp "https://binaries.soliditylang.org/linux-arm64/${FILE}"
FULL=$(curl -sSf https://binaries.soliditylang.org/linux-arm64/list.json | python3 -c "import sys,json; print([b['sha256'] for b in json.load(sys.stdin)['builds'] if b['path']=='${FILE}'][0][2:])")
case "$FULL" in ${SHA256}*) ;; *) echo "목록의 체크섬이 고정값과 다름" >&2; exit 1;; esac
echo "${FULL}  bin/solc.tmp" | sha256sum -c --quiet
chmod +x bin/solc.tmp && mv bin/solc.tmp bin/solc
bin/solc --version | tail -1
