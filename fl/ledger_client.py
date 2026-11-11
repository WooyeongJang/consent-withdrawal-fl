"""원장 연동 서비스(ledger/service.py) 클라이언트. 학습 컨테이너에서 표준 라이브러리만으로 호출한다."""

import hashlib
import json
import os
import time
import urllib.request

PURPOSE_HASH = None
# 승인된 처리 방법의 종류(ConsentLedger v3, 3장 반영 44번 (a)): 1 정확(L1·L2로 완료), 2 근사(L3 통과 필요).
# 체크포인트 재수행은 동등 조건을 충족하면 "U1-eq"(정확), 아니면 "U1"(근사)로 등록한다. 목록에 없는 이름은 근사로 둔다.
EXACT_METHODS = ("U0", "U1-eq", "U5")


def method_kind(name):
    return 1 if name in EXACT_METHODS else 2


def b32(hexstr):
    """64자리 16진수 해시 → '0x'…(bytes32)."""
    h = hexstr.removeprefix("0x")
    assert len(h) == 64, hexstr
    return "0x" + h


def text32(s):
    return "0x" + hashlib.sha256(s.encode()).hexdigest()


class Ledger:
    def __init__(self, url="http://127.0.0.1:8600"):
        self.url = url
        self.log = []

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def tx(self, role, fn, *args, expect_ok=True):
        t = time.time()
        out = self._post("/tx", {"role": role, "fn": fn, "args": list(args)})
        out.update(role=role, fn=fn, wall_s=time.time() - t)
        self.log.append(out)
        if expect_ok and (out.get("error") or out.get("status") != 1):
            raise RuntimeError(f"ledger tx failed: {fn} {out.get('error') or out}")
        return out

    def batch(self, txs, expect_ok=True):
        """txs: [(role, fn, args), ...]. 서로 다른 계정은 동시에, 같은 계정은 nonce 순서로 연달아 제출."""
        t = time.time()
        out = self._post("/batch", {"txs": [{"role": r, "fn": f, "args": list(a)} for r, f, a in txs]})
        if "results" not in out:
            raise RuntimeError(f"ledger batch failed: {out}")
        wall = time.time() - t
        for x in out["results"]:
            x["wall_s"] = wall
            self.log.append(x)
            if expect_ok and x.get("status") != 1:
                raise RuntimeError(f"ledger tx failed in batch: {x['fn']} {x}")
        return out["results"]

    def sign_subject(self, fn, *args):
        return self._post("/sign_subject", {"fn": fn, "args": list(args)})

    def submit_feepayer(self, raw, label="submit_feepayer"):
        out = self._post("/submit_feepayer", {"raw": raw})
        out.update(role="platform(fee-payer)", fn=label)
        self.log.append(out)
        if out.get("status") != 1:
            raise RuntimeError(f"fee-payer submit failed: {out}")
        return out

    def deploy(self, name="ConsentLedger", label=""):
        # 데이터센터 수 K는 학습 쪽 설정(FL_K, 기본 3)과 같게 둔다
        out = self._post("/deploy", {"name": name, "delta": 0, "label": label, "k": int(os.environ.get("FL_K", "3"))})
        if "address" not in out:
            raise RuntimeError(f"deploy failed: {out}")
        self.log.append({**out, "fn": f"deploy:{name}", "role": "platform", "status": 1})
        return out

    def approve_method(self, name, prog):
        """처리 방법을 (방법, prog) 쌍으로 승인 목록에 올린다(플랫폼, 이 배포에서 처음 쓸 때 한 번, v3.1)."""
        self.approved = getattr(self, "approved", set())
        if (name, prog) not in self.approved:
            self.tx("platform", "approveMethod", text32(name), b32(prog), method_kind(name))
            self.approved.add((name, prog))

    def tx_other(self, name, role, fn, *args):
        out = self._post("/tx_other", {"name": name, "role": role, "fn": fn, "args": list(args)})
        out.update(role=role, fn=f"{name}.{fn}")
        self.log.append(out)
        if out.get("status") != 1:
            raise RuntimeError(f"tx_other failed: {out}")
        return out

    KEY_FNS = ("deploy:ConsentLedger", "deploy:CompletionRecord", "commitTrainingConfig", "approveMethod", "createJob", "completeJob",
               "recordVerification", "deploy", "revoke", "sampleAudit")

    def key_txs(self):
        """논문에 명시할 주요 거래 ID(3장 8.6): 배포, 처리 방법 승인, 처리 작업, 검증 기록, 배포 전이, 감사 표본."""
        return [{"fn": x["fn"], "hash": x.get("hash"), "block": x.get("block")} for x in self.log
                if x.get("fn") in self.KEY_FNS and x.get("hash")]

    def last_block(self):
        return max((x.get("block") or 0) for x in self.log) if self.log else None

    def view(self, fn, *args, at=None):
        """at=None이면 이 클라이언트가 마지막으로 기록한 거래의 블록 기준으로 조회한다(읽기-쓰기 일관성)."""
        blk = at if at is not None else self.last_block()
        out = self._post("/view", {"fn": fn, "args": list(args), "block": blk})
        if "error" in out:
            raise RuntimeError(f"view failed: {fn} {out}")
        return out.get("result")

    def info(self):
        with urllib.request.urlopen(self.url + "/info", timeout=60) as r:
            return json.loads(r.read())
