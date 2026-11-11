"""비교 감사 구조의 기록기(3장 8.3절).

B2: 운영자가 서명한 중앙 감사 로그(해시 체인). 사후 수정은 체인 검증으로 탐지된다.
B5: B2 + 외부 증인 공동 서명. 증인은 받은 항목마다 자기 키로 서명해 사본을 보관한다.
    운영자가 참여자마다 다른 로그를 보여 주거나 사후 수정하면 증인 사본과의 대조로 드러난다.
서명은 HMAC-SHA256으로 모사한다(키는 실행마다 새로 만든다). 서명 체계의 안전성은 비교 대상이 아니다.
"""

import hashlib
import hmac
import json
import os


class SignedLog:
    def __init__(self, path, key=None, witness_key=None):
        self.path, self.key, self.wkey = path, key or os.urandom(32), witness_key
        self.entries, self.witness_copy = [], []
        self.head = "0" * 64

    def append(self, kind, **payload):
        body = json.dumps({"kind": kind, **payload, "prev": self.head}, sort_keys=True)
        h = hashlib.sha256(body.encode()).hexdigest()
        sig = hmac.new(self.key, h.encode(), "sha256").hexdigest()
        e = {"body": body, "hash": h, "sig": sig}
        if self.wkey is not None:
            e["witness_sig"] = hmac.new(self.wkey, h.encode(), "sha256").hexdigest()
            self.witness_copy.append(dict(e))
        self.entries.append(e)
        self.head = h
        return e

    def verify(self, entries=None):
        prev = "0" * 64
        for e in entries or self.entries:
            if json.loads(e["body"])["prev"] != prev or hashlib.sha256(e["body"].encode()).hexdigest() != e["hash"]:
                return False
            if not hmac.compare_digest(e["sig"], hmac.new(self.key, e["hash"].encode(), "sha256").hexdigest()):
                return False
            prev = e["hash"]
        return True

    def matches_witness(self, entries):
        # 제시된 로그의 본문에서 해시를 다시 계산해 증인이 서명·보관한 해시와 비교한다
        return self.wkey is not None and \
            [hashlib.sha256(e["body"].encode()).hexdigest() for e in entries] == [e["hash"] for e in self.witness_copy]

    def save(self):
        with open(self.path, "w") as f:
            for e in self.entries:
                f.write(json.dumps(e) + "\n")
