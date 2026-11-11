"""주체 사이 통신(표준 라이브러리 HTTP)과 직렬화. 전송량은 실제 HTTP 본문 바이트로 센다."""

import io
import json
import os
import struct
import threading
import time
import urllib.request

import torch


def pack(meta, blob=b""):
    """본문 = [메타 JSON 길이(8바이트)] + 메타 JSON + 이진 덩어리(모델 상태 등)."""
    m = json.dumps(meta, sort_keys=True).encode()
    return struct.pack(">Q", len(m)) + m + blob


def unpack(body):
    n = struct.unpack(">Q", body[:8])[0]
    return json.loads(body[8:8 + n]), body[8 + n:]


def state_bytes(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def bytes_state(b):
    return torch.load(io.BytesIO(b), weights_only=True)


class Link:
    """한 상대(데이터센터)와의 연결. 보낸·받은 본문 바이트와 시간을 기록한다.
    delay_s: 인위적 단방향 지연(요청·응답 각각), rate_bps: 인위적 대역폭(0이면 제한 없음) — 8.2 지연 민감도 분석용."""

    def __init__(self, url, delay_s=0.0, rate_bps=0.0):
        self.url, self.delay_s, self.rate_bps = url, delay_s, rate_bps
        self.sent = self.recv = 0
        self.calls = []
        self.lock = threading.Lock()

    def _shape(self, nbytes):
        t = self.delay_s + (nbytes * 8 / self.rate_bps if self.rate_bps else 0.0)
        if t > 0:
            time.sleep(t)
        return t

    def post(self, path, meta, blob=b"", timeout=3600):
        body = pack(meta, blob)
        t0 = time.time()
        shaped = self._shape(len(body))
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.read()
        shaped += self._shape(len(out))
        with self.lock:
            self.sent += len(body); self.recv += len(out)
            self.calls.append({"path": path, "sent": len(body), "recv": len(out), "wall_s": time.time() - t0,
                               "shaped_s": shaped})
        return unpack(out)


def wait_ready(url, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"not ready: {url}")


def env_k():
    return os.environ.get("DC_K")
