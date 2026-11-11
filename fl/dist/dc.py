"""데이터센터 컨테이너: 자기 기관의 학습 기록 조각(/shard, 읽기 전용)으로 로컬 학습과 입력 커밋을 한다.

  DC_K=0 python -m fl.dist.dc --shard /shard --store /store --port 9000
받은 전역 모델(global_r{r-1}.pt)과 라운드 커밋(commit_r{r}.json)을 /store에 남긴다(감사자가 읽기 전용으로 본다).
"""

import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

from fl import evidence as EV
from fl import train as T
from fl.dist.common import bytes_state, pack, state_bytes, unpack
from fl.run import INIT_SEED, Policy, prog_id, state_cpu


class ShardData:
    """fl.train.Data와 같은 접근 방식(X, Y, pid, ecg, train_idx)을 기관 하나의 학습 기록 조각에 제공한다."""

    def __init__(self, shard, k):
        X = np.load(os.path.join(shard, "X.npy"), mmap_mode="r")
        sub = T.lead_subset()
        self.X = np.ascontiguousarray(X[:, sub, :]) if sub is not None else X
        with open(os.path.join(shard, "meta.csv")) as f:
            rows = list(csv.DictReader(f))
        self.ecg = np.array([int(r["ecg_id"]) for r in rows])
        self.pid = np.array([int(r["patient_id"]) for r in rows])
        self.Y = np.array([[float(r[c]) for c in T.CATS] for r in rows], dtype=np.float32)
        self.train_idx = {k: np.arange(len(rows))}
        self.mean = self.std = None


class ShardStore:
    """fl.run.Store와 같은 rid 규칙(정보주체별 학습 기록의 ecg_id 오름차순 순번)."""

    def __init__(self, data, k):
        self.data, self.k = data, k
        by_pid = {}
        for i in data.train_idx[k]:
            by_pid.setdefault(int(data.pid[i]), []).append(int(i))
        self.rid, self.by_pid = {}, {}
        for p, idxs in by_pid.items():
            m = EV.rids_for_patient(p, [int(data.ecg[i]) for i in idxs])
            for i in idxs:
                self.rid[i] = m[int(data.ecg[i])]
            self.by_pid[p] = sorted(m.values())

    def input_set(self, policy, r):
        return np.array([i for i in self.data.train_idx[self.k] if policy.valid(int(self.data.pid[i]), r)], dtype=np.int64)

    def rids(self, idx):
        return [self.rid[int(i)] for i in idx]


def policy_from(d):
    ints = lambda m: {int(p): int(v) for p, v in (m or {}).items()}
    return Policy(ints(d.get("enroll")), ints(d.get("withdraw")), d.get("version", 1), ints(d.get("reconsent")))


class DC:
    def __init__(self, k, shard, store):
        self.k, self.store_dir = k, store
        self.device, self.max_pool = T.configure("gpu_det")
        self.data = ShardData(shard, k)
        self.store = ShardStore(self.data, k)
        self.prog = prog_id("gpu_det")
        self.trees = {}
        os.makedirs(store, exist_ok=True)

    def registry(self, meta, blob):
        """정보주체별 레코드 식별자(정책 등록 때 원장에 남기는 rid 약속)."""
        return {"k": self.k, "subjects": {str(p): r for p, r in self.store.by_pid.items()}}, b""

    def norm(self, meta, blob):
        """표준화 통계용 합(학습 개시 시점에 유효한 기록). 기관별 합·제곱합·원소 수만 보낸다."""
        S = self.store.input_set(policy_from(meta["policy"]), meta.get("r", 1))
        x = np.asarray(self.data.X[np.sort(S)], dtype=np.float64)
        return {"k": self.k, "sum": float(x.sum()), "sumsq": float((x ** 2).sum()), "size": int(x.size), "n": int(len(S)),
                "scope_root": EV.SortedMerkle(self.store.rids(S)).root}, b""

    def round(self, meta, blob):
        r, policy = meta["r"], policy_from(meta["policy"])
        start = bytes_state(blob)
        torch.save(start, os.path.join(self.store_dir, f"global_r{r - 1}.pt"))
        S = self.store.input_set(policy, r)
        if len(S) == 0:
            return {"k": self.k, "r": r, "skipped": True}, b""
        tree = EV.SortedMerkle(self.store.rids(S))
        self.trees[r] = tree
        C = {"k": self.k, "r": r, "root_S": tree.root, "n": len(S), "sigma": policy.version, "purpose": meta["purpose"]}
        m = T.new_model(self.max_pool, self.device, INIT_SEED)
        m.load_state_dict(start)
        h_start = T.model_hash(m)[0]
        m, _ = T.local_train(m, self.data, self.k, r, meta["base_seed"], self.device, meta["rounds"], idx=S,
                             norm=tuple(meta["norm"]))
        h_delta = T.model_hash(m)[0]
        U = {"k": self.k, "r": r, "h_delta": h_delta, "h_start": h_start, "h_C": EV.commit_hash(C)}
        B = EV.binding_value(self.k, r, h_start, tree.root, policy.version, meta["purpose"], self.prog, h_delta)
        out = {"k": self.k, "r": r, "C": C, "U": U, "B": B}
        with open(os.path.join(self.store_dir, f"commit_r{r}.json"), "w") as f:
            json.dump({**out, "policy": meta["policy"], "base_seed": meta["base_seed"], "rounds": meta["rounds"],
                       "norm": meta["norm"]}, f)
        return out, state_bytes(state_cpu(m))

    def proofs(self, meta, blob):
        tree = self.trees[meta["r"]]
        return {"proofs": {q: tree.non_membership(q) for q in meta["rids"]}}, b""


def serve(dc, port):
    routes = {"/registry": dc.registry, "/norm": dc.norm, "/round": dc.round, "/proofs": dc.proofs}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404); self.end_headers()

        def do_POST(self):
            meta, blob = unpack(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                body = pack(*routes[self.path](meta, blob))
                self.send_response(200)
            except Exception as e:                                  # 오류도 본문으로 돌려준다
                body = pack({"error": repr(e)})
                self.send_response(500)
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

    HTTPServer(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="/shard")
    ap.add_argument("--store", default="/store")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    k = os.environ["DC_K"]
    dc = DC(k, a.shard, a.store)
    print(f"dc{k} ready: records {len(dc.data.ecg)}", flush=True)
    serve(dc, a.port)


if __name__ == "__main__":
    main()
