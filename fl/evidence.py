"""증거 구조 (3장 v1.6 2.2·3.4·3.5·5.2·6.2절).

- 레코드 식별자 rid = H(k_s ∥ j): k_s는 정보주체 단말의 비밀값, j는 제출 순번(2.2절).
  시뮬레이션에서는 k_s = H("ks", 연구 솔트, patient_id), j = 환자 기록의 ecg_id 오름차순 순번으로 만든다.
- 정렬 머클 트리와 비소속 증명(3.5절): 잎은 H("leaf", rid). 비소속 증명은 q의 앞뒤로 인접한 두 잎과 각 경로다.
- 결속값 B(k, r) = H(k ∥ r ∥ h_start ∥ root_S ∥ σ ∥ π ∥ prog ∥ seed_rule ∥ H(Δ)) (5.2절).
모든 해시는 SHA-256이며 구분자를 넣어 도메인을 분리한다.
"""

import hashlib
import json
from bisect import bisect_left

STUDY_SALT = "dissertation-ptbxl-v1.0.3"
SEED_RULE = "seed_{k,r} = int(SHA256('seed|base|k|r')[:16],16) mod 2^63"


def H(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x1f")
    return h.hexdigest()


def subject_secret(patient_id):
    return H("ks", STUDY_SALT, patient_id)


def rids_for_patient(patient_id, ecg_ids):
    """환자의 기록들에 제출 순번 j(ecg_id 오름차순)를 매겨 rid를 만든다. {ecg_id: rid}"""
    ks = subject_secret(patient_id)
    return {e: H("rid", ks, j) for j, e in enumerate(sorted(ecg_ids))}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def commit_hash(obj):
    return H("commit", canonical(obj))


class SortedMerkle:
    def __init__(self, items):
        self.leaves = sorted(set(items))
        lv = [H("leaf", x) for x in self.leaves] or [H("empty")]
        self.levels = [lv]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            self.levels.append([H("node", cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i])
                                for i in range(0, len(cur), 2)])
        self.root = self.levels[-1][0]

    def __len__(self):
        return len(self.leaves)

    def path(self, i):
        out = []
        for lv in self.levels[:-1]:
            j = i ^ 1
            out.append(lv[j] if j < len(lv) else lv[i])
            i //= 2
        return out

    def membership(self, x):
        i = bisect_left(self.leaves, x)
        if i == len(self.leaves) or self.leaves[i] != x:
            return None
        return {"index": i, "leaf": x, "path": self.path(i)}

    def non_membership(self, q):
        """q가 집합에 없으면 비소속 증명, 있으면 None(증명 불가)."""
        i = bisect_left(self.leaves, q)
        if i < len(self.leaves) and self.leaves[i] == q:
            return None
        left = {"index": i - 1, "leaf": self.leaves[i - 1], "path": self.path(i - 1)} if i > 0 else None
        right = {"index": i, "leaf": self.leaves[i], "path": self.path(i)} if i < len(self.leaves) else None
        return {"q": q, "n": len(self.leaves), "left": left, "right": right}


def verify_membership(root, proof):
    if proof is None:
        return False
    h, idx = H("leaf", proof["leaf"]), proof["index"]
    for sib in proof["path"]:
        h = H("node", h, sib) if idx % 2 == 0 else H("node", sib, h)
        idx //= 2
    return h == root


def verify_non_membership(root, proof):
    if proof is None:
        return False
    q, L, R, n = proof["q"], proof["left"], proof["right"], proof["n"]
    if L is None and R is None:
        return n == 0
    if L is not None and not (L["leaf"] < q and verify_membership(root, L)):
        return False
    if R is not None and not (q < R["leaf"] and verify_membership(root, R)):
        return False
    if L is not None and R is not None:
        return R["index"] == L["index"] + 1          # 인접한 두 잎
    if L is None:
        return R["index"] == 0                       # q가 가장 작은 잎보다 작음
    return L["index"] == n - 1                       # q가 가장 큰 잎보다 큼


def binding_value(k, r, h_start, root_s, sigma, purpose, prog, h_delta, seed_rule=SEED_RULE):
    return H("B", k, r, h_start, root_s, sigma, purpose, prog, seed_rule, h_delta)
