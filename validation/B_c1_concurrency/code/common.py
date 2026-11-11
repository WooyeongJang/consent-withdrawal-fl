"""실험 B 공용 모듈(순수 파이썬): 시행 계획, 입력 집합, 사후 탐지기, 판정.

- 컨트랙트 소스는 고치지 않는다. 정렬 머클 트리와 비소속 증명은 기존 구현(fl/evidence.py)을 읽기 전용으로 쓴다.
- 탐지기(detect)는 원장 이벤트와 (a) 증명자가 내놓는 커밋된 트리, (b) 정보주체가 가진 자기 rid만 입력으로 받는다.
  하네스가 아는 참값(누가 어느 기관 소속인지, 제출·조회 시점)은 받지 않는다.
"""
import hashlib
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from fl.evidence import SortedMerkle, H, rids_for_patient, verify_non_membership  # noqa: E402  (읽기 전용 사용)

SEED = 20261109
FAR = 2 ** 40
LOCAL_CONDITIONS = ("L1", "L2", "L3", "L4", "L5", "L6")
N_PER_CELL = 50


def keccak_text(s):
    from eth_utils import keccak
    return keccak(text=s)


def subject_of(pid):
    """정보주체 가명(bytes32)과 rid 목록. pid는 가상의 환자 번호(실제 데이터가 아니다)."""
    rids = list(rids_for_patient(pid, [1, 2]).values())
    return keccak_text(f"validation-B-subject-{pid}"), rids


def build_set(pid_filler_base, related_rids):
    """데이터센터의 입력 집합 S: 채움 rid 8개 + (같은 기관 철회자이면) 그 정보주체의 rid."""
    filler = [H("rid", "validation-B-filler", pid_filler_base, j) for j in range(8)]
    tree = SortedMerkle(filler + list(related_rids))
    return tree


def root_bytes(tree):
    return bytes.fromhex(tree.root)


def local_plan(seed=SEED, n=N_PER_CELL):
    """로컬 EVM 시행 계획. 무작위 요소는 실행 전에 모두 뽑는다."""
    rng = random.Random(seed)
    plan = []
    tid = 0
    for version in ("v3.1", "v3.2"):
        for relation in ("same", "other"):
            for cond in LOCAL_CONDITIONS:
                for i in range(n):
                    tid += 1
                    p = {"tid": tid, "version": version, "relation": relation, "cond": cond, "k": rng.randrange(3)}
                    if cond == "L2":
                        p["gap"] = rng.choice((1, 2, 3))
                    elif cond == "L3":
                        p["first"] = rng.choice(("W", "D")); p["same_block"] = rng.random() < 0.5
                    elif cond == "L4":
                        p["signed_first"] = rng.choice(("D", "W")); p["same_block"] = rng.random() < 0.5
                    elif cond == "L5":
                        p["first"] = rng.choice(("W", "D"))
                    elif cond == "L6":
                        p["lag"] = rng.choice((1, 2, 5)); p["K"] = rng.choice((1, 2, 3))
                        p["w_blocks"] = sorted(rng.randrange(1, p["lag"] + 1) for _ in range(p["K"]))   # 조회 뒤 몇 번째 블록에 들어가는가
                    plan.append(p)
    return plan


def testnet_plan(version, n=150, seed=SEED):
    """테스트넷 시행 계획: 제출 시각 차 o_D − o_W ~ U(−1.5, 1.5)초, 같은 기관 여부 확률 0.5."""
    rng = random.Random(f"{seed}-{version}")
    return [{"tid": i + 1, "version": version, "relation": "same" if rng.random() < 0.5 else "other",
             "k": rng.randrange(3), "offset_s": rng.uniform(-1.5, 1.5)} for i in range(n)]


# ------------------------------------------------------------------ 사후 탐지기(이벤트만)
def detect(events, trees, subject_rids):
    """events: [{"event", "args", "block", "tx_index", "log_index", "ts"}] (한 컨트랙트의 전체 이벤트)
    원장 순서는 (블록, 거래 색인, 로그 색인)이다. 로그 색인이 블록 단위가 아니라 거래 단위로 매겨지는 노드(eth-tester)가 있어
    거래 색인을 함께 쓴다(시운전에서 확인, DEVIATIONS.md).
    trees: {root_hex: SortedMerkle} — 증명자(데이터센터)가 커밋한 집합. subject_rids: {subject_hex: [rid]} — 정보주체가 가진 것.
    반환: {(k, r): {...}} 수락된 단위마다
      version_recorded, eff_before_pos, version_consistent, viol_pos[subject], viol_time[subject]"""
    req = {}
    eff = []
    inputs = {}
    for e in events:
        a = e["args"]
        if e["event"] == "WithdrawalRequested":
            req[a["reqId"]] = a["subject"]
        elif e["event"] == "RoundInputCommitted":
            inputs[(a["k"], a["r"])] = {"root": a["rootS"], "sigma": a["sigma"]}
    for e in events:
        if e["event"] == "WithdrawalEffective":
            eff.append({"reqId": e["args"]["reqId"], "tEff": e["args"]["tEff"], "pos": (e["block"], e["tx_index"], e["log_index"]),
                        "subject": req[e["args"]["reqId"]]})
    out = {}
    for e in events:
        if e["event"] != "UpdateAccepted":
            continue
        a = e["args"]
        kr = (a["k"], a["r"])
        pos = (e["block"], e["tx_index"], e["log_index"])
        root = inputs[kr]["root"]
        tree = trees.get(root)
        before_pos = [w for w in eff if w["pos"] < pos]
        before_time = [w for w in eff if w["tEff"] <= a["tau"]]

        def members(ws):
            v = []
            for w in ws:
                for rid in subject_rids.get(w["subject"], []):
                    # 정보주체의 검사: 커밋된 root_S에 대해 자기 rid의 비소속 증명을 요구하고 검증한다
                    proof = tree.non_membership(rid) if tree is not None else None
                    if not verify_non_membership(root, proof):
                        v.append(w["subject"]); break
            return v
        out[kr] = {"version_recorded": a["withdrawalVersion"], "eff_before_pos": len(before_pos),
                   "version_consistent": a["withdrawalVersion"] == len(before_pos), "sigma": inputs[kr]["sigma"],
                   "viol_pos": members(before_pos), "viol_time": members(before_time), "tau": a["tau"], "pos": pos}
    return out


def ledger_class(w_pos, d_pos):
    """(블록, 거래 색인) 순서로 나눈다. w_pos가 여러 개면 가장 이른 것 기준."""
    if w_pos is None or d_pos is None:
        return "incomplete"
    if w_pos[0] == d_pos[0]:
        return "W<D same block" if w_pos[1] < d_pos[1] else "D<W same block"
    return "W<D other block" if w_pos[0] < d_pos[0] else "D<W other block"


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
