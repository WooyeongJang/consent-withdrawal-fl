"""검증 L1(재실행)·L2(입력 집합) (3장 v1.6 알고리즘 3·4).

감사자는 관할 데이터센터 안에서 저장소를 읽기 전용으로 보고, 원장(여기서는 커밋 기록)과 체크포인트를 받는다.
"""

import os

import numpy as np
import torch

from fl import evidence as EV
from fl import train as T
from fl.run import INIT_SEED, PURPOSE


def reexecute(data, store, policy, run_dir, commit, base_seed, device, max_pool, rounds, norm, prog):
    """알고리즘 3: 커밋된 입력·출발 상태로 지정 프로그램을 다시 실행해 업데이트 해시를 비교한다."""
    C, U = commit["C"], commit["U"]
    k, r = C["k"], C["r"]
    S = store.input_set(policy, k, r)                              # 1. 저장소에서 입력 재구성
    if EV.SortedMerkle(store.rids(S)).root != C["root_S"]:
        return {"result": "불일치", "reason": "입력 재구성 실패"}
    start = torch.load(os.path.join(run_dir, f"ckpt_r{r - 1}.pt"))  # 3. 출발 상태
    m = T.new_model(max_pool, device, INIT_SEED)
    m.load_state_dict(start)
    if T.model_hash(m)[0] != U["h_start"]:
        return {"result": "불일치", "reason": "출발 상태 불일치"}
    m, _ = T.local_train(m, data, k, r, base_seed, device, rounds, idx=S, norm=(norm["mean"], norm["std"]))
    h = T.model_hash(m)[0]                                          # 4–5. 재실행 출력 비교
    ok = h == U["h_delta"]
    return {"result": "통과" if ok else "불일치", "reason": "" if ok else "출력 불일치",
            "binding": EV.binding_value(k, r, U["h_start"], C["root_S"], C["sigma"], PURPOSE, prog, U["h_delta"])}


def verify_input_set(store, policy_expected, job, subject_rids_by_k):
    """알고리즘 4 + 정보주체 자기 검증: 처리 작업의 각 라운드 입력 커밋이 기대 입력 집합과 같고,
    철회 레코드가 어느 커밋에도 없는지(비소속 증명) 확인한다."""
    mismatch, member = [], []
    for c in job["commits"]:
        C = c["C"]
        exp = store.input_set(policy_expected, C["k"], C["r"])
        tree = EV.SortedMerkle(store.rids(exp))
        if tree.root != C["root_S"]:
            mismatch.append((C["k"], C["r"]))
        for q in subject_rids_by_k.get(C["k"], []):
            if not EV.verify_non_membership(C["root_S"], tree.non_membership(q)):
                member.append((C["k"], C["r"]))
    return {"result": "통과" if not mismatch and not member else "불일치",
            "units_checked": len(job["commits"]), "root_mismatch": mismatch, "withdrawn_included": member}
