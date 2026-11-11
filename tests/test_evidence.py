"""fl/evidence.py 단위 테스트(표준 라이브러리만): python3 tests/test_evidence.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fl.evidence import (SortedMerkle, H, binding_value, rids_for_patient, verify_membership,  # noqa: E402
                         verify_non_membership)


def test_merkle():
    rng = random.Random(0)
    for n in (0, 1, 2, 3, 5, 8, 33, 1000):
        items = [H("x", rng.random()) for _ in range(n)]
        t = SortedMerkle(items)
        for x in items[:20]:
            assert verify_membership(t.root, t.membership(x)), "소속 증명"
            assert t.non_membership(x) is None, "소속 원소는 비소속 증명 불가"
        for _ in range(20):
            q = H("q", rng.random())
            p = t.non_membership(q)
            assert verify_non_membership(t.root, p), f"비소속 증명 n={n}"
            assert not verify_non_membership(H("other"), p) or n == 0, "다른 루트에서는 실패"
    # 경계: 가장 작은 값보다 작은 q, 가장 큰 값보다 큰 q
    t = SortedMerkle(["b", "d", "f"])
    assert verify_non_membership(t.root, t.non_membership("a"))
    assert verify_non_membership(t.root, t.non_membership("z"))
    # 위조: 인접하지 않은 두 잎으로 만든 증명은 실패
    p = t.non_membership("c")
    p_bad = dict(p, right={"index": 2, "leaf": "f", "path": t.path(2)})
    assert not verify_non_membership(t.root, p_bad), "인접하지 않은 잎 위조"
    # 위조: 집합에 있는 원소를 없는 척(왼쪽 잎을 바꿔치기)
    t2 = SortedMerkle(["b", "c", "d"])
    forged = {"q": "c", "n": 3, "left": t2.membership("b"), "right": t2.membership("d")}
    assert not verify_non_membership(t2.root, forged), "실제 원소에 대한 비소속 위조"


def test_rid_and_binding():
    a = rids_for_patient(17, [30, 10, 20])
    assert list(a) == [10, 20, 30] and len(set(a.values())) == 3
    assert rids_for_patient(17, [10, 20, 30]) == a, "결정적"
    assert rids_for_patient(18, [10])[10] != a[10], "환자가 다르면 다른 rid"
    b1 = binding_value(0, 1, "hs", "root", 1, "train", "prog", "hd")
    assert b1 == binding_value(0, 1, "hs", "root", 1, "train", "prog", "hd")
    assert b1 != binding_value(0, 1, "hs", "root", 1, "train", "prog2", "hd"), "prog가 다르면 다른 결속값"


if __name__ == "__main__":
    test_merkle(); test_rid_and_binding(); print("evidence tests: OK")
