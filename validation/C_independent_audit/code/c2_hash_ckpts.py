"""C-3·C-4 준비: 저장된 유형 B 체크포인트의 모델 해시와 U0까지의 상대 거리를 다시 계산한다(학습 없음, CPU).

모델 해시의 정의(fl/train.py model_hash를 읽고 옮김): 이름순으로 정렬한 state_dict의 (이름 UTF-8, 텐서 바이트)를
차례로 SHA-256에 넣은 값. 상대 거리(fl/t11.py rel_dist): 부동소수점 텐서 전체의 ‖a − b‖ / ‖b‖.

  docker run --rm --user "$(id -u):$(id -g)" -v ~/data/runs/final_e3b:/runs:ro -v <code>:/code:ro -v <results>:/out \
      nvcr.io/nvidia/pytorch@sha256:3becd0... python /code/c2_hash_ckpts.py
"""
import glob
import hashlib
import json
import os
import sys
import time

import torch

RUNS, OUT = "/runs", "/out"


def load(path):
    return torch.load(path, map_location="cpu")


def mhash(sd):
    h = hashlib.sha256()
    for name, t in sorted(sd.items()):
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rel_dist(a, b):
    num = den = 0.0
    for k in b:
        if b[k].is_floating_point():
            num += float(((a[k].double() - b[k].double()) ** 2).sum())
            den += float((b[k].double() ** 2).sum())
    return (num ** 0.5) / (den ** 0.5)


def main():
    res = {"header": {"datetime": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__, "python": sys.version.split()[0],
                      "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                      "image_digest_env": os.environ.get("IMAGE_DIGEST", "")}, "seeds": {}}
    for s in range(1, 6):
        d = f"{RUNS}/B_s{s}"
        out = {"orig": {}, "final": {}, "file_sha256": {}}
        for r in range(0, 51):
            out["orig"][r] = mhash(load(f"{d}/orig/ckpt_r{r}.pt"))
        u0 = load(f"{d}/U0/ckpt_r50.pt")
        p1 = load(f"{d}/orig/ckpt_r50.pt")
        targets = {"U0": f"{d}/U0/ckpt_r50.pt", "U0_audit": f"{d}/U0_audit/ckpt_r50.pt", "T6": f"{d}/T6/ckpt_r50.pt",
                   "T6_audit": f"{d}/T6_audit/ckpt_r50.pt", "T11b": f"{d}/T11b/ckpt_r50.pt", "T11d": f"{d}/T11d/ckpt_r50.pt",
                   "T11f": f"{d}/T11f/ckpt_r51.pt"}
        for p in sorted(glob.glob(f"{d}/T11c_*")):
            targets[os.path.basename(p)] = f"{p}/ckpt_r50.pt"
        for p in sorted(glob.glob(f"{d}/T11e_c*")):
            targets[os.path.basename(p)] = f"{p}/ckpt_r50.pt"
        for name, path in targets.items():
            if not os.path.exists(path):
                out["final"][name] = {"missing": path}
                continue
            sd = load(path)
            out["final"][name] = {"hash": mhash(sd), "rel_dist_to_U0": rel_dist(sd, u0), "rel_dist_to_P1": rel_dist(sd, p1),
                                  "file": path, "file_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        out["final"]["P1"] = {"hash": out["orig"][50], "rel_dist_to_U0": rel_dist(p1, u0)}
        # U0 사슬의 라운드별 해시(작업 입력 커밋과 events.jsonl 대조용)
        out["U0_rounds"] = {r: mhash(load(f"{d}/U0/ckpt_r{r}.pt")) for r in range(0, 51) if os.path.exists(f"{d}/U0/ckpt_r{r}.pt")}
        out["U0_audit_rounds"] = {r: mhash(load(f"{d}/U0_audit/ckpt_r{r}.pt")) for r in range(0, 51)
                                  if os.path.exists(f"{d}/U0_audit/ckpt_r{r}.pt")}
        res["seeds"][s] = out
        print("seed", s, "done", len(targets), flush=True)
    json.dump(res, open(f"{OUT}/ckpt_hashes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
