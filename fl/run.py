"""정책을 반영한 연합학습과 증거 기록 (3장 v1.6 3.3–3.6, 알고리즘 1·2, 6.2–6.3절).

- 정책: 환자별 참여 시작 라운드 e와 철회 효력 라운드 w. 판정 기준 시점은 집계 수락 시점이며,
  라운드 r의 업데이트는 그 라운드의 수락 시점에 유효한 기록만 써야 한다(I1).
  레코드 x는 e ≤ r 이고 (w 없음 또는 r < w)일 때 라운드 r에서 유효하다.
- 표준화 통계는 학습 시작 시점(라운드 1)에 유효한 기록으로 계산해 고정하고, 그 범위를 커밋한다(8.1).
- 라운드마다 기관별 입력 커밋 C, 업데이트 커밋 U, 결속값 B, 수락 기록, 집계 기록 G와 체크포인트를 남긴다.
  원장 연동 전이므로 이벤트는 JSON Lines로 기록한다(이벤트 이름은 6.3절 스키마를 따른다).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from fl import evidence as EV
from fl import train as T

INIT_SEED = 20261025
PURPOSE = "ptbxl-superdiag-train"
INSTS = T.inst_keys()                    # 환경변수 FL_K(기본 3)


class Policy:
    def __init__(self, enroll=None, withdraw=None, version=1, reconsent=None):
        self.enroll = dict(enroll or {})      # patient_id -> e (기본 1)
        self.withdraw = dict(withdraw or {})  # patient_id -> w
        self.reconsent = dict(reconsent or {})  # patient_id -> c (c > w, 재동의 라운드부터 다시 유효)
        self.version = version

    def valid(self, pid, r):
        e = self.enroll.get(pid, 1)
        w = self.withdraw.get(pid)
        c = self.reconsent.get(pid)
        return e <= r and (w is None or r < w or (c is not None and r >= c))

    def without(self, pids, version_bump=1):
        """처음부터의 재학습용: 지정 환자를 처음부터 없는 것으로 둔 정책."""
        en = dict(self.enroll)
        for p in pids:
            en[p] = 10 ** 9
        return Policy(en, {k: v for k, v in self.withdraw.items() if k not in pids}, self.version + version_bump,
                      {k: v for k, v in self.reconsent.items() if k not in pids})

    def describe(self):
        d = {"version": self.version, "enroll": {str(k): v for k, v in sorted(self.enroll.items()) if v != 1},
             "withdraw": {str(k): v for k, v in sorted(self.withdraw.items())}}
        if self.reconsent:
            d["reconsent"] = {str(k): v for k, v in sorted(self.reconsent.items())}
        return d


class Store:
    """데이터센터 저장소 관점: 학습 집합 기록 위치 → rid."""

    def __init__(self, data):
        self.data = data
        self.rid = {}
        by_pid = {}
        for k in INSTS:
            for i in data.train_idx[k]:
                by_pid.setdefault(int(data.pid[i]), []).append(int(i))
        for p, idxs in by_pid.items():
            m = EV.rids_for_patient(p, [int(data.ecg[i]) for i in idxs])
            for i in idxs:
                self.rid[i] = m[int(data.ecg[i])]

    def input_set(self, policy, k, r):
        return np.array([i for i in self.data.train_idx[k] if policy.valid(int(self.data.pid[i]), r)], dtype=np.int64)

    def rids(self, idx):
        return [self.rid[int(i)] for i in idx]

    def subject_rids(self, pid, k=None):
        insts = INSTS if k is None else (k,)
        return sorted(self.rid[int(i)] for kk in insts for i in self.data.train_idx[kk] if int(self.data.pid[i]) == pid)


def code_hash():
    h = hashlib.sha256()
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def prog_id(cond):
    """결속값의 prog: 학습 프로그램과 실행 환경 식별자(E0 결과: 이미지·장치·설정·스레드가 모두 결과에 영향)."""
    return EV.H("prog", code_hash(), os.environ.get("IMAGE_DIGEST", "unknown"), cond,
                torch.__version__, torch.get_num_threads())


def state_cpu(model):
    return {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}


def run_fl(data, store, policy, base_seed, device, max_pool, cond, rounds, run_dir, label,
           start_state=None, start_round=1, norm=None, log_every=10, accept_check=True, tamper=None,
           retain_every=None, ledger=None, stop_round=None, on_round_start=None):
    """연합학습을 start_round부터 rounds까지 실행한다.
    tamper: {(k, r): 'hidden_withdrawn'} — 시험용 부정행위 주입(커밋에서 빼고 실제 학습에는 철회 기록 사용).
    retain_every: Δt. 지정하면 r ≡ 1 (mod Δt)인 라운드의 기관별 로컬 모델 상태를 보존한다(FedEraser U2용).
    ledger: fl.ledger_client.Ledger. 지정하면 라운드 개시·입력/업데이트 커밋·수락/거절·집계·감사 표본을
            ConsentLedger(Kairos)에 역할 계정 서명 + 플랫폼 대납으로 기록한다(6.3절)."""
    os.makedirs(run_dir, exist_ok=True)
    events = []
    ev = lambda name, **kw: events.append({"event": name, **kw})
    prog = prog_id(cond)

    if norm is None:
        scope = np.concatenate([store.input_set(policy, k, 1) for k in INSTS])
        mean, std = T.norm_stats(data, scope)
        norm = {"mean": mean, "std": std, "scope_root": EV.SortedMerkle(store.rids(scope)).root, "n": int(len(scope))}
        ev("NormStatsCommitted", **norm)
    nm = (norm["mean"], norm["std"])

    g = T.new_model(max_pool, device, INIT_SEED)
    if start_state is not None:
        g.load_state_dict(start_state)
    h_prev = T.model_hash(g)[0]
    ckpts = {start_round - 1: h_prev}
    torch.save(state_cpu(g), os.path.join(run_dir, f"ckpt_r{start_round - 1}.pt"))
    commits, round_hashes = [], {}

    if ledger is not None:
        from fl.ledger_client import b32, text32
        purpose32 = text32(PURPOSE)
    last = stop_round or rounds
    import time as _t
    # 학습 설정 커밋(3장 반영 34번 (a)): 라운드 시드는 커밋된 기준 시드와 시드 규칙에서 결정적으로 도출된다.
    # 원장 연결 학습의 첫 구간(start_round = 1)에서, 첫 라운드와 첫 철회 요청보다 먼저 기록한다(ConsentLedger v3.1).
    config = EV.H("config", base_seed, EV.SEED_RULE, prog, norm.get("scope_root", ""))
    if ledger is not None:
        if start_round == 1:
            ev("TrainingConfigCommitted", config=config, base_seed=base_seed, seed_rule=EV.SEED_RULE, prog=prog)
            ledger.tx("platform", "commitTrainingConfig", b32(config))
        ledger.tx("platform", "openRound", start_round, int(_t.time()) + 3600, b32(h_prev))
    for r in range(start_round, last + 1):
        states, weights, accepted = [], [], []
        phase1, phase2 = [], []
        if on_round_start is not None:          # 예: 이 라운드에 효력을 갖는 철회 요청의 원장 등록
            on_round_start(r)
        for k in INSTS:
            S = store.input_set(policy, k, r)
            # 공격 주입(E3): {"mode": included|stale|hidden, "pids": [...], "collude": bool}
            spec = (tamper or {}).get((k, r))
            if spec == "hidden_withdrawn":                       # 이전 형식 호환
                spec = {"mode": "hidden", "pids": None}
            sigma = policy.version
            committed = used = S
            if spec:
                extra = np.array([i for i in data.train_idx[k] if policy.withdraw.get(int(data.pid[i]), 10 ** 9) <= r
                                  and policy.enroll.get(int(data.pid[i]), 1) <= r
                                  and (spec.get("pids") is None or int(data.pid[i]) in spec["pids"])], dtype=np.int64)
                used = np.sort(np.concatenate([S, extra]))
                if spec["mode"] in ("included", "stale"):
                    committed = used
                if spec["mode"] == "stale":
                    sigma = max(policy.version - 1, 0)           # 철회 반영 전 정책 스냅샷
            if len(used) == 0:                                   # 입력 집합이 빈 기관(예: 기관 탈퇴)은 이 라운드에 참여하지 않는다
                ev("InstitutionSkipped", k=k, r=r, reason="empty_input_set")
                continue
            S_rids = store.rids(committed)
            tree = EV.SortedMerkle(S_rids)
            C = {"k": k, "r": r, "root_S": tree.root, "n": len(committed), "sigma": sigma, "purpose": PURPOSE}
            m = T.new_model(max_pool, device, INIT_SEED)
            m.load_state_dict(g.state_dict())
            m, _ = T.local_train(m, data, k, r, base_seed, device, rounds, idx=used, norm=nm)
            h_delta = T.model_hash(m)[0]
            U = {"k": k, "r": r, "h_delta": h_delta, "h_start": h_prev, "h_C": EV.commit_hash(C)}
            B = EV.binding_value(k, r, h_prev, tree.root, sigma, PURPOSE, prog, h_delta)
            ev("RoundInputCommitted", **C); ev("UpdateCommitted", **U)
            if ledger is not None:
                phase1 += [(f"dc{k}", "commitInput", (int(k), r, b32(tree.root), len(committed), sigma, purpose32)),
                           (f"dc{k}", "commitUpdate", (int(k), r, b32(h_delta), b32(h_prev)))]
            # 알고리즘 2: 수락 시점에 무효인(철회 효력이 발생한) 이 기관 기록의 비소속 확인
            status, reason, invalid = "accepted", "", []
            if accept_check and not (spec and spec.get("collude")):   # T10: 공모한 집계 주체는 검사를 건너뛴다
                invalid = [store.rid[int(i)] for i in data.train_idx[k]
                           if policy.withdraw.get(int(data.pid[i]), 10 ** 9) <= r and not policy.valid(int(data.pid[i]), r)]
                for q in invalid:
                    if not EV.verify_non_membership(tree.root, tree.non_membership(q)):
                        status, reason = "rejected", ("stale_snapshot" if sigma < policy.version else "policy_conflict")
                        break
            ev("UpdateAccepted" if status == "accepted" else "UpdateRejected", k=k, r=r, reason=reason,
               withdrawn_checked=len(invalid))
            if ledger is not None:
                phase2.append(("aggregator", "decide", (int(k), r, status == "accepted", text32(reason or "ok"))))
            commits.append({"C": C, "U": U, "B": B, "status": status, "tampered": used is not S, "attack": spec,
                            "_committed_idx": committed})
            if status == "accepted":
                states.append(state_cpu(m)); weights.append(len(committed)); accepted.append(k)
                if retain_every and (r - 1) % retain_every == 0:
                    torch.save({"state": states[-1], "n": len(committed), "h_start": U["h_start"]},
                               os.path.join(run_dir, f"upd_r{r}_k{k}.pt"))
        g.load_state_dict(T.fedavg(states, weights))
        h_prev = T.model_hash(g)[0]
        round_hashes[r] = h_prev
        torch.save(state_cpu(g), os.path.join(run_dir, f"ckpt_r{r}.pt"))
        ev("RoundAggregated", r=r, accepted=accepted, h_out=h_prev)
        if ledger is not None:
            # 1단계: 데이터센터 커밋(기관별 계정, 동시) → 2단계: 수락 판정·집계(집계 주체, 연속)
            # → 3단계: 감사 표본(감사자)과 다음 라운드 개시(플랫폼, 동시)
            ledger.batch(phase1)
            ledger.batch(phase2 + [("aggregator", "aggregate", (r, b32(h_prev), b32(h_prev)))])
            phase3 = [("auditor", "sampleAudit", (r,))]
            if r < last:
                phase3.append(("platform", "openRound", (r + 1, int(_t.time()) + 3600, b32(h_prev))))
            ledger.batch(phase3)
        if log_every and r % log_every == 0:
            print(f"[{label}] round {r}: {h_prev[:12]}", flush=True)

    with open(os.path.join(run_dir, "events.jsonl"), "w") as f:
        for e in events:
            f.write(EV.canonical(e) + "\n")
    return {"label": label, "final_hash": h_prev, "round_hashes": round_hashes, "norm": norm, "prog": prog, "config": config,
            "commits": commits, "policy": policy.describe(), "start_round": start_round, "model": g}
