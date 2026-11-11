# Testing the Promises of Blockchain-Based Consent Withdrawal in Multi-Hospital Federated Learning

Code, smart contracts, configurations, and result files for the paper

> W. Jang, H. Park, and K. A. Kim, *Testing the Promises of Blockchain-Based Consent Withdrawal in Multi-Hospital Federated Learning: An Empirical Study With Electrocardiogram Data*, submitted to IEEE Journal of Biomedical and Health Informatics. Preprint: arXiv:TODO.
>
> Repository: https://github.com/WooyeongJang/consent-withdrawal-fl

The paper is an empirical study. On PTB-XL electrocardiograms in a simulated three-hospital setting, it tests three assumptions that blockchain-based medical federated learning (FL) frameworks make about consent withdrawal:

- **C1** Checking a patient's consent before training starts is enough to honor a withdrawal (experiment E1).
- **C2** Excluding a withdrawn patient from later training, or running an unlearning method, establishes that the patient's influence was removed, and a membership-inference test can verify it (E2).
- **C3** An immutable ledger record declaring remediation complete is sufficient evidence that the declared processing was performed (E3).

The code here is the reference implementation that served as the experimental instrument. It is not proposed as a new security protocol or unlearning algorithm. It decides consent when the aggregator accepts an update and separates what is owed after a withdrawal into three remediation stages:

- **S1** participation blocking
- **S2** lineage and deployment management
- **S3** influence removal

The data subject and an independent auditor check these stages at four levels:

- **L1** bit-exact sampled re-execution of training and unlearning
- **L2** input-set equivalence
- **L3** a powered residual-influence test for approximate unlearning
- **L4** an on-device deployment-state check (simulated in software in the experiments)

Withdrawals and evidence are recorded on the Kaia Kairos testnet (chain ID 1001).

**Data are not redistributed.** PTB-XL and the PhysioNet/CinC Challenge 2021 data must be obtained from PhysioNet under their licenses (see [Data](#data)).

> Citation metadata: `CITATION.cff` (the arXiv identifier is added there after posting).

## Repository layout

| Path | Contents |
|---|---|
| `contracts/` | `ConsentLedger.sol` (v3.1, used in all experiments), `ConsentLedger_v3_2.sol` (v3.2, rejects seven v3.1 fault paths in unit tests, four paths remain open; not used in experiments E1–E4; its acceptance version check was also measured on Kairos, see `validation/`) and `CompletionRecord.sol` (baseline B3) |
| `fl/` | FL training, evidence layer (Merkle commitments, non-membership proofs), auditor (L1–L3), unlearning methods (U0–U5), experiment drivers E0–E4 and T11 |
| `ledger/` | Ledger service: signs role-account transactions, fee delegation by the platform, block-pinned reads (Dockerfile included) |
| `scripts/` | Experiment runners, aggregation, tables and figures (`run_final.sh` runs the whole final rerun) |
| `tests/` | Contract state transitions, evidence layer, comparison of our FedOSD with the official implementation |
| `tools/` | `toolchain.json` (pinned container digest, solc version and SHA-256, EVM version, ledger libraries), `get_solc.sh` |
| `site_check/` | PTB-XL site inspection and institution assignment tables for K = 3, 5, 10, with exclusion lists and hashes |
| `c2021/` | Challenge 2021 diagnosis-code to superclass mapping, checksum and invalid-sample exclusion lists |
| `results/consent/` | Consent-event scenarios for E2 (hash-fixed) |
| `results/final/` | Final rerun results (JSON) and aggregates used in the paper |
| `results/` (other) | Preliminary and pilot runs, kept for provenance; the paper uses only `results/final/`, except the E3-B calibration inputs in `results/late/` and `results/t11/` (see below) |
| `supplement/` | Supplementary Document of the paper (PDF) |
| `docs/` | `extended_results.pdf`: extended results distributed with the repository (Tables R1–R23, Figs. R1–R5, R for repository: implementation lessons, contract v3.2 details, implementation and experiment details, E0/E1/L3-sensitivity/auxiliary/subgroup/FedOSD-variant/ablation tables, E4 micro-benchmarks, latency and bandwidth, L1 sampling cost, ledger event table, policy-state diagram, system overview, notation table, and three post hoc validation sections: L3 operating characteristics, the acceptance race under concurrent submission, and an independent re-derivation of the E3 and L3 claims); `extended_src/` holds its LaTeX source (cross-references to the paper and Supplementary Document resolve only when compiled next to them) |
| `validation/` | Post hoc validation behind Sections R10–R12 of the extended results: pre-registered plan (`PREREG.md`, hashed before execution), deviations and failed runs (`DEVIATIONS.md`), code, logs, and results for (A) L3 power and decision reliability, (B) the acceptance race on a local EVM and on Kairos with contracts v3.1 and v3.2, (C) an independent re-derivation of the E3 and L3 claims from ledger events, checkpoints, and stored statistics, (D) an independent re-execution of the 206 sampled type-B training units from the ledger's sampling seeds and commitments (all reproduced their committed hashes), and (E) a deterministic regeneration of the 20 type-A runs checked against the ledger's round hashes (1,000 of 1,000) with re-execution of the 899 sampled units (877 honest passed, 22 injected mismatched). Reports are in Korean; see `validation/README.md` |
| `REPRODUCIBILITY.md`, `key_transactions.csv` | Hashes, contract addresses, and key transaction IDs of the final rerun |
| `requirements-host.txt`, `requirements-test.txt` | Pinned host-side Python packages and pinned contract-test packages |

## Environment

All numbers in the paper come from one machine and one software stack:

- **Hardware:** NVIDIA GB10 (ARM64), single node.
- **Container:** NGC PyTorch 26.08, pinned by digest in `tools/toolchain.json`:
  `nvcr.io/nvidia/pytorch@sha256:3becd068f49bd2ad38f90db5f9a4803019a76933a24e63d821376c44e7a9200a`
  (Python 3.12.3, PyTorch 2.14.0a0 nv26.08, CUDA 13.4, cuDNN 9.25.0).
- **Determinism:** GPU deterministic mode. Bit-exact re-execution (assumption A6) was confirmed only for the same image digest, device type, deterministic settings, and CPU thread count (experiment E0). Hashes differ across GPU/CPU, software stacks, and thread counts. Do not expect bit-identical results on other hardware.
- **Ledger:** Kaia Kairos, public RPC `https://public-en-kairos.node.kaia.io`, contracts compiled with solc 0.8.38 (EVM `prague`), accessed with web3py-ext 2.1.0 / web3 6.3.0.
- **ARM64 only as released:** the container digest is the arm64 image, and `tools/get_solc.sh` fetches the arm64 solc build. On x86_64, use the matching x86_64 image and solc release (bit-exact hashes are then not expected).
- **Required local paths:** the run scripts expect, under `$HOME/data/`, the raw PTB-XL files at `physionet.org/files/ptb-xl/1.0.3/`, the caches at `cache/ptbxl_k{3,5,10}/` and `cache/c2021/` (built below), working directories under `runs/` (created by the scripts), and, for E3 type-B runs, `runs/lira/shadows_late.npz` (see "E3-B calibration inputs").
- **Language of comments:** code comments, docstrings, and log messages are in Korean; identifiers, this README, and the paper are in English.
- **Host Python 3.12** (data preparation, aggregation, figures): `pip install -r requirements-host.txt`.

## Data

1. **PTB-XL v1.0.3**
   - Download from PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/
   - The AWS mirror `s3://physionet-open/ptb-xl/1.0.3/` is faster.
   - Verify the files against the distribution's `SHA256SUMS.txt`.
2. **PhysioNet/CinC Challenge 2021 v1.0.3** (auxiliary)
   - Get it with `scripts/fetch_c2021.sh`.
   - Cohorts used: ningbo, georgia, chapman_shaoxing, cpsc_2018 + cpsc_2018_extra.
   - Records whose server copy did not match the published checksum are listed in `c2021/excluded_checksum.csv` and skipped.
   - The Challenge's SNOMED CT code tables, used as reference for our mapping `c2021/mapping_superclass_v1.csv`, are not included. Get them with `scripts/fetch_c2021_maps.sh` (pinned commit, SHA-256 checked). They are not needed to build the caches.
3. **Build the training caches** (host):

```bash
# PTB-XL, 100 Hz, institutions from site_check/patient_map_K{3,5,10}.csv
python fl/prepare_data.py --root <ptb-xl-1.0.3> --out ~/data/cache/ptbxl_k3  --k 3
python fl/prepare_data.py --root <ptb-xl-1.0.3> --out ~/data/cache/ptbxl_k5  --k 5  --signals ~/data/cache/ptbxl_k3/ptbxl_100hz.npy
python fl/prepare_data.py --root <ptb-xl-1.0.3> --out ~/data/cache/ptbxl_k10 --k 10 --signals ~/data/cache/ptbxl_k3/ptbxl_100hz.npy
# Challenge 2021 (cohort = institution, record-level withdrawal)
python fl/prepare_c2021.py --root <challenge-2021-1.0.3>/training --out ~/data/cache/c2021
```

How institutions are formed:

- PTB-XL `site` values are grouped into institutions. The rules and resulting sizes are described in `site_check/README.md`, and the resulting tables are committed with hashes in `site_check/hashes.txt`.
- This split is a simulation. It is not equivalent to real multi-institution data.
- Note on counts: `site_check/README.md` and `site_check/summary.txt` report the training set before the superclass filter (15,012 patients, 17,407 records). The paper reports the training set after excluding the 411 records without a diagnostic superclass (14,812 patients, 17,073 records; 6,275 / 4,812 / 3,725 patients per institution for K = 3). Both counts come from the same assignment tables.

## Ledger setup

The experiments need Kairos testnet accounts for these roles:

- platform (fee payer)
- aggregator
- auditor
- one account per data center (`dc0`, `dc1`, …)

Subject accounts are created per request. Testnet KAIA is available from the Kairos faucet.

1. **Keys:** one JSON file per role, `{"role": "<role>", "private_key": "0x…"}`, in a directory outside the repository (e.g. `~/.config/<name>/kairos/exp/`, mode 600). **Never commit keys.**
2. **Compiler:** `bash tools/get_solc.sh` (downloads solc 0.8.38 for ARM64 and checks its SHA-256).
3. **Contract artifacts:** `python scripts/build_artifacts.py <artifact-dir>` compiles `ConsentLedger.sol` and `CompletionRecord.sol` (solc 0.8.38, `prague`, optimizer on). The bytecode, including the metadata hash, is identical to the artifacts used in the experiments; `--check <dir>` verifies this against another artifact directory.
4. **Ledger service:**

```bash
docker build -t dissertation-ledger ledger/
docker run -d --name dissertation-ledger --network host \
  -v <keys-dir>:/keys:ro -v <artifact-dir>:/artifact:ro -v <state-dir>:/state \
  dissertation-ledger
# listens on 127.0.0.1:8600 (POST /deploy, /tx, /view; GET /info)
```

Each experiment deploys a fresh contract. Deployment addresses are appended to `<state-dir>/deployments.jsonl`.

## Reproducing the paper

The final rerun is one script. Every stage skips results that already exist, so it can be resumed.

```bash
bash scripts/run_final.sh all       # e0 → [e2main → e2aux] ∥ [e1 → e3] → e4 → summary
bash scripts/run_final.sh <stage>   # e0 | e1 | e2main | e2aux | e3 | e4 | summary
```

When run with `all`, the script first checks two things: that the working tree has no uncommitted code changes, and that HEAD is the tag `final-rerun-v1`. That tag exists only in the authors' working repository, so in this repository `all` stops at the tag check; run the stages one by one instead (`bash scripts/run_final.sh <stage>`). Relative to the tagged commit, `ledger/`, `tools/`, and `contracts/ConsentLedger.sol` are unchanged; `fl/e3b.py` differs only by renaming a module alias (`F` → `TM`) that shadowed another import and crashed T5 when run together with T11; `contracts/ConsentLedger_v3_2.sol`, the table and figure scripts in `scripts/`, and the v3.2 tests in `tests/` were added after the rerun. Eight files were corrected after the rerun following a code review, with nothing re-run, so `results/final/` comes from the earlier code (the positive-control change is stated in Supplementary Section S7 (Table S8) and the E1 Kairos change in the footnote of Table R6 in `docs/extended_results.pdf`; the other corrections are listed only here):

- `fl/l3.py`: the positive control requires P1 to be classed as exceeding the margin (before: lower confidence bound above zero), and the `result` field is updated together with the class.
- `fl/e2.py`, `scripts/summarize_e2.py`: the run records the index of its own model among the U0 references and the aggregation uses it, falling back to AUC equality for the final-rerun files; U5 is treated like U0 where it is the same model. Re-aggregation changed no L3 class.
- `fl/exp_lira.py`: guard for a record that is a member (or a non-member) in every shadow model; one such record occurred in the final rerun (10% early-withdrawal scenario; see "Scope of the results").
- `fl/e1chain.py`: the on-chain policy snapshot is the withdrawal version read at selection (before: a constant), and the `during` condition submits each withdrawal once per round (before: once per data center, so the final rerun sent each of its 24 withdrawals three times, 72 transactions).
- `fl/e0.py`, `fl/e1.py`, `fl/e3.py`: hard-coded K = 3 removed.

Results are written to `results/final/`. The ledger stages (E1 on Kairos, E3, E4) spend testnet KAIA; the script stops if the platform balance falls below `MIN_KAIA`.

| Stage | Paper | Approximate time on GB10 | Ledger |
|---|---|---|---|
| E0 reproducibility | Sec. IV-C (opening), Table R4 | < 1 h | – |
| E1 consent-check timing: selection-time check (`sel`), acceptance-time check (`acc`), strict consent-version comparison (`strict`); simulation + Kairos (`acc` only) | Sec. IV-A, Fig. 2, Tables R5–R6 | ~1 h | Kairos |
| E2 unlearning and influence verification (19 scenarios × 10 seeds, auxiliary × 5 seeds) | Sec. IV-B, Table II, Fig. 3, Supplementary Tables S4–S5 and Section S6 (positive-control rule provenance, Table S7), Tables R7–R11 | ~3 days | – |
| E3 verification of withdrawal execution: detection of injected faults T1–T9 and concealment case T10; component ablation (`E3-A` in the code and result files, recomputed from the E3 evidence); concealment cases T11a–f | Sec. IV-C, Table III, Supplementary Table S6, Fig. R1, Table R12 | ~1 day | Kairos |
| E4 operational cost | Sec. V, Fig. R2, Tables R13–R15 | ~25 min | Kairos |

Times are rough figures from our run. Section, table, and figure numbers refer to the submitted IEEE JBHI manuscript; S-numbers refer to its Supplementary Document and R-numbers to `docs/extended_results.pdf`.

**Reproducibility record.** `REPRODUCIBILITY.md` lists the container digest, data/split/mapping and consent-event hashes, contract artifact and bytecode hashes, and the Kairos contract address of every run behind the paper's numbers; `key_transactions.csv` lists their key transaction IDs (4,584 rows: deployment, training-configuration commitment, job creation and completion, verification records, deployment transitions). Both can be checked on the Kairos explorer without running anything.

**E3-B calibration inputs.** The type-B runs of E3 (`fl/e3b.py`) test residual influence (L3) with the 16 shadow models and five U0 reference models of the pilot run that set the L3 margin, not with models trained in the final rerun. Two of the three inputs are in this repository: `results/late/late_methods.json` (U0 reference AUCs, from `fl/exp_late_methods.py`) and `results/t11/t11_eps.json` (T11 tolerances, from `fl/exp_t11.py`). The third, `shadows_late.npz` (per-record shadow-model statistics over the withdrawal cohort and its non-member set, 16 × 1,674 values), is derived from PTB-XL and is not included; regenerate it in the training container with `python -m fl.exp_lira --out <out> --runs ~/data/runs/lira --l3runs <L3 run directory> --l3json <L3 summary JSON> --shadows 16 --late 45` (the module writes `shadows_late.npz` into `--runs`; late condition, 16 shadow models, shadow seeds 1000–1015). `scripts/run_final.sh e3` checks all three against the SHA-256 values in `REPRODUCIBILITY.md` and stops on a mismatch. Bit-identical regeneration is expected only in the environment described above.

**Fields re-judged at aggregation.** In the per-run E2 files (`results/final/e2/*.json`), the `L3.class` and `deploy` fields were written during the run under an earlier, weaker positive-control rule. `scripts/summarize_e2.py` re-judges both under the rule in the paper (P1 must itself exceed), and every L3 count in the paper comes from that aggregation (`results/final/E2_e2.json`), not from the per-run fields.

Each processing time reported in the E2 base scenario table was re-measured with no other job on the GPU:

```bash
bash scripts/run_e2_timing.sh && python scripts/check_e2_timing.py   # model hashes must match results/final/e2
```

### Checking a release with preliminary runs (no ledger)

`run_final.sh` writes to `results/final/` and skips results that already exist, so in this repository it re-runs nothing. Set `F=<directory>` to run a stage elsewhere, or call the lower-level scripts with a separate output directory. Bit-identical hashes are expected only on the setup described under "Environment".

```bash
IMG=$(python3 -c "import json;print(json.load(open('tools/toolchain.json'))['training_image']['repo_digest'])")
# E0, one local round (seconds): the hash must equal results/final/e0/unit_gpu_det_s1_r1.json
mkdir -p ~/check/e0 && docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp --ipc=host \
  -v "$PWD":/w:ro -v ~/data/cache/ptbxl_k3:/data:ro -v ~/check/e0:/out -w /w "$IMG" \
  python -m fl.e0 --mode unit --cond gpu_det --run 1 --out /out
# E0, all conditions (about 20 min with REPS=2)
OUT=~/check/e0_full REPS=2 bash scripts/run_e0.sh && python scripts/summarize_e0.py ~/check/e0_full ~/check/E0.json
# E2 base scenario, seed 1 (about 40 min; the first run also trains 16 shadow and 5 reference models)
SEEDS=1 bash scripts/run_e2.sh check/e2 check_e2 E2-base   # compare each method's hash and AUROC with results/final/e2/E2-base_s1.json
```

Stages that send Kairos transactions (E1 on Kairos, E3, E4, `kairos_*_probe.py`) need role keys and testnet KAIA; set `PLATFORM` to your fee-payer address for the balance check and `KAIROS_KEY_DIR` for the probes.

### From results to tables and figures

```bash
python scripts/l3_sensitivity.py results/final/e2 results/final/L3_sensitivity.json
python scripts/make_tables.py --out tables --aux   # supplementary tables (LaTeX)
python scripts/make_figures.py --final --jbhi --out figures
python scripts/posthoc_checks.py   # post hoc numbers quoted in the paper -> results/final/posthoc_checks.json
python scripts/l3_rule_comparison.py results/final   # Supplementary Table S9: approximate-method L3 passes under no control, the run-time control rule, and the rule as written
python scripts/shadow_count_sensitivity.py --runs <E2 run folder> --out results/final/shadow_count_sensitivity.json   # extended results Table R21: needs the shadows.npz files written by the E2 runs (not redistributed); --selftest checks the formula on synthetic data
python scripts/compare_c4_rerun.py   # stored results vs. the rerun of the 10% early-withdrawal scenario (results/c4_rerun/)
python scripts/check_manuscript_numbers.py   # recomputes 173 figures quoted in the main text, the Supplementary Document, and the extended results from results/, validation/, and site_check/
python scripts/recount_l3_deploy.py results/final   # Supplementary Section S6 and Table S7: 53 of 269 deployable-on-L3-pass fields in the run logs become inconclusive when the positive-control rule is re-applied (all reported L3 outcomes already use the rule as written)
```

`scripts/posthoc_checks.py` recomputes the statistics that the paper reports beyond the per-scenario aggregates: U0's L3 classes and its rate outside ±1.645 standard errors, the prediction-interval variant, Holm correction across all scenarios, the effect of the positive control on the early- and mid-withdrawal grid, the T1 check against the on-chain withdrawal count, and the number of honest re-executions. In `results/final/E3.json`, the field `A_l1_sampled_honest_units` (899) counts all sampled type-A units, including 22 units with an injected fault; the paper's 877 honest re-executions are in `posthoc_checks.json`.

`scripts/repro_manifest.py` is the authors' tool that produced `REPRODUCIBILITY.md` (in Korean, from the working repository); it needs the contract artifacts and the `final-rerun-v1` tag, which are not part of this repository. Set `REPRO_TXS` and pass an output path to run it elsewhere.

### Tests

```bash
pip install -r requirements-test.txt  # pinned; a clean environment with these versions passed the four tests below
bash tools/get_solc.sh                # solc 0.8.38, SHA-256 checked
python tests/test_contract.py         # v3.1 state transitions; needs web3, eth-tester[py-evm], tools/bin/solc
python tests/test_contract_v3_2.py    # v3.2: signed withdrawal/re-consent, replay, acceptance race, r0, resolved scope, lineage
python tests/test_contract_v31_gaps.py  # shows the same fault paths succeed on v3.1
python tests/test_evidence.py
docker run --rm -v "$PWD":/w -w /w "$IMG" python tests/test_fedosd_official.py  # needs PyTorch: run in the training container
```

## Scope of the results

- The results are empirical tests of the three assumptions under the trust assumptions stated in the paper. They do not establish end-to-end security or certify complete influence removal.
- Detection rates hold only for the faults and concealment cases we injected.
- Re-execution checks hold only in the environment where E0 confirmed bit-exact replay.
- All consent events are simulated, and the institution split is synthetic.
- No result should be read as evidence about real deployments or clinical outcomes.
- The following are discussed in the paper and are **not prevented** by contract v3.1, which all experiments used:
  - the acceptance race and the deployment-gating bypasses (`docs/extended_results.pdf`, Table R2)
  - correctness of regular-round aggregation, which is assumed
  - forged non-membership proofs from a malformed commitment (unsorted tree or misstated |S|), which only sampled L1 and the auditor's L2 detect; the experiments used honestly built trees
- Contract v3.2 closes the listed paths in the first group (signed withdrawal and re-consent, replay, acceptance version check, first-use round, resolved scope, round numbering) in local unit tests; its acceptance version check was also measured on Kairos with two concurrent submitters (`docs/extended_results.pdf`, Section R11: no stale acceptance in 150 trials, at the cost of reverting acceptances after unrelated withdrawals). Four paths remain open in v3.2: job outputs are matched to withdrawals by count, not by set; the subject's data center can over-declare its first r0; the platform registers the subject key; and for delta > 0 deployment gating relies on a call to `effectuate`. v3.2 was not used in experiments E1–E4. Aggregation correctness remains an assumption.
- The L3 equivalence test was not sized in advance. Its post hoc operating characteristics (no model can pass at 1% withdrawal; a method whose true difference equals the margin passes with probability up to 0.11 because the interval omits the tested model's training variability) and intervals at the nominal level are in `docs/extended_results.pdf`, Section R10.
- Known difference between stored results and the released code: in the 10% early-withdrawal scenario of E2, one record of the withdrawn set was a member in all 16 shadow models; the run-time code gave it an undefined LiRA score, which the released code guards against, so five stored reference AUCs are up to 5.9e-4 above what the released code returns (no L3 class changes). The comment in `fl/exp_lira.py` saying that this did not occur in the final rerun is incorrect; the experiment code is kept as it was run (Section R12). Repeating the ten seeds of that scenario with the released code, on the stored shadow and reference models, reproduced all 90 model hashes, lowered every model's AUC by 0.0002–0.0006, and changed none of the 80 test outcomes; these files are in `results/c4_rerun/` (`python scripts/compare_c4_rerun.py`), and the tables of the paper still read `results/final/`.
- The non-membership verifier (`fl/evidence.py`) does not bind a proof's tree size to the committed set size, so a prover can forge a non-membership proof. In the experiments the verifier built every proof from the committed tree itself.
- Docstrings and comments in `fl/`, `ledger/`, `scripts/`, `tests/`, and `validation/` cite internal design notes by path (`reports/...`, `3장 ...`, `jbhi/`). Those documents belong to the authors' working repository and are not part of this release; the code was published unchanged rather than re-edited.
- This release corresponds to the manuscript version tagged `v1.0` here and `jbhi-v83` in the working repository. The `validation/` folder and Sections R10–R12 of `docs/extended_results.pdf` were added after the experiments (2026-11-09/10); the run files in `results/final/` are unchanged since the final rerun. Two derived files were added on 2026-11-11 without new training of the reported models: `results/final/shadow_count_sensitivity.json` (attack discrimination with 4, 8, or 15 of the stored shadow models) and `results/c4_rerun/` (above).

## Comparators

- **U2 FedEraser, U3 SISA, U4 FedOSD:** reimplemented for patient-level withdrawal. `tests/test_fedosd_official.py` compares our FedOSD with the computation parts of the official implementation (github.com/zibinpan/FedOSD, commit 41cc106, MIT License, Copyright (c) 2025 zibinpan), copied into the test with its notice.
- **Baselines B0–B6:** the audit structures defined in the paper. B3 does not claim to reproduce the prior design it is modelled on.
- **Model:** `fl/model.py` re-implements `resnet1d_wang` from the architecture description of Strodthoff et al. (2021) and the benchmark documentation; no code was copied from the benchmark repository. Hyperparameters are in `fl/train.py` and summarized in `docs/extended_results.pdf`, Section R3.

## Citation

See `CITATION.cff` (GitHub renders it as "Cite this repository"). The arXiv identifier and, after acceptance, the journal DOI are added there.

## License

- Code and smart contracts: MIT License (`LICENSE`).
- Data-derived tables, configurations, consent scenarios, results, and the Supplementary Document: CC BY 4.0 (`LICENSE-data`).
- The FedOSD excerpt in `tests/` keeps its MIT license; see `THIRD_PARTY_NOTICES.md`.
- PTB-XL and Challenge 2021 are licensed by PhysioNet under CC BY 4.0 and are not redistributed here. Please cite them as listed in `THIRD_PARTY_NOTICES.md`.
