# Handover: K-project cluster work (as of 2026-10-19)

You are taking over engineering work on the PNAS study **"The Intelligence of
Non-Action: Can a Language Model Cease Without a Verbal Controller?"**
(PI: Dan Vilenchik, BGU). Repo: `github.com/sdannyvi/K-project`, branch
**`cluster-wp0`** (not yet merged to `main`). The repo is **public**: never
commit tokens, API keys, or confirmatory label keys.

## Read first, in this order

1. `cluster/CLUSTER.md`: site facts, env, partitions, and the GPU P2P hazard.
2. `cluster/PNAS_CLUSTER_EXECUTION_PLAN.md`: the operational plan, including
   **Amendment A1** in §1.
3. `cluster/P0_PROTOCOL_DRAFT.md`: the open protocol decisions (DRAFT v2, revised after an
   external review; awaiting the PI. Its "Review status" table says what is settled).
4. `cluster/config/experiment_manifest.yaml` and `cluster/COMPUTE_BUDGET.csv`.
5. `paper/main.tex`: the manuscript (methods/preregistration draft).

## Environment

- Cluster: BGU ECE HPC, `ece-hpc.bgu.ac.il`, node `hpc-vilenchik` (9× RTX A5000
  24 GB, 96 CPUs, 500 GB RAM). Slurm, account `research_vilenchik`, partition
  `vilenchik_part`, QOS `vilenchik_owned`, **max 4 concurrent jobs per user**,
  3-day walltime. CPU-only work goes to `CPUonly` (1-day limit).
- Repo clone: `/truenas/home/vilenchi/K-project`. Every shell and job runs
  `source cluster/env.sh` first, which activates conda env `kproject` and pins
  `HF_HOME`, the HF token path and `NCCL_P2P_DISABLE=1`.
- Heavy outputs: `$K_PROJECT_SCRATCH=/truenas/projects/vilenchik_shared/k-project`.
  Model cache: `/truenas/projects/vilenchik_shared/hf_cache`.
- Hugging Face is logged in (account `sdannyvi`); all 7 open models are accessible
  and pinned in `reports/model_access.json`.
- There is no LaTeX on the cluster; the paper is built elsewhere.
- git has no identity on the cluster. Commit with
  `git -c user.name="danv" -c user.email="danv@playtika.com" commit ...`
  (matches history) unless the PI says otherwise.

## Hard rules

1. **Never shard one model across GPUs** (`device_map="auto"`, tensor/pipeline
   parallel). GPU peer-to-peer on `hpc-vilenchik` is broken: `cuda:i -> cuda:j`
   tensor copies are **silently corrupted**, and NCCL hangs unless P2P is
   disabled. `models.check_single_device` rejects any multi-device `device_map`. One model per GPU; extra GPUs run independent seeds/shards or DDP
   (NCCL is correct with `NCCL_P2P_DISABLE=1`). Diagnostics:
   `pnas_pipeline/diagnostics/{peer_copy,nccl_smoke}.py`.
2. GPU jobs: 8 CPUs + 45 GB RAM per GPU on `vilenchik_part`. CPU-only work never
   goes on a GPU partition. Download models with a CPU job first
   (`cluster/slurm/prefetch_model.sbatch`), never inside a GPU job.
3. Scientific stages (`h0_construct` … `e5_transition_exploratory`) are
   registered in `pnas_pipeline/stages/__init__.py` but **blocked** (exit 3)
   until the P0 protocol is frozen and the stage code is written and tested.
   Do not run confirmatory work, and do not set `confirmatory_frozen: true`
   yourself. That is a PI decision with a signed certificate.
4. Do not change scientific choices (models, arms, seeds, margins, prompts)
   without the PI. Engineering choices are yours.
5. Run `bash cluster/scripts/preflight.sh` before submitting, and
   `python -m pytest -q pnas_pipeline/tests neural_poc/tests` before committing.

## What exists (`pnas_pipeline/`)

- `run_stage.py`: idempotent dispatcher used by `cluster/slurm/job_template.sbatch`.
  Exit codes: 0 done or already complete; 2 bad manifest or prerequisite; 3 blocked;
  99 stopped by USR1/TERM. Reusing a `RUN_ID` resumes it.
- `manifest.py` (validation, hashing, DAG, immutable expansion), `io.py`
  (atomic and write-once writes), `shards.py` (checksummed `status.json`),
  `provenance/` (run identity, model-revision access check).
- `budget.py` (COMPUTE_BUDGET.csv launch gate; currently FAIL on `TBD` rows),
  `gates.py` (signed gate-certificate validator).
- `models.py`: the shared text-only loader. Gemma-3 and Mistral-3.1 are
  vision-language checkpoints; LoRA attaches only to language-model layers; the
  Mistral tokenizer needs `fix_mistral_regex=True` and its chat template comes
  from `chat_template.json`.
- Utility stages: `dry_run`, `calibrate_throughput`, `calibrate_qlora`.
- Submit a stage:
  `STAGE_ARGS="--model M --revision SHA" RUN_ID=X bash cluster/scripts/submit_stage.sh STAGE [dep_jobid] [sbatch args]`.
  Results land in `$K_PROJECT_SCRATCH/runs/<RUN_ID>/<stage>/metrics.json`.

## Measured so far (one A5000)

| Model | Generation (4-bit, batch 8) | QLoRA r=16 |
|---|---|---|
| Qwen2.5-3B (bf16) | 167.5 tok/s | fits easily |
| Qwen3-14B | 105.0 tok/s, 10.3 GB | seq 384 / batch 2, 20.7 GB, 828 train tok/s |
| Qwen3-32B | 50.8 tok/s, 19.7 GB | **does not fit**: OOM at seq 256 |
| Llama-3.1-8B | 207.5 tok/s | seq 384 / batch 4, 19.0 GB, 1,670 train tok/s |
| Gemma-3-27B | 61.0 tok/s, 17.9 GB | **fails 10% headroom**: seq 256 / batch 1 reaches 23.7 GB; seq 384 OOM |
| Mistral-Small-3.1-24B | 72.0 tok/s, 16.0 GB | seq 384 / batch 1, 21.3 GB, 499 train tok/s |

The 3B neural POC reran on the cluster and matches the Mac pilot. Amendment A1
follows from this table: **Qwen3-14B carries the full design (E1–E4); Qwen3-32B is
inference-only (E1, E2, E3a–c).**

## Calibration completion update (2026-10-20)

Prefetch jobs 10176/10177 completed. The first Gemma/Mistral calibration jobs
10178--10181 failed before model loading because Transformers attempted an online
Hugging Face metadata request and the compute-node proxy returned HTTP 500. Their
logs and failed provenance are retained. Offline-cache retries 10190--10193 used
the same pinned revisions and completed successfully. Results are recorded in
`cluster/COMPUTE_BUDGET.csv` and the table above. No calibration jobs remain in
flight.

## Your next tasks

1. Present the completed Gemma/Mistral fit facts to the PI. Gemma fails the
   one-A5000 10% headroom gate; Mistral fits at sequence 256 and 384 with batch 1.
   Any resulting E4 scope change remains a PI scientific decision under Amendment A1.
2. Implement P0.5 in code once the PI accepts it: a single prompt-rendering
   function with a fixed system prompt, Llama `date_string` pinned, and Qwen3
   `enable_thinking=False`. Store a hash of each rendered prompt. Use it in
   `calibrate.py` and all future stages.
3. When the PI returns `P0_PROTOCOL_DRAFT.md` with decisions, transfer them into
   the manifest, and fill the E4 rows of `COMPUTE_BUDGET.csv` from P0.6.
4. Then start WP1 code (`pnas_pipeline/data/build_sncc.py`, `data/split.py`)
   with unit tests, reusing `neural_poc/src` logic where it is tested.

## Open items for the PI (do not resolve yourself)

- P0 decisions: expert and annotator names, capability benchmarks, all margins.
- Report the GPU P2P fault to the BGU HPC admins (likely IOMMU/ACS on the node).
- Merge `cluster-wp0` into `main` after review.
