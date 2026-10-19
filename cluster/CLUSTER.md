# BGU ECE HPC cluster — K-project guide

Site-specific companion to `PNAS_CLUSTER_EXECUTION_PLAN.md`. Shares conventions
with `~/Voxwell/CLUSTER.md` and `~/holocaust/CLUSTER.md`; values below were
checked against `sacctmgr`/`sinfo` on 2026-10-19.

## Access and paths

| Item | Value |
|---|---|
| Host | `ece-hpc.bgu.ac.il` (user `vilenchi`) |
| Repo (cluster-native clone) | `/truenas/home/vilenchi/K-project` |
| Heavy artifacts (`$K_PROJECT_SCRATCH`) | `/truenas/projects/vilenchik_shared/k-project` |
| HF cache (`$HF_HOME`, group-shared) | `/truenas/projects/vilenchik_shared/hf_cache` |
| Python env | conda env `kproject` (`~/miniforge3`) |
| SLURM logs | `slurm_logs/` in the repo root (gitignored) |

`source cluster/env.sh` sets all of the above and activates the env. Every batch
script sources it.

## The `kproject` env

Python 3.11, `torch==2.6.0+cu124`, `transformers<5`, `accelerate`, `peft`,
`bitsandbytes`, scikit-learn, pandas, scipy, statsmodels, pytest. Rebuild with:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n kproject python=3.11 && conda activate kproject
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r neural_poc/requirements.txt -r pnas_pipeline/requirements.txt
```

Exact pins live in `cluster/env_lock/` (`requirements-lock.txt`, conda export and
explicit spec); regenerate them whenever the env changes.

`HF_HOME` is pinned in `env.sh` rather than defaulted: `~/.bashrc` exports the
global `/truenas/projects/hugging_face_cache`, which is not writable by us and
would otherwise be inherited by every sbatch job.

Gated models (`google/gemma-3-27b-it`, `meta-llama/Llama-3.1-8B-Instruct`) need
`hf auth login` once on the login node plus an accepted license on
huggingface.co. Check access and pin revisions with
`python -m pnas_pipeline.provenance.revisions` (writes `reports/model_access.json`).

Do not reuse the Holocaust `gemma4` env: it carries a transformers 5.x dev build,
which `neural_poc/requirements.txt` excludes.

## Partitions and QOS

| Use | Flags |
|---|---|
| GPU work | `--account=research_vilenchik --partition=vilenchik_part --qos=vilenchik_owned` |
| CPU-only work | `--partition=CPUonly --gres=none` (1-day limit) |

- `hpc-vilenchik` (single node): **9× RTX A5000 (24 GB), 96 CPUs, 500 GB RAM**,
  3-day max walltime. Request **8 CPUs + 45 GB per GPU**.
- `vilenchik_owned`: up to 9 GPUs per job, **max 4 concurrent jobs per user**
  (array tasks count individually). Default QOS on the account is
  `vilenchik_4jobs` (4 GPUs, 3 days); pin `vilenchik_owned` explicitly.
- Never run CPU-only stages on `vilenchik_part`.
- If the group partition is saturated, `shared_a5000` (1-day limit) is the
  fallback for 1-GPU jobs.

## Batch scripts

| Script | Purpose |
|---|---|
| `slurm/poc_qwen3b.sbatch` | Reruns the 3B neural POC on one A5000 and logs per-step wall time |
| `slurm/job_template.sbatch` | Generic `pnas_pipeline.run_stage` stage runner; submit via `scripts/submit_stage.sh` |
| `slurm/nccl_smoke.sbatch` | WP0 4-GPU NCCL all-reduce correctness and bandwidth check |
| `slurm/failure_notifier.sbatch` | `afternotok` companion; runs on `CPUonly` |
| `scripts/preflight.sh` | Env, paths, secret scan; warns (not fails) if no GPU on login node |

Submit from the repo root:

```bash
cd /truenas/home/vilenchi/K-project && mkdir -p slurm_logs
sbatch cluster/slurm/poc_qwen3b.sbatch
STEPS="smoke" sbatch cluster/slurm/poc_qwen3b.sbatch   # subset of steps
```

Pipeline stages go through `submit_stage.sh`; stage arguments ride in
`STAGE_ARGS`, and reusing a `RUN_ID` resumes (completed stages are skipped):

```bash
STAGE_ARGS="--model Qwen/Qwen3-32B --quantization 4bit" RUN_ID=wp0-calibration \
  bash cluster/scripts/submit_stage.sh calibrate_throughput --time=0-04:00:00
python -m pnas_pipeline.budget          # COMPUTE_BUDGET.csv launch gate
python -m pnas_pipeline.gates CERT.yaml # signed gate-certificate check
python -m pytest -q pnas_pipeline/tests
```

Scientific stages (`h0_construct` ... `e5_transition_exploratory`) are registered
but exit 3 with the missing P0/WP input until implemented. Resume a failed
`poc_qwen3b` step into an earlier job's results with
`K_POC_RESULTS_DIR=$K_PROJECT_SCRATCH/poc_qwen3b/<jobid> STEPS=native_lora sbatch ...`.

Cluster runs of `neural_poc` write to `$K_POC_RESULTS_DIR` (set by the sbatch
to `$K_PROJECT_SCRATCH/poc_qwen3b/<jobid>`), so the versioned Mac pilot evidence
in `neural_poc/results/` is never overwritten. Use
`configs/poc_qwen3b_cuda.yaml` (CUDA, bf16) on the cluster; the original
`poc_qwen3b.yaml` is the Mac/MPS config and is kept for provenance.

## GPU peer-to-peer is broken on `hpc-vilenchik` (checked 2026-10-19)

| Test | Result |
|---|---|
| NCCL all-reduce, 4 GPUs, P2P on (jobs 10129, 10146) | hangs until the watchdog timeout |
| NCCL all-reduce, `NCCL_P2P_DISABLE=1` (job 10146) | correct, 4.6 GB/s bus bandwidth |
| `tensor.to("cuda:1")` from `cuda:0` (`diagnostics/peer_copy.py`) | **silently corrupted**, although `can_device_access_peer` is True |

The GPUs sit behind PCIe switches (`nvidia-smi topo -m`: PIX/PXB). Silent wrong
values on P2P copies usually mean the IOMMU or PCIe ACS settings block direct
transfers; fixing that needs the cluster admins. Until it is fixed:

- `env.sh` sets `NCCL_P2P_DISABLE=1`, so DDP/torchrun with one GPU per process is safe.
- **Never shard one model across GPUs** (`device_map="auto"`, accelerate/pipeline
  parallel). Fit each model on one A5000 (32B in 4-bit), and use extra GPUs as
  independent processes or DDP.
- After any admin fix, rerun `python -m pnas_pipeline.diagnostics.peer_copy` on
  2 GPUs and `sbatch cluster/slurm/nccl_smoke.sbatch` before lifting this rule.
- `nvidia-smi` inside a job lists GPUs that are not yours; trust
  `CUDA_VISIBLE_DEVICES` / `scontrol show job -d` (`GRES=...(IDX:n)`).

## Known issues (inherited from the Holocaust/LLMSurvey runs)

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available() is False` with a granted GPU | Transient contention on `hpc-vilenchik` when sibling jobs start together; resubmit (solo if repeated). `common.load_model` now fails loudly instead of silently falling back to CPU. |
| `Job dependency problem` at submission | `afterok:<id>` on a job already purged from the Slurm table; only depend on jobs still in `squeue`. |
| Multi-GPU `device_map="auto"` very slow | Per-token pipeline hops across GPUs, and on this node the hops also corrupt data (see above). Do not use it. |
| `QOSMaxSubmitJobPerUserLimit` / pending on `QOSMaxJobsPerUserLimit` | 4 concurrent jobs per user; throttle arrays with `%N` and count jobs from other projects too. |
| Stdout buffered in logs | `PYTHONUNBUFFERED=1` is set in `env.sh`; scripts also use `python -u`. |
