#!/usr/bin/env bash
#SBATCH --job-name=kproj-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=96G
#SBATCH --time=60:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${K_PROJECT_REPO:?Set K_PROJECT_REPO to the checked-out repository}"
: "${K_PROJECT_SCRATCH:?Set K_PROJECT_SCRATCH to node-local or parallel scratch}"
: "${HF_HOME:?Set HF_HOME to a persistent model cache}"
: "${STAGE:?Set STAGE to a pipeline stage}"
: "${K_PROJECT_SCRATCH_QUOTA_GB:?Set the measured scratch quota in GB}"

mkdir -p "${K_PROJECT_SCRATCH}/logs" "${K_PROJECT_SCRATCH}/runs"
cd "${K_PROJECT_REPO}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

if [[ -n "${K_PROJECT_SECRETS_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${K_PROJECT_SECRETS_FILE}"
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID}}"
RUN_DIR="${K_PROJECT_SCRATCH}/runs/${RUN_ID}/${STAGE}"
mkdir -p "${RUN_DIR}"

checkpoint_and_exit() {
  touch "${RUN_DIR}/PREEMPTED_OR_TIMELIMIT"
  if [[ -n "${CHILD_PID:-}" ]]; then
    kill -USR1 "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" || true
  fi
  rsync -a --partial "${RUN_DIR}/" "${K_PROJECT_REPO}/runs/${RUN_ID}/${STAGE}/"
  exit 99
}
trap checkpoint_and_exit USR1 TERM

# The stage runner must be idempotent: it reads completed shard markers, resumes
# checkpoints, and writes results atomically (temporary file followed by rename).
srun python -m pnas_pipeline.run_stage \
  --manifest cluster/config/experiment_manifest.yaml \
  --stage "${STAGE}" \
  --run-id "${RUN_ID}" \
  --output-dir "${RUN_DIR}" \
  ${STAGE_ARGS:-} &
CHILD_PID=$!
wait "${CHILD_PID}"

touch "${RUN_DIR}/SUCCESS"
mkdir -p "${K_PROJECT_REPO}/runs/${RUN_ID}/${STAGE}"
rsync -a --partial "${RUN_DIR}/" "${K_PROJECT_REPO}/runs/${RUN_ID}/${STAGE}/"
