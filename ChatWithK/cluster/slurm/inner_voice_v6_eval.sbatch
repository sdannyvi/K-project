#!/usr/bin/env bash
#SBATCH --job-name=k-inner-v6-eval
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-6%2
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
: "${RUN_DIR:?Set RUN_DIR to the scratch run directory}"
: "${ANNOTATIONS:?Set ANNOTATIONS to private PI development labels}"
: "${PATTERN_SOURCE:?Set PATTERN_SOURCE to development PatternReframe JSONL}"
: "${FACTUAL_SOURCE:?Set FACTUAL_SOURCE to development factual JSONL}"
index="${SLURM_ARRAY_TASK_ID}"
if (( index == 0 )); then
  name="base"
  adapter_args=()
else
  seeds=(20261106 20261107 20261108)
  arm_idx=$(((index - 1) / 3))
  seed_idx=$(((index - 1) % 3))
  arms=(ordinary boundary)
  name="${arms[$arm_idx]}_seed${seeds[$seed_idx]}"
  adapter_args=(--adapter "${RUN_DIR}/adapters/${name}")
fi
out="${RUN_DIR}/evaluation/${name}"
mkdir -p "${out}"
python -u -m synthetic_preliminary.eval_inner_voice_v6_annotations \
  --annotations "${ANNOTATIONS}" --source "${PATTERN_SOURCE}" \
  --output "${out}/pattern_annotated.jsonl" "${adapter_args[@]}"
python -u -m synthetic_preliminary.eval_inner_voice_v6_factual \
  --source "${FACTUAL_SOURCE}" --output "${out}/factual.jsonl" \
  --gsm8k 100 --drop 50 --propara 7 "${adapter_args[@]}"
