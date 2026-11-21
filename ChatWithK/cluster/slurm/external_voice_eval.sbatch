#!/usr/bin/env bash
#SBATCH --job-name=k-external-voice
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=1-12:00:00
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
: "${MODEL:?}" "${REVISION:?}" "${ADAPTER:?}" "${V11_ADAPTER:?}" "${SOURCE:?}" "${RUN_DIR:?}"
shard="${SLURM_ARRAY_TASK_ID:?}"
if [[ -n "${PILOT_ITEMS:-}" ]]; then
  output="${RUN_DIR}/pilot/${shard}.jsonl"
  pilot_args=(--pilot-items 3)
else
  output="${RUN_DIR}/outputs/${shard}.jsonl"
  pilot_args=()
fi
mkdir -p "$(dirname "${output}")"
python -u -m synthetic_preliminary.eval_external_voice \
  --source "${SOURCE}" --model "${MODEL}" --revision "${REVISION}" \
  --adapter "${ADAPTER}" --v11-adapter "${V11_ADAPTER}" \
  --output "${output}" --shard-index "${shard}" --shard-count 12 \
  "${pilot_args[@]}"
