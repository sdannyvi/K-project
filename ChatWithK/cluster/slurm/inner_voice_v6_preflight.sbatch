#!/usr/bin/env bash
#SBATCH --job-name=k-inner-v6-check
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
: "${RUN_DIR:?Set RUN_DIR to the scratch run directory}"
: "${DATA:?Set DATA to the candidate corpus JSONL path}"
python -u -m synthetic_preliminary.preflight_inner_voice_v6 \
  --data "${DATA}" --output "${RUN_DIR}/token_preflight.json" --max-length 384
