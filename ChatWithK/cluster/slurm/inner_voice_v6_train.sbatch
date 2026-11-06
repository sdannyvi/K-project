#!/usr/bin/env bash
#SBATCH --job-name=k-inner-v6-train
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
: "${RUN_DIR:?Set RUN_DIR to the scratch run directory}"
: "${DATA:?Set DATA to the vetted corpus JSONL path}"
arms=(ordinary boundary)
seeds=(20261106 20261107 20261108)
arm="${arms[$((SLURM_ARRAY_TASK_ID / 3))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
output="${RUN_DIR}/adapters/${arm}_seed${seed}"
mkdir -p "${RUN_DIR}/adapters"
python -u -m synthetic_preliminary.train_inner_voice_v6 \
  --data "${DATA}" --output "${output}" --arm "${arm}" --seed "${seed}" \
  --max-length 384 --rank 16 --gradient-accumulation 16
