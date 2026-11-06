#!/usr/bin/env bash
#SBATCH --job-name=k-inner-v6-smoke
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
out="${K_PROJECT_SCRATCH}/runs/inner-voice-v6-smoke-${SLURM_JOB_ID}"
python -u -m synthetic_preliminary.train_inner_voice_v6 \
  --data synthetic_preliminary/data/inner_voice_v5b_pilot.jsonl \
  --output "${out}/adapter" --arm boundary --seed 20261106 --limit 16 \
  --max-length 384 --rank 16 --gradient-accumulation 16
