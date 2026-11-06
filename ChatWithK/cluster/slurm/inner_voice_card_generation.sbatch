#!/usr/bin/env bash
#SBATCH --job-name=k-inner-cards
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=0-10:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
run_dir="${K_PROJECT_SCRATCH}/runs/inner-voice-v6-${SLURM_JOB_ID}"
mkdir -p "${run_dir}"
python -u -m synthetic_preliminary.generate_inner_voice_cards \
  --output "${run_dir}/raw_cards.jsonl" \
  --metadata "${run_dir}/generation_metadata.json" \
  --requests-per-topic "${REQUESTS_PER_TOPIC:-25}" \
  --cards-per-request 4
