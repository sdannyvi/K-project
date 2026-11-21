#!/usr/bin/env bash
#SBATCH --job-name=k-external-summary
#SBATCH --partition=CPUonly
#SBATCH --gres=none
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
run_root="${K_PROJECT_SCRATCH}/runs/external-voice-20261121"
python -u -m synthetic_preliminary.summarize_external_voice \
  --run-root "${run_root}" \
  --source "${run_root}/input/external_voice_items.json"
