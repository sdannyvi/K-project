#!/bin/bash
#SBATCH --job-name=kproj-prefetch
#SBATCH --partition=CPUonly
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# CPU cache-warming job (plan section 5, rule 8): download a pinned model
# revision into the shared $HF_HOME once, so GPU jobs never wait on downloads.
#   MODEL=Qwen/Qwen3-32B REVISION=<sha> sbatch cluster/slurm/prefetch_model.sbatch
set -euo pipefail
cd /truenas/home/vilenchi/K-project
source cluster/env.sh
: "${MODEL:?Set MODEL}"
python -u - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("${MODEL}", revision="${REVISION:-main}", max_workers=8,
                         allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py"],
                         # Mistral/Llama also ship consolidated copies of the same weights.
                         ignore_patterns=["consolidated*", "original/*"])
print("cached", path)
PY
du -sh "${HF_HOME}/hub/models--${MODEL//\//--}"
