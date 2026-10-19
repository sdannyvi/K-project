# Source from every K-project batch script and interactive session on the BGU
# ECE HPC cluster (ece-hpc.bgu.ac.il):   source cluster/env.sh
# See cluster/CLUSTER.md for the rationale behind each value.

export K_PROJECT_REPO="${K_PROJECT_REPO:-/truenas/home/vilenchi/K-project}"
# Large artifacts (activations, checkpoints, generations). /truenas/projects has
# >100 TB free; home is shared with other projects, so keep heavy output here.
export K_PROJECT_SCRATCH="${K_PROJECT_SCRATCH:-/truenas/projects/vilenchik_shared/k-project}"
# Self-imposed budget, not a filesystem quota (none is enforced on this share).
export K_PROJECT_SCRATCH_QUOTA_GB="${K_PROJECT_SCRATCH_QUOTA_GB:-2000}"
# Group-shared Hugging Face cache (also used by the Holocaust project). Pinned,
# not defaulted: ~/.bashrc exports the read-only global cache, which sbatch
# would otherwise inherit.
export HF_HOME=/truenas/projects/vilenchik_shared/hf_cache
# Optional mode-600 file with API keys for closed-model E1 runs.
export K_PROJECT_SECRETS_FILE="${K_PROJECT_SECRETS_FILE:-}"

export TOKENIZERS_PARALLELISM=false
# GPU peer-to-peer is broken on hpc-vilenchik (2026-10-19): NCCL collectives hang
# with P2P on (job 10129) and direct torch cuda:i -> cuda:j copies return corrupted
# data with no error. Stage NCCL through host memory. This does NOT fix plain
# tensor copies, so never shard one model across GPUs (device_map="auto") here.
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source /truenas/home/vilenchi/miniforge3/etc/profile.d/conda.sh
conda activate kproject
