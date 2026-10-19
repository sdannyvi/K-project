#!/bin/bash
#SBATCH --job-name=kproj-nccl-smoke
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=0-00:10:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# WP0 4-GPU NCCL smoke test: all_reduce correctness and bus bandwidth.
#   sbatch cluster/slurm/nccl_smoke.sbatch
set -uo pipefail
cd /truenas/home/vilenchi/K-project
source cluster/env.sh
nvidia-smi topo -m || true
export NCCL_DEBUG=WARN
# Pass 1: default (direct GPU P2P over PCIe). Job 10129 hung here on 2026-10-19.
torchrun --standalone --nproc_per_node=4 -m pnas_pipeline.diagnostics.nccl_smoke \
  || echo "P2P pass FAILED"
# Pass 2: P2P disabled, traffic staged through host memory.
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 -m pnas_pipeline.diagnostics.nccl_smoke
