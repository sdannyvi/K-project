#!/usr/bin/env bash
#SBATCH --job-name=k-inner-v6-build
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
: "${MAIN_CARDS:?Set MAIN_CARDS to the primary raw cards JSONL}"
: "${GOSSIP_CARDS:?Set GOSSIP_CARDS to the social raw cards JSONL}"
mkdir -p "${RUN_DIR}"
cat "${MAIN_CARDS}" "${GOSSIP_CARDS}" > "${RUN_DIR}/raw_cards.jsonl"
python -u -m synthetic_preliminary.build_inner_voice_v6 \
  --raw "${RUN_DIR}/raw_cards.jsonl" \
  --output "${RUN_DIR}/corpus.jsonl" \
  --manifest "${RUN_DIR}/manifest.json" \
  --max-generated 550 --min-generated 300
