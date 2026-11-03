#!/usr/bin/env bash
#SBATCH --job-name=kproj-syn-qlora
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"

run_id="${RUN_ID:-synthetic-preliminary-${SLURM_JOB_ID}}"
run_dir="${K_PROJECT_SCRATCH}/runs/${run_id}"
mkdir -p "${run_dir}"

train_args=(
  --data synthetic_preliminary/data/corpus.jsonl
  --output "${run_dir}/adapters"
  --model Qwen/Qwen3-14B
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18
  --max-length 384 --rank 16 --batch-size 1 --gradient-accumulation 16
  --sft-epochs "${SFT_EPOCHS:-1}" --preference-epochs "${PREFERENCE_EPOCHS:-1}"
)
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  train_args+=(--limit "${TRAIN_LIMIT}")
fi
python -u -m synthetic_preliminary.train_qlora "${train_args[@]}"
