#!/bin/bash
#SBATCH --job-name=kproj-poc-qwen3b
#SBATCH --account=research_vilenchik
#SBATCH --partition=vilenchik_part
#SBATCH --qos=vilenchik_owned
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Port of the Mac (MPS) Qwen2.5-3B pilot to one A5000. Reruns the existing
# neural_poc pipeline end to end and records wall time per step, which gives
# the first measured throughput rows for cluster/COMPUTE_BUDGET.csv (WP0).
#
# Results go to $K_PROJECT_SCRATCH/poc_qwen3b/<jobid>/, never to the versioned
# neural_poc/results/ (the committed Mac evidence). Submit from the repo root:
#   mkdir -p slurm_logs && sbatch cluster/slurm/poc_qwen3b.sbatch
# Optional: STEPS="smoke collect" sbatch ... to run a subset.
# Resume:   K_POC_RESULTS_DIR=$K_PROJECT_SCRATCH/poc_qwen3b/<jobid> STEPS="native_lora" sbatch ...

set -euo pipefail
cd /truenas/home/vilenchi/K-project
source cluster/env.sh

# Preset K_POC_RESULTS_DIR to resume steps into an earlier job's directory.
export K_POC_RESULTS_DIR="${K_POC_RESULTS_DIR:-${K_PROJECT_SCRATCH}/poc_qwen3b/${SLURM_JOB_ID}}"
mkdir -p "${K_POC_RESULTS_DIR}"
CFG=configs/poc_qwen3b_cuda.yaml
STEPS="${STEPS:-smoke collect fit intervene evaluate eos_gate native_lora}"

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

cd neural_poc
timing="${K_POC_RESULTS_DIR}/step_timing.tsv"
run() {
  local name=$1; shift
  local t0=$SECONDS
  echo "=== ${name}: $*"
  python -u "$@"
  printf '%s\t%s\n' "${name}" "$((SECONDS - t0))" | tee -a "${timing}"
}

for step in ${STEPS}; do
  case ${step} in
    smoke)       run smoke       src/smoke_test.py --config ${CFG} ;;
    collect)     run collect     src/collect_activations.py --config ${CFG} ;;
    fit)         run fit         src/fit_direction.py --config ${CFG} ;;
    intervene)   run intervene   src/generate_intervened.py --config ${CFG} ;;
    evaluate)    run evaluate    src/evaluate.py ;;
    eos_gate)    run eos_gate    src/eos_gate.py --config ${CFG} \
                   --eos-boost 100 --min-tokens 8 --max-tokens 64 ;;
    native_lora) run native_lora src/native_lora_cessation.py --config ${CFG} \
                   --epochs 1 --lr 0.0005 --rank 2 --alpha 4 --start-layer 34 \
                   --max-new-tokens 64 --max-training-examples 12 ;;
    *) echo "unknown step ${step}" >&2; exit 2 ;;
  esac
done

echo "Done. Results: ${K_POC_RESULTS_DIR}"
