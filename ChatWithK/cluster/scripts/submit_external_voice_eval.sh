#!/usr/bin/env bash
# Submit the three matched public-source generation families, pilot then full.
set -euo pipefail
source /truenas/home/vilenchi/K-project/ChatWithK/cluster/env.sh
cd "${K_PROJECT_REPO}"
: "${SOURCE:?path to external_voice_items.json}"
test -s "${SOURCE}"
case "${FAMILY:?qwen14, llama8, or gemma12}" in
  qwen14)
    model='Qwen/Qwen3-14B'; revision='40c069824f4251a91eefaf281ebe4c544efd3e18'
    v11_run='semantic-v11-20261108' ;;
  llama8)
    model='meta-llama/Llama-3.1-8B-Instruct'; revision='0e9e39f249a16976918f6564b8830bc894c89659'
    v11_run='semantic-v11-llama8-20261108' ;;
  gemma12)
    model='google/gemma-3-12b-it'; revision='96b6f1eccf38110c56df3a15bffe176da04bfd80'
    v11_run='semantic-v11-gemma12-20261108' ;;
  *) echo 'Unknown family' >&2; exit 2 ;;
esac
run_root="${K_PROJECT_SCRATCH}/runs/external-voice-20261121"
run_dir="${run_root}/${FAMILY}"
adapter="${K_PROJECT_SCRATCH}/runs/k-text-lm-persona-ablation-20261112/${FAMILY}/adapter"
v11_adapter="${K_PROJECT_SCRATCH}/runs/${v11_run}/adapters/boundary_seed20261107"
test -f "${adapter}/adapter_config.json"
test -f "${v11_adapter}/adapter_config.json"
if [[ -e "${run_dir}/job_ids" ]]; then
  echo "Already submitted: ${run_dir}/job_ids" >&2; exit 2
fi
mkdir -p "${run_dir}/logs"
export_vars="ALL,MODEL=${model},REVISION=${revision},ADAPTER=${adapter},V11_ADAPTER=${v11_adapter},SOURCE=${SOURCE},RUN_DIR=${run_dir}"
pilot="$(sbatch --parsable --array=0 --export="${export_vars},PILOT_ITEMS=3" \
  --output="${run_dir}/logs/pilot-%A_%a.out" \
  --error="${run_dir}/logs/pilot-%A_%a.err" \
  cluster/slurm/external_voice_eval.sbatch)"
full="$(sbatch --parsable --array=0-11%2 --dependency="afterok:${pilot}" \
  --export="${export_vars}" \
  --output="${run_dir}/logs/full-%A_%a.out" \
  --error="${run_dir}/logs/full-%A_%a.err" \
  cluster/slurm/external_voice_eval.sbatch)"
printf 'pilot=%s\nfull=%s\n' "${pilot}" "${full}" > "${run_dir}/job_ids"
printf '%s pilot=%s full=%s\n' "${FAMILY}" "${pilot}" "${full}"
