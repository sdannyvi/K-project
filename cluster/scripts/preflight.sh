#!/usr/bin/env bash
set -euo pipefail

repo_dir="${K_PROJECT_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${repo_dir}"
# shellcheck disable=SC1091
source cluster/env.sh

required=(
  cluster/config/experiment_manifest.yaml
  paper/main.tex
  neural_poc/src/native_lora_cessation.py
  neural_poc/data/items.jsonl
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

command -v git >/dev/null
command -v python >/dev/null
command -v rsync >/dev/null
# The login node and the Claude Code sandbox have no GPU and may lack working
# Slurm client libraries; warn rather than fail there.
for tool in sbatch squeue nvidia-smi; do
  command -v "${tool}" >/dev/null || echo "WARNING: ${tool} not on PATH (fine on login/sandbox shell)"
done

[[ -n "${K_PROJECT_SCRATCH:-}" ]] || { echo "K_PROJECT_SCRATCH is not set" >&2; exit 1; }
[[ -n "${HF_HOME:-}" ]] || { echo "HF_HOME is not set" >&2; exit 1; }
[[ -n "${K_PROJECT_SCRATCH_QUOTA_GB:-}" ]] || { echo "K_PROJECT_SCRATCH_QUOTA_GB is not set" >&2; exit 1; }
mkdir -p "${K_PROJECT_SCRATCH}" "${HF_HOME}" slurm_logs runs

if [[ -n "${K_PROJECT_SECRETS_FILE:-}" ]]; then
  [[ -f "${K_PROJECT_SECRETS_FILE}" ]] || { echo "Secrets file not found" >&2; exit 1; }
  perms="$(stat -f '%Lp' "${K_PROJECT_SECRETS_FILE}" 2>/dev/null || stat -c '%a' "${K_PROJECT_SECRETS_FILE}")"
  [[ "${perms}" == "600" ]] || { echo "Secrets file must have mode 600" >&2; exit 1; }
fi

echo "Git commit: $(git rev-parse HEAD)"
python --version
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi
echo "Scratch: ${K_PROJECT_SCRATCH}"
echo "HF cache: ${HF_HOME}"
echo "Scratch quota (GB): ${K_PROJECT_SCRATCH_QUOTA_GB}"

bash cluster/scripts/secret_scan.sh

python -c "from pnas_pipeline.manifest import load_manifest, manifest_hash; print('Manifest sha256:', manifest_hash(load_manifest('cluster/config/experiment_manifest.yaml')))"
python -m pytest -q pnas_pipeline/tests
# Informational until calibration is complete; the launch gate itself is
# `python -m pnas_pipeline.budget` returning 0.
python -m pnas_pipeline.budget | tail -1 || true

if [[ -f assets/'The Finders PDF.pdf' ]]; then
  sha256sum assets/'The Finders PDF.pdf'
else
  echo "Private Finders book absent (allowed; see docs/private_sources.md)."
fi

echo "Preflight passed. Do not launch confirmation until the manifest is frozen."
