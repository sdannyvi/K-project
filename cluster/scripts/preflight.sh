#!/usr/bin/env bash
set -euo pipefail

repo_dir="${K_PROJECT_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${repo_dir}"

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
command -v sbatch >/dev/null
command -v squeue >/dev/null
command -v rsync >/dev/null
command -v nvidia-smi >/dev/null

[[ -n "${K_PROJECT_SCRATCH:-}" ]] || { echo "K_PROJECT_SCRATCH is not set" >&2; exit 1; }
[[ -n "${HF_HOME:-}" ]] || { echo "HF_HOME is not set" >&2; exit 1; }
[[ -n "${K_PROJECT_SCRATCH_QUOTA_GB:-}" ]] || { echo "K_PROJECT_SCRATCH_QUOTA_GB is not set" >&2; exit 1; }
mkdir -p "${K_PROJECT_SCRATCH}" "${HF_HOME}" logs runs

if [[ -n "${K_PROJECT_SECRETS_FILE:-}" ]]; then
  [[ -f "${K_PROJECT_SECRETS_FILE}" ]] || { echo "Secrets file not found" >&2; exit 1; }
  perms="$(stat -f '%Lp' "${K_PROJECT_SECRETS_FILE}" 2>/dev/null || stat -c '%a' "${K_PROJECT_SECRETS_FILE}")"
  [[ "${perms}" == "600" ]] || { echo "Secrets file must have mode 600" >&2; exit 1; }
fi

echo "Git commit: $(git rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Scratch: ${K_PROJECT_SCRATCH}"
echo "HF cache: ${HF_HOME}"
echo "Scratch quota (GB): ${K_PROJECT_SCRATCH_QUOTA_GB}"

bash cluster/scripts/secret_scan.sh

if [[ -f assets/'The Finders PDF.pdf' ]]; then
  shasum -a 256 assets/'The Finders PDF.pdf'
else
  echo "Private Finders book absent (allowed; see docs/private_sources.md)."
fi

echo "Preflight passed. Do not launch confirmation until the manifest is frozen."
