#!/usr/bin/env bash
#SBATCH --job-name=kproj-failure
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
failed_job="${FAILED_JOB_ID:?Set FAILED_JOB_ID}"
reason="$(sacct -j "${failed_job}" --format=JobID,State,ExitCode,Elapsed -n -P 2>/dev/null || true)"
mkdir -p logs
printf '%s\n' "${reason}" > "logs/failed-${failed_job}.txt"
echo "Upstream job ${failed_job} failed. Dependents must not be released."
