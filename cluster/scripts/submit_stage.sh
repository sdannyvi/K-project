#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: submit_stage.sh STAGE [dependency-job-id] [extra sbatch args...]}"
dependency="${2:-}"
shift || true
shift || true

dep_args=()
if [[ -n "${dependency}" ]]; then
  dep_args=(--dependency="afterok:${dependency}")
fi

export STAGE="${stage}"
sbatch --kill-on-invalid-dep=yes --export=ALL,STAGE="${stage}" "${dep_args[@]}" "$@" cluster/slurm/job_template.sbatch
