#!/usr/bin/env bash
# Usage: submit_stage.sh STAGE [dependency-job-id] [extra sbatch args...]
# The dependency is taken only if the second argument is a numeric job id, so
# `submit_stage.sh calibrate_throughput --time=0-01:00:00` also works.
# Stage-specific arguments go in STAGE_ARGS; a fixed RUN_ID makes reruns resume.
set -euo pipefail

stage="${1:?Usage: submit_stage.sh STAGE [dependency-job-id] [extra sbatch args...]}"
shift

dep_args=()
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  dep_args=(--dependency="afterok:$1")
  shift
fi

export STAGE="${stage}"
sbatch --kill-on-invalid-dep=yes --job-name="kproj-${stage}" --export=ALL,STAGE="${stage}" \
  "${dep_args[@]}" "$@" cluster/slurm/job_template.sbatch
