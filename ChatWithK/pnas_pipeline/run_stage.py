"""Idempotent stage dispatcher called by cluster/slurm/job_template.sbatch.

    python -m pnas_pipeline.run_stage --manifest M --stage S --run-id R \
        --output-dir $K_PROJECT_SCRATCH/runs/R/S [stage args...]

Exit codes: 0 done (or already complete), 2 bad manifest/prerequisite,
3 stage blocked, 99 stopped by USR1/TERM before completion (resubmit to resume).
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .io import atomic_write_json
from .manifest import ManifestError, load_manifest, manifest_hash, prerequisites
from .provenance.snapshot import write_run_identity
from .shards import is_complete, mark_complete
from .stages import SCIENTIFIC, UTILITY, StageBlocked

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Context:
    stage: str
    run_id: str
    output_dir: Path
    manifest: dict
    stage_args: list[str]
    repo: Path = REPO
    _stop: list = field(default_factory=lambda: [False])

    def stop_requested(self) -> bool:
        return self._stop[0]

    def log(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.stage}: {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args, stage_args = parser.parse_known_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"manifest invalid: {exc}", file=sys.stderr)
        return 2
    runner = UTILITY.get(args.stage) or SCIENTIFIC.get(args.stage)
    if runner is None:
        print(f"unknown stage {args.stage!r}; known: {sorted(UTILITY) + sorted(SCIENTIFIC)}", file=sys.stderr)
        return 2

    out = args.output_dir
    if is_complete(out):
        print(f"{args.stage} already complete in {out}; nothing to do")
        return 0

    run_root = out.parent
    if args.stage in SCIENTIFIC:
        if not manifest["confirmatory_frozen"]:
            print("note: manifest is not frozen; outputs are pilot-only", file=sys.stderr)
        missing = [dep for dep in prerequisites(manifest["stages"][args.stage])
                   if not is_complete(run_root / dep)]
        if missing:
            print(f"prerequisites not complete in {run_root}: {missing}", file=sys.stderr)
            return 2

    ctx = Context(args.stage, args.run_id, out, manifest, stage_args)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: ctx._stop.__setitem__(0, True))

    write_run_identity(run_root, REPO, manifest, manifest_hash(manifest))
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        metrics = runner(ctx) or {}
    except StageBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3
    if ctx.stop_requested():
        ctx.log("stopped before completion; resubmit with the same RUN_ID to resume")
        return 99
    metrics["wall_seconds"] = round(time.time() - started, 1)
    atomic_write_json(out / "metrics.json", metrics)
    mark_complete(out, ["metrics.json"], {"stage": args.stage, "run_id": args.run_id,
                                          "args": stage_args})
    ctx.log(f"complete: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
