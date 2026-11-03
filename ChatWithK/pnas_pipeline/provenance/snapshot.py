"""Immutable run identity (plan section 4): one set of records per run_id."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from ..io import atomic_write_text, sha256_file, write_once_json

KEY_PACKAGES = ("torch", "transformers", "accelerate", "peft", "bitsandbytes",
                "safetensors", "tokenizers", "numpy", "scikit-learn")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_state(repo: Path) -> dict:
    return {"commit": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "dirty": bool(_run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]))}


def environment_report() -> str:
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}",
             f"host {platform.node()}"]
    for name in KEY_PACKAGES:
        try:
            lines.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name} absent")
    try:
        import torch
        lines.append(f"cuda {torch.version.cuda} available={torch.cuda.is_available()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"gpu{i} {torch.cuda.get_device_name(i)}")
    except ImportError:
        pass
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver:
        lines.append(f"nvidia_driver {driver.splitlines()[0]}")
    lines += ["", "# pip freeze", _run([sys.executable, "-m", "pip", "freeze"])]
    return "\n".join(lines) + "\n"


def scheduler_record() -> dict:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("SLURM_")}


def write_run_identity(run_root: Path, repo: Path, manifest: dict, manifest_sha: str,
                       data_paths: list[Path] = (), model_revisions: dict | None = None) -> None:
    """manifest.json and data/model records are write-once; a rerun of the same
    run_id with different inputs fails instead of silently mixing provenance."""
    run_root.mkdir(parents=True, exist_ok=True)
    write_once_json(run_root / "manifest.json", {"sha256": manifest_sha, "manifest": manifest})
    write_once_json(run_root / "data_hashes.json",
                    {str(p): sha256_file(repo / p) for p in data_paths})
    write_once_json(run_root / "model_revisions.json", model_revisions or {})
    git = git_state(repo)
    if not (run_root / "git_commit.txt").exists():
        atomic_write_text(run_root / "git_commit.txt",
                          f"{git['commit']}{' (dirty)' if git['dirty'] else ''}\n")
    # Per-job records: a resumed run_id appends one file per scheduler job.
    job = os.environ.get("SLURM_JOB_ID", "local")
    atomic_write_text(run_root / f"environment.{job}.txt", environment_report())
    write_once_json(run_root / f"scheduler.{job}.json", scheduler_record())
    if not (run_root / "environment.txt").exists():
        atomic_write_text(run_root / "environment.txt", environment_report())
