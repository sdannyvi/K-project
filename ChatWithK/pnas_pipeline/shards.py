"""Shard completion records: a shard is complete only if status.json validates."""
from __future__ import annotations

import json
from pathlib import Path

from .io import atomic_write_json, sha256_file

STATUS = "status.json"


def mark_complete(shard_dir: str | Path, outputs: list[str], extra: dict | None = None) -> dict:
    shard_dir = Path(shard_dir)
    status = {"complete": True,
              "outputs": {name: sha256_file(shard_dir / name) for name in sorted(outputs)},
              **(extra or {})}
    atomic_write_json(shard_dir / STATUS, status)
    return status


def is_complete(shard_dir: str | Path) -> bool:
    path = Path(shard_dir) / STATUS
    if not path.exists():
        return False
    try:
        status = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if not status.get("complete"):
        return False
    for name, digest in status.get("outputs", {}).items():
        target = Path(shard_dir) / name
        if not target.exists() or sha256_file(target) != digest:
            return False
    return True
