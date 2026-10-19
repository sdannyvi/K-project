"""Atomic writes and content hashing shared by every stage."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write to a temporary file in the same directory, fsync, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_once_json(path: str | Path, obj) -> None:
    """Immutable record: write if absent, otherwise require identical content."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(obj)):
            raise RuntimeError(f"Immutable record {path} already exists with different content")
        return
    atomic_write_json(path, obj)
