"""Signed gate-certificate validator (plan sections 4 and 6).

A certificate is a YAML/JSON file:

    gate: h0a_elicitation
    manifest_sha256: <sha of the expanded manifest>
    artifacts: {relative/path: <sha256>, ...}
    signatures:
      - {name: ..., role: PI, date: 2026-10-19}

The checker recomputes every artifact hash and requires the roles named for
the gate. Downstream waves are submitted only after it passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .io import sha256_file

REQUIRED_ROLES = {
    "compute_budget": {"PI", "compute_operator"},
    "h0a_elicitation": {"PI", "expert_1", "expert_2"},
    "protocol_freeze": {"PI"},
}
DEFAULT_ROLES = {"PI"}


def check(cert_path: str | Path, root: str | Path = ".", manifest_sha256: str | None = None) -> list[str]:
    cert = yaml.safe_load(Path(cert_path).read_text()) or {}
    problems = []
    for key in ("gate", "manifest_sha256", "artifacts", "signatures"):
        if key not in cert:
            problems.append(f"missing field {key}")
    if problems:
        return problems
    if manifest_sha256 and cert["manifest_sha256"] != manifest_sha256:
        problems.append("certificate signs a different manifest")
    for rel, digest in cert["artifacts"].items():
        path = Path(root) / rel
        if not path.exists():
            problems.append(f"artifact missing: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"artifact hash mismatch: {rel}")
    signed = {s.get("role") for s in cert["signatures"] if s.get("name") and s.get("date")}
    missing = REQUIRED_ROLES.get(cert["gate"], DEFAULT_ROLES) - signed
    if missing:
        problems.append(f"unsigned roles: {sorted(missing)}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a signed gate certificate.")
    parser.add_argument("certificate")
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    problems = check(args.certificate, args.root, args.manifest_sha256)
    for problem in problems:
        print(f"BLOCKED: {problem}")
    print("Gate: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
