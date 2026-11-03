"""Manifest loading, schema validation, hashing and immutable expansion."""
from __future__ import annotations

from pathlib import Path

import yaml

from .io import canonical_json, sha256_bytes

PLACEHOLDER = "REQUIRED_BEFORE_LAUNCH"
REQUIRED_TOP = ("project", "schema_version", "confirmatory_frozen", "seed",
                "cluster", "paths", "models", "stages", "hard_gates")
RESOURCES = {"cpu", "cpu_api", "cpu_human", "gpu", "gpu_or_api"}


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict:
    manifest = yaml.safe_load(Path(path).read_text())
    validate(manifest)
    return manifest


def prerequisites(stage: dict) -> list[str]:
    prereq = stage.get("prerequisite", "none")
    if prereq in (None, "none"):
        return []
    return [prereq] if isinstance(prereq, str) else list(prereq)


def validate(manifest: dict) -> None:
    missing = [key for key in REQUIRED_TOP if key not in manifest]
    if missing:
        raise ManifestError(f"Manifest missing keys: {missing}")
    if not isinstance(manifest["confirmatory_frozen"], bool):
        raise ManifestError("confirmatory_frozen must be a boolean")
    stages = manifest["stages"]
    for name, stage in stages.items():
        if stage.get("resource") not in RESOURCES:
            raise ManifestError(f"Stage {name}: unknown resource {stage.get('resource')!r}")
        if not stage.get("outputs"):
            raise ManifestError(f"Stage {name}: declares no outputs")
        for dep in prerequisites(stage):
            if dep not in stages:
                raise ManifestError(f"Stage {name}: unknown prerequisite {dep!r}")
    topological_order(stages)


def topological_order(stages: dict) -> list[str]:
    order, state = [], {}

    def visit(name, path):
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise ManifestError(f"Stage dependency cycle: {' -> '.join(path + [name])}")
        state[name] = "visiting"
        for dep in prerequisites(stages[name]):
            visit(dep, path + [name])
        state[name] = "done"
        order.append(name)

    for name in stages:
        visit(name, [])
    return order


def manifest_hash(manifest: dict) -> str:
    return sha256_bytes(canonical_json(manifest))


def unresolved_placeholders(obj, prefix="") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found += unresolved_placeholders(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found += unresolved_placeholders(value, f"{prefix}[{i}]")
    elif obj == PLACEHOLDER:
        found.append(prefix)
    return found


def expand(manifest: dict, model_revisions: dict[str, str]) -> dict:
    """Immutable expanded manifest: pins every open model to a commit SHA.

    Confirmatory expansion refuses any unresolved placeholder or unpinned model.
    """
    open_models = sorted({m for group, ids in manifest["models"].items()
                          if group not in ("negative_closed", "required_revisions")
                          for m in ids})
    unpinned = [m for m in open_models if not model_revisions.get(m)]
    expanded = {"source_manifest": manifest,
                "source_manifest_sha256": manifest_hash(manifest),
                "model_revisions": {m: model_revisions.get(m) for m in open_models}}
    if manifest["confirmatory_frozen"]:
        problems = unresolved_placeholders(manifest) + [f"unpinned model {m}" for m in unpinned]
        if problems:
            raise ManifestError(f"Cannot expand a confirmatory manifest: {problems}")
    expanded["expanded_sha256"] = manifest_hash(expanded)
    return expanded
