import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from pnas_pipeline import budget, gates, run_stage
from pnas_pipeline.io import sha256_file, write_once_json
from pnas_pipeline.manifest import ManifestError, expand, load_manifest, manifest_hash, validate
from pnas_pipeline.shards import is_complete, mark_complete

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "cluster/config/experiment_manifest.yaml"


@pytest.fixture
def manifest():
    return load_manifest(MANIFEST)


def test_repo_manifest_validates(manifest):
    assert "e4_native_adaptation" in manifest["stages"]


def test_cycle_and_unknown_prerequisite_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad["stages"]["h0_construct"]["prerequisite"] = "e5_transition_exploratory"
    with pytest.raises(ManifestError, match="cycle"):
        validate(bad)
    bad = copy.deepcopy(manifest)
    bad["stages"]["h0_construct"]["prerequisite"] = "nope"
    with pytest.raises(ManifestError, match="unknown prerequisite"):
        validate(bad)


def test_hash_is_key_order_independent(manifest):
    reordered = dict(reversed(list(manifest.items())))
    assert manifest_hash(reordered) == manifest_hash(manifest)
    changed = copy.deepcopy(manifest)
    changed["seed"] += 1
    assert manifest_hash(changed) != manifest_hash(manifest)


def test_frozen_manifest_refuses_placeholders(manifest):
    frozen = copy.deepcopy(manifest)
    frozen["confirmatory_frozen"] = True
    with pytest.raises(ManifestError, match="REQUIRED|unpinned"):
        expand(frozen, {})
    assert expand(manifest, {})["expanded_sha256"]


def test_write_once_refuses_changes(tmp_path):
    write_once_json(tmp_path / "m.json", {"a": 1})
    write_once_json(tmp_path / "m.json", {"a": 1})
    with pytest.raises(RuntimeError, match="Immutable"):
        write_once_json(tmp_path / "m.json", {"a": 2})


def test_shard_completion_detects_tampering(tmp_path):
    (tmp_path / "out.jsonl").write_text("x\n")
    assert not is_complete(tmp_path)
    mark_complete(tmp_path, ["out.jsonl"])
    assert is_complete(tmp_path)
    (tmp_path / "out.jsonl").write_text("y\n")
    assert not is_complete(tmp_path)


def _run(tmp_path, stage, *extra):
    return run_stage.main(["--manifest", str(MANIFEST), "--stage", stage, "--run-id", "t",
                           "--output-dir", str(tmp_path / "t" / stage), *extra])


def test_dry_run_is_idempotent_and_writes_identity(tmp_path, capsys):
    assert _run(tmp_path, "dry_run") == 0
    root = tmp_path / "t"
    for name in ("manifest.json", "git_commit.txt", "environment.txt", "data_hashes.json"):
        assert (root / name).exists(), name
    status = (root / "dry_run/status.json").read_text()
    assert _run(tmp_path, "dry_run") == 0
    assert "already complete" in capsys.readouterr().out
    assert (root / "dry_run/status.json").read_text() == status


def test_scientific_stage_checks_prerequisites_then_blocks(tmp_path):
    assert _run(tmp_path, "e3b_native_policy") == 2  # e3a missing
    assert _run(tmp_path, "h0_construct") == 3       # no prerequisite, but blocked
    assert _run(tmp_path, "not_a_stage") == 2


def _budget_csv(path, tps, hours):
    fields = ["stage", "model", "role", "source_scenarios", "streams_per_scenario", "conditions",
              "arms", "seeds", "completions", "mean_new_tokens", "measured_tokens_per_second",
              "gpu_count", "estimated_gpu_hours", "status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["E1", "m", "r", 600, 6, 1, 1, 1, 10, 150, tps, 1, hours, "ok"])))


def test_budget_formula_and_tbd(tmp_path):
    path = tmp_path / "b.csv"
    _budget_csv(path, "TBD", "TBD")
    assert any("TBD" in p for p in budget.check(path, 4, 42))
    hours = 600 * 6 * 10 * 150 / 100 / 3600  # 15.0
    _budget_csv(path, 100, hours)
    assert budget.check(path, 4, 42) == []
    _budget_csv(path, 100, 99)
    assert any("formula" in p for p in budget.check(path, 4, 42))
    assert any("ceiling" in p for p in budget.check(path, 1, 0.1))


def test_repo_budget_is_blocked_until_calibrated():
    assert budget.check(REPO / "cluster/COMPUTE_BUDGET.csv", 4, 42)


def test_gate_certificate(tmp_path):
    artifact = tmp_path / "a.json"
    artifact.write_text(json.dumps({"x": 1}))
    cert = {"gate": "compute_budget", "manifest_sha256": "abc",
            "artifacts": {"a.json": sha256_file(artifact)},
            "signatures": [{"name": "Dan", "role": "PI", "date": "2026-10-19"}]}
    path = tmp_path / "cert.yaml"
    path.write_text(yaml.safe_dump(cert))
    assert gates.check(path, tmp_path) == ["unsigned roles: ['compute_operator']"]
    cert["signatures"].append({"name": "Op", "role": "compute_operator", "date": "2026-10-19"})
    path.write_text(yaml.safe_dump(cert))
    assert gates.check(path, tmp_path, manifest_sha256="abc") == []
    assert gates.check(path, tmp_path, manifest_sha256="other") == ["certificate signs a different manifest"]
    artifact.write_text("tampered")
    assert gates.check(path, tmp_path) == ["artifact hash mismatch: a.json"]


def test_lora_targets_skip_vision_tower():
    import re
    from types import SimpleNamespace
    from pnas_pipeline.models import lora_targets
    text_only = SimpleNamespace(config=SimpleNamespace())
    assert lora_targets(text_only, ["q_proj"]) == ["q_proj"]
    mm = SimpleNamespace(config=SimpleNamespace(text_config=1, vision_config=1))
    pattern = re.compile(lora_targets(mm, ["q_proj", "v_proj"]))
    assert pattern.fullmatch("model.language_model.layers.3.self_attn.q_proj")
    assert not pattern.fullmatch("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")
    assert not pattern.fullmatch("model.language_model.layers.3.self_attn.q_proj.lora_A")


def test_model_sharding_is_rejected():
    from pnas_pipeline.models import ShardingForbidden, check_single_device
    check_single_device({"": 0})
    check_single_device(0)
    for bad in ("auto", "balanced", {"model.layers.0": 0, "lm_head": 1}):
        with pytest.raises(ShardingForbidden):
            check_single_device(bad)
