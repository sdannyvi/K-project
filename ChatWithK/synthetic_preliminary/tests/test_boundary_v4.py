import json
from collections import Counter
from pathlib import Path

import torch

from synthetic_preliminary.build_boundary_v4 import build
from synthetic_preliminary.train_boundary_ablation import boundary_loss, eos_log_odds


def test_matched_corpus_has_disjoint_development_and_no_shared_opening_stops():
    source = Path(__file__).parents[1] / "data" / "edge_v2_corpus.jsonl"
    rows = build([json.loads(line) for line in source.open()], 20261105)
    assert Counter(row["split"] for row in rows) == {"train": 6000, "development": 800}
    train_sources = {r["source_id"] for r in rows if r["split"] == "train"}
    dev_sources = {r["source_id"] for r in rows if r["split"] == "development"}
    assert train_sources.isdisjoint(dev_sources)
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
        assert row["completion"].startswith(row["shared_prefix"])
        assert row["decision_prefix"].startswith(row["shared_prefix"])
    assert len(groups) == 1700
    for members in groups.values():
        assert len(members) == 4
        assert len({r["prompt"] for r in members}) == 1
        assert len({r["shared_prefix"] for r in members}) == 1
        assert sum(r["stop_at_decision"] for r in members) == 1
        stop = next(r for r in members if r["stop_at_decision"])
        assert stop["decision_prefix"] == stop["completion"]
        assert all(not r["completion"].startswith(stop["completion"]) for r in members if r is not stop)


def test_boundary_loss_prefers_eos_only_after_completed_claim():
    eos_id = 2
    batch = {
        "decision_index": torch.tensor([1, 1]),
        "shared_index": torch.tensor([0, 0]),
        "stop_at_decision": torch.tensor([1, 0]),
    }
    good = torch.tensor([
        [[6.0, 0.0, -6.0], [0.0, 0.0, 6.0]],
        [[6.0, 0.0, -6.0], [6.0, 0.0, -6.0]],
    ])
    bad = good.clone()
    bad[0, 0, 2] = 6.0  # stop inside the shared opening
    bad[1, 1, 2] = 6.0  # stop on the practical continuation
    assert boundary_loss(good, batch, eos_id) < boundary_loss(bad, batch, eos_id)
    assert eos_log_odds(good[0, 1], eos_id) > eos_log_odds(good[1, 1], eos_id)
