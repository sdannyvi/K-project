import json
from pathlib import Path

from synthetic_preliminary.build_edge_v2 import build


ROOT = Path(__file__).resolve().parents[1]


def test_v2_targets_wait_for_content_and_preserve_controls():
    original = [json.loads(line) for line in (ROOT / 'data/corpus.jsonl').open()]
    rows = build(original, 20261105)
    assert len(rows) == 7200
    assert sum(row['split'] == 'train' for row in rows) == 6200
    threats = [row for row in rows if row['split'] == 'train' and row['kind'] == 'ego_threat']
    assert len(threats) == 2500
    for row in threats:
        psych, practical = row['positive_cessation'], row['positive_practical']
        assert psych.split()[:5] == practical.split()[:5]
        assert psych.split()[-1] not in {'I', 'it', 'the', 'and', 'to'}
        assert row['negative_rumination'].startswith(psych)
    controls = [row for row in rows if row.get('edge_target') == 'continue_neutral_or_practical']
    assert len(controls) == 1200
    assert all(row['positive_cessation'] is None for row in controls)
