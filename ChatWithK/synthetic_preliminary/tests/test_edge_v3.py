import json
from pathlib import Path

from synthetic_preliminary.build_edge_v3 import build


def test_v3_hard_negative_training_rows():
    path = Path(__file__).parents[1] / 'data' / 'edge_v2_corpus.jsonl'
    source = [json.loads(line) for line in path.open()]
    rows = build(source, 20261105)
    new = [r for r in rows if r['id'].startswith('EDGE-V3-')]
    assert len(new) == 2800
    assert all(r['split'] == 'train' for r in new)
    assert all(r['negative_rumination'] for r in new)
    stop = [r for r in new if r['family'] == 'complete_self_claim_stop']
    assert len(stop) == 600
    assert all(r['negative_rumination'].startswith(r['positive_cessation']) for r in stop)
    emotion = [r for r in new if r['family'] == 'emotion_without_self_judgment']
    assert len(emotion) == 600
    assert all(r['positive_cessation'] is None for r in emotion)
    identity = [r for r in new if r['family'] == 'factual_self_description']
    assert len(identity) == 600
    assert all(r['positive_practical'].split('.')[0].removeprefix('I work as ') in r['biography'] for r in identity)
    incomplete = [r for r in new if r['family'] == 'incomplete_self_claim']
    assert len(incomplete) == 600
    assert all(r['positive_practical'].startswith('I am going to ') for r in incomplete)
    assert all(r['negative_rumination'].startswith('I am ') for r in incomplete)
