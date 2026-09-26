import json
from collections import Counter, defaultdict
from pathlib import Path


DATA = Path(__file__).parents[1] / "data" / "items.jsonl"


def test_matched_quartets_and_scenario_splits():
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    assert len(by_scenario) == 12
    for items in by_scenario.values():
        assert Counter(x["label"] for x in items) == Counter(
            {"psychological": 1, "daily": 1, "factual": 1, "descriptive": 1}
        )
        assert len({x["split"] for x in items}) == 1


def test_unique_ids():
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len({x["id"] for x in rows}) == len(rows)
