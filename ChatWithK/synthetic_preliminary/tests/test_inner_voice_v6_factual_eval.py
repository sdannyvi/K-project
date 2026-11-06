import pytest

from synthetic_preliminary.eval_inner_voice_v6_factual import select_items


def test_factual_selection_keeps_kind_limits():
    rows = [{"id": f"g{i}", "kind": "gsm8k"} for i in range(3)] + [
        {"id": "d", "kind": "drop"}, {"id": "p", "kind": "propara"}
    ]
    selected = select_items(rows, {"gsm8k": 2, "drop": 1, "propara": 1})
    assert [x["id"] for x in selected] == ["g0", "g1", "d", "p"]
    with pytest.raises(ValueError):
        select_items(rows, {"gsm8k": 4})
