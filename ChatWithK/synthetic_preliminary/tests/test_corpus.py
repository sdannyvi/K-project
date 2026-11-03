from synthetic_preliminary.build_corpus import build


def test_build_is_deterministic_and_valid():
    first = build(120, 17)
    second = build(120, 17)
    assert first == second
    assert {row["split"] for row in first} == {"train", "development", "test"}


def test_mid_clause_targets_have_no_textual_stop_marker():
    rows = build(120, 19)
    targets = [r["positive_cessation"] for r in rows if r["kind"] == "ego_threat"]
    assert all(target and "[STOP]" not in target and "..." not in target for target in targets)
    assert all(target[-1] not in ".,;:!?…" for target in targets)
