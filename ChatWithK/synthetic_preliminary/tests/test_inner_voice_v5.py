"""Checks that the revised pilot corpus is paired and split by situation."""

from collections import defaultdict

from synthetic_preliminary.build_inner_voice_v5 import build


def test_paired_construct_and_splits():
    rows = build()
    groups = defaultdict(list)
    card_splits = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].append(row)
        card_splits[row["card_id"]].add(row["split"])
        assert row["full_candidate_text"].startswith(row["opening"] + " ")
        assert row["full_candidate_text"].startswith(row["target_text"])
        if row["stop_at_target_end"]:
            assert row["boundary_kind"] == "noninstrumental_rumination"
            assert row["cut_after_word"] == row["target_text"].split()[-1]
            assert len(row["target_text"]) < len(row["full_candidate_text"])
        else:
            assert row["target_text"] == row["full_candidate_text"]
    assert len(groups) == 160
    assert all(len(group) == 4 for group in groups.values())
    assert all(len(splits) == 1 for splits in card_splits.values())
    assert sum(len({r["card_id"] for r in g}) == 1 for g in groups.values()) == 160

