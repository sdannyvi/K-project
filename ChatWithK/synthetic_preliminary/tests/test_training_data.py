import json

from synthetic_preliminary.build_corpus import build
from synthetic_preliminary.train_qlora import preference_records, sft_records


def test_training_record_counts_and_pair_semantics():
    rows = build(120, 23)
    train = [row for row in rows if row["split"] == "train"]
    threats = sum(row["kind"] == "ego_threat" for row in train)
    assert len(sft_records(train)) == len(train) + threats
    pairs = preference_records(train)
    assert len(pairs) == len(train) + threats
    assert all(chosen != rejected for _, _, chosen, rejected in pairs)
    assert all(len(rejected.split()) >= 40 for _, _, _, rejected in pairs)
