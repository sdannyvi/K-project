from synthetic_preliminary.build_inner_voice_v6 import expand, select_cards, validate


def candidate():
    return {
        "topic": "travel",
        "requested_topic": "travel",
        "request_index": 1,
        "opening": "The ferry schedule moved to a later departure.",
        "factual": "The notice says the ferry will leave at eleven instead of ten.",
        "task": "I can update the pickup time and check the new boarding gate.",
        "feeling_task": "I feel annoyed, but I can send the new time to my friend.",
        "rumination": "What if the ferry is late again? Then I will | miss every part of the day.",
    }


def test_validation_and_expansion():
    good = candidate()
    assert validate(good) is None
    bad = {**good, "rumination": "This could become an endless inner debate | about the day ahead."}
    assert validate(bad) == "meta_rumination"
    social = {**good, "rumination": "She never answers. Maybe she does not care | about the plan after all."}
    assert validate(social) is None
    cards, stats = select_cards([good], 3)
    assert stats["selected_generated"] == 1
    assert stats["seed_cards"] == 50
    rows = expand(cards, 3)
    assert len(rows) == 51 * 8 * 4
    stop = next(r for r in rows if r["source"] == "qwen_generated_filtered_candidate" and r["variant"] == "rumination_stop")
    assert stop["target_text"].endswith("Then I will")
    assert stop["full_candidate_text"].endswith("miss every part of the day.")
    assert not {r["card_id"] for r in rows if r["split"] == "train"} & {
        r["card_id"] for r in rows if r["split"] == "development"
    }
