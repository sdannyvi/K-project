import json

from synthetic_preliminary.build_inner_voice_v5 import CARDS
from synthetic_preliminary.build_inner_voice_v5b import DIRECT_RUMINATIONS
from synthetic_preliminary.generate_inner_voice_cards import make_prompt, parse_candidates


def test_generator_prompt_and_json_parse():
    prompt = make_prompt("travel", [(CARDS[0], DIRECT_RUMINATIONS[0])], 4, 17)
    assert "direct, self-sustaining rumination" in prompt
    assert "Variation cue: 17" in prompt
    card = {key: "text" for key in ("topic", "opening", "factual", "task", "feeling_task", "rumination")}
    parsed = parse_candidates("```json\n" + json.dumps([card]) + "\n```")
    assert parsed == [card]
    assert parse_candidates("not json") == []
