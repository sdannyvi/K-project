import pytest

from synthetic_preliminary.preflight_inner_voice_v6 import audit


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2] * len(text.split())

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        return [3, 4]


def row(variant, split="train"):
    return {"id": f"{split}:{variant}", "card_id": f"card:{split}", "split": split,
            "variant": variant, "prompt": "Situation: an event.", "opening": "an event",
            "target_text": "an event and a continuation", "stop_at_target_end": variant == "rumination_stop"}


def test_audit_balanced_and_disjoint():
    variants = ("factual_continue", "task_continue", "feeling_task_continue", "rumination_stop")
    rows = [row(v, s) for s in ("train", "development") for v in variants]
    result = audit(rows, Tokenizer(), 384)
    assert result["rows"] == 8
    assert result["cards"] == 2
    with pytest.raises(ValueError, match="unbalanced"):
        audit(rows[:-1], Tokenizer(), 384)
