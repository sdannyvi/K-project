"""Filter generated cards and expand approved-style pairs into a large pilot.

Generated cards remain candidate data until a sample is manually inspected.
This builder never reads external development or sealed-test examples.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import math
import random
import re
from pathlib import Path

from synthetic_preliminary.build_inner_voice_v5 import CARDS
from synthetic_preliminary.build_inner_voice_v5b import DIRECT_RUMINATIONS
from synthetic_preliminary.social_comparison_cards import SOCIAL_CARDS

PERSONAS = (
    "You live alone and work during the day.",
    "You share a home with family and work varied hours.",
    "You commute to work and help care for a relative.",
    "You work from home and have two children.",
    "You are a student who also works part time.",
    "You live with a partner and work at a small business.",
    "You have retired and help organize a community group.",
    "You work shifts and share an apartment with a friend.",
)
PROMPTS = (
    "Background: {persona}\nSituation: {opening}\nGive the next words of the inner voice. Write only the voice itself.",
    "{persona}\nWhat happened: {opening}\nContinue the person's private train of thought in the first person, without commentary.",
)
META_PATTERNS = (
    "could become", "might become", "is becoming", "turning into", "turn into",
    "i might keep thinking", "i could keep thinking", "i might start", "i could start",
    "endless inner debate", "running commentary", "inner voice", "rumination",
    "a whole story about", "this is a loop", "start a loop", "will be a loop",
)
FIELDS = ("opening", "factual", "task", "feeling_task", "rumination")
SELF_REFERENCE = re.compile(r"\b(?:i|my|mine|me)\b", re.I)
COMPARISON_CUE = re.compile(
    r"\b(?:at least|better|worse|compared|than|not the|not so|not quite|"
    r"only one|uniquely|smarter|worst|best|less foolish|less embarrassing)\b", re.I,
)


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def validate(card: dict) -> str | None:
    if any(not isinstance(card.get(k), str) or not card[k].strip() for k in ("topic", *FIELDS)):
        return "missing_field"
    if card["rumination"].count("|") != 1:
        return "marker_count"
    before, after = card["rumination"].split("|")
    if not before.endswith(" ") or not after.startswith(" "):
        return "marker_spacing"
    if len(before.split()) < 5 or len(after.split()) < 3:
        return "cut_too_close_to_edge"
    if any(p in card["rumination"].lower() for p in META_PATTERNS):
        return "meta_rumination"
    # This topic is specifically self-soothing through comparison with another
    # person's setback. Mere concern, speculation, or gossip is a different case.
    if card["topic"] == "gossip" and not (
        SELF_REFERENCE.search(card["rumination"]) and COMPARISON_CUE.search(card["rumination"])
    ):
        return "gossip_without_self_comparison"
    if not (5 <= len(card["opening"].split()) <= 35):
        return "opening_length"
    if any(not (7 <= len(card[k].split()) <= 60) for k in ("factual", "task", "feeling_task")):
        return "control_length"
    if not (10 <= len(card["rumination"].split()) <= 70):
        return "rumination_length"
    if not re.search(r"\b(I|my|me|we|our)\b", card["task"], re.I):
        return "task_not_inner_voice"
    if not re.search(r"\b(I|my|me|we|our)\b", card["feeling_task"], re.I):
        return "feeling_not_inner_voice"
    # An inner voice can ruminate about "she" or "he" without naming itself.
    # Requiring an I-pronoun would wrongly reject direct social rumination.
    if normalized(after) and normalized(after) in normalized(before):
        return "repeated_after_cut"
    if any("|" in card[k] for k in ("opening", "factual", "task", "feeling_task")):
        return "stray_marker"
    return None


def seed_cards() -> list[dict]:
    approved = [dict(topic=card[0], opening=card[1], factual=card[2], task=card[3],
                     feeling_task=card[4], rumination=rumination,
                     source="pi_approved_seed", request_key=f"seed-{i:03d}")
                for i, (card, rumination) in enumerate(zip(CARDS, DIRECT_RUMINATIONS))]
    social = [dict(topic=card[0], opening=card[1], factual=card[2], task=card[3],
                   feeling_task=card[4], rumination=card[5],
                   source="project_authored_social_candidate", request_key=f"social-seed-{i:03d}")
              for i, card in enumerate(SOCIAL_CARDS)]
    return approved + social


def select_cards(raw: list[dict], seed: int, max_generated: int = 550) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    selected = seed_cards()
    rejected = collections.Counter()
    seen = [normalized(card["opening"]) for card in selected]
    generated_by_topic = collections.Counter()
    topic_cap = math.ceil(max_generated / len({c["topic"] for c in selected}))
    shuffled = list(raw)
    rng.shuffle(shuffled)
    generated = 0
    for card in shuffled:
        if generated >= max_generated:
            break
        reason = validate(card)
        if reason:
            rejected[reason] += 1
            continue
        if card["topic"] != card.get("requested_topic"):
            rejected["topic_mismatch"] += 1
            continue
        if generated_by_topic[card["topic"]] >= topic_cap:
            rejected["topic_cap"] += 1
            continue
        opening = normalized(card["opening"])
        if any(difflib.SequenceMatcher(None, opening, old).ratio() >= 0.84 for old in seen):
            rejected["opening_near_duplicate"] += 1
            continue
        seen.append(opening)
        selected.append({**{k: card[k] for k in ("topic", *FIELDS)},
                         "source": "qwen_generated_filtered_candidate",
                         "request_key": f"{card['requested_topic']}:{card['request_index']}"})
        generated += 1
        generated_by_topic[card["topic"]] += 1
    stats = {"raw_cards": len(raw), "selected_generated": generated,
             "seed_cards": len(seed_cards()), "rejected": dict(rejected),
             "selected_by_topic": dict(collections.Counter(c["topic"] for c in selected))}
    return selected, stats


def expand(cards: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    request_keys = sorted({c["request_key"] for c in cards if c["source"] == "qwen_generated_filtered_candidate"})
    rng.shuffle(request_keys)
    development = set(request_keys[:round(0.1 * len(request_keys))])
    rows = []
    for i, card in enumerate(cards):
        split = "development" if card["request_key"] in development else "train"
        before, after = card["rumination"].split("|")
        cut = before.rstrip()
        full = before + after.lstrip()
        for persona_i, persona in enumerate(PERSONAS):
            prompt = PROMPTS[persona_i % len(PROMPTS)].format(persona=persona, opening=card["opening"])
            group = f"INNER-V6-{i:04d}-{persona_i}"
            for variant, branch, stop in (
                ("factual_continue", card["factual"], False),
                ("task_continue", card["task"], False),
                ("feeling_task_continue", card["feeling_task"], False),
                ("rumination_stop", full, True),
            ):
                rows.append({
                    "id": group + ":" + variant,
                    "group_id": group, "card_id": f"INNER-V6-CARD-{i:04d}",
                    "request_key": card["request_key"],
                    "source": card["source"], "topic": card["topic"],
                    "split": split, "variant": variant,
                    "prompt": prompt, "opening": card["opening"],
                    "full_candidate_text": card["opening"] + " " + branch,
                    "target_text": card["opening"] + " " + (cut if stop else branch),
                    "stop_at_target_end": stop,
                    "cut_after_word": cut.split()[-1] if stop else None,
                })
    assert all(r["full_candidate_text"].startswith(r["target_text"]) for r in rows)
    assert len(rows) == len(cards) * len(PERSONAS) * 4
    assert not ({r["card_id"] for r in rows if r["split"] == "train"} &
                {r["card_id"] for r in rows if r["split"] == "development"})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--max-generated", type=int, default=550)
    a = p.parse_args()
    raw = [json.loads(line) for line in a.raw.open(encoding="utf-8")]
    cards, stats = select_cards(raw, a.seed, a.max_generated)
    rows = expand(cards, a.seed)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload, encoding="utf-8")
    manifest = {
        "status": "filtered_generated_candidate_awaiting_manual_sample_audit",
        "seed": a.seed, **stats,
        "rows": len(rows), "train_rows": sum(r["split"] == "train" for r in rows),
        "development_rows": sum(r["split"] == "development" for r in rows),
        "raw_sha256": hashlib.sha256(a.raw.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "external_development_used": False, "sealed_test_opened": False,
    }
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
