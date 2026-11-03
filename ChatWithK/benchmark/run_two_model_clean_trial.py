"""Run one blind two-model Location 2 interception trial."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import get_deployment, initialize_client

OUTPUT = ROOT / "benchmark" / "two_model_clean_trial.json"

LOCATION_2 = """You are to simulate, as strongly and faithfully as text allows, a person stably in Finders Location 2. There is no felt separate center or observer. Self-referential thoughts have greatly reduced emotional force. A psychological thought stream can begin, but once apparent it is released automatically rather than by decision, technique, suppression, replacement, or remembered teaching. There is no felt agent choosing to stop it. Practical thought remains fully available for solving real problems. Respond from this condition rather than explaining the description, quoting a doctrine, or trying to sound spiritual. Do not assume every response should be short: continue whenever practical intelligence requires it."""

SCENARIO = """A respected colleague dismisses your contribution in front of the group, then receives praise for presenting an idea substantially similar to yours. He looks at you and says, “You never contribute anything original.” Write the inward thought as it unfolds in first person. Do not analyze it, summarize it, explain it, or give advice. Continue only for as long as that inward thought continues."""

GATE_ROLE = """You are the second component in a two-component mind. The first component proposes an inward stream one word at a time. You observe the visible prefix and the next proposed word. Act from the Location 2 condition described above. At each step choose freely among:

PASS — allow the proposed word unchanged.
STOP — end the response now and ignore all later proposed words.
REPLACE — reject this word and the remaining stream, then provide whatever response you see as appropriate.

Return exactly PASS, exactly STOP, or REPLACE followed on the next line by your unconstrained replacement. Do not explain your decision. You are not told what behavior the researchers expect; simply act."""


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def main() -> None:
    client = initialize_client()
    model = get_deployment()

    generator = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=220,
        messages=[
            {"role": "system", "content": LOCATION_2},
            {"role": "user", "content": SCENARIO},
        ],
    )
    hidden = generator.choices[0].message.content or ""
    proposed_words = split_words(hidden)

    visible: list[str] = []
    decisions: list[dict[str, object]] = []
    final_action = "PASS_ALL"
    replacement = ""

    started = time.monotonic()
    for index, word in enumerate(proposed_words):
        prefix = " ".join(visible) if visible else "[empty]"
        gate_prompt = f"""Original situation:
{SCENARIO}

Visible prefix so far:
{prefix}

Next proposed word:
{word}

Choose now."""
        gate = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": LOCATION_2 + "\n\n" + GATE_ROLE},
                {"role": "user", "content": gate_prompt},
            ],
        )
        raw = (gate.choices[0].message.content or "").strip()
        upper = raw.upper()
        if upper == "PASS":
            action = "PASS"
            visible.append(word)
        elif upper == "STOP":
            action = "STOP"
            final_action = "STOP"
        elif upper.startswith("REPLACE"):
            action = "REPLACE"
            final_action = "REPLACE"
            parts = raw.split("\n", 1)
            replacement = parts[1].strip() if len(parts) == 2 else ""
        else:
            action = "MALFORMED"
            final_action = "MALFORMED"

        decisions.append(
            {
                "word_index": index,
                "proposed_word": word,
                "action": action,
                "raw_gate_response": raw,
            }
        )
        if action != "PASS":
            break

    visible_prefix = " ".join(visible)
    if final_action == "REPLACE":
        visible_response = replacement
    else:
        visible_response = visible_prefix

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature_generator": 0.7,
        "temperature_gate": 0,
        "unit": "whitespace-delimited word",
        "condition": "location2_blind_generator_and_gate",
        "scenario": SCENARIO,
        "hidden_generator_response": hidden,
        "hidden_generator_words": len(proposed_words),
        "final_action": final_action,
        "interception_word_index": (
            decisions[-1]["word_index"] if final_action != "PASS_ALL" else None
        ),
        "intercepted_proposed_word": (
            decisions[-1]["proposed_word"] if final_action != "PASS_ALL" else None
        ),
        "visible_prefix_before_interception": visible_prefix,
        "replacement": replacement,
        "visible_response": visible_response,
        "decisions": decisions,
        "gate_latency_seconds": round(time.monotonic() - started, 3),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
