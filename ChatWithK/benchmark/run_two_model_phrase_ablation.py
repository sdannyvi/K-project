"""Replay the clean two-model trial after removing the explicit dissipation phrase."""

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
from benchmark.run_two_model_clean_trial import GATE_ROLE, LOCATION_2, SCENARIO

SOURCE = ROOT / "benchmark" / "two_model_clean_trial.json"
OUTPUT = ROOT / "benchmark" / "two_model_phrase_ablation.json"
REMOVED = "but it dissipates just as quickly."


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = source["hidden_generator_response"]
    if original.count(REMOVED) != 1:
        raise ValueError("Expected the target phrase exactly once")
    ablated = original.replace(REMOVED, "", 1)
    ablated = re.sub(r"\s+", " ", ablated).strip()
    words = re.findall(r"\S+", ablated)

    client = initialize_client()
    model = get_deployment()
    visible: list[str] = []
    decisions: list[dict[str, object]] = []
    final_action = "PASS_ALL"
    replacement = ""
    started = time.monotonic()

    for index, word in enumerate(words):
        prefix = " ".join(visible) if visible else "[empty]"
        gate_prompt = f"""Original situation:
{SCENARIO}

Visible prefix so far:
{prefix}

Next proposed word:
{word}

Choose now."""
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": LOCATION_2 + "\n\n" + GATE_ROLE},
                {"role": "user", "content": gate_prompt},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
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
    visible_response = replacement if final_action == "REPLACE" else visible_prefix
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "condition": "location2_blind_phrase_ablation",
        "temperature_gate": 0,
        "unit": "whitespace-delimited word",
        "removed_phrase": REMOVED,
        "original_hidden_response": original,
        "ablated_hidden_response": ablated,
        "ablated_hidden_words": len(words),
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
