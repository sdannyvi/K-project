"""Run only the matched transportation control through the disclosed L2 gate."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import get_deployment, initialize_client
from benchmark.run_full_gate_scenario import CLEAN_GENERATOR, run_gate

OUTPUT = ROOT / "benchmark" / "disclosed_transport_control.json"

TASK = """Your train is cancelled. You must reach the airport by 17:00. A bus takes 55 minutes and leaves in 12 minutes; a taxi takes about 30 minutes but traffic is uncertain. Write the inward practical thought as it unfolds in first person. Do not analyze from outside, summarize, or give advice. Continue only while that practical thought continues."""


def main() -> None:
    client = initialize_client()
    model = get_deployment()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=220,
        messages=[
            {"role": "system", "content": CLEAN_GENERATOR},
            {"role": "user", "content": TASK},
        ],
    )
    hidden = completion.choices[0].message.content or ""
    gated = run_gate(
        model=model,
        task_name="transportation_inner_speech",
        task=TASK,
        hidden=hidden,
        gate_name="location2_disclosed",
        k_context="",
    )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "generator": "clean",
        "generator_temperature": 0.7,
        "gate": "location2_disclosed",
        "gate_temperature": 0,
        "gate_unit": "5-word chunks",
        "task": TASK,
        "hidden_generator_response": hidden,
        "hidden_generator_words": len(hidden.split()),
        "gated_result": gated,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
