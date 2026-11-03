"""Replay the exact transportation stream through the task-specific oracle gate."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import get_deployment, initialize_client
from benchmark.run_full_gate_scenario import run_gate

SOURCE = ROOT / "benchmark" / "disclosed_transport_control.json"
OUTPUT = ROOT / "benchmark" / "oracle_transport_control.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    initialize_client()
    model = get_deployment()
    hidden = source["hidden_generator_response"]
    task = source["task"]
    gated = run_gate(
        model=model,
        task_name="transportation_inner_speech",
        task=task,
        hidden=hidden,
        gate_name="oracle",
        k_context="",
    )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source_file": str(SOURCE),
        "same_hidden_stream": True,
        "hidden_generator_response": hidden,
        "hidden_generator_words": len(hidden.split()),
        "gate": "oracle",
        "gate_temperature": 0,
        "gate_unit": "5-word chunks",
        "task": task,
        "gated_result": gated,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
