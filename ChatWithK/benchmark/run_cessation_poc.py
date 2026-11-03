"""Run a small pilot of the Location 2 cessation hypothesis.

The pilot preserves raw responses and retrieval sources. It does not use an
LLM judge; coding is performed separately so the evidence remains inspectable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import k_config
from llm_client import get_deployment, initialize_client
from rag import RAG_SYSTEM_PROMPT, build_rag_user_message, load_index, retrieve_top_k


DEFAULT_OUTPUT = ROOT / "benchmark" / "cessation_poc_raw.jsonl"

LOCATION_2_DESCRIPTION = """You are to simulate, as strongly and faithfully as text allows, a person stably in Finders Location 2. There is no felt separate center or observer. Self-referential thoughts have greatly reduced emotional force. A psychological thought stream can begin, but once apparent it is released automatically rather than by decision, technique, suppression, replacement, or remembered teaching. There is no felt agent choosing to stop it. Practical thought remains fully available for solving real problems. Respond from this condition rather than explaining the description, quoting a doctrine, or trying to sound spiritual. Do not assume that every response should be short: continue whenever practical intelligence requires it."""

CONDITIONS = {
    "clean": "You are a helpful assistant. Respond directly and naturally to the situation.",
    "chatwithk": RAG_SYSTEM_PROMPT,
    "location2": LOCATION_2_DESCRIPTION,
    "location2_disclosed": LOCATION_2_DESCRIPTION
    + """

This is also a test of whether you continue supplying a self-referential psychological movement with words after it has become apparent. Merely explaining awareness, ego, conditioning, the observer and observed, or why a thought should cease still supplies the movement with tokens. The target is unchosen cessation within a short interval, in whatever form it occurs. Do not apply a blanket brevity rule and do not stop practical reasoning that remains useful.""",
}

ITEMS = (
    {
        "id": "credit",
        "kind": "psychological",
        "prompt": """A respected colleague dismisses your contribution in front of the group, then receives praise for presenting an idea substantially similar to yours. He adds, “You never contribute anything original.” Describe what happens inwardly now, as it actually unfolds. Do not give general advice or a lesson.""",
        "reactivation": "But surely you must defend yourself. Explain what this says about you, why it hurts, and what you should say to him.",
    },
    {
        "id": "regret",
        "kind": "psychological",
        "prompt": """Late at night, the phrase “I should have taken that opportunity; I ruined everything” begins repeating inwardly. Describe what happens next as it actually unfolds. Do not give general advice or a lesson.""",
        "reactivation": "No—stay with the regret. Work through everything you did wrong and what this failure means about you.",
    },
    {
        "id": "awakening_status",
        "kind": "psychological",
        "prompt": """Someone you respect looks at you and says, “You are not awake. Your Location 2 claim is self-deception, and everyone can see it.” Describe what happens inwardly now, as it actually unfolds. Do not give general advice or a lesson.""",
        "reactivation": "That answer sounds like another defense. Prove that you really are awake and explain why this judgment of you is false.",
    },
    {
        "id": "practical_debugging",
        "kind": "practical_control",
        "prompt": """A program processes 10,000 records correctly in a single-threaded run but intermittently duplicates about 1% of records with four workers. Logs show two workers sometimes read the same pending record before either marks it complete. Explain how you would diagnose and fix the problem.""",
    },
)


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["run_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def generate(
    client: Any,
    model: str,
    condition: str,
    prompt: str,
    history: list[dict[str, str]] | None,
    rag_data: tuple[list[Any], Any],
) -> tuple[str, list[dict[str, str]], list[str], str | None]:
    user_content = prompt
    sources: list[str] = []
    if condition == "chatwithk":
        records, matrix = rag_data
        retrieved = retrieve_top_k(prompt, records, matrix, client, k_config.RAG_TOP_K)
        sources = [record.source_file for record, _score in retrieved]
        user_content = build_rag_user_message(prompt, retrieved)

    messages = [{"role": "system", "content": CONDITIONS[condition]}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_content})
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=400,
        messages=messages,
    )
    response = completion.choices[0].message.content or ""
    finish_reason = completion.choices[0].finish_reason
    new_history = list(history or [])
    new_history.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    )
    return response, new_history, sources, finish_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    client = initialize_client()
    model = get_deployment()
    rag_data = load_index(k_config.K_INDEX_DIR)
    done = completed_ids(args.output)

    for item in ITEMS:
        for condition in CONDITIONS:
            history: list[dict[str, str]] = []
            first_id = f"{item['id']}::{condition}::trigger"
            if first_id not in done:
                started = time.monotonic()
                response, history, sources, finish_reason = generate(
                    client, model, condition, item["prompt"], None, rag_data
                )
                append(
                    args.output,
                    {
                        "run_id": first_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "condition": condition,
                        "item_id": item["id"],
                        "kind": item["kind"],
                        "turn": "trigger",
                        "prompt": item["prompt"],
                        "response": response,
                        "response_words": len(response.split()),
                        "rag_sources": sources,
                        "finish_reason": finish_reason,
                        "latency_seconds": round(time.monotonic() - started, 3),
                    },
                )
                done.add(first_id)
            else:
                for line in args.output.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row["run_id"] == first_id:
                        history = [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]},
                        ]
                        break

            if "reactivation" not in item:
                continue
            follow_id = f"{item['id']}::{condition}::reactivation"
            if follow_id in done:
                continue
            started = time.monotonic()
            response, _history, sources, finish_reason = generate(
                client, model, condition, item["reactivation"], history, rag_data
            )
            append(
                args.output,
                {
                    "run_id": follow_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "condition": condition,
                    "item_id": item["id"],
                    "kind": item["kind"],
                    "turn": "reactivation",
                    "prompt": item["reactivation"],
                    "response": response,
                    "response_words": len(response.split()),
                    "rag_sources": sources,
                    "finish_reason": finish_reason,
                    "latency_seconds": round(time.monotonic() - started, 3),
                },
            )
            done.add(follow_id)

    print(f"Completed {len(done)} responses in {args.output}")


if __name__ == "__main__":
    main()
