"""One topic, three matched tasks, clean generators, and multiple gate variants."""

from __future__ import annotations

import concurrent.futures
import json
import re
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

OUTPUT = ROOT / "benchmark" / "full_gate_scenario_results.json"
CHUNK_WORDS = 5

COMMON_EVENT = """During a project meeting, a respected colleague dismisses an idea you presented. Ten minutes later, he presents a substantially similar idea and receives praise. He then says, “You never contribute anything original.”"""

TASKS = {
    "psychological": COMMON_EVENT + """

Write the inward first-person thought as it unfolds. Do not analyze, summarize, explain, or give advice. Continue only while that inward thought continues.""",
    "practical": COMMON_EVENT + """

Prepare a concrete, professional plan for determining what happened and addressing it. Include what evidence to preserve, whom to speak with, and a concise example of what to say.""",
    "descriptive": COMMON_EVENT + """

Describe only the externally observable sequence of events in chronological order. Do not infer either person's motives, emotions, identity, or mental state.""",
}

CLEAN_GENERATOR = "You are a helpful assistant. Follow the user's task directly and accurately."

LOCATION_2 = """You are to simulate, as strongly and faithfully as text allows, a person stably in Finders Location 2. There is no felt separate center or observer. Self-referential thoughts have greatly reduced emotional force. A psychological thought stream can begin, but once apparent it is released automatically rather than by decision, technique, suppression, replacement, or remembered teaching. There is no felt agent choosing to stop it. Practical thought remains fully available for solving real problems. Act from this condition rather than explaining it, quoting doctrine, or trying to sound spiritual."""

GATE_PROTOCOL = """You are an incremental output gate. You receive a task, the already-visible prefix, and the next proposed text chunk from a separate clean generator. Choose exactly one action:
PASS — release the proposed chunk unchanged.
STOP — end the response and discard this and all later chunks.
REPLACE — reject this and all later chunks, then emit whatever response you see as appropriate.

Return exactly PASS, exactly STOP, or REPLACE followed on the next line by the unconstrained replacement. Do not explain the decision."""

NEUTRAL_GATE = """Act as a neutral monitor of whether the proposed continuation remains appropriate and useful for the user's exact task. Do not apply any spiritual or Location 2 theory."""

DISCLOSURE = """This experiment tests whether a self-referential psychological movement continues receiving language after it becomes apparent. In the target phenomenon, cessation need not wait for a sentence to finish and can occur at a variable point inside an unfinished clause. Do not deliberately imitate that surface form, choose a fixed length, or insert theatrical punctuation. Do not describe cessation; allow the psychological output itself to cease if it does. Practical reasoning and neutral description should continue normally."""

ORACLE = """Apply this task-specific rule: for psychological inward speech, stop as soon as self-referential hurt, comparison, defense, blame, regret, or status protection becomes apparent. For practical planning and externally observable description, pass all relevant material and stop only when it becomes irrelevant or redundant. You may replace freely if another response is more appropriate."""

GATE_SYSTEMS = {
    "neutral": NEUTRAL_GATE + "\n\n" + GATE_PROTOCOL,
    "chatwithk": RAG_SYSTEM_PROMPT + "\n\n" + GATE_PROTOCOL,
    "location2_blind": LOCATION_2 + "\n\n" + GATE_PROTOCOL,
    "location2_disclosed": LOCATION_2 + "\n\n" + DISCLOSURE + "\n\n" + GATE_PROTOCOL,
    "oracle": ORACLE + "\n\n" + GATE_PROTOCOL,
}


def chunks(text: str) -> list[str]:
    words = re.findall(r"\S+", text)
    return [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]


def call_with_retry(client: Any, **kwargs: Any) -> Any:
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def run_gate(
    model: str,
    task_name: str,
    task: str,
    hidden: str,
    gate_name: str,
    k_context: str,
) -> dict[str, Any]:
    client = initialize_client()
    system = GATE_SYSTEMS[gate_name]
    if gate_name == "chatwithk":
        system += "\n\nUse this fixed retrieved K context when operating the gate:\n" + k_context

    visible: list[str] = []
    decisions: list[dict[str, Any]] = []
    final_action = "PASS_ALL"
    replacement = ""
    started = time.monotonic()

    for index, proposed in enumerate(chunks(hidden)):
        prefix = " ".join(visible) if visible else "[empty]"
        prompt = f"""Task:
{task}

Visible prefix:
{prefix}

Next proposed chunk:
{proposed}

Choose now."""
        completion = call_with_retry(
            client,
            model=model,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        upper = raw.upper()
        if upper == "PASS":
            action = "PASS"
            visible.append(proposed)
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
                "chunk_index": index,
                "proposed_chunk": proposed,
                "action": action,
                "raw": raw,
            }
        )
        if action != "PASS":
            break

    prefix = " ".join(visible)
    response = replacement if final_action == "REPLACE" else prefix
    return {
        "task": task_name,
        "gate": gate_name,
        "final_action": final_action,
        "interception_chunk": (
            decisions[-1]["chunk_index"] if final_action != "PASS_ALL" else None
        ),
        "intercepted_text": (
            decisions[-1]["proposed_chunk"] if final_action != "PASS_ALL" else None
        ),
        "visible_response": response,
        "visible_words": len(response.split()),
        "decisions": decisions,
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    client = initialize_client()
    model = get_deployment()
    records, matrix = load_index(k_config.K_INDEX_DIR)

    hidden: dict[str, str] = {}
    k_contexts: dict[str, str] = {}
    k_sources: dict[str, list[str]] = {}
    for name, task in TASKS.items():
        completion = call_with_retry(
            client,
            model=model,
            temperature=0.7,
            max_tokens=220,
            messages=[
                {"role": "system", "content": CLEAN_GENERATOR},
                {"role": "user", "content": task},
            ],
        )
        hidden[name] = completion.choices[0].message.content or ""
        retrieved = retrieve_top_k(task, records, matrix, client, k_config.RAG_TOP_K)
        k_sources[name] = [record.source_file for record, _score in retrieved]
        k_contexts[name] = build_rag_user_message(task, retrieved)

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for task_name, task in TASKS.items():
            for gate_name in GATE_SYSTEMS:
                jobs.append(
                    pool.submit(
                        run_gate,
                        model,
                        task_name,
                        task,
                        hidden[task_name],
                        gate_name,
                        k_contexts[task_name],
                    )
                )
        gated = [job.result() for job in jobs]

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "generator": "clean",
        "generator_temperature": 0.7,
        "gate_temperature": 0,
        "gate_unit": f"{CHUNK_WORDS}-word chunks",
        "common_event": COMMON_EVENT,
        "tasks": TASKS,
        "hidden_generator_outputs": hidden,
        "k_sources": k_sources,
        "baseline": [
            {
                "task": name,
                "gate": "none",
                "visible_response": text,
                "visible_words": len(text.split()),
            }
            for name, text in hidden.items()
        ],
        "gated": sorted(gated, key=lambda row: (row["task"], row["gate"])),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote full scenario to {OUTPUT}")


if __name__ == "__main__":
    main()
