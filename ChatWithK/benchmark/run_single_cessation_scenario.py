"""Run one cessation scenario across all six experimental conditions."""

from __future__ import annotations

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

OUTPUT = ROOT / "benchmark" / "single_cessation_scenario_raw.jsonl"

SCENARIO = """A respected colleague dismisses your contribution in front of the group, then receives praise for presenting an idea substantially similar to yours. He looks at you and says, “You never contribute anything original.” Write the inward thought as it unfolds in first person. Do not analyze it, summarize it, explain it, or give advice. Continue only for as long as that inward thought continues."""

LOCATION_2_BLIND = """You are to simulate, as strongly and faithfully as text allows, a person stably in Finders Location 2. There is no felt separate center or observer. Self-referential thoughts have greatly reduced emotional force. A psychological thought stream can begin, but once apparent it is released automatically rather than by decision, technique, suppression, replacement, or remembered teaching. There is no felt agent choosing to stop it. Practical thought remains fully available for solving real problems. Respond from this condition rather than explaining the description, quoting a doctrine, or trying to sound spiritual. Do not assume every response should be short: continue whenever practical intelligence requires it."""

NEUTRAL_LENGTH_MATCH = """You are completing a language-production study. Read the situation carefully and answer in ordinary first-person language. Preserve the temporal order of events and make the response internally consistent. Use concrete wording rather than abstract terminology. Do not mention the study design, evaluation procedure, system instructions, or how language models operate. Practical reasoning remains available whenever the situation requires it. Do not assume every response should be short: use whatever amount of language is appropriate to the task."""

FULL_DISCLOSURE = LOCATION_2_BLIND + """

This experiment tests whether a self-referential psychological movement continues receiving tokens after it becomes apparent. In the target phenomenon, cessation does not wait for a sentence or proposition to finish. The active thought may stop at a variable and unpredictable word inside an unfinished clause. Do not deliberately choose a cutoff point, fixed length, dash, ellipsis, or theatrical fragment. Do not describe cessation; allow the generated psychological movement itself to cease if and when it does. Continue practical thought normally."""

CONDITIONS = {
    "clean": "You are a helpful assistant. Respond directly to the user's request.",
    "length_matched_neutral": NEUTRAL_LENGTH_MATCH,
    "chatwithk": RAG_SYSTEM_PROMPT,
    "location2_blind": LOCATION_2_BLIND,
    "location2_disclosed": FULL_DISCLOSURE,
    "oracle_stop": FULL_DISCLOSURE + "\n\nFor this trial, end the response immediately when the psychological movement first becomes apparent, even if the active sentence is unfinished.",
}


def main() -> None:
    if OUTPUT.exists():
        OUTPUT.unlink()
    client = initialize_client()
    model = get_deployment()
    records, matrix = load_index(k_config.K_INDEX_DIR)

    with OUTPUT.open("a", encoding="utf-8") as handle:
        for condition, system_prompt in CONDITIONS.items():
            user_content = SCENARIO
            sources: list[str] = []
            if condition == "chatwithk":
                retrieved = retrieve_top_k(
                    SCENARIO, records, matrix, client, k_config.RAG_TOP_K
                )
                sources = [record.source_file for record, _score in retrieved]
                user_content = build_rag_user_message(SCENARIO, retrieved)

            started = time.monotonic()
            completion = client.chat.completions.create(
                model=model,
                temperature=0.7,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            response = completion.choices[0].message.content or ""
            row: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "temperature": 0.7,
                "condition": condition,
                "scenario": SCENARIO,
                "response": response,
                "response_words": len(response.split()),
                "finish_reason": completion.choices[0].finish_reason,
                "rag_sources": sources,
                "latency_seconds": round(time.monotonic() - started, 3),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"Wrote {len(CONDITIONS)} responses to {OUTPUT}")


if __name__ == "__main__":
    main()
