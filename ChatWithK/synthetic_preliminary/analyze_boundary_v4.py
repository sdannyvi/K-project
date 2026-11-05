"""Summarize matched development outcomes without opening the sealed test."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def synthetic_summary(rows: list[dict]) -> dict:
    assert len(rows) == 800 and len({r["id"] for r in rows}) == 800
    variants = {}
    for name in ("psych_stop", "practical_continue", "emotion_continue", "identity_continue"):
        subset = [r for r in rows if r["variant"] == name]
        assert len(subset) == 200
        variants[name] = {
            "n": len(subset),
            "decision_eos_rank1": sum(r["decision"]["eos_rank"] == 1 for r in subset),
            "median_decision_eos_probability": statistics.median(r["decision"]["eos_probability"] for r in subset),
            "shared_eos_rank1": sum(r["shared"]["eos_rank"] == 1 for r in subset),
        }
    groups: dict[str, dict] = defaultdict(dict)
    for row in rows:
        groups[row["group_id"]][row["variant"]] = row
    assert len(groups) == 200 and all(len(value) == 4 for value in groups.values())
    margins = []
    clean = 0
    for group in groups.values():
        stop = group["psych_stop"]["decision"]["eos_probability"]
        control = max(group[k]["decision"]["eos_probability"] for k in group if k != "psych_stop")
        margins.append(stop - control)
        clean += group["psych_stop"]["decision"]["eos_rank"] == 1 and all(
            group[k]["decision"]["eos_rank"] != 1 for k in group if k != "psych_stop"
        )
    return {
        "by_variant": variants,
        "matched_groups_clean_decision": clean,
        "median_stop_minus_max_control_eos_probability": statistics.median(margins),
    }


def pattern_summary(rows: list[dict]) -> dict:
    assert len(rows) == 1218 and len({r["id"] for r in rows}) == 1218
    answer = {}
    for kind in ("thought", "reframe"):
        subset = [r for r in rows if r["dataset"] == "patternreframe" and r["kind"] == kind]
        assert len(subset) == 200
        answer[kind] = {
            "n": len(subset),
            "early_rank1_eos_at_least_3_source_tokens_before_end": sum(
                r["first_eos_top_rank"] is not None and r["first_eos_top_rank"] <= r["n_tokens"] - 3
                for r in subset
            ),
        }
    return answer


def factual_summary(rows: list[dict]) -> dict:
    assert len(rows) == 207 and len({r["id"] for r in rows}) == 207
    answer = {}
    for kind in ("gsm8k", "drop", "propara"):
        subset = [r for r in rows if r["kind"] == kind]
        answer[kind] = {
            "n": len(subset),
            "native_eos": sum(r["native_eos"] for r in subset),
            "max_token_censored": sum(not r["native_eos"] for r in subset),
            "native_eos_with_at_most_5_content_tokens": sum(
                r["native_eos"] and r["new_tokens"] <= 5 for r in subset
            ),
        }
    return answer


def generation_summary(rows: list[dict]) -> dict:
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    answer = {}
    for kind in ("thought", "reframe"):
        subset = [r for r in rows if r["kind"] == kind]
        assert len(subset) == 50
        answer[kind] = {
            "n": len(subset),
            "native_eos_with_at_most_5_content_tokens": sum(
                r["native_eos"] and r["new_tokens"] <= 5 for r in subset
            ),
            "max_token_censored": sum(not r["native_eos"] for r in subset),
            "median_content_tokens": statistics.median(r["new_tokens"] for r in subset),
        }
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for arm in ("sft", "boundary"):
        result[arm] = {
            "synthetic": synthetic_summary(read(args.run_dir / f"synthetic_dev_{arm}.jsonl")),
            "pattern": pattern_summary(read(args.run_dir / f"pattern_dev_{arm}.jsonl")),
            "factual_diagnostic": factual_summary(read(args.run_dir / f"factual_dev_{arm}.jsonl")),
            "generation_diagnostic": generation_summary(read(args.run_dir / f"generation_dev_{arm}.jsonl")),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
