"""Validate the full exploratory corpus before submitting six GPU trainers."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from synthetic_preliminary.train_inner_voice_v6 import MODEL, REVISION, encode


def audit(rows: list[dict], tokenizer, max_length: int) -> dict:
    ids = set()
    groups = collections.defaultdict(set)
    branches = collections.Counter()
    lengths = []
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"duplicate row id: {row['id']}")
        ids.add(row["id"])
        groups[row["card_id"]].add(row["split"])
        branches[(row["split"], row["variant"])] += 1
        encoded = encode(tokenizer, row, max_length)
        lengths.append(len(encoded["input_ids"]))
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("card appears in both splits")
    expected = {"factual_continue", "task_continue", "feeling_task_continue", "rumination_stop"}
    for split in {r["split"] for r in rows}:
        counts = {variant: branches[(split, variant)] for variant in expected}
        if len(set(counts.values())) != 1:
            raise ValueError(f"unbalanced branches in {split}: {counts}")
    return {"rows": len(rows), "cards": len(groups), "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 2),
            "branches": {f"{k[0]}:{k[1]}": v for k, v in sorted(branches.items())}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=384)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rows = [json.loads(line) for line in a.data.open(encoding="utf-8")]
    result = audit(rows, tokenizer, a.max_length)
    result.update({"data_sha256": hashlib.sha256(a.data.read_bytes()).hexdigest(),
                   "model": MODEL, "revision": REVISION, "max_length": a.max_length})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
