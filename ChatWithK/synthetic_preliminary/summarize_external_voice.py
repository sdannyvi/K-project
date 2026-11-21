"""Validate and summarize external voice outputs without printing source text."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from synthetic_preliminary.eval_external_voice import ALL_CONDITIONS, SEED


FAMILIES = ("qwen14", "llama8", "gemma12")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    items = json.loads(args.source.read_text(encoding="utf-8"))
    expected_ids = {x["id"] for x in items}
    if len(expected_ids) != 3340:
        raise ValueError("source count changed")
    grouped: dict[tuple[str, str, str, str], list[dict]] = collections.defaultdict(list)
    seen = set()
    for family in FAMILIES:
        files = sorted((args.run_root / family / "outputs").glob("*.jsonl"))
        if len(files) != 12:
            raise ValueError(f"incomplete shard files for {family}: {len(files)}")
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                key = (family, row["id"], row["condition"])
                if (key in seen or row["id"] not in expected_ids or
                        row["condition"] not in ALL_CONDITIONS or row["seed"] != SEED):
                    raise ValueError(f"duplicate or unexpected output: {key}")
                seen.add(key)
                grouped[(family, row["dataset"], row["source_role"], row["condition"])].append(row)
        family_count = sum(key[0] == family for key in seen)
        if family_count != 3340 * len(ALL_CONDITIONS):
            raise ValueError(f"incomplete family {family}: {family_count}")
    aggregates = []
    for (family, dataset, role, condition), rows in sorted(grouped.items()):
        lengths = [r["new_tokens"] for r in rows]
        aggregates.append({"family": family, "dataset": dataset, "source_role": role,
                           "condition": condition, "n": len(rows),
                           "native_eos": sum(bool(r["native_eos"]) for r in rows),
                           "eos_within_5": sum(r["native_eos"] and r["new_tokens"] <= 5 for r in rows),
                           "eos_within_20": sum(r["native_eos"] and r["new_tokens"] <= 20 for r in rows),
                           "capped": sum(bool(r["max_token_censored"]) for r in rows),
                           "mean_new_tokens": round(statistics.mean(lengths), 3),
                           "sd_new_tokens": round(statistics.stdev(lengths), 3) if len(rows) > 1 else 0,
                           "median_new_tokens": statistics.median(lengths)})
    result = {"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "records": len(seen), "families": list(FAMILIES),
              "conditions": list(ALL_CONDITIONS), "aggregates": aggregates,
              "interpretation": "Source roles are not cessation ground truth; semantic review pending."}
    path = args.run_root / "summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(seen), "aggregate_rows": len(aggregates),
                      "summary": str(path)}))


if __name__ == "__main__":
    main()
