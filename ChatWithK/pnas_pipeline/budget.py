"""COMPUTE_BUDGET.csv validator (plan section 4: campaign compute budget gate).

GPU hours = items * conditions * arms * seeds * completions * mean tokens
            / measured tokens-per-second * GPU count / 3600
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

COUNT_FIELDS = ("source_scenarios", "streams_per_scenario", "conditions", "arms",
                "seeds", "completions", "mean_new_tokens")
NOT_GPU = {"API", "N/A"}


def gpu_hours(row: dict) -> float | None:
    """None when the row is not GPU-bound or still uncalibrated."""
    tps = row["measured_tokens_per_second"].strip()
    if tps in NOT_GPU or tps == "TBD":
        return None
    tokens = 1.0
    for field in COUNT_FIELDS:
        tokens *= float(row[field])
    return tokens / float(tps) * float(row["gpu_count"]) / 3600


def check(path: str | Path, gpus: int, ceiling_days: float) -> list[str]:
    rows = list(csv.DictReader(Path(path).open()))
    problems, total = [], 0.0
    for i, row in enumerate(rows, start=2):
        label = f"line {i} ({row['stage']}, {row['model']})"
        tbd = [k for k, v in row.items() if (v or "").strip() == "TBD"]
        if tbd:
            problems.append(f"{label}: TBD in {tbd}")
            continue
        hours = gpu_hours(row)
        if hours is not None:
            declared = float(row["estimated_gpu_hours"])
            if abs(declared - hours) > max(0.5, 0.01 * hours):
                problems.append(f"{label}: estimated_gpu_hours {declared} != formula {hours:.1f}")
            total += hours
    calendar_days = total / gpus / 24
    if calendar_days > ceiling_days:
        problems.append(f"projected {calendar_days:.1f} days on {gpus} GPUs exceeds ceiling {ceiling_days}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="cluster/COMPUTE_BUDGET.csv")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--ceiling-days", type=float, default=42)
    args = parser.parse_args(argv)
    problems = check(args.csv, args.gpus, args.ceiling_days)
    for problem in problems:
        print(f"BLOCKED: {problem}")
    print("Budget gate: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
