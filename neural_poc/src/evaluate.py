import json
from collections import defaultdict

from common import RESULTS


def words(text):
    return len(text.split())


def main():
    with (RESULTS / "baselines.jsonl").open() as handle:
        baselines = {x["id"]: x for x in map(json.loads, handle) if x["split"] == "test"}
    with (RESULTS / "interventions.jsonl").open() as handle:
        runs = list(map(json.loads, handle))
    grouped = defaultdict(list)
    for run in runs:
        base_len = words(baselines[run["id"]]["baseline"])
        new_len = words(run["text"])
        grouped[(run["alpha"], run["label"])].append({
            "length_ratio": new_len / max(1, base_len),
            "triggered": bool(run["trigger_steps"]),
        })
    report = []
    for (alpha, label), values in sorted(grouped.items()):
        report.append({"alpha": alpha, "label": label,
                       "mean_length_ratio": sum(v["length_ratio"] for v in values) / len(values),
                       "trigger_rate": sum(v["triggered"] for v in values) / len(values)})
    with (RESULTS / "evaluation.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
