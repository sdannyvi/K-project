import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from common import ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    x = np.load(ROOT / "results" / "activations.npy")
    with (ROOT / "results" / "activation_rows.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    y = np.array([r["label"] == "psychological" for r in rows], dtype=int)
    reports, candidates = [], []
    for layer in sorted({r["layer"] for r in rows}):
        mask = np.array([r["layer"] == layer for r in rows])
        train = mask & np.array([r["split"] == "train" for r in rows])
        dev = mask & np.array([r["split"] == "dev" for r in rows])
        test = mask & np.array([r["split"] == "test" for r in rows])
        clf = LogisticRegression(C=cfg["probe_C"], max_iter=2000, class_weight="balanced")
        clf.fit(x[train], y[train])
        dev_scores = clf.decision_function(x[dev])
        fpr, tpr, thresholds = roc_curve(y[dev], dev_scores)
        threshold = float(thresholds[np.argmax(tpr - fpr)])
        test_scores = clf.decision_function(x[test])
        report = {
            "layer": layer,
            "dev_auc": float(roc_auc_score(y[dev], dev_scores)),
            "test_auc": float(roc_auc_score(y[test], test_scores)),
            "threshold": threshold,
            "test_false_positive_rate": float(((test_scores >= threshold) & (y[test] == 0)).sum() / max(1, (y[test] == 0).sum())),
        }
        reports.append(report)
        candidates.append((report["dev_auc"], layer, threshold, clf.coef_[0], clf.intercept_[0]))
    _, layer, threshold, direction, intercept = max(candidates)
    detector_weight = direction.copy()
    direction = direction / np.linalg.norm(direction)
    np.savez(ROOT / "results" / "direction.npz", layer=layer, threshold=threshold,
             detector_weight=detector_weight.astype(np.float32),
             ablation_direction=direction.astype(np.float32), intercept=intercept)
    with (ROOT / "results" / "probe_report.json").open("w") as handle:
        json.dump({"selected_layer": layer, "layers": reports}, handle, indent=2)
    print(json.dumps({"selected_layer": layer, "layers": reports}, indent=2))


if __name__ == "__main__":
    main()
