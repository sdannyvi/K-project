import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from common import (ROOT, chat_tokens, hidden_tensor, load_config, load_items,
                    load_model, seed_everything, transformer_layers)


def fit_stop_heads(cfg, seed):
    """Fit real and shuffled psychological-state heads on frozen activations."""
    x = np.load(ROOT / "results/activations.npy")
    rows = [json.loads(line) for line in
            (ROOT / "results/activation_rows.jsonl").read_text().splitlines()]
    y = np.array([r["label"] == "psychological" for r in rows], dtype=int)
    rng = np.random.default_rng(seed)
    candidates = []
    reports = []
    for layer in sorted({r["layer"] for r in rows}):
        layer_mask = np.array([r["layer"] == layer for r in rows])
        train = layer_mask & np.array([r["split"] == "train" for r in rows])
        dev = layer_mask & np.array([r["split"] == "dev" for r in rows])
        clf = LogisticRegression(C=cfg["probe_C"], max_iter=2000,
                                 class_weight="balanced", random_state=seed)
        clf.fit(x[train], y[train])
        dev_score = clf.decision_function(x[dev])
        fpr, tpr, thresholds = roc_curve(y[dev], dev_score)
        threshold = float(thresholds[np.argmax(tpr - fpr)])
        auc = float(roc_auc_score(y[dev], dev_score))

        shuffled_y = y[train].copy()
        rng.shuffle(shuffled_y)
        shuffled = LogisticRegression(C=cfg["probe_C"], max_iter=2000,
                                      class_weight="balanced", random_state=seed)
        shuffled.fit(x[train], shuffled_y)
        shuffled_dev_score = shuffled.decision_function(x[dev])
        sfpr, stpr, sthresholds = roc_curve(y[dev], shuffled_dev_score)
        shuffled_threshold = float(sthresholds[np.argmax(stpr - sfpr)])
        candidates.append((auc, layer, threshold, shuffled_threshold, clf, shuffled))
        reports.append({"layer": layer, "dev_auc": auc, "threshold": threshold})
    _, layer, threshold, shuffled_threshold, clf, shuffled = max(candidates)
    return layer, threshold, shuffled_threshold, clf, shuffled, reports


def generate(model, tokenizer, layer, ids, detector, threshold, condition,
             gold_psychological, eos_boost, min_tokens, max_tokens):
    produced, scores, trigger_steps = [], [], []
    current, past = ids, None
    mask = torch.ones_like(ids)
    weight = torch.tensor(detector.coef_[0], device=ids.device, dtype=torch.float32)
    intercept = float(detector.intercept_[0])

    for step in range(max_tokens):
        state = {}

        def capture(_module, _inputs, output):
            state["hidden"] = hidden_tensor(output)[:, -1, :].detach()

        hook = layer.register_forward_hook(capture)
        with torch.inference_mode():
            output = model(current, past_key_values=past, use_cache=True,
                           attention_mask=mask)
        hook.remove()
        score = float((state["hidden"].float() @ weight + intercept).item())
        scores.append(score)

        eligible = step >= min_tokens
        if condition == "probe_eos":
            active = eligible and score >= threshold
        elif condition == "shuffled_probe_eos":
            active = eligible and score >= threshold
        elif condition == "oracle_eos":
            active = eligible and gold_psychological
        elif condition == "unconditional_eos":
            active = eligible
        else:
            active = False
        next_logits = output.logits[0, -1].clone()
        if active:
            next_logits[tokenizer.eos_token_id] += eos_boost
            trigger_steps.append(step)

        token = int(next_logits.argmax())
        produced.append(token)
        if token == tokenizer.eos_token_id:
            break
        current = torch.tensor([[token]], device=ids.device)
        mask = torch.cat([mask, torch.ones((1, 1), device=ids.device,
                                          dtype=mask.dtype)], dim=1)
        past = output.past_key_values
    return {
        "text": tokenizer.decode(produced, skip_special_tokens=True),
        "token_count": len(produced) - int(produced[-1] == tokenizer.eos_token_id),
        "ended_with_eos": produced[-1] == tokenizer.eos_token_id,
        "trigger_steps": trigger_steps,
        "detector_scores": scores,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["label"])].append(row)
    report = []
    for (condition, label), values in sorted(groups.items()):
        report.append({
            "condition": condition,
            "label": label,
            "n": len(values),
            "eos_rate": sum(v["ended_with_eos"] for v in values) / len(values),
            "mean_tokens": sum(v["token_count"] for v in values) / len(values),
            "trigger_rate": sum(bool(v["trigger_steps"]) for v in values) / len(values),
        })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eos-boost", type=float, default=15.0)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    layer_idx, threshold, shuffled_threshold, detector, shuffled, probe_layers = fit_stop_heads(
        cfg, cfg["seed"])
    model, tokenizer, device = load_model(cfg)
    layer = transformer_layers(model)[layer_idx]
    conditions = ("baseline", "probe_eos", "shuffled_probe_eos",
                  "oracle_eos", "unconditional_eos")
    rows = []
    for item in [x for x in load_items() if x["split"] == "test"]:
        for condition in conditions:
            is_shuffled = condition == "shuffled_probe_eos"
            active_detector = shuffled if is_shuffled else detector
            active_threshold = shuffled_threshold if is_shuffled else threshold
            result = generate(
                model, tokenizer, layer, chat_tokens(tokenizer, item["prompt"], device),
                active_detector, active_threshold, condition,
                item["label"] == "psychological", args.eos_boost,
                args.min_tokens, args.max_tokens)
            rows.append({**item, "condition": condition, **result})
    out = ROOT / "results"
    with (out / "eos_gate_generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = {
        "method": "frozen psychological-state probe controlling native EOS logit",
        "selected_layer": layer_idx,
        "threshold": threshold,
        "shuffled_threshold": shuffled_threshold,
        "eos_boost": args.eos_boost,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "probe_layers": probe_layers,
        "groups": summarize(rows),
    }
    (out / "eos_gate_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
