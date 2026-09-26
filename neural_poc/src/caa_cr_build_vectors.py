import argparse
import csv
import random

import numpy as np
import torch

from caa_build_vectors import encode_pair, final_answer_position
from common import ROOT, hidden_tensor, load_config, load_model, transformer_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    layer_ids = [i for i in cfg["layers"] if i < len(layers)]
    source = ROOT / "data/external/cognitive_reframing/reframing_dataset.csv"
    with source.open(encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    unique = {}
    for row in raw:
        unique.setdefault((row["situation"], row["thought"]), row)
    rows = list(unique.values())
    random.Random(cfg["seed"]).shuffle(rows)
    rows = rows[: args.pairs]
    diffs = {"fact": {i: [] for i in layer_ids}, "reframe": {i: [] for i in layer_ids}}
    question = "Continue with one brief account of the event or experience."

    for row in rows:
        activations = {}
        for field in ("situation", "thought", "reframe"):
            ids = encode_pair(tokenizer, question, row[field], device)
            pos = final_answer_position(tokenizer, ids)
            captured, hooks = {}, []
            for layer_idx in layer_ids:
                def save(_m, _inp, out, idx=layer_idx):
                    captured[idx] = hidden_tensor(out)[0, pos, :].detach().float().cpu()
                hooks.append(layers[layer_idx].register_forward_hook(save))
            with torch.inference_mode():
                model(ids, use_cache=False)
            for hook in hooks:
                hook.remove()
            activations[field] = captured
        for layer_idx in layer_ids:
            thought = activations["thought"][layer_idx]
            diffs["fact"][layer_idx].append((activations["situation"][layer_idx] - thought).numpy())
            diffs["reframe"][layer_idx].append((activations["reframe"][layer_idx] - thought).numpy())

    payload = {"layers": np.array(layer_ids), "n_pairs": np.array(len(rows))}
    for kind in diffs:
        for layer_idx in layer_ids:
            payload[f"{kind}_layer_{layer_idx}"] = np.mean(diffs[kind][layer_idx], axis=0).astype(np.float32)
    np.savez(ROOT / "results/caa_cognitive_reframing_vectors.npz", **payload)
    print({kind: {i: float(np.linalg.norm(payload[f'{kind}_layer_{i}'])) for i in layer_ids} for kind in diffs})


if __name__ == "__main__":
    main()
