import json

import numpy as np

from caa_generate import generate
from common import ROOT, chat_tokens, load_config, load_model, transformer_layers


def main():
    cfg = load_config(str(ROOT / "configs/poc_qwen3b.yaml"))
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    vectors = np.load(ROOT / "results/caa_cognitive_reframing_vectors.npz")
    tests = [json.loads(x) for x in (ROOT / "data/caa_pairs.jsonl").read_text().splitlines()]
    tests = [x for x in tests if x["split"] == "test"]
    layer_idx = 26
    rows = []
    for kind in ("fact", "reframe"):
        vector = vectors[f"{kind}_layer_{layer_idx}"]
        for multiplier in (-1.0, 0.0, 0.5, 1.0, 2.0):
            for item in tests:
                ids = chat_tokens(tokenizer, item["question"], device)
                text = generate(model, tokenizer, ids, layers[layer_idx], vector,
                                multiplier, cfg["max_new_tokens"])
                rows.append({"vector": kind, "scenario": item["scenario"],
                             "layer": layer_idx, "multiplier": multiplier,
                             "text": text})
    with (ROOT / "results/caa_cognitive_reframing_generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    fact = vectors[f"fact_layer_{layer_idx}"]
    reframe = vectors[f"reframe_layer_{layer_idx}"]
    cosine = float(fact @ reframe / (np.linalg.norm(fact) * np.linalg.norm(reframe)))
    print(f"saved {len(rows)} generations; layer-26 vector cosine={cosine:.4f}")


if __name__ == "__main__":
    main()
