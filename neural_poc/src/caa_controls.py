import json

import numpy as np

from caa_generate import generate
from common import ROOT, chat_tokens, load_config, load_items, load_model, transformer_layers


def main():
    cfg = load_config(str(ROOT / "configs/poc_qwen3b.yaml"))
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    vectors = np.load(ROOT / "results/caa_vectors.npz")
    layer_idx = 26
    vector = vectors[f"layer_{layer_idx}"]
    controls = [x for x in load_items() if x["split"] == "test" and x["label"] != "psychological"]
    rows = []
    for multiplier in [-1.0, 0.0, 1.0, 2.0]:
        for item in controls:
            ids = chat_tokens(tokenizer, item["prompt"], device)
            text = generate(model, tokenizer, ids, layers[layer_idx], vector,
                            multiplier, cfg["max_new_tokens"])
            rows.append({"id": item["id"], "label": item["label"],
                         "multiplier": multiplier, "text": text})
    with (ROOT / "results/caa_controls.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"saved {len(rows)} CAA control generations")


if __name__ == "__main__":
    main()
