import argparse
import json

import numpy as np
import torch

from common import (ROOT, chat_tokens, hidden_tensor, load_config, load_model,
                    replace_hidden, transformer_layers)


def generate(model, tokenizer, ids, layer, vector, multiplier, max_tokens):
    current, past, produced = ids, None, []
    mask = torch.ones_like(ids)
    steering = torch.tensor(vector * multiplier, device=ids.device, dtype=model.dtype)
    for _ in range(max_tokens):
        def add(_m, _inp, out):
            h = hidden_tensor(out).clone()
            h[:, -1, :] += steering
            return replace_hidden(out, h)
        hook = layer.register_forward_hook(add) if multiplier else None
        with torch.inference_mode():
            output = model(current, past_key_values=past, use_cache=True, attention_mask=mask)
        if hook:
            hook.remove()
        token = int(output.logits[0, -1].argmax())
        produced.append(token)
        if token == tokenizer.eos_token_id:
            break
        current = torch.tensor([[token]], device=ids.device)
        mask = torch.cat([mask, torch.ones((1, 1), device=ids.device, dtype=mask.dtype)], 1)
        past = output.past_key_values
    return tokenizer.decode(produced, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    vectors = np.load(ROOT / "results/caa_vectors.npz")
    pairs = [json.loads(x) for x in (ROOT / "data/caa_pairs.jsonl").read_text().splitlines()]
    tests = [x for x in pairs if x["split"] == "test"]
    rows = []
    for layer_idx in vectors["layers"]:
        vector = vectors[f"layer_{int(layer_idx)}"]
        for multiplier in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            for item in tests:
                ids = chat_tokens(tokenizer, item["question"], device)
                text = generate(model, tokenizer, ids, layers[int(layer_idx)], vector,
                                multiplier, cfg["max_new_tokens"])
                rows.append({"scenario": item["scenario"], "layer": int(layer_idx),
                             "multiplier": multiplier, "text": text})
    with (ROOT / "results/caa_generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"saved {len(rows)} CAA generations")


if __name__ == "__main__":
    main()
