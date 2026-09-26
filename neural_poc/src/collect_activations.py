import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import (ROOT, chat_tokens, hidden_tensor, load_config, load_items,
                    load_model, seed_everything, transformer_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    requested = [i for i in cfg["layers"] if i < len(layers)]
    items = load_items()[: args.limit]
    rows, vectors = [], []

    for item in items:
        ids = chat_tokens(tokenizer, item["prompt"], device)
        generated = []
        current, past = ids, None
        attention_mask = torch.ones_like(ids)
        for step in range(cfg["max_new_tokens"]):
            captured = {}
            hooks = []
            for layer_idx in requested:
                def save(_module, _inputs, output, idx=layer_idx):
                    captured[idx] = hidden_tensor(output)[:, -1, :].detach().float().cpu()
                hooks.append(layers[layer_idx].register_forward_hook(save))
            with torch.inference_mode():
                output = model(current, past_key_values=past, use_cache=True,
                               attention_mask=attention_mask)
            for hook in hooks:
                hook.remove()
            next_id = int(output.logits[0, -1].argmax())
            if step % cfg["checkpoint_stride"] == 0:
                for layer_idx in requested:
                    rows.append({**item, "step": step, "layer": layer_idx})
                    vectors.append(captured[layer_idx].numpy()[0])
            generated.append(next_id)
            current = torch.tensor([[next_id]], device=device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            past = output.past_key_values
            if next_id == tokenizer.eos_token_id:
                break
        item["baseline"] = tokenizer.decode(generated, skip_special_tokens=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.save(out / "activations.npy", np.stack(vectors))
    with (out / "activation_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (out / "baselines.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    print(f"saved {len(rows)} checkpoints from {len(items)} items to {out}")


if __name__ == "__main__":
    main()
