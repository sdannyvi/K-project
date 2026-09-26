import argparse
import json

import numpy as np
import torch

from common import (ROOT, chat_tokens, hidden_tensor, load_config, load_items,
                    load_model, replace_hidden, seed_everything, transformer_layers)


def generate(model, tokenizer, layers, ids, layer_idx, detector_weight,
             ablation_direction, threshold, intercept, alpha, max_tokens,
             conditional=True):
    produced, trigger_steps = [], []
    detector = torch.tensor(detector_weight, device=ids.device, dtype=torch.float32)
    d = torch.tensor(ablation_direction, device=ids.device, dtype=model.dtype)
    current, past = ids, None
    attention_mask = torch.ones_like(ids)

    for step in range(max_tokens):
        state = {"score": None}
        def intervene(_module, _inputs, output):
            hidden = hidden_tensor(output)
            last = hidden[:, -1, :]
            score = torch.matmul(last.float(), detector) + intercept
            state["score"] = float(score.item())
            active = (state["score"] >= threshold) if conditional else True
            if active:
                projection = torch.matmul(last, d).unsqueeze(-1) * d
                changed = hidden.clone()
                changed[:, -1, :] = last - alpha * projection
                trigger_steps.append(step)
                return replace_hidden(output, changed)
            return output
        hook = layers[layer_idx].register_forward_hook(intervene)
        with torch.inference_mode():
            output = model(current, past_key_values=past, use_cache=True,
                           attention_mask=attention_mask)
        hook.remove()
        next_id = int(output.logits[0, -1].argmax())
        produced.append(next_id)
        current = torch.tensor([[next_id]], device=ids.device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=ids.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        past = output.past_key_values
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(produced, skip_special_tokens=True), trigger_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    artifact = np.load(ROOT / "results" / "direction.npz")
    layer_idx = int(artifact["layer"])
    results = []
    for item in [x for x in load_items() if x["split"] == "test"]:
        for alpha in cfg["alphas"]:
            ids = chat_tokens(tokenizer, item["prompt"], device)
            text, triggers = generate(model, tokenizer, layers, ids, layer_idx,
                                      artifact["detector_weight"], artifact["ablation_direction"],
                                      float(artifact["threshold"]),
                                      float(artifact["intercept"]), alpha,
                                      cfg["max_new_tokens"], conditional=True)
            results.append({**item, "condition": "conditional_ablation",
                            "alpha": alpha, "text": text,
                            "trigger_steps": triggers})
    with (ROOT / "results" / "interventions.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    print(f"saved {len(results)} intervention runs")


if __name__ == "__main__":
    main()
