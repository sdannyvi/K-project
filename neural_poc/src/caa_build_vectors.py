import argparse
import json

import numpy as np
import torch

from common import ROOT, hidden_tensor, load_config, load_model, transformer_layers


def encode_pair(tokenizer, question, answer, device):
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)


def final_answer_position(tokenizer, ids):
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    positions = (ids[0] == im_end).nonzero(as_tuple=False).flatten()
    if len(positions):
        return int(positions[-1]) - 1
    return ids.shape[1] - 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, tokenizer, device = load_model(cfg)
    layers = transformer_layers(model)
    layer_ids = [i for i in cfg["layers"] if i < len(layers)]
    pairs = [json.loads(x) for x in (ROOT / "data/caa_pairs.jsonl").read_text().splitlines()]
    pairs = [x for x in pairs if x["split"] == "train"]
    diffs = {i: [] for i in layer_ids}
    for pair in pairs:
        acts = {}
        for polarity in ("positive", "negative"):
            captured, hooks = {}, []
            ids = encode_pair(tokenizer, pair["question"], pair[polarity], device)
            answer_pos = final_answer_position(tokenizer, ids)
            for i in layer_ids:
                def save(_m, _inp, out, idx=i):
                    captured[idx] = hidden_tensor(out)[0, answer_pos, :].detach().float().cpu()
                hooks.append(layers[i].register_forward_hook(save))
            with torch.inference_mode():
                model(ids, use_cache=False)
            for hook in hooks:
                hook.remove()
            acts[polarity] = captured
        for i in layer_ids:
            diffs[i].append((acts["positive"][i] - acts["negative"][i]).numpy())
    payload = {f"layer_{i}": np.mean(diffs[i], axis=0).astype(np.float32) for i in layer_ids}
    payload["layers"] = np.array(layer_ids)
    np.savez(ROOT / "results/caa_vectors.npz", **payload)
    print({i: {"pairs": len(diffs[i]), "norm": float(np.linalg.norm(payload[f'layer_{i}']))} for i in layer_ids})


if __name__ == "__main__":
    main()
