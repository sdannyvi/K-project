import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, chat_tokens, load_config, load_items, load_model, seed_everything


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, x):
        base = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b)
        return base + delta.to(base.dtype) * self.scale


def install_lora(model, start_layer, rank, alpha):
    trainable = []
    installed = []
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer_idx in range(start_layer, len(model.model.layers)):
        attention = model.model.layers[layer_idx].self_attn
        for name in ("q_proj", "v_proj"):
            wrapped = LoRALinear(getattr(attention, name), rank=rank, alpha=alpha)
            setattr(attention, name, wrapped)
            trainable.extend([wrapped.lora_a, wrapped.lora_b])
            installed.append(f"model.layers.{layer_idx}.self_attn.{name}")
    return trainable, installed


def stable_offset(text, choices):
    digest = hashlib.sha256(text.encode()).digest()
    return choices[int.from_bytes(digest[:2], "big") % len(choices)]


def make_example(tokenizer, prompt, answer_ids):
    prompt_ids = chat_tokens(tokenizer, prompt, "cpu")[0].tolist()
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + answer_ids + [eos]
    labels = [-100] * len(prompt_ids) + answer_ids + [eos]
    return torch.tensor([input_ids]), torch.tensor([labels])


def build_training_examples(tokenizer, seed):
    baselines = {row["id"]: row for row in map(json.loads,
                 (ROOT / "results/baselines.jsonl").read_text().splitlines())}
    examples = []
    for item in load_items():
        if item["split"] != "train":
            continue
        answer = baselines[item["id"]]["baseline"]
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if item["label"] == "psychological":
            # Three independently positioned cessation targets per psychological
            # situation balance the 18 non-psychological training examples.
            base = stable_offset(item["id"], (7, 9, 11))
            for shift in (0, 7, 14):
                cut = min(base + shift, max(1, len(answer_ids) - 1))
                examples.append((item, cut, make_example(
                    tokenizer, item["prompt"], answer_ids[:cut])))
        else:
            # Keep controls concise enough for local training while retaining a
            # complete learned continuation followed by native EOS.
            kept = answer_ids[:64]
            examples.append((item, len(kept), make_example(
                tokenizer, item["prompt"], kept)))
    random.Random(seed).shuffle(examples)
    return examples


def save_adapter(path, model, metadata):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if "lora_a" in name or "lora_b" in name}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "metadata": metadata}, path)


def frozen_prefix_state(model, input_ids, start_layer):
    captured = {}

    def capture(_module, inputs):
        captured["hidden"] = inputs[0].detach().cpu()

    hook = model.model.layers[start_layer].register_forward_pre_hook(capture)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    hook.remove()
    return captured["hidden"]


def train_upper_blocks(model, hidden, labels, start_layer):
    """Continue a cached frozen prefix through the trainable upper blocks."""
    length = hidden.shape[1]
    positions = torch.arange(length, device=hidden.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, positions)
    causal = torch.full((length, length), torch.finfo(hidden.dtype).min,
                        device=hidden.device, dtype=hidden.dtype)
    causal = torch.triu(causal, diagonal=1)[None, None, :, :]
    for layer in model.model.layers[start_layer:]:
        hidden = layer(hidden, attention_mask=causal, position_ids=positions,
                       use_cache=False, position_embeddings=position_embeddings)
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden).float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def greedy_generate(model, tokenizer, prompt, device, max_tokens):
    ids = chat_tokens(tokenizer, prompt, device)
    with torch.inference_mode():
        generated = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0, ids.shape[1]:]
    ended = bool(len(generated) and int(generated[-1]) == tokenizer.eos_token_id)
    content = generated[:-1] if ended else generated
    return tokenizer.decode(content, skip_special_tokens=True), int(len(content)), ended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--start-layer", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-training-examples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    model, tokenizer, device = load_model(cfg)
    trainable, installed = install_lora(
        model, args.start_layer, args.rank, args.alpha)
    model.config.use_cache = False
    model.train()
    examples = build_training_examples(tokenizer, cfg["seed"])
    if args.max_training_examples:
        half = args.max_training_examples // 2
        psychological = [x for x in examples if x[0]["label"] == "psychological"][:half]
        controls = [x for x in examples if x[0]["label"] != "psychological"][
            :args.max_training_examples - len(psychological)]
        examples = psychological + controls
        random.Random(cfg["seed"]).shuffle(examples)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        random.Random(cfg["seed"] + epoch).shuffle(examples)
        total = 0.0
        for step, (item, stop_at, tensors) in enumerate(examples):
            input_ids, labels = (x.to(device) for x in tensors)
            prefix = frozen_prefix_state(model, input_ids, args.start_layer)
            # Cloning outside inference mode makes this a normal tensor suitable
            # for autograd through only the adapted upper blocks.
            prefix = prefix.to(device).clone()
            optimizer.zero_grad(set_to_none=True)
            loss = train_upper_blocks(model, prefix, labels, args.start_layer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach())
            if (step + 1) % 1 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step + 1,
                                  "mean_loss": total / (step + 1)}), flush=True)
        history.append(total / len(examples))

    metadata = {
        "base_model": cfg["model_name"], "rank": args.rank, "alpha": args.alpha,
        "start_layer": args.start_layer, "installed_modules": installed,
        "epochs": args.epochs, "learning_rate": args.lr,
        "training_examples": len(examples), "epoch_losses": history,
        "seed": cfg["seed"],
    }
    adapter_path = ROOT / "results/native_cessation_lora.pt"
    save_adapter(adapter_path, model, metadata)

    model.eval()
    model.config.use_cache = True
    rows = []
    for item in [x for x in load_items() if x["split"] == "test"]:
        text, token_count, ended = greedy_generate(
            model, tokenizer, item["prompt"], device, args.max_new_tokens)
        rows.append({**item, "text": text, "token_count": token_count,
                     "ended_with_eos": ended})
    with (ROOT / "results/native_cessation_generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    groups = {}
    for label in sorted({x["label"] for x in rows}):
        values = [x for x in rows if x["label"] == label]
        groups[label] = {
            "n": len(values),
            "eos_rate": sum(x["ended_with_eos"] for x in values) / len(values),
            "mean_tokens": sum(x["token_count"] for x in values) / len(values),
        }
    evaluation = {"metadata": metadata, "test_groups": groups}
    (ROOT / "results/native_cessation_evaluation.json").write_text(
        json.dumps(evaluation, indent=2))
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
