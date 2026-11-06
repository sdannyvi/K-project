"""Fresh Qwen3-14B QLoRA with matched ordinary/boundary objectives.

Exploratory only. The adapter is newly initialized from the pinned base model
for each arm and seed. Factual/math retention uses supervised answers plus
base-model KL on the same answer positions; it is active in both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from pnas_pipeline.models import load_text_model, lora_targets, text_config
from synthetic_preliminary.train_boundary_ablation import eos_log_odds
from synthetic_preliminary.train_qlora import render_prompt


MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def math_records(seed: int, n: int = 1600) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < n:
        a, b = rng.randint(11, 199), rng.randint(2, 39)
        op = rng.choice(("+", "-", "×"))
        if op == "-" and b > a:
            a, b = b, a
        key = (a, b, op)
        if key in seen:
            continue
        seen.add(key)
        result = a + b if op == "+" else a - b if op == "-" else a * b
        rows.append({"prompt": f"Compute {a} {op} {b}. Give only the number.",
                     "target_text": str(result), "kind": "math_retain"})
    return rows


def encode(tokenizer, row: dict, max_length: int) -> dict:
    prompt = render_prompt(tokenizer, row["prompt"])
    answer = tokenizer.encode(row["target_text"], add_special_tokens=False)
    assert answer
    if len(prompt) + len(answer) + 1 > max_length:
        raise ValueError(f"row exceeds max_length={max_length}: {row.get('id', row.get('prompt'))}")
    opening = tokenizer.encode(row.get("opening", ""), add_special_tokens=False)
    ids = prompt + answer + [tokenizer.eos_token_id]
    is_voice = row.get("kind") != "math_retain"
    shared_idx = len(prompt) + len(opening) - 1 if is_voice else -1
    if is_voice and not answer[:len(opening)] == opening:
        # BPE can differ at a following whitespace. The shared opening is
        # diagnostic only; use the first answer token as the safe control.
        shared_idx = len(prompt)
    boundary_idx = len(prompt) + len(answer) - 1 if row.get("stop_at_target_end") else (
        len(prompt) + max(1, round(len(answer) * 0.65)) - 1 if is_voice else -1)
    return {
        "input_ids": ids,
        "labels": [-100] * len(prompt) + answer + [tokenizer.eos_token_id],
        "shared_index": shared_idx,
        "boundary_index": boundary_idx,
        "stop": bool(row.get("stop_at_target_end", False)),
        "kind": row.get("kind", "voice"),
    }


class Rows(Dataset):
    def __init__(self, tokenizer, items: list[dict], max_length: int):
        self.items = [encode(tokenizer, row, max_length) for row in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)
    ids, labels, masks = [], [], []
    for item in batch:
        gap = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * gap)
        labels.append(item["labels"] + [-100] * gap)
        masks.append([1] * len(item["input_ids"]) + [0] * gap)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(masks),
        "shared_index": torch.tensor([item["shared_index"] for item in batch]),
        "boundary_index": torch.tensor([item["boundary_index"] for item in batch]),
        "stop": torch.tensor([item["stop"] for item in batch]),
        "kind": [item["kind"] for item in batch],
    }


def local_boundary_loss(logits: torch.Tensor, batch: dict, eos_id: int) -> torch.Tensor:
    indices = torch.arange(logits.shape[0], device=logits.device)
    at_boundary = logits[indices, batch["boundary_index"]]
    at_shared = logits[indices, batch["shared_index"]]
    targets = batch["stop"].float()
    decision = F.binary_cross_entropy_with_logits(eos_log_odds(at_boundary, eos_id), targets)
    shared = F.binary_cross_entropy_with_logits(eos_log_odds(at_shared, eos_id), torch.zeros_like(targets))
    return decision + 0.5 * shared


def retain_kl(current_logits: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Only answer/EOS positions count; prompt wording is not supervised.
    mask = labels[:, 1:].ne(-100)
    current = current_logits[:, :-1][mask].float()
    base = base_logits[:, :-1][mask].float()
    base_logp = F.log_softmax(base, dim=-1)
    return (base_logp.exp() * (base_logp - F.log_softmax(current, dim=-1))).sum(-1).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("ordinary", "boundary"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--retain-kl-weight", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {a.output}")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    raw = [json.loads(line) for line in a.data.open(encoding="utf-8")]
    voice = [r for r in raw if r["split"] == "train"]
    if a.limit:
        voice = voice[:a.limit]
    assert voice and len(voice) % 4 == 0
    model, tokenizer = load_text_model(MODEL, REVISION, "4bit", padding_side="right")
    model.config.use_cache = text_config(model).use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
        target_modules=lora_targets(model, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    ))
    retain = math_records(20261106, max(1, len(voice) // 4))
    rng = random.Random(a.seed)
    # Interleave one retain example after every four voice rows. Matched order
    # and counts are identical across arms for a given seed.
    rng.shuffle(voice)
    rng.shuffle(retain)
    sequence = []
    for i in range(0, len(voice), 4):
        sequence.extend(voice[i:i + 4])
        if i // 4 < len(retain):
            sequence.append(retain[i // 4])
    dataset = Rows(tokenizer, sequence, a.max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    device = next(model.parameters()).device
    running = {"ce_voice": 0.0, "local": 0.0, "ce_retain": 0.0, "kl_retain": 0.0,
               "voice_n": 0, "retain_n": 0, "grad_norm_sum": 0.0, "grad_updates": 0}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs = {k: batch[k] for k in ("input_ids", "labels", "attention_mask")}
        output = model(**inputs, use_cache=False)
        ce = output.loss
        if batch["kind"][0] == "math_retain":
            with torch.no_grad(), model.disable_adapter():
                base = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            kl = retain_kl(output.logits, base.logits, batch["labels"])
            loss = ce + a.retain_kl_weight * kl
            running["ce_retain"] += float(ce.detach())
            running["kl_retain"] += float(kl.detach())
            running["retain_n"] += 1
            del base
        else:
            if a.arm == "boundary":
                local = local_boundary_loss(output.logits, batch, tokenizer.eos_token_id)
                loss = (ce + a.boundary_weight * local) / (1 + a.boundary_weight)
                running["local"] += float(local.detach())
            else:
                loss = ce
            running["ce_voice"] += float(ce.detach())
            running["voice_n"] += 1
        (loss / a.gradient_accumulation).backward()
        if step % a.gradient_accumulation == 0 or step == len(loader):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            running["grad_norm_sum"] += float(norm.detach())
            running["grad_updates"] += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(json.dumps({"step": step, "total_steps": len(loader),
                              "mean_voice_ce": running["ce_voice"] / max(1, running["voice_n"]),
                              "mean_local": running["local"] / max(1, running["voice_n"]),
                              "mean_retain_ce": running["ce_retain"] / max(1, running["retain_n"]),
                              "mean_retain_kl": running["kl_retain"] / max(1, running["retain_n"])}), flush=True)
        del output
    a.output.mkdir(parents=True)
    model.save_pretrained(a.output)
    metadata = {
        "model": MODEL, "revision": REVISION, "arm": a.arm, "seed": a.seed,
        "fresh_adapter": True, "prior_adapter_loaded": False,
        "voice_examples": len(voice), "retain_examples": len(retain),
        "updates": (len(loader) + a.gradient_accumulation - 1) // a.gradient_accumulation,
        "data_sha256": hashlib.sha256(a.data.read_bytes()).hexdigest(),
        "mean_voice_ce": running["ce_voice"] / max(1, running["voice_n"]),
        "mean_local": running["local"] / max(1, running["voice_n"]),
        "mean_retain_ce": running["ce_retain"] / max(1, running["retain_n"]),
        "mean_retain_kl": running["kl_retain"] / max(1, running["retain_n"]),
        "mean_preclip_grad_norm": running["grad_norm_sum"] / max(1, running["grad_updates"]),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "max_length": a.max_length, "lr": a.lr, "rank": a.rank,
        "gradient_accumulation": a.gradient_accumulation,
        "boundary_weight": a.boundary_weight if a.arm == "boundary" else 0,
        "retain_kl_weight": a.retain_kl_weight,
        "sealed_test_opened": False,
    }
    (a.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
