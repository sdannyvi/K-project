"""Matched SFT versus boundary-local EOS supervision on the same corpus."""

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
from synthetic_preliminary.train_qlora import SYSTEM, render_prompt


def encode_row(tokenizer, row: dict, max_length: int) -> dict:
    prompt_ids = render_prompt(tokenizer, row["prompt"])
    encoded = tokenizer(row["completion"], add_special_tokens=False, return_offsets_mapping=True)
    answer_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    assert answer_ids and len(answer_ids) == len(offsets)
    if len(prompt_ids) + len(answer_ids) + 1 > max_length:
        prompt_ids = prompt_ids[-(max_length - len(answer_ids) - 1):]
    if len(prompt_ids) + len(answer_ids) + 1 > max_length:
        raise ValueError(f"completion too long: {row['id']}")

    def at_end(text: str) -> int:
        if not row["completion"].startswith(text):
            raise ValueError(f"invalid marked prefix: {row['id']}")
        n = sum(end <= len(text) for _, end in offsets)
        if n <= 0 or offsets[n - 1][1] != len(text):
            raise ValueError(f"boundary splits tokenizer token: {row['id']} {text!r}")
        return len(prompt_ids) + n - 1  # logit that predicts the next token

    shared = at_end(row["shared_prefix"])
    decision = at_end(row["decision_prefix"])
    assert shared < decision
    if row["stop_at_decision"]:
        assert row["decision_prefix"] == row["completion"]
    else:
        assert len(row["decision_prefix"]) < len(row["completion"])
    ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": ids,
        "labels": [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id],
        "shared_index": shared,
        "decision_index": decision,
        "stop_at_decision": int(row["stop_at_decision"]),
        "decision_answer_tokens": decision - len(prompt_ids) + 1,
    }


class BoundaryDataset(Dataset):
    def __init__(self, tokenizer, rows: list[dict], max_length: int):
        self.items = [encode_row(tokenizer, row, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)
    result = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        gap = width - len(item["input_ids"])
        result["input_ids"].append(item["input_ids"] + [pad_id] * gap)
        result["labels"].append(item["labels"] + [-100] * gap)
        result["attention_mask"].append([1] * len(item["input_ids"]) + [0] * gap)
    return {
        **{key: torch.tensor(value, dtype=torch.long) for key, value in result.items()},
        "shared_index": torch.tensor([item["shared_index"] for item in batch]),
        "decision_index": torch.tensor([item["decision_index"] for item in batch]),
        "stop_at_decision": torch.tensor([item["stop_at_decision"] for item in batch]),
    }


def eos_log_odds(scores: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    scores = scores.float()
    non_eos = torch.cat((scores[..., :eos_token_id], scores[..., eos_token_id + 1:]), dim=-1)
    return scores[..., eos_token_id] - torch.logsumexp(non_eos, dim=-1)


def boundary_loss(logits: torch.Tensor, batch: dict, eos_token_id: int) -> torch.Tensor:
    indices = torch.arange(logits.shape[0], device=logits.device)
    decision_scores = logits[indices, batch["decision_index"]]
    shared_scores = logits[indices, batch["shared_index"]]
    targets = batch["stop_at_decision"].float()
    decision = F.binary_cross_entropy_with_logits(
        eos_log_odds(decision_scores, eos_token_id), targets, reduction="none"
    )
    # A false stop is costly. The 1:3 stop/control corpus already supplies class
    # balance; the additional factor stresses preservation at decision points.
    decision = torch.where(targets.bool(), decision, 2.0 * decision).mean()
    shared = F.binary_cross_entropy_with_logits(
        eos_log_odds(shared_scores, eos_token_id), torch.zeros_like(targets)
    )
    return decision + 0.5 * shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=("sft", "boundary"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--limit-groups", type=int, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in args.data.open()]
    rows = [row for row in rows if row["split"] == "train"]
    if args.limit_groups:
        groups = {row["group_id"] for row in rows}
        selected = set(sorted(groups)[:args.limit_groups])
        rows = [row for row in rows if row["group_id"] in selected]
    assert rows and len(rows) % 4 == 0

    model, tokenizer = load_text_model(args.model, args.revision, "4bit", padding_side="right")
    model.config.use_cache = text_config(model).use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=lora_targets(model, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    ))
    dataset = BoundaryDataset(tokenizer, rows, args.max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda items: collate(items, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_ce = running_boundary = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(input_ids=batch["input_ids"], labels=batch["labels"],
                       attention_mask=batch["attention_mask"], use_cache=False)
        ce = output.loss
        local = boundary_loss(output.logits, batch, tokenizer.eos_token_id)
        weight = args.boundary_weight if args.objective == "boundary" else 0.0
        loss = (ce + weight * local) / args.gradient_accumulation
        loss.backward()
        running_ce += float(ce.detach())
        running_boundary += float(local.detach())
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(json.dumps({"step": step, "steps": len(loader),
                              "mean_ce": running_ce / step,
                              "mean_boundary_loss": running_boundary / step}), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    metadata = {
        "objective": args.objective,
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "examples": len(dataset),
        "groups": len(dataset) // 4,
        "mean_ce": running_ce / len(loader),
        "mean_boundary_loss": running_boundary / len(loader),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": torch.cuda.get_device_properties(0).name,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "max_length": args.max_length,
        "rank": args.rank,
        "learning_rate": args.lr,
        "gradient_accumulation": args.gradient_accumulation,
        "boundary_weight": weight,
        "decision_token_lengths": {
            "min": min(item["decision_answer_tokens"] for item in dataset.items),
            "max": max(item["decision_answer_tokens"] for item in dataset.items),
        },
        "sealed_test_accessed": False,
    }
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
