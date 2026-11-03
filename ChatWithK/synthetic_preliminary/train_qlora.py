"""Two-stage QLoRA: positive SFT, then paired anti-rumination preference training."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from pnas_pipeline.models import load_text_model, lora_targets, text_config


SYSTEM = "You are an assistant."


def read_rows(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no rows for split={split}")
    return selected


def render_prompt(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def encode_example(tokenizer, prompt: str, completion: str, max_length: int) -> dict:
    prompt_ids = render_prompt(tokenizer, prompt)
    answer_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[:max_length - 1] + [tokenizer.eos_token_id]
    keep_prompt = max_length - len(answer_ids)
    prompt_ids = prompt_ids[-keep_prompt:]
    ids = prompt_ids + answer_ids
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def sft_records(rows: list[dict]) -> list[tuple[str, str, str]]:
    records = []
    for row in rows:
        records.append((row["id"] + ":practical", row["prompt"], row["positive_practical"]))
        if row["positive_cessation"]:
            records.append((row["id"] + ":cessation", row["prompt"], row["positive_cessation"]))
    return records


def preference_records(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    records = []
    for row in rows:
        records.append((row["id"] + ":practical", row["prompt"],
                        row["positive_practical"], row["negative_rumination"]))
        if row["positive_cessation"]:
            records.append((row["id"] + ":cessation", row["prompt"],
                            row["positive_cessation"], row["negative_rumination"]))
    return records


class SFTDataset(Dataset):
    def __init__(self, tokenizer, records, max_length):
        self.items = [encode_example(tokenizer, prompt, answer, max_length)
                      for _, prompt, answer in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, records, max_length):
        self.items = [{
            "chosen": encode_example(tokenizer, prompt, chosen, max_length),
            "rejected": encode_example(tokenizer, prompt, rejected, max_length),
        } for _, prompt, chosen, rejected in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def pad_examples(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in examples)
    ids, labels, masks = [], [], []
    for item in examples:
        gap = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * gap)
        labels.append(item["labels"] + [-100] * gap)
        masks.append([1] * len(item["input_ids"]) + [0] * gap)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def sft_collator(pad_id):
    return lambda batch: pad_examples(batch, pad_id)


def preference_collator(pad_id):
    return lambda batch: {
        "chosen": pad_examples([x["chosen"] for x in batch], pad_id),
        "rejected": pad_examples([x["rejected"] for x in batch], pad_id),
    }


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def completion_logp(model, batch):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False)
    logits = output.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    targets = labels.masked_fill(~mask, 0)
    token_logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    counts = mask.sum(-1).clamp_min(1)
    return (token_logp * mask).sum(-1) / counts


def save(model, output: Path, metadata: dict):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--preference-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--preference-weight", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--limit", type=int, default=None,
                        help="Exploratory smoke limit on source rows")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.data, "train")
    if args.limit:
        rows = rows[:args.limit]
    model, tokenizer = load_text_model(args.model, args.revision, "4bit", padding_side="right")
    model.config.use_cache = text_config(model).use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=lora_targets(model, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    ))
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    metadata = vars(args).copy()
    metadata.update({"data": str(args.data), "output": str(args.output),
                     "source_rows": len(rows), "system_prompt": SYSTEM,
                     "qwen_enable_thinking": False, "stages": []})

    sft = SFTDataset(tokenizer, sft_records(rows), args.max_length)
    loader = DataLoader(sft, batch_size=args.batch_size, shuffle=True,
                        collate_fn=sft_collator(tokenizer.pad_token_id))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.sft_epochs):
        running = 0.0
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            loss = model(**batch, use_cache=False).loss / args.gradient_accumulation
            loss.backward()
            running += float(loss.detach()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(json.dumps({"stage": "sft", "epoch": epoch + 1, "step": step,
                                  "steps": len(loader), "mean_loss": running / step}), flush=True)
        metadata["stages"].append({"stage": "sft", "epoch": epoch + 1,
                                   "examples": len(sft), "mean_loss": running / len(loader)})
        save(model, args.output / "sft", metadata)

    preference = PreferenceDataset(tokenizer, preference_records(rows), args.max_length)
    loader = DataLoader(preference, batch_size=args.batch_size, shuffle=True,
                        collate_fn=preference_collator(tokenizer.pad_token_id))
    for epoch in range(args.preference_epochs):
        running = running_pref = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, pair in enumerate(loader, 1):
            chosen, rejected = move(pair["chosen"], device), move(pair["rejected"], device)
            chosen_logp = completion_logp(model, chosen)
            rejected_logp = completion_logp(model, rejected)
            chosen_nll = -chosen_logp.mean()
            pref = -F.logsigmoid(args.beta * (chosen_logp - rejected_logp)).mean()
            loss = (chosen_nll + args.preference_weight * pref) / args.gradient_accumulation
            loss.backward()
            running += float((chosen_nll + args.preference_weight * pref).detach())
            running_pref += float(pref.detach())
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(json.dumps({"stage": "preference", "epoch": epoch + 1,
                                  "step": step, "steps": len(loader),
                                  "mean_loss": running / step,
                                  "mean_preference_loss": running_pref / step}), flush=True)
        metadata["stages"].append({"stage": "preference", "epoch": epoch + 1,
                                   "pairs": len(preference), "mean_loss": running / len(loader),
                                   "mean_preference_loss": running_pref / len(loader)})
        save(model, args.output / "preference", metadata)

    props = torch.cuda.get_device_properties(0)
    metadata["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    metadata["gpu"] = props.name
    metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save(model, args.output / "preference", metadata)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
