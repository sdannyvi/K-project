from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_items(path: Path | None = None) -> list[dict]:
    source = path or ROOT / "data" / "items.jsonl"
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model(config: dict):
    device = config.get("device", "mps")
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    dtype = getattr(torch, config.get("dtype", "float16"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, device


def chat_tokens(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    return ids.to(device)


def transformer_layers(model):
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for parent, child in candidates:
        obj = getattr(model, parent, None)
        if obj is not None and hasattr(obj, child):
            return getattr(obj, child)
    raise ValueError("Unsupported model architecture: cannot locate transformer layers")


def hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden
