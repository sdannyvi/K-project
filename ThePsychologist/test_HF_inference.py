"""
Smoke test: load a model and answer one prompt. Checks HF access, GPU sharding and generation.

Usage:
    python test_inference.py                                  # DATA_MODEL from config.py
    python test_inference.py meta-llama/Llama-3.1-8B-Instruct # any model id
"""
import os
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import whoami
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()  # reads HF_TOKEN from .env

from config import DATA_MODEL, DEVICE_MAP, DTYPE

name = sys.argv[1] if len(sys.argv) > 1 else DATA_MODEL
token = os.getenv("HF_TOKEN")
print("HF_TOKEN:", f"set, {len(token)} chars, starts {token[:3]}" if token else "NOT set in the environment")
try:  # does the token reach huggingface_hub, and is it valid?
    print("HF user:", whoami()["name"])
except Exception as e:
    print("HF token missing or invalid:", type(e).__name__, str(e).splitlines()[0])
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, DTYPE), device_map=DEVICE_MAP)
print("devices used:", sorted(set(map(str, getattr(model, "hf_device_map", {}).values()))))

text = tok.apply_chat_template([{"role": "user", "content": "Say hello in one sentence."}],
                               add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
