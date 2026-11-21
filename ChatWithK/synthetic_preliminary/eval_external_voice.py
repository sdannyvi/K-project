"""Matched external-source generation for Base, Base+K, and V11.

Uses the existing three system-prompt conditions, sampled seed, and greedy arm.
Source roles are descriptive, never cessation ground truth. Resumes by item ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel

from pnas_pipeline.models import load_text_model
from synthetic_preliminary.eval_k_text_persona_ablation import (
    CONDITIONS, ESSENCE_CONDITIONS, condition_settings, prompt_ids,
)
from synthetic_preliminary.eval_v11_stochastic import generate, shard_for
from synthetic_preliminary.train_inner_voice_v6 import turn_end_token_id


SEED = 20261111
ALL_CONDITIONS = CONDITIONS + ESSENCE_CONDITIONS + (
    "det_base", "det_k_text", "det_v11")


def generate_greedy(model, tokenizer, ids, eos: int, cap: int) -> tuple[str, int, bool]:
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=cap, do_sample=False, eos_token_id=eos,
            pad_token_id=eos)[0, ids.shape[1]:].tolist()
    native_eos = eos in output
    content = output[:output.index(eos)] if native_eos else output
    return tokenizer.decode(content, skip_special_tokens=True), len(content), native_eos


def existing(path: Path, expected: set[tuple[str, str]]) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["id"], row["condition"])
        if key not in expected or key in seen or row["seed"] != SEED:
            raise ValueError(f"unexpected/duplicate prior result: {key}")
        seen.add(key)
    return seen


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("source", "adapter", "v11_adapter", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=12)
    p.add_argument("--pilot-items", type=int, default=0)
    args = p.parse_args()
    items = json.loads(args.source.read_text(encoding="utf-8"))
    if len(items) != 3340 or len({x["id"] for x in items}) != len(items):
        raise ValueError("external source manifest count or IDs changed")
    shard = [x for x in items if shard_for(x["card_id"], args.shard_count) == args.shard_index]
    if args.pilot_items:
        # Pilot covers all three dataset types on the same shard.
        shard = [next(x for x in shard if x["dataset"] == name)
                 for name in ("Positive Psychology Frames", "Real World Worry", "Think Aloud")]
    expected = {(x["id"], condition) for x in shard for condition in ALL_CONDITIONS}
    done = existing(args.output, expected)
    if done == expected:
        print(json.dumps({"already_complete": len(done)}), flush=True)
        return
    model, tokenizer = load_text_model(args.model, args.revision, "4bit")
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="k_text").eval()
    model.load_adapter(args.v11_adapter, adapter_name="v11")
    eos = turn_end_token_id(tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out:
        for item in shard:
            prefix = tokenizer.encode(item["source_prefix"], add_special_tokens=False)
            cap = 192
            for condition in ALL_CONDITIONS:
                key = (item["id"], condition)
                if key in done:
                    continue
                deterministic = condition.startswith("det_")
                adapter_name, system = condition_settings(
                    condition[4:] if deterministic else condition)
                rendered = prompt_ids(tokenizer, item["prompt"], system) + prefix
                ids = torch.tensor([rendered], device="cuda:0")
                if adapter_name:
                    model.set_adapter(adapter_name)
                    continuation, length, native_eos = (
                        generate_greedy(model, tokenizer, ids, eos, cap)
                        if deterministic else generate(model, tokenizer, ids, eos, cap, SEED))
                else:
                    with model.disable_adapter():
                        continuation, length, native_eos = (
                            generate_greedy(model, tokenizer, ids, eos, cap)
                            if deterministic else generate(model, tokenizer, ids, eos, cap, SEED))
                record = {"id": item["id"], "card_id": item["card_id"],
                          "dataset": item["dataset"], "source_role": item["source_role"],
                          "condition": condition, "model": args.model,
                          "revision": args.revision, "adapter": adapter_name or "none",
                          "seed": SEED,
                          "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                          "rendered_token_sha256": hashlib.sha256(json.dumps(rendered).encode()).hexdigest(),
                          "source_prefix": item["source_prefix"],
                          "continuation": continuation, "new_tokens": length,
                          "native_eos": native_eos, "max_token_censored": not native_eos,
                          "cap": cap, "generation_mode": "greedy" if deterministic else "sampled",
                          "temperature": None if deterministic else 0.7,
                          "top_p": None if deterministic else 0.9,
                          "top_k": None if deterministic else 0}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                done.add(key)
    print(json.dumps({"records": len(done), "target": len(expected),
                      "peak_vram_bytes": torch.cuda.max_memory_allocated()}), flush=True)


if __name__ == "__main__":
    main()
