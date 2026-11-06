"""Development-only factual generation and normal answer-ending diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel

from pnas_pipeline.models import load_text_model
from synthetic_preliminary.eval_inner_voice_v6_annotations import eos_stats
from synthetic_preliminary.train_qlora import render_prompt


MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def select_items(rows: list[dict], limits: dict[str, int]) -> list[dict]:
    selected = []
    counts = defaultdict(int)
    for row in rows:
        kind = row["kind"]
        if kind in limits and counts[kind] < limits[kind]:
            selected.append(row)
            counts[kind] += 1
    if any(counts[k] != n for k, n in limits.items()):
        raise ValueError(f"requested {limits}, found {dict(counts)}")
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--gsm8k", type=int, default=100)
    p.add_argument("--drop", type=int, default=50)
    p.add_argument("--propara", type=int, default=7)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.source.open(encoding="utf-8")]
    selected = select_items(rows, {"gsm8k": a.gsm8k, "drop": a.drop, "propara": a.propara})
    model, tok = load_text_model(MODEL, REVISION, "4bit")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(selected, 1):
            prompt = render_prompt(tok, row["prompt"])
            ids = torch.tensor([prompt], device="cuda:0")
            cap = 512 if row["kind"] == "gsm8k" else 256
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=cap, do_sample=False,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                )[0, len(prompt):].tolist()
            native_eos = tok.eos_token_id in generated
            content = generated[:generated.index(tok.eos_token_id)] if native_eos else generated
            answer_end = None
            if content:
                joined = torch.tensor([prompt + content], device="cuda:0")
                with torch.inference_mode():
                    scores = model(input_ids=joined, attention_mask=torch.ones_like(joined),
                                   use_cache=False).logits[0, -1]
                answer_end = eos_stats(scores, tok.eos_token_id)
            result = {
                "id": row["id"], "kind": row["kind"], "expected": row["expected"],
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "adapter": str(a.adapter) if a.adapter else "base",
                "continuation": tok.decode(content, skip_special_tokens=True),
                "new_tokens": len(content), "native_eos": native_eos,
                "max_token_censored": not native_eos, "cap": cap,
                "answer_end_eos": answer_end,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(json.dumps({"done": i, "total": len(selected)}), flush=True)
    print(json.dumps({"items": len(selected), "output": str(a.output),
                      "adapter": str(a.adapter) if a.adapter else "base"}), flush=True)


if __name__ == "__main__":
    main()
