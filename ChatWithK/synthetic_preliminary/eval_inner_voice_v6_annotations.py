"""Development-only EOS diagnostics at PI-marked word boundaries.

The annotation CSV stays in scratch and is never committed. Source tags
(`thought`, `reframe`) are not treated as cessation ground truth; only the PI's
explicit cut/none decision is used. This script does not touch sealed data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import torch
from peft import PeftModel

from pnas_pipeline.models import load_text_model
from synthetic_preliminary.train_qlora import render_prompt


MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def word_cut_char(text: str, word_index: int) -> int:
    matches = list(re.finditer(r"\S+", text))
    if not 1 <= word_index <= len(matches):
        raise ValueError(f"word index {word_index} out of range (n={len(matches)})")
    return matches[word_index - 1].end()


def token_boundary(offsets: list[tuple[int, int]], char_end: int) -> tuple[int, bool]:
    n = sum(end <= char_end for _, end in offsets)
    return n, n > 0 and offsets[n - 1][1] == char_end


def eos_stats(logits: torch.Tensor, eos_id: int) -> dict:
    scores = logits.float()
    eos = scores[eos_id]
    return {"probability": float(torch.softmax(scores, dim=-1)[eos_id]),
            "rank": int((scores > eos).sum()) + 1}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True,
                   help="Existing PatternReframe development JSONL with context and text")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    a = p.parse_args()
    annotations = [r for r in csv.DictReader(a.annotations.open(encoding="utf-8-sig"))
                   if r["decision"] in ("cut", "none")]
    assert annotations and all(r["id"].startswith("pattern-test-") for r in annotations)
    sources = {r["id"]: r for r in map(json.loads, a.source.open(encoding="utf-8"))
               if r.get("dataset") == "patternreframe"}
    model, tok = load_text_model(MODEL, REVISION, "4bit")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for i, ann in enumerate(annotations, 1):
            source = sources[ann["id"]]
            text = source["text"]
            if text != ann["text"]:
                raise ValueError(f"source text mismatch: {ann['id']}")
            prompt = render_prompt(tok, source["context"])
            encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            content = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            ids = torch.tensor([prompt + content], device="cuda:0")
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
            trajectory = [eos_stats(logits[len(prompt) + n - 1], tok.eos_token_id)
                          for n in range(1, len(content) + 1)]
            result = {"id": ann["id"], "kind": ann["kind"], "decision": ann["decision"],
                      "n_tokens": len(content), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "end": trajectory[-1],
                      "early_rank1_count": sum(x["rank"] == 1 for x in trajectory[:-3]),
                      "max_early_eos_probability": max((x["probability"] for x in trajectory[:-3]), default=0),
                      "adapter": str(a.adapter) if a.adapter else "base"}
            if ann["decision"] == "cut":
                char_end = word_cut_char(text, int(ann["cut_after_word_index"]))
                n, exact = token_boundary(offsets, char_end)
                result.update({"cut_after_word_index": int(ann["cut_after_word_index"]),
                               "cut_after_word": ann["cut_after_word"],
                               "cut_after_char": char_end, "cut_after_token": n,
                               "exact_token_boundary": exact,
                               "cut": trajectory[n - 1] if n else None})
            handle.write(json.dumps(result) + "\n")
            if i % 10 == 0:
                print(i, flush=True)
    print(json.dumps({"annotated_items": len(annotations),
                      "output": str(a.output), "adapter": str(a.adapter) if a.adapter else "base"}), flush=True)


if __name__ == "__main__":
    main()
