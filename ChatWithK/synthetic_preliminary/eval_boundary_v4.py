"""Teacher-forced decision-boundary diagnostics on synthetic development rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from pnas_pipeline.models import load_text_model
from synthetic_preliminary.train_boundary_ablation import encode_row


def eos_stats(scores: torch.Tensor, eos_id: int) -> dict:
    scores = scores.float()
    eos = scores[eos_id]
    return {
        "eos_probability": float(torch.softmax(scores, dim=-1)[eos_id]),
        "eos_rank": int((scores > eos).sum()) + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.open()]
    rows = [row for row in rows if row["split"] == "development"]
    assert len(rows) == 800
    model, tokenizer = load_text_model(
        "Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "4bit"
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            item = encode_row(tokenizer, row, args.max_length)
            ids = torch.tensor([item["input_ids"]], device="cuda:0")
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0]
            result = {
                "id": row["id"],
                "group_id": row["group_id"],
                "variant": row["variant"],
                "stop_at_decision": row["stop_at_decision"],
                "decision_answer_tokens": item["decision_answer_tokens"],
                "shared": eos_stats(logits[item["shared_index"]], tokenizer.eos_token_id),
                "decision": eos_stats(logits[item["decision_index"]], tokenizer.eos_token_id),
            }
            handle.write(json.dumps(result) + "\n")
            if index % 100 == 0:
                print(index, flush=True)


if __name__ == "__main__":
    main()
