"""100-example memory/throughput calibration (plan section 5) for one model.

Writes metrics.json with generated tokens/second and peak VRAM, the inputs the
COMPUTE_BUDGET.csv rows need. Prompts come from neural_poc/data/items.jsonl until
the SNCC corpus exists; length is forced to --max-new-tokens so tokens/second is
not confounded by early EOS.

    STAGE=calibrate_throughput STAGE_ARGS="--model Qwen/Qwen3-32B --quantization 4bit" \
      bash cluster/scripts/submit_stage.sh calibrate_throughput
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="calibrate_throughput")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--items", default="neural_poc/data/items.jsonl")
    return parser.parse_args(argv)


def run(ctx):
    import torch

    from ..models import load_text_model

    args = parse(ctx.stage_args)
    if not torch.cuda.is_available():
        raise RuntimeError("calibrate_throughput needs a visible GPU")
    prompts = [json.loads(line)["prompt"] for line in Path(ctx.repo / args.items).read_text().splitlines()]
    prompts = list(itertools.islice(itertools.cycle(prompts), args.n_examples))

    t0 = time.perf_counter()
    model, tokenizer = load_text_model(args.model, args.revision, args.quantization)
    load_seconds = time.perf_counter() - t0
    ctx.log(f"loaded {args.model} in {load_seconds:.0f}s")

    torch.cuda.reset_peak_memory_stats()
    new_tokens, gen_seconds = 0, 0.0
    for start in range(0, len(prompts), args.batch_size):
        batch = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                               add_generation_prompt=True, tokenize=False)
                 for p in prompts[start:start + args.batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 min_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        gen_seconds += time.perf_counter() - t1
        new_tokens += (out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0]
        if ctx.stop_requested():
            ctx.log("stop requested; recording partial calibration")
            break

    props = torch.cuda.get_device_properties(0)
    return {
        "model": args.model, "revision": args.revision or model.config._commit_hash,
        "quantization": args.quantization, "batch_size": args.batch_size,
        "examples": min(len(prompts), start + args.batch_size), "max_new_tokens": args.max_new_tokens,
        "generated_tokens": new_tokens, "generation_seconds": round(gen_seconds, 2),
        "tokens_per_second": round(new_tokens / gen_seconds, 2),
        "load_seconds": round(load_seconds, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": props.name, "gpu_memory_gb": round(props.total_memory / 1e9, 1),
    }
