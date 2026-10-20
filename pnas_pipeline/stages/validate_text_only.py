"""Validate removal of unused multimodal modules for text-only execution."""
from __future__ import annotations

import argparse
import gc
import time


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="validate_text_only")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--prompt", default="Describe how to prepare a cup of tea.")
    return parser.parse_args(argv)


def run(ctx):
    import torch

    from ..models import disable_unused_vision_modules, load_text_model

    args = parse(ctx.stage_args)
    if not torch.cuda.is_available():
        raise RuntimeError("validate_text_only needs a visible GPU")
    model, tokenizer = load_text_model(
        args.model, args.revision, args.quantization, disable_vision=False)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(0)
    with torch.inference_mode():
        before = model(**encoded, logits_to_keep=1).logits.detach().float().cpu()
    allocated_before = torch.cuda.memory_allocated()
    removed = disable_unused_vision_modules(model)
    if not removed:
        raise RuntimeError("no vision modules were removed")
    gc.collect()
    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    with torch.inference_mode():
        after = model(**encoded, logits_to_keep=1).logits.detach().float().cpu()
    max_abs_diff = (before - after).abs().max().item()
    exactly_equal = torch.equal(before, after)
    close = torch.allclose(before, after, rtol=0, atol=0)
    ctx.log(f"removed {removed}; freed {(allocated_before - allocated_after) / 1e9:.2f} GB; "
            f"max_abs_diff={max_abs_diff}")
    if not close:
        raise RuntimeError(f"text logits changed after disabling vision modules: {max_abs_diff}")
    return {
        "model": args.model,
        "revision": args.revision or model.config._commit_hash,
        "quantization": args.quantization,
        "removed_modules": removed,
        "allocated_before_gb": round(allocated_before / 1e9, 3),
        "allocated_after_gb": round(allocated_after / 1e9, 3),
        "freed_gb": round((allocated_before - allocated_after) / 1e9, 3),
        "max_abs_logit_difference": max_abs_diff,
        "exactly_equal": exactly_equal,
        "allclose_rtol_0_atol_0": close,
        "checked_at_unix": time.time(),
    }
