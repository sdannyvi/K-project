"""QLoRA memory/throughput calibration for one model on one GPU.

Answers whether E4 adaptation fits on a single 24 GB A5000 (multi-GPU sharding
is unsafe on hpc-vilenchik; see cluster/CLUSTER.md). For each batch size it runs
a few optimizer steps on fixed-length sequences and records peak VRAM and
seconds/step; an OOM is recorded, not fatal.

    STAGE_ARGS="--model Qwen/Qwen3-32B --revision <sha>" \
      bash cluster/scripts/submit_stage.sh calibrate_qlora
"""
from __future__ import annotations

import argparse
import time


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="calibrate_qlora")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq-len", type=int, default=512, help="prompt + ~150 generated tokens, padded up")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    return parser.parse_args(argv)


def run(ctx):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    args = parse(ctx.stage_args)
    if not torch.cuda.is_available():
        raise RuntimeError("calibrate_qlora needs a visible GPU")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    # Not peft.prepare_model_for_kbit_training: it upcasts embeddings and lm_head
    # to fp32 (~6 GB extra at 32B), which alone can break the 24 GB budget.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=args.target_modules.split(","), task_type="CAUSAL_LM"))
    load_seconds = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 1e9
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctx.log(f"loaded in {load_seconds:.0f}s, {weights_gb:.1f} GB, {trainable / 1e6:.1f}M trainable")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    vocab = model.config.vocab_size
    results = []
    for batch in map(int, args.batch_sizes.split(",")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, vocab, (batch, args.seq_len), device=0)
        try:
            times = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t1)
            step = sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else times[0]  # median, skip warm-up
            row = {"batch_size": batch, "ok": True, "seconds_per_step": round(step, 2),
                   "train_tokens_per_second": round(batch * args.seq_len / step, 1),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": batch, "ok": False, "error": "OOM",
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            optimizer.zero_grad(set_to_none=True)
        ctx.log(str(row))
        results.append(row)
        if not row["ok"] or ctx.stop_requested():
            break

    props = torch.cuda.get_device_properties(0)
    fits = [r for r in results if r["ok"] and r["peak_vram_gb"] <= 0.9 * props.total_memory / 1e9]
    return {
        "model": args.model, "revision": args.revision or model.config._commit_hash,
        "seq_len": args.seq_len, "rank": args.rank, "target_modules": args.target_modules,
        "trainable_params": trainable, "load_seconds": round(load_seconds, 1),
        "weights_vram_gb": round(weights_gb, 2), "batches": results,
        # Plan section 5 rule 7: keep 10% GPU-memory headroom.
        "max_batch_within_10pct_headroom": max((r["batch_size"] for r in fits), default=0),
        "gpu": props.name, "gpu_memory_gb": round(props.total_memory / 1e9, 1),
    }
