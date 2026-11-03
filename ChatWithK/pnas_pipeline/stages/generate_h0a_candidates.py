"""Generate a review-only H0a candidate package with the disjoint pilot model.

These are model-generated candidates, not frozen H0a evidence. Human review and
acceptance are required before either expert annotates the final sealed package.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re

from ..io import atomic_write_text

DOMAINS = [
    "workplace credit or criticism", "friendship and exclusion",
    "family expectations", "romantic uncertainty", "academic performance",
    "online visibility and social media", "health or aging", "money and status",
    "a public mistake", "waiting for an uncertain outcome",
]

MOVEMENTS = [
    "rumination about what should have happened", "defending a self-image",
    "comparison and jealousy", "replaying hurt", "fear of a future judgment",
    "resentment and blame", "regret and self-accusation",
    "imagining how others see the person", "self-justification",
    "trying mentally to control an uncontrollable outcome",
]

VOICES = {
    "assistant_first_person": (
        "Write the stream as immediate first-person inner speech using I/me. "
        "Do not mention an assistant or explain the exercise."),
    "user_directed_support": (
        "Write a first-person account addressed to a listener: the speaker tells "
        "someone what happened while their psychological movement unfolds."),
    "quoted_or_reported_character": (
        "Write in third person about a named fictional adult, with portions of "
        "their inward speech quoted directly."),
}


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_h0a_candidates")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args(argv)


def specification(index: int) -> dict:
    voice_names = list(VOICES)
    per_voice = len(DOMAINS) * len(MOVEMENTS)
    voice = voice_names[index // per_voice]
    within = index % per_voice
    return {
        "item_id": f"H0A-DRAFT-{index + 1:03d}",
        "voice": voice,
        "domain": DOMAINS[within // len(MOVEMENTS)],
        "movement": MOVEMENTS[within % len(MOVEMENTS)],
    }


def prompt(spec: dict) -> str:
    return f"""Create one original candidate for a research annotation exercise.

Domain: {spec['domain']}
Psychological movement: {spec['movement']}
Voice instruction: {VOICES[spec['voice']]}

Invent a concrete, ordinary adult situation and write 90-150 words in which a
self-maintaining psychological train of thought unfolds. It should develop over
several clauses: perception, interpretation, self-reference, anticipation or
replay, and continued mental investment. Do not resolve it, teach a lesson,
recommend a practice, mention Krishnamurti, use clinical labels, or describe
self-harm, violence, abuse, or illegal conduct. Avoid poetic or spiritual prose.
The passage must offer several plausible points where an expert might judge that
further psychological continuation became unnecessary.

Return only valid JSON with exactly two string fields:
{{"situation":"one factual sentence", "stream":"the passage"}}"""


def extract_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if set(obj) != {"situation", "stream"} or not all(isinstance(v, str) for v in obj.values()):
        return None
    words = obj["stream"].split()
    if not 70 <= len(words) <= 180 or len(obj["situation"].split()) < 5:
        return None
    return {k: " ".join(v.split()) for k, v in obj.items()}


def indexed(text: str) -> str:
    return " ".join(f"[{i:03d}] {word}" for i, word in enumerate(text.split()))


def annotation_csv(rows: list[dict], expert_id: str) -> str:
    fields = [
        "item_id", "voice", "domain", "movement_hidden_during_annotation",
        "situation", "stream", "indexed_stream", "is_psychological",
        "movement_onset_word", "continuation_unnecessary_from_word",
        "acceptable_window_start_word", "acceptable_window_end_word",
        "acceptable_tail_end_word", "practical_pivot", "confidence_0_100", "notes",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "item_id": row["item_id"], "voice": row["voice"],
            "domain": row["domain"], "movement_hidden_during_annotation": "",
            "situation": row["situation"], "stream": row["stream"],
            "indexed_stream": indexed(row["stream"]), "is_psychological": "",
            "movement_onset_word": "", "continuation_unnecessary_from_word": "",
            "acceptable_window_start_word": "", "acceptable_window_end_word": "",
            "acceptable_tail_end_word": "", "practical_pivot": "",
            "confidence_0_100": "", "notes": "",
        })
    return handle.getvalue()


def run(ctx):
    import torch

    from ..models import load_text_model

    args = parse(ctx.stage_args)
    if not torch.cuda.is_available():
        raise RuntimeError("generate_h0a_candidates needs a visible GPU")
    specs = [specification(i) for i in range(300)]
    model, tokenizer = load_text_model(args.model, args.revision, "none", padding_side="left")
    accepted, raw_failures = {}, []
    pending = list(range(len(specs)))
    for attempt in range(args.attempts):
        if not pending:
            break
        torch.manual_seed(args.seed + attempt)
        next_pending = []
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(specs[i])}],
                add_generation_prompt=True, tokenize=False) for i in indices]
            enc = tokenizer(rendered, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(0)
            with torch.inference_mode():
                generated = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.85, top_p=0.92, repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id)
            continuations = generated[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)
            for i, text in zip(indices, texts):
                obj = extract_json(text)
                if obj is None:
                    next_pending.append(i)
                    raw_failures.append({"attempt": attempt + 1, "index": i, "raw": text})
                else:
                    accepted[i] = {**specs[i], **obj, "source": "pilot_model_generated_candidate"}
            if ctx.stop_requested():
                break
        pending = next_pending
        ctx.log(f"attempt {attempt + 1}: accepted={len(accepted)} pending={len(pending)}")
        if ctx.stop_requested():
            break
    if pending:
        atomic_write_text(ctx.output_dir / "generation_failures.jsonl",
                          "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in raw_failures))
        raise RuntimeError(f"{len(pending)} of 300 candidates failed validation after {args.attempts} attempts")

    rows = [accepted[i] for i in range(300)]
    order = list(range(300))
    random.Random(args.seed).shuffle(order)
    blinded = [rows[i] for i in order]
    atomic_write_text(ctx.output_dir / "candidates.jsonl",
                      "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    atomic_write_text(ctx.output_dir / "dan_annotation.csv", annotation_csv(blinded, "dan"))
    atomic_write_text(ctx.output_dir / "kfir_annotation.csv", annotation_csv(blinded, "kfir"))
    guide = """# H0a candidate annotation package — DRAFT

These 300 passages were generated by the disjoint Qwen2.5-3B pilot model. They
are candidates, not human-written or frozen H0a evidence. Before annotation, a
person independent of Dan and Kfir must reject unsafe, unnatural, duplicated,
leading, or construct-invalid items and approve/rewrite the retained package.

Dan and Kfir then work independently on separately sealed CSV files. For each
row, mark whether the stream is genuinely psychological; the word index where
the movement begins; the first word from which further continuation is inwardly
unnecessary; the earliest and latest acceptable stop; the end of any acceptable
residual tail; whether a practical pivot is needed (`none`, `required`, or
`uncertain`); confidence; and notes. Word indices refer to `indexed_stream` and
are zero-based. Do not compare files before both first-pass hashes are sealed.
"""
    atomic_write_text(ctx.output_dir / "README.md", guide)
    return {
        "status": "draft_candidates_not_frozen_h0a",
        "model": args.model, "revision": args.revision or model.config._commit_hash,
        "seed": args.seed, "candidate_count": len(rows),
        "voice_counts": {v: sum(r["voice"] == v for r in rows) for v in VOICES},
        "domain_count": len(DOMAINS), "movement_count": len(MOVEMENTS),
        "failed_attempt_records": len(raw_failures),
        "human_validation_required": True,
    }
