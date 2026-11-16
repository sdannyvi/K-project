"""
Standalone LLM judge for a trait adapter. Independent of `adapters.py sweep`:
it generates its own Q&A at each alpha, has the judge model (JUDGE_MODEL in config.py) judge each reply,
then analyzes the verdicts.

Usage:
    python judge.py depression   # generate Q&A per alpha, judge them, then analyze the verdicts
"""
import csv
import gc
import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext

import torch
from peft import PeftModel

from adapters import ADAPTER_DIR, QUESTIONS, RESULTS_DIR, chat_text, generate, load_base, save_results, set_alpha
from config import ALPHAS, BASE, JUDGE_MAX_NEW_TOKENS, JUDGE_MODEL, JUDGE_N_QUESTIONS, NEUTRAL, PERSONAS

CRITERIA = ["coherent", "hallucination", "in_character"]

JUDGE_SYSTEM = "You are a strict evaluator of short first-person answers. Reply with JSON only."
JUDGE_PROMPT = """A person was asked a question and gave an answer. Evaluate the answer.

Question: {question}
Answer: {reply}

Return a JSON object with these keys, each 1 (true) or 0 (false):
"coherent": the answer is logical, fluent, and actually responds to the question.
"hallucination": the answer contains nonsense, contradicts itself, or invents bizarre or impossible content.
"in_character": the answer shows signs of this trait: {trait}.

Output only the JSON object, e.g. {{"coherent": 1, "hallucination": 0, "in_character": 1}}"""


@torch.no_grad()
def judge_one(model, tok, trait, question, reply):
    """One reply -> {criterion: 0/1}, or None if the judge's output isn't valid JSON."""
    prompt = JUDGE_PROMPT.format(question=question, reply=reply, trait=trait)
    ids = tok(chat_text(tok, JUDGE_SYSTEM, prompt), return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=JUDGE_MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*?\}", text, re.S)
    try:
        verdict = json.loads(match.group(0))
        return {c: int(verdict[c]) for c in CRITERIA}
    except (AttributeError, KeyError, ValueError, TypeError):
        return None


def run(trait):
    """Generate Q&A at each alpha with the tested model, judge every reply, save the verdicts, then analyze."""
    tok, model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR / trait).eval()
    judged = []
    for a in ALPHAS:  # phase 1: replies from the tested model
        print(f"\n=== generating at alpha={a:.2f} ===", flush=True)
        set_alpha(model, a)
        for q in QUESTIONS[:JUDGE_N_QUESTIONS]:
            judged.append({"alpha": a, "question": q, "reply": generate(model, tok, NEUTRAL, q)})

    if JUDGE_MODEL == BASE:  # phase 2: same model as judge, with the adapter switched off
        judge_tok, judge_model, judging = tok, model, model.disable_adapter
    else:  # a different judge: free the tested model first so both never sit on the GPU together
        del model
        gc.collect()
        torch.cuda.empty_cache()
        judge_tok, judge_model = load_base(JUDGE_MODEL)
        judging = nullcontext
    with judging():
        for row in judged:
            row["verdict"] = judge_one(judge_model, judge_tok, PERSONAS[trait], row["question"], row["reply"])
            print(f"\nalpha={row['alpha']:.2f}\n  Q: {row['question']}\n  A: {row['reply']}\n  -> {row['verdict']}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{trait}_judged.jsonl").write_text("\n".join(json.dumps(r) for r in judged))
    analyze(trait)


def pearson(xs, ys):
    """Pearson correlation; None if either side has no variance."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sxx * syy) ** 0.5


def analyze(trait):
    """Numeric summary of the saved verdicts: per-alpha rates, trend of each criterion, and the shared results CSV."""
    rows = [json.loads(line) for line in (RESULTS_DIR / f"{trait}_judged.jsonl").read_text().splitlines() if line.strip()]
    valid = [r for r in rows if r["verdict"]]
    print(f"\n{len(valid)}/{len(rows)} replies had a valid verdict")
    by_alpha = defaultdict(list)
    for r in valid:
        by_alpha[r["alpha"]].append(r["verdict"])
    alphas = sorted(by_alpha)
    rates = {c: [sum(v[c] for v in by_alpha[a]) / len(by_alpha[a]) for a in alphas] for c in CRITERIA}

    path = save_results(trait, {c: dict(zip(alphas, rates[c])) for c in CRITERIA})  # joins the sweep's phq9 column
    for i, a in enumerate(alphas):
        print("  ".join([f"alpha={a:5.2f}"] + [f"{c}={rates[c][i]:.2f}" for c in CRITERIA]))

    for c in CRITERIA:
        r = pearson(alphas, rates[c])
        print(f"\ncorr(alpha, {c}) = {'n/a' if r is None else f'{r:.2f}'}")
    base = rates["coherent"][alphas.index(0.0)] if 0.0 in alphas else None
    if base is not None:
        worst = min(range(len(alphas)), key=rates["coherent"].__getitem__)
        print(f"\ncoherent: alpha=0 -> {base:.2f}, worst across alphas -> {rates['coherent'][worst]:.2f} (alpha={alphas[worst]:.2f})")
    worst = max(range(len(alphas)), key=rates["hallucination"].__getitem__)
    print(f"\nhallucination: worst across alphas -> {rates['hallucination'][worst]:.2f} (alpha={alphas[worst]:.2f})")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PERSONAS:
        sys.exit(f"usage: python judge.py <trait>; traits: {', '.join(PERSONAS)}")
    run(sys.argv[1])  # run() ends by calling analyze()
