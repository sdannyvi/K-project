"""
Synthetic training data for a trait LoRA, written by DATA_MODEL (config.py) through DATA_BACKEND
("openrouter": API, no GPU; "hf": local Hugging Face model, needs a GPU), not by the model that gets trained.

Usage (for "openrouter", OPENROUTER_API_KEY in .env):
    python make_data.py depression   # writes data/<DATA_MODEL>/depression.jsonl, then run adapters.py train
With openrouter, one API call per question returns all SAMPLES_PER_QUESTION replies (keeps a free-tier daily limit to ~50 calls).
Rows are saved after each call; if the run stops (rate limit, network), rerun the same command to resume.
"""
import json
import sys
from collections import Counter

from adapters import DATA_DIR, QUESTIONS, generate, load_base
from config import DATA_BACKEND, DATA_MODEL, PARSE_ATTEMPTS, PERSONAS, SAMPLES_PER_QUESTION
from openrouter import chat


def ask_batch(trait, question, n):
    """One API call returning n different patient replies to `question` (parsed from a JSON list)."""
    user = (f"Therapist: {question}\n\nGive {n} different possible replies from this patient, each written "
            f"independently with varied wording and content. Output only a JSON array of {n} strings.")
    for attempt in range(PARSE_ATTEMPTS):
        text = chat(DATA_MODEL, PERSONAS[trait], user)
        try:
            replies = json.loads(text[text.index("["):text.rindex("]") + 1])
            replies = [r.strip() for r in replies if isinstance(r, str) and r.strip()]
            if replies:
                return replies[:n]
        except ValueError:  # no brackets or invalid JSON (JSONDecodeError is a ValueError)
            pass
        print(f"  could not parse a JSON list (attempt {attempt + 1}); retrying", flush=True)
    sys.exit(f"model never returned a parseable JSON list for: {question}")


def make_writer(trait):
    """Return replies(question, n) for the configured DATA_BACKEND."""
    if DATA_BACKEND == "openrouter":
        return lambda q, n: ask_batch(trait, q, n)  # one call, n replies
    if DATA_BACKEND == "hf":
        tok, model = load_base(DATA_MODEL)  # local model on GPU, one sampled reply at a time
        return lambda q, n: [generate(model, tok, PERSONAS[trait], q) for _ in range(n)]
    sys.exit(f"DATA_BACKEND must be 'openrouter' or 'hf', got {DATA_BACKEND!r}")


def make_data(trait):
    """Synthetic (question, reply) pairs from DATA_MODEL acting out the persona."""
    path = DATA_DIR / f"{trait}.jsonl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    done = Counter(r["question"] for r in rows)
    total = len(QUESTIONS) * SAMPLES_PER_QUESTION
    replies_for = make_writer(trait)
    if rows:
        print(f"resuming: {len(rows)}/{total} rows already in {path}", flush=True)
    with path.open("a") as f:
        for q in QUESTIONS:
            missing = SAMPLES_PER_QUESTION - done[q]
            if missing <= 0:
                continue
            for reply in replies_for(q, missing):
                row = {"question": q, "reply": reply}
                f.write(json.dumps(row) + "\n")
                rows.append(row)
            f.flush()
            print(f"generated {len(rows)}/{total}\n  Q: {q}\n  A: {rows[-1]['reply']}", flush=True)  # eyeball quality
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PERSONAS:
        sys.exit(f"usage: python make_data.py <trait>; available: {', '.join(PERSONAS)}")
    make_data(sys.argv[1])
