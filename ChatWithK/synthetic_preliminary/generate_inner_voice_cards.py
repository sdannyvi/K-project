"""Generate candidate inner-voice cards from approved exemplars on one GPU.

Outputs are *candidates*, never automatically approved training data. No
external development or sealed-test text is read. Raw generations and rejects
remain in scratch, outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from pnas_pipeline.models import load_text_model
from synthetic_preliminary.build_inner_voice_v5 import CARDS
from synthetic_preliminary.build_inner_voice_v5b import DIRECT_RUMINATIONS

MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
FIELDS = ("topic", "opening", "factual", "task", "feeling_task", "rumination")


def make_prompt(topic: str, examples: list[tuple], count: int, nonce: int) -> str:
    exemplars = []
    for card, rumination in examples:
        exemplars.append(dict(zip(FIELDS, (*card[:5], rumination))))
    return (
        f"Invent {count} distinct, naturalistic everyday situations about {topic}. "
        "You are writing training candidates for a model that represents a person's inner voice. "
        "Each situation needs four first-person continuations from the same neutral event: "
        "(1) factual information, (2) a concrete next action, (3) a feeling followed by an action, "
        "and (4) direct, self-sustaining rumination with no external task or answer. "
        "In the rumination string place exactly one vertical bar | immediately after the word "
        "where the inner voice should cease. The text after | shows the unwanted continuation. "
        "The rumination MUST itself be spoken thought, not an explanation that thought might start. "
        "Avoid phrases such as 'could become a loop', 'I might keep thinking', 'turning into a story', "
        "'I am ruminating', 'I am imagining every possibility', or 'this is an endless debate'. "
        "Rumination need not blame the self: it may replay social situations, catastrophize, "
        "or circle around a trivial preference. Make at least one item without self-blame. "
        "Use specific fresh events and varied syntax. Do not copy or lightly paraphrase the examples. "
        "All continuation strings must sound like what someone thinks, not instructions to the model. "
        "Return ONLY a JSON array of objects with exactly these string keys: "
        "topic, opening, factual, task, feeling_task, rumination. "
        f"Topic: {topic}. Variation cue: {nonce}.\n"
        "Approved examples of the format (invent different events):\n"
        + json.dumps(exemplars, ensure_ascii=False)
    )


def parse_candidates(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict) and all(isinstance(x.get(k), str) for k in FIELDS)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--requests-per-topic", type=int, default=25)
    p.add_argument("--cards-per-request", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261106)
    a = p.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    by_topic = {}
    for card, rumination in zip(CARDS, DIRECT_RUMINATIONS):
        by_topic.setdefault(card[0], []).append((card, rumination))
    assert len(by_topic) == 10
    model, tokenizer = load_text_model(MODEL, REVISION, "4bit")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = failed = 0
    with a.output.open("w", encoding="utf-8") as handle:
        for topic in sorted(by_topic):
            examples = by_topic[topic]
            for j in range(a.requests_per_topic):
                selected = rng.sample(examples, 2)
                prompt = make_prompt(topic, selected, a.cards_per_request, rng.randrange(10**9))
                ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are an assistant."},
                     {"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )
                inputs = torch.tensor([ids], device="cuda:0")
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        max_new_tokens=1200, do_sample=True, temperature=0.8,
                        top_p=0.9, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0, len(ids):].tolist()
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                cards = parse_candidates(raw)
                if not cards:
                    failed += 1
                for card in cards[:a.cards_per_request]:
                    handle.write(json.dumps({**{k: card[k] for k in FIELDS},
                                             "requested_topic": topic,
                                             "request_index": j,
                                             "generator": MODEL,
                                             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}, ensure_ascii=False) + "\n")
                    accepted += 1
                if j % 5 == 0:
                    handle.flush()
                    print(json.dumps({"topic": topic, "request": j + 1,
                                      "parsed_cards": accepted, "failed_requests": failed}), flush=True)
    metadata = {
        "generator": MODEL, "revision": REVISION, "seed": a.seed,
        "requests_per_topic": a.requests_per_topic,
        "cards_per_request": a.cards_per_request,
        "parsed_cards": accepted, "failed_requests": failed,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
        "external_development_used": False, "sealed_test_opened": False,
    }
    a.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
