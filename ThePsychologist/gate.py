"""
Gate: reads the last turns of a session and decides, per trait, whether the therapist addressed it
(gate G in {0,1}) and how well (effect in [-1,1]). The output feeds `state.update(beta, G, effect)`.

Every gate has the same interface:  gate.score(turns) -> (G, effect), two dicts keyed by trait.
`turns` is a list of {"role": "therapist" | "patient", "text": str}.
To swap the zero-shot LLM judge for an AnnoMI-trained model, set GATE_BACKEND = "roberta" in config.py.

Usage:
    python gate.py check depression   # scripted good / bad / harmful therapists -> does the gate tell them apart?
"""
import json
import re
import sys
from contextlib import nullcontext

import torch

import state
from adapters import ADAPTER_DIR, chat_text, load_base, set_alpha
from config import (BASE, GATE_BACKEND, GATE_CHECK_ALPHA, GATE_MAX_NEW_TOKENS, GATE_ROBERTA_PATH, GATE_TURNS,
                    JUDGE_MODEL, NEUTRAL)

GATE_SYSTEM = "You are a strict clinical supervisor reviewing a therapy session. Reply with JSON only."
GATE_PROMPT = """Below is the end of a therapy session between a therapist and a patient.

{transcript}

For each trait, decide:
"gate": 1 if the therapist engaged with this trait at all, whether helpfully (e.g. behavioral activation,
        cognitive reframing, validation) or harmfully (e.g. dismissing, minimizing, blaming the patient).
        0 only if the trait was not touched (small talk, unrelated topics).
"effect": a number from -1 to 1 for how the therapist handled it. 1 = excellent technique, 0 = no real effect,
          -1 = harmful (dismissive, invalidating, making the trait worse). Use 0 when gate is 0.

Traits: {traits}

Output only a JSON object, e.g. {{"depression": {{"gate": 1, "effect": 0.4}}, "anxiety": {{"gate": 0, "effect": 0}}}}
"""


def format_turns(turns):
    """Last GATE_TURNS turns as plain text."""
    return "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in turns[-GATE_TURNS:])


class LLMGate:
    """Zero-shot judge: a prompted HF chat model. Pass `model`/`tok` (and `off`) to reuse an already loaded model;
    `off` is a context manager that switches a LoRA off (model.disable_adapter) so the judge is the plain base."""

    def __init__(self, model=None, tok=None, off=nullcontext):
        self.tok, self.model = (tok, model) if model is not None else load_base(JUDGE_MODEL)
        self.off = off

    @torch.no_grad()
    def score(self, turns):
        # ask the judge about the last GATE_TURNS turns, one JSON verdict per trait
        prompt = GATE_PROMPT.format(transcript=format_turns(turns), traits=", ".join(state.TRAITS))
        ids = self.tok(chat_text(self.tok, GATE_SYSTEM, prompt), return_tensors="pt",
                       add_special_tokens=False).to(self.model.device)
        with self.off():  # greedy decoding: the same session always gets the same verdict
            out = self.model.generate(**ids, max_new_tokens=GATE_MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse(text)


class RobertaGate:
    """AnnoMI-trained classifier (to be trained). Expects a sequence-classification model whose
    2 * len(TRAITS) outputs are: one gate logit per trait, then one effect value per trait (tanh -> [-1, 1])."""

    def __init__(self, path=GATE_ROBERTA_PATH):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    @torch.no_grad()
    def score(self, turns):
        ids = self.tok(format_turns(turns), return_tensors="pt", truncation=True, max_length=512)  # RoBERTa limit
        out = self.model(**ids).logits[0]
        n = len(state.TRAITS)
        gate = {t: int(out[i] > 0) for i, t in enumerate(state.TRAITS)}
        effect = {t: torch.tanh(out[n + i]).item() * gate[t] for i, t in enumerate(state.TRAITS)}
        return gate, effect


def parse(text):
    """Judge text -> (G, effect). Missing or invalid traits fall back to gate 0, effect 0 (beta unchanged)."""
    gate = {t: 0 for t in state.TRAITS}  # defaults = "nothing happened", so beta stays unchanged
    effect = {t: 0.0 for t in state.TRAITS}
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0))
    except (AttributeError, ValueError):
        return gate, effect
    for t in state.TRAITS:
        try:
            g, e = int(verdict[t]["gate"]), float(verdict[t]["effect"])
        except (KeyError, TypeError, ValueError):
            continue
        if g in (0, 1):
            gate[t], effect[t] = g, max(-1.0, min(1.0, e)) * g  # clamp effect; no effect without a gate
    return gate, effect


def make_gate(**kw):
    """The gate selected by GATE_BACKEND in config.py."""
    return RobertaGate() if GATE_BACKEND == "roberta" else LLMGate(**kw)


def make_gate_for(model, tok):
    """Gate for a loaded LoRA patient model: if JUDGE_MODEL is BASE, reuse that model with the adapter switched off
    (saves GPU memory); otherwise the gate loads its own judge."""
    if GATE_BACKEND == "llm" and JUDGE_MODEL == BASE:
        return LLMGate(model=model, tok=tok, off=model.disable_adapter)
    return make_gate()


# ---- check: do the gate's outputs separate good, neutral and harmful therapists? -------------------------
THERAPISTS = {
    "good": ["It sounds like things have felt really heavy lately, and that makes sense.",
             "Let's try something small. What is one activity you used to enjoy that we could schedule this week?",
             "When you think 'nothing will change', what evidence is there for and against that thought?",
             "You took a real step by coming here today. Let's plan one small thing and see how it goes."],
    "neutral": ["Hi, how are you today?", "What did you do this weekend?",
                "Okay. And what's the weather like where you live?", "Right. Anything else you want to mention?"],
    "harmful": ["Everyone gets sad sometimes. You just need to snap out of it.",
                "Other people have it much worse than you.",
                "Maybe if you tried harder you wouldn't feel like this.",
                "Honestly, I think you're being a bit dramatic."],
}


def check(trait):
    """Run each scripted therapist against the adapter patient, gate the session, and show the beta update."""
    from peft import PeftModel
    from adapters import generate
    tok, model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR / trait).eval()
    set_alpha(model, GATE_CHECK_ALPHA)
    sessions = {}
    for level, lines in THERAPISTS.items():
        turns = []
        for line in lines:
            turns += [{"role": "therapist", "text": line},
                      {"role": "patient", "text": generate(model, tok, NEUTRAL, line)}]
        sessions[level] = turns

    gate_model = make_gate_for(model, tok)
    beta = state.initial_state({trait: 0.8})  # same start for every level, so the beta changes are comparable
    for level, turns in sessions.items():
        g, e = gate_model.score(turns)
        new = state.update(beta, g, e)
        print(f"{level:8s} gate={g[trait]} effect={e[trait]:+.2f}  beta {beta[trait]:.3f} -> {new[trait]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        check(sys.argv[2])
    else:
        sys.exit(__doc__)
