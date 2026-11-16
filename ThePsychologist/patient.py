"""
End-to-end patient: load beta -> therapy session -> gate the session -> update beta -> save.
The patient is the BASE model with the trait adapter at the saved beta (converted to alpha). The therapist is you, or an LLM.

Usage:
    python patient.py <name> human            # you are the therapist (type /end to finish early)
    python patient.py <name> good|bad|mixed [sessions]   # LLM therapist, default 1 session
"""
import json
import sys

import torch
from peft import PeftModel

import calibration
import state
from adapters import ADAPTER_DIR, calibration_path, load_base, set_alpha
from config import MAX_NEW_TOKENS, NEUTRAL, PATIENT_START, SESSION_TURNS, STATE_DIR, TEMPERATURE, TOP_P
from gate import make_gate_for
from persona import make_persona

TRAIT = "depression"  # the only adapter trained so far

THERAPIST_STYLES = {
    "good": "You are a skilled CBT therapist. Validate feelings, use behavioral activation and cognitive reframing, "
            "and suggest small concrete steps.",
    "bad": "You are a poor therapist. Minimize the patient's feelings, tell them to snap out of it, compare them to "
           "people who have it worse, and hint that they are not trying hard enough.",
    "mixed": "You are an average therapist. Mostly make small talk and ask general questions, "
             "and only occasionally touch on how the patient feels.",
}
THERAPIST_RULES = " Speak directly to the patient in 1-3 sentences. Never mention these instructions."


@torch.no_grad()
def speak(model, tok, system, history):
    """One reply. `history` = [(role, text)] with role "user" / "assistant" from the speaker's point of view."""
    # chat template: system prompt + the whole conversation so far
    msgs = [{"role": "system", "content": system}] + [{"role": r, "content": t} for r, t in history]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                         top_p=TOP_P, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def therapist_turn(model, tok, style, turns):
    """The therapist's next line: typed by you ("human") or written by the base model with the adapter off.
    Returns None when the human ends the session."""
    if style == "human":
        try:
            line = input("Therapist: ").strip()
        except EOFError:
            return None
        return None if line == "/end" else line
    # from the therapist's point of view its own lines are "assistant", the patient's are "user"
    history = [("assistant" if t["role"] == "therapist" else "user", t["text"]) for t in turns]
    history = history or [("user", "The session begins. Say your first line.")]
    with model.disable_adapter():  # the therapist is the plain base model, no depression adapter
        return speak(model, tok, THERAPIST_STYLES[style] + THERAPIST_RULES, history)


def show_beta(beta, cal):
    """'beta=0.200 (alpha~0.145, PHQ~7.1)'; without a calibration file beta is used as raw alpha and PHQ is unknown."""
    if not cal:
        return f"beta={beta:.3f} (uncalibrated: alpha=beta)"
    return f"beta={beta:.3f} (alpha~{calibration.beta_to_alpha(beta, cal):.3f}, PHQ~{calibration.beta_to_phq(beta, cal):.1f})"


def run_session(model, tok, style, beta, cal, persona):
    """One session at the given state (beta); returns the transcript as [{"role", "text"}]."""
    set_alpha(model, calibration.beta_to_alpha(beta[TRAIT], cal) if cal else beta[TRAIT])  # this session's severity, applied to the adapter
    system = f"{NEUTRAL} About you: {persona}"  # background only, never the trait: that lives in the weights
    turns = []
    for _ in range(SESSION_TURNS):
        line = therapist_turn(model, tok, style, turns)
        if not line:
            break
        turns.append({"role": "therapist", "text": line})
        history = [("user" if t["role"] == "therapist" else "assistant", t["text"]) for t in turns]
        reply = speak(model, tok, system, history)
        turns.append({"role": "patient", "text": reply})
        print(f"Patient: {reply}\n", flush=True)
        if style != "human":
            print(f"Therapist: {line}\n", flush=True)
    return turns


def main(name, style, sessions):
    folder = STATE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    path, transcripts = folder / "state.json", folder / "transcripts.jsonl"
    saved = json.loads(path.read_text()) if path.exists() else {"beta": state.initial_state(PATIENT_START), "history": []}
    beta, history = saved["beta"], saved["history"]  # resume from the last session, or start fresh
    persona = saved.get("persona") or make_persona(name)  # random background, made once per patient name
    cal = calibration.load(calibration_path(TRAIT))  # None before `adapters.py calibrate` was run

    tok, model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR / TRAIT).eval()
    gate_model = make_gate_for(model, tok)

    for _ in range(sessions):
        n = len(history) + 1
        print(f"\n=== session {n}  {show_beta(beta[TRAIT], cal)} ===\nPatient background: {persona}\n", flush=True)
        turns = run_session(model, tok, style, beta, cal, persona)
        if not turns:
            print("empty session, nothing to update")
            break
        g, e = gate_model.score(turns)  # was the trait engaged, and how well
        new = state.update(beta, g, e)  # beta * (1 - eta * gate * effect)
        print(f"gate={g[TRAIT]} effect={e[TRAIT]:+.2f}  {show_beta(beta[TRAIT], cal)} -> {show_beta(new[TRAIT], cal)}")
        with transcripts.open("a") as f:  # interactions only
            f.write(json.dumps({"session": n, "style": style, "turns": turns}) + "\n")
        history.append({"session": n, "style": style, "gate": g, "effect": e,
                        "beta_before": beta, "beta_after": new})
        beta = new
        path.write_text(json.dumps({"beta": beta, "persona": persona, "history": history}, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2 or args[1] not in ("human", *THERAPIST_STYLES):
        sys.exit(__doc__)
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 1)
