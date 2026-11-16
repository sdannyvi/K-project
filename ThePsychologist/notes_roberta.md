# Notes: RobertaGate (AnnoMI-trained gate)

Status: `RobertaGate` in `gate.py` is a stub. No model is trained. Switch with `GATE_BACKEND = "roberta"` in `config.py`.

## Contract (must match `gate.py`)
- Input: last `GATE_TURNS` turns as text (`format_turns`), truncated to 512 tokens.
- Model: `AutoModelForSequenceClassification` with `2 * len(state.TRAITS)` outputs:
  - outputs `0..n-1`: gate logit per trait (`> 0` means G = 1)
  - outputs `n..2n-1`: effect per trait, passed through `tanh` to [-1, 1], zeroed when G = 0
- Saved with `save_pretrained` to `GATE_ROBERTA_PATH`, so `RobertaGate` can load it.
- Output of `score(turns)` is `(G, effect)` dicts, the same as `LLMGate`.

## To do
1. **Labels.** AnnoMI marks therapist moves and quality (high/low), not traits.
   - Needs a technique -> trait mapping (CBT literature). This mapping is itself a thesis contribution.
   - AnnoMI is motivational interviewing, not CBT (see `docs/critique.md`). Consider a small clinician-annotated set.
2. **Bootstrap labels with the LLM gate.** Run `LLMGate` on AnnoMI windows, then compare with the mapped annotations.
   - Agreement or disagreement between the two is a result worth reporting.
3. **Loss.** BCE on gate logits, MSE on effect (only where G = 1). Effect needs a signed label: map low-quality or harmful moves to negative values.
4. **Windows.** Cut transcripts into `GATE_TURNS`-sized windows, matching what the gate sees at inference. The 512-token limit applies.
5. **Evaluate.**
   - Run `python gate.py check` with this backend: good, neutral and harmful therapists should separate.
   - Measure classifier error against held-out annotations, since the gate becomes the patient's ground truth.
6. **Domain shift.** AnnoMI transcripts are human-human. Check that the gate still works on LLM-patient sessions.

## Open questions
- Is `gate` a hard 0/1 or a probability? A soft gate may train and update better.
- One shared classifier head set, or one model per trait?
- Learned alpha chooser (critique note #2): train it on this gate's outputs after the fixed `state.update` baseline works.
