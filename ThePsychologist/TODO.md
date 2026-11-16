# TODO

- [ ] Optional: after creating LoRa adapter, test it's results with SAE architecture, view the concept that is been steered. Test it on a natural llm. View how alpha changes the results of the SAE. // Look at the attention.
- [ ] When adapters are done, check the traits that been used and validate them with the SAE results. (Think of a hexagonal shape where each trait is pulling)


- [ ] Run `python test_state.py`, then `python adapters.py train depression` and
      `python adapters.py sweep depression`. Go/no-go: PHQ-9 must rise monotonically with alpha.
      If it does not, rethink weights-as-state (try higher LoRA rank, more/better data, more epochs).
- [ ] Add personas to `PERSONAS` in `adapters.py` for `anxiety`, `anhedonia`, `hopelessness`
      (the CLI only accepts traits listed there) and train one adapter each.
      `sweep` currently scores PHQ-9 only; non-depression traits need their own questionnaire
      (e.g. GAD-7 for anxiety).
- [ ] Multi-adapter mixing: combine several trait adapters with per-trait alpha from `state.py`
      (`set_alpha` currently scales a single adapter).
- [ ] Write `evaluate.py`: compare static vs prompt-state vs weights-state patients across sessions.
- [ ] Expand `QUESTIONS` in `adapters.py` (only 20 therapist turns now) or supply real data in `data/<trait>.jsonl`.
