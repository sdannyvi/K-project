
# patient_sim

Proof of concept: a simulated psychotherapy patient whose clinical state lives in the model weights (one LoRA adapter per trait, mixed by a coefficient alpha) and changes across sessions only when a gate says the session addressed that trait.

## Files

| File | Purpose |
|---|---|
| `state.py` | Beta vector per trait, session update rule, JSON save/load. |
| `test_state.py` | Unit tests for `state.py`. |
| `make_data.py` | Generates the synthetic training data with the large `DATA_MODEL` via OpenRouter and saves it to `data/<DATA_MODEL>/<trait>.jsonl`. |
| `adapters.py` | Trains a trait LoRA on `BASE` (8B) from that data, scales it by alpha, checks it with PHQ-9 scoring (to see effect of alpha on trait severity). |
| `gate.py` | Gate: reads the last turns of a session and outputs, per trait, G (was the trait engaged?) and effect (-1..1, how well). Zero-shot HF LLM judge now; `GATE_BACKEND = "roberta"` swaps in an AnnoMI classifier (stub, see `notes_roberta.md`). `python gate.py check depression` tests it on scripted good/neutral/harmful therapists. |
| `patient.py` | End-to-end loop: load beta, run a session (you or an LLM as therapist), gate it, update beta, save to `states/`. |
| `notes_roberta.md` | Notes for training the AnnoMI RoBERTa gate. |
| `requirements.txt` | Python dependencies. |

* Planned, not written yet: `evaluate.py`.

## Project steps

| # | Step | What it does | Status |
|---|---|---|---|
| 1 | Train trait adapters | One LoRA per trait (depression, anxiety, anhedonia, hopelessness). Training data is synthetic: the base model plays a persona, then the persona prompt is removed so the trait lives in the weights (`adapters.py train`). | depression only, untested |
| 2 | Validate interpolation | Scale an adapter by alpha (0 = base model, 1 = full trait) and check that the PHQ-9 score rises steadily. This is the go/no-go check (`adapters.py sweep`; see `PHQ9.md`). | written, not run |

| 3 | Gating classifier | Reads a session transcript and outputs, per trait, a gate G (was the trait engaged?) and an effect (how well?) (`gate.py`). LLM judge works on the scripted check (harmful/good OK, neutral gets too high a gate/effect); AnnoMI RoBERTa still to train. | LLM judge done, RoBERTa not started |
| 4 | State controller | Holds one beta per trait, applies `beta <- beta * (1 - eta * G * effect)` after each session, saves/loads it as JSON (`state.py`, tested by `test_state.py`). | done |
| 5 | End-to-end patient | Chat loop: load beta, set the adapter (via alpha), talk to the therapist, then gate the session and update beta (`patient.py`). Depression adapter only. | written, not run |
| 6 | Evaluate | Compare static-prompt, prompt-state and weights-state patients across sessions (`evaluate.py`). | not started |

## Setup

```
pip install -r requirements.txt
```
Llama-3.2-3B-Instruct is gated: accept its license on Hugging Face, then paste your
token into `.env` (`HF_TOKEN=hf_...`). `.env` is loaded automatically by `adapters.py` (alternatively, run `hf auth login`). Needs a CUDA GPU.

## Train


```
python make_data.py depression
```


The training data is written by a larger model (`DATA_MODEL`, an OpenRouter model, key in `.env` as `OPENROUTER_API_KEY`; interrupted runs resume). The LoRA is then trained on the smaller `BASE` model (Llama-3.1-8B) from that file, so the data model is never trained.

* Syntetic data created once (per model for now)
* LoRa will always overwrite
* sweeper result will overwrite

```
python adapters.py train depression
```
Saves the adapter to `adapters/<BASE>/<DATA_MODEL>/depression/`, so adapters trained on different data never overwrite each other (the folder name is the model that wrote the data; e.g. `adapters/Llama-3.1-8B-Instruct/nemotron-3-ultra-550b-a55b-free/depression/`).

## Evaluate
depression LoRa:

```
python adapters.py sweep depression
```

Prints the patient's PHQ-9 score (0-27) for alpha = 0, 0.25, 0.5, 0.75, 1 and whether it rises monotonically. 
* This is the go/no-go check. 
* The full comparison (static vs prompt-state vs weights) will live in `evaluate.py`.

Saves results as csv to `results/<BASE>/<DATA_MODEL>/`, mirroring the adapter folder.


## Judge
```
python judge.py depression
```



## Gate and patient loop
```
python gate.py check depression          # do good / neutral / harmful therapists get different effects?
python patient.py anna human             # you are the therapist (/end to finish early)
python patient.py anna good|bad|mixed 5  # LLM therapist, 5 sessions
```
* Gate: G = 1 if the therapist engaged the trait at all (helpfully or harmfully), effect carries the quality. G = 0 leaves beta unchanged.
* `states/<name>/state.json` = current beta + patient persona + per-session history (gate, effect, beta before/after). `states/<name>/transcripts.jsonl` = only the therapist/patient turns, one line per session. Settings (`STATE_DIR`, `PATIENT_START`, `SESSION_TURNS`, `GATE_*`) are in `config.py`.
* The adapter used is chosen by `DATA_MODEL` in `config.py`.
* Calibration: `python adapters.py calibrate depression` sweeps alpha finely and saves a beta -> alpha map (`results/.../depression_calibration.json`) built by inverting the measured PHQ(alpha) curve. The patient state is now beta in [0, 1]: PHQ rises linearly from the alpha=0 baseline (beta=0) to near the plateau (beta=1), fixing the saturation around alpha 0.5. Run it after every retrain.
* Idea for later: a learned model that chooses alpha (critique note 2), compared against the fixed rule in `state.py`.

## Beta: how it maps back to alpha

`calibration.py` is not run directly: it is a helper that `python adapters.py calibrate depression` calls (and `patient.py` reads).

Alpha is the raw LoRA scale and its effect on PHQ is very uneven (steep near 0, flat after ~0.5). Beta is a 0-1 severity knob that is linear in PHQ.

1. `calibrate` measures `f(alpha)` = PHQ-9 on a fine grid (0, 0.05, ... 1) and forces it to be non-decreasing.
2. Anchors: `PHQ_lo = f(0)` (beta = 0) and `PHQ_hi = PHQ_lo + CALIB_FRACTION * (f_max - PHQ_lo)` (beta = 1).
3. Wanted PHQ for a beta: `target(beta) = PHQ_lo + beta * (PHQ_hi - PHQ_lo)` (a straight line).
4. Back to alpha: invert the measured curve, `alpha(beta) = f^-1(target(beta))`, by linear interpolation between grid points (`calibration.beta_to_alpha`).

So `PHQ(alpha(beta)) = target(beta)`: equal steps in beta give equal steps in PHQ. `state.py` and the gate only ever see beta; `patient.py` converts to alpha right before setting the adapter. Re-run `calibrate` after every retrain, since the curve changes.

**Why beta (note for the future learned policy):** each model / adapter reacts differently to the same alpha (different baseline, different saturation point). If a ML model later learns to choose the severity, it should not have to learn that per-model quirk. Beta puts every model on the same scale: 0 = its own baseline, 1 = its own near-max, and equal steps in beta = equal steps in PHQ. The policy outputs beta, and `calibrate` (run once per model / retrain) translates it into that model's alpha. Limit: beta is relative to each model's own range, so if one model peaks at PHQ 24 and another at 19, the same beta means a different absolute PHQ (beta=1 is 24 in one, 19 in the other).

## Training data

Path: `data/<trait>.jsonl` (e.g. `data/depression.jsonl`), one JSON object per line:

```
{"question": "How have you been sleeping?", "reply": "Not well. I lie awake for hours and still feel exhausted."}
```

- `question`: the therapist's turn. `reply`: the patient's answer showing the trait.
- If the file exists, training uses it. If not, `train` generates it synthetically
  with the base model. Delete the file to regenerate it.
- To use real data, write your own file in this format before running `train`.



### My open problems:

* Chat with patient - not good enough, LoRa doesnt work perfectly.
  bigger LoRa rank + new dataset? Check with AI. (TODO states so.)
* Train Robeta - that will know to change gate and effect, not a llm judge.
* Ceck the prossible dataset for futer work.