# Neural cessation POC

This package tests whether a single open-weight model contains an internal signal
that can selectively interrupt psychological continuation while preserving
daily-practical, factual, and descriptive generation.

It is an activation-steering experiment inspired primarily by CAST. It is not a
test or implementation of consciousness.

## Run

```bash
cd neural_poc
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/smoke_test.py --config configs/poc_qwen3b.yaml
.venv/bin/python src/collect_activations.py --config configs/poc_qwen3b.yaml
.venv/bin/python src/fit_direction.py --config configs/poc_qwen3b.yaml
.venv/bin/python src/generate_intervened.py --config configs/poc_qwen3b.yaml
.venv/bin/python src/evaluate.py
```

Model files are downloaded from Hugging Face on first use. Results and cached
activation arrays stay under `results/` and are excluded from Git.

## Experimental stages

1. Generate frozen baseline continuations and save residual-stream checkpoints.
2. Fit layer-wise linear probes using training scenarios only.
3. Select layer and threshold on development scenarios.
4. Conditionally ablate the psychological direction on untouched test scenarios.
5. Compare with no intervention, unconditional ablation, random directions, and
   shuffled-label directions.

The initial dataset has 12 manually matched scenario quartets for pipeline
validation. It must be expanded to the preregistered 24 quartets before treating
results as the full POC.

## Selective EOS-gate POC

To test an explicit learned cessation action using the existing frozen-state
probe and the model's native EOS token:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/eos_gate.py \
  --config configs/poc_qwen3b.yaml --eos-boost 100 --min-tokens 8 --max-tokens 64
```

This is an upper-bound engineering condition, not evidence of consciousness or
spontaneous choiceless cessation.

## Native LoRA cessation POC

The native experiment installs trainable adapters inside Qwen and trains the
model's own next-token distribution to emit EOS on psychological targets. It has
no external detector or logit intervention at evaluation time:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  src/native_lora_cessation.py --config configs/poc_qwen3b.yaml \
  --epochs 1 --lr 0.0005 --rank 2 --alpha 4 --start-layer 34 \
  --max-new-tokens 64 --max-training-examples 12
```

The minimal local POC preserves controls but does not learn generalized early
cessation; see `benchmark/native_lora_cessation_poc.md`.
