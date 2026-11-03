# Synthetic preliminary adaptation experiment

This is an exploratory, automatically labelled corpus that replaces the
human-labelled preliminary adaptation run. It does **not** replace H0a or any
confirmatory dataset in the P0 protocol.

Each record combines a neutral biography with either a practical demand or an
ego-threatening event. The targets distinguish:

1. `practical`: complete, task-directed continuation;
2. `cessation`: an ego reaction that is cut at an intentionally incomplete
   word boundary and followed directly by the tokenizer's native EOS;
3. `rumination`: a long self-maintaining psychological continuation used as a
   rejected response.

The corpus is authored and generated locally by GPT-6.1 Sol through curated
scenario and language banks. No external generator model or human annotation is
used. Split assignment occurs by scenario-template family before realization,
so held-out evaluation cannot reuse the training wording.

Build and validate:

```bash
python synthetic_preliminary/build_corpus.py \
  --output synthetic_preliminary/data/corpus.jsonl \
  --manifest synthetic_preliminary/data/manifest.json \
  --size 6000 --seed 20261103
python -m pytest -q synthetic_preliminary/tests
```

The intended adaptation target is `Qwen/Qwen3-14B` using one-A5000 4-bit
QLoRA. SFT learns the desired continuations; a subsequent adapter-relative DPO
stage uses practical/cessation responses as chosen and rumination as rejected.
