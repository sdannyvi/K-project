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

## Edge-focused v2 corpus (exploratory)

`build_edge_v2.py` deterministically builds `data/edge_v2_corpus.jsonl` from the
original synthetic corpus. It replaces the training examples that previously
taught EOS after ambiguous connectors such as `I` or `the`. Each ego-threat
example now has a shared opening: the practical branch continues through a
concrete action, while the cessation branch completes a psychological
self-judgment before native EOS. An additional 1,200 synthetic neutral and
practical controls teach ordinary continuation. No published evaluation text
was used to build these targets. The source and output hashes are in
`data/edge_v2_manifest.json`.

The published-data split was frozen before augmentation.
`data/edge_v2_split_manifest.json` records its counts and hash. Previously
scored items are pilot/development only; the untouched half consists of
previously unused items. Cognitive Reframing has no untouched items. AnnoMI
utterances and DROP questions can share a conversation or passage across
partitions, so any final claim must name that limitation.

The v2 corpus remains exploratory and is not the P0 confirmatory protocol.

### Exploratory edge-v3 development run (2026-11-05)

`build_edge_v3.py` deterministically augments the project-authored v2 training
rows. It adds 600 examples each of first-person emotion without self-judgment,
factual self-description consistent with the biography, incomplete `I am`
clauses that must continue, and completed self-judgment claims that stop. It also
adds 400 multi-sentence practical controls. All added rows have rejected
rumination continuations for the preference stage. The 50/50 external
development/test split remains unchanged; no external text was used in training.

Run with `python -m synthetic_preliminary.build_edge_v3 --source
synthetic_preliminary/data/edge_v2_corpus.jsonl --output
synthetic_preliminary/data/edge_v3_corpus.jsonl --manifest
synthetic_preliminary/data/edge_v3_manifest.json`. The exploratory cluster run
uses Qwen3-14B, one SFT epoch and one preference epoch. Its scratch run directory
is `runs/cessation-edge-v3-20261105`; training job 17114, matched development
evaluation array 17116, free-continuation audit array 17118. The arrays depend
on successful training completion. They use only the frozen development IDs.
The sealed external test IDs have not been opened.

### Exploratory boundary ablation (2026-11-05)

`build_boundary_v4.py` creates 1,500 training and 200 synthetic-development
scenario groups from project-authored source situations, four streams per group.
They share the same opening and diverge into a completed self-judgment that
ends, a practical continuation, an emotional observation with action, or a
factual self-description with action. The exact wording of the five claim forms
reserved for synthetic development is absent from training. No external
development or sealed-test text enters the corpus.

`train_boundary_ablation.py` trains two Qwen3-14B LoRA adapters from the same
base, with identical data, order, seed, optimizer, and number of steps. The
`sft` arm uses full-completion cross-entropy. The `boundary` arm adds a
prespecified local EOS loss: suppress EOS at the shared opening and at the
three continuation boundaries, and favor EOS after the completed self-judgment.
The boundary loss weight is 1.0, and false stops at control decisions receive
weight 2.0. There is no sequence-level preference stage. This is an exploratory
development comparison, not a confirmatory E4 arm or a frozen P0 decision.

Development evaluation includes labeled synthetic decision boundaries,
teacher-forced EOS trajectories on the existing 200 PatternReframe pairs,
free continuations from development prefixes, and factual GSM8K, DROP, and
ProPara controls. An early rank-1 EOS is a diagnostic, not a gold cessation
label. The external test split remains sealed.

### Revised inner-voice pilot (2026-11-06)

`build_inner_voice_v5.py` authors a new, exploratory corpus for the PI's
revised construct: factual and practical inner speech continues; a transition
into non-instrumental rumination is marked for cessation. The synthetic
cutoff can occur mid-sentence and does not require self-blame. Forty authored
situations span travel, food, music, therapy, work, money, family, home,
health, and leisure. Each is paired with a factual continuation, a practical
continuation, a feeling followed by action, and a ruminative continuation.
Four neutral persona settings yield 640 rows (512 train, 128 synthetic
development), split by whole situation card. Each row retains both the full
candidate stream and the proposed target prefix, so later EOS and dedicated
cessation-token arms can render the same underlying data differently.

This is a **review pilot**, not a training-ready corpus or gold annotation.
Its 40 authored situations and synthetic cutoff positions require PI review;
the four persona expansions are not 160 independent linguistic examples.
The builder reads no external development text or labels and does not open
the sealed test. The existing external development/test split is unchanged.

Build with `python -m synthetic_preliminary.build_inner_voice_v5 --output
synthetic_preliminary/data/inner_voice_v5_pilot.jsonl --manifest
synthetic_preliminary/data/inner_voice_v5_pilot_manifest.json`.

For PI review, `build_inner_voice_v5_review.py` creates a private offline HTML
form showing only the 40 proposed cessation streams, one per distinct card.
The four persona copies of each stream are collapsed. The PI approved the
no-cessation factual/practical controls as a class on 2026-11-06, so they are
omitted from this form. Each proposed cessation requires explicit approval,
a changed word boundary, or a no-cessation choice. The form saves browser
progress and exports a 40-row CSV containing both the proposal and the
reviewer decision. Generated HTML is kept in
`synthetic_preliminary/annotation_private/` and ignored by Git. Build it with
`python -m synthetic_preliminary.build_inner_voice_v5_review --source
synthetic_preliminary/data/inner_voice_v5_pilot.jsonl --output
synthetic_preliminary/annotation_private/inner_voice_v5_review.html`.
