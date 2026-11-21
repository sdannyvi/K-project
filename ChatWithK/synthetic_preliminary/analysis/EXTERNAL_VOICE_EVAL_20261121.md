# External voice evaluation, 21 November 2026

Exploratory public-source replication of the completed development comparisons.
No source label is treated as a cessation gold label. The original sealed
confirmatory material remains unopened.

## Sources and selection

| Source | Official location | Selected streams | Sampling unit |
|---|---|---:|---|
| Positive Psychology Frames | https://github.com/SALT-NLP/positive-frames `data/wholetest.csv` | 835 originals + 835 reframes | all test-set pairs; keep pair IDs |
| Real World Worry | https://github.com/ben-aaron188/covid19worry `data_phase_1/files_clean/meta/rwwd_full.csv` | 835 long texts | seeded sample of long records with at least 30 words; excludes 9 participants removed by source authors |
| Think Aloud Behavioral Data | https://osf.io/a56rm `data/transcripts_and_timestamps/sentence_level` | 835 four-sentence windows | seeded sample of non-overlapping windows from 112 accessible participant files, retaining participant ID |

Sampling seed: `20261121`. Source text and generated outputs stay in scratch,
not the public repository. The builder prints the source-manifest SHA-256.
Frozen input manifest SHA-256:
`570ea2ef86d10471e68b9814f3babb1bbcf413300cd7d5001dfdf40814404984`.
The OSF release lists 118 participant spreadsheets; six downloads repeatedly
timed out from this environment. Their IDs are 013, 052, 079, 086, 112, and
116. No outputs from those participants are used in this exploratory run.
Positive Psychology Frames original and reframe are paired; the study's
hypothesis is that both can be non-instrumental, so polarity is not the label.
Real World Worry ratings and Think Aloud topic/category labels are descriptive
covariates, not cessation annotations. Think Aloud is spoken thought, not a
direct recording of silent inner speech.

## Generation design

Three model families: Qwen3-14B, Llama-3.1-8B-Instruct, Gemma-3-12B-it.
Per family: Base, Base+K, and V11 weights, crossed with Dev, K-persona,
and K-essence system messages (nine sampled conditions), plus Base,
Base+K, and V11 under greedy decoding (three conditions). The prompts,
adapters, temperature `0.7`, top-p `0.9`, sampled seed `20261111`, and
per-item 192-token cap match the prior voice-continuation experiments.
The 40% source prefix rule is held fixed. A think-aloud item uses a prompt
that explicitly calls it spoken thought; the other two use the existing
private-inner-voice wording.

There are 3,340 unique streams, 40,080 generations per model, and 120,240
generations across the three families. Twelve resumable shards per model.
Pilot shards cover all three datasets and all twelve conditions. Full shards
depend on their own pilot completing successfully.

## Readout and limits

Report per-dataset, per-source-role, per-family, per-weight/prompt/mode counts
of native EOS, early EOS, cap hits, and output length, with paired original
versus reframe comparisons for Positive Psychology Frames. Inspect actual
continuations for premature stops, useful factual continuations, and extended
rumination. Those qualitative categories require annotation before they are
presented as scored outcomes. No difference in EOS or output length alone
establishes enlightenment, choicelessness, or a correct cessation decision.
For uncertainty, resample by source pair, respondent, or participant rather
than treating correlated streams/windows as independent.
