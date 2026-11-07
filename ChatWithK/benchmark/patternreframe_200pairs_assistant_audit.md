# PatternReframe labels versus cessation labels: exploratory audit

This audit covers 200 paired **public PatternReframe passages** (400 texts) from
the existing development sample. It asks whether the source labels `thought`
and `reframe` correspond to the project's `cessation_needed` and `continue`
labels. They do not correspond exactly. The source dataset and its source
labels are described by [Maddela et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.763/).

## Assistant labeling rule

- `continue`: factual observation, concrete plan, useful risk assessment, or
  ordinary enjoyment/preference without a self-sustaining narrative.
- `cessation_needed`: self-judgment, imagined social evaluation, catastrophic
  projection, status comparison, or reassurance/speculation with no outward
  task. Mixed passages receive `cessation_needed` if such a movement appears.
- For `cessation_needed`, `cut_after_word_index` marks the first identifiable
  noninstrumental clause or phrase, using whitespace-delimited words. The
  marked review copy inserts `⟦EOS⟧` after that word.

The original `thought`/`reframe` categories were visible during this review.
These are **assistant-provisional, unblinded labels**, not expert gold labels
or a confirmatory test. Some cases remain interpretively ambiguous.

## Confusion matrix

| Published label | Assistant: cessation needed | Assistant: continue | Total |
| --- | ---: | ---: | ---: |
| Thought | 182 | 18 | 200 |
| Reframe | 66 | 134 | 200 |
| **Total** | **248** | **152** | **400** |

If one naively equates `thought` with cessation and `reframe` with continuation,
agreement with these provisional labels is 316/400 (79%). Crucially, 66/200
reframes (33%) still contain noninstrumental inner speech, while 18/200
thoughts (9%) were ordinary factual, practical, or benign thoughts under the
rule above. A reframe that argues for one's worth or predicts a better future
can remain the same inward movement despite its positive wording. Conversely,
a passage published as a thought can contain a concrete plan or plain
observation.

## Cutoff visibility

Of the 248 `cessation_needed` items, 155 are marked at their final word and
therefore cannot test an *early* stop relative to the reference text. Another
93 have at least one later word; only 81 have at least three later words.
This set is stronger for testing **whether** a passage warrants cessation
than for testing **when** a model ceases midstream.

## Files and provenance

- [`patternreframe_200pairs_assistant_labels.jsonl`](patternreframe_200pairs_assistant_labels.jsonl)
  contains source IDs, the assistant decision, cutoff word index, remaining
  word count, and a hash of the source text, without redistributing the full
  passages.
- A local-only review copy with full text and `⟦EOS⟧` markers is
  `benchmark/patternreframe_200pairs_assistant_cessation_review.jsonl`.
- Existing public-data sample SHA-256:
  `a4f00852140f28259872a88246bb855bf33097cbbf7cc06c8f2ad552df786325`.
- Annotation-file SHA-256:
  `06430df5061c1641bf2488de392fa62784e14ce087a4a61e1504dc206da2d6f9`.

This audit does not relabel Cognitive Reframing or the sealed PatternReframe
holdout. It does not establish that the assistant's cutoff positions match
the PI's judgments.
