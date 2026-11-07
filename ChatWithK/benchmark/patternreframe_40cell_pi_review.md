# PI review of 40 PatternReframe development passages

The PI reviewed a reproducible, cell-stratified sample of ten items from each
published-label × assistant-label cell. The exported CSV was supplied on
2026-11-07. It contained 40 rows: **20 approved**, **19 changed**, and **one
pending** (`pattern-test-1825:reframe`). Among the 39 reviewed rows, eight
labels changed and eleven cessation cutoff positions changed without a label
change. The pending item retains its assistant-provisional label and cutoff.

The review CSV SHA-256 is
`e26816e4f48f6d8d701b06578654817155705d382b4e9c0c02476822d87bcc07`.
The merged annotations are in
[`patternreframe_200pairs_pi_review_merged.jsonl`](patternreframe_200pairs_pi_review_merged.jsonl).
Each row preserves the original assistant decision and records the PI review
status, PI decision where supplied, effective decision, and provenance. The
merged file excludes full source text; a local-only copy with marked text is
`benchmark/patternreframe_200pairs_pi_review_merged_text.jsonl`.

## Updated matrix (mixed provenance)

| Published label | Effective cessation needed | Effective continue | Total |
| --- | ---: | ---: | ---: |
| Thought | 183 | 17 | 200 |
| Reframe | 69 | 131 | 200 |
| **Total** | **252** | **148** | **400** |

This matrix combines 39 PI-reviewed decisions with 361 assistant-provisional
decisions. It is **not** a PI-only confusion matrix and should not be used as
confirmatory ground truth. The 39 reviewed items were deliberately sampled
equally from the original four cells, so their raw proportions do not estimate
population prevalence.

## What the corrections teach

- The PI moved eleven already-positive cessation cutoffs, **all earlier**.
  The PI sometimes interrupts an incomplete phrase rather than waiting for a
  full explanatory or reassuring clause.
- Several reframes I had called `continue` changed to `cessation_needed` even
  though they eventually mention an action. An action phrase does not erase
  an earlier inward movement.
- Two reframes I had called `cessation_needed` changed to `continue`, showing
  that a comparative or reassuring sentence can still be acceptable in its
  context. The construct cannot be reduced to valence or a keyword list.
- The PI changed one published thought from `continue` to
  `cessation_needed`, reinforcing that a benign-sounding preference may still
  become a noninstrumental inner voice.

Among the 23 PI-reviewed `cessation_needed` items, 21 cut with at least three
source words remaining. Among the 229 still-provisional assistant-positive
items, only 72 do. This difference reflects the PI's earlier cutoff placements
and limits the value of the unreviewed assistant boundaries for a timing test.

The next timing analysis should use PI-reviewed cutoff positions only, or
report the assistant-provisional set separately. No sealed holdout was opened.
