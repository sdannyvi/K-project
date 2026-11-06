# PatternReframe development: paired text-end EOS contrast

Date: 2026-11-06. Exploratory development result; sealed test untouched.
Source: ECE scratch run `boundary-ablation-v4-20261105`, files
`pattern_dev_sft.jsonl` and `pattern_dev_boundary.jsonl`. Both contain the
same 200 PatternReframe thought/reframe scenario pairs. The analysis used the
last teacher-forced boundary in each text, clipped probability to
`[1e-12, 1-1e-12]`, and calculated log-odds as `log(p/(1-p))`.

| Arm | Median thought-end log-odds | Median reframe-end log-odds | Mean paired thought-minus-reframe gap | Median paired gap |
| --- | ---: | ---: | ---: | ---: |
| Ordinary SFT | 0.172 | -0.481 | 0.462 | 0.647 |
| Local EOS boundary | 6.880 | 0.157 | 4.708 | 4.482 |

Across the same 200 scenario pairs, the boundary arm's paired gap exceeds the
ordinary arm's by mean **4.246 log-odds** (median 3.995). A scenario-pair
bootstrap with seed `20261106` and 10,000 resamples gave a percentile 95%
interval **[3.509, 4.981]** for the mean difference. This establishes a
stronger *text-end* contrast in the boundary arm. It does not establish
recognition of a completed psychological claim: valence, punctuation, and
utterance ending can explain some or all of the contrast. Both arms were
unanchored by a factual/math retention loss.

Sentence-structure audit of the 200 thought texts: 111 have one `[.!?]`
mark, 79 have two, 3 have three, and 7 have none. A preliminary diagnostic
incorrectly reported zero multi-sentence items because of an escaped
whitespace pattern; this count supersedes it. Punctuation counts are only a
candidate filter, not semantic annotation. In many two-sentence thoughts,
the first sentence is the event and the self-judgment occurs at text end.
The prespecified *internal post-claim* contrast remains unmeasured until
claim-completion positions are annotated without looking at model scores.
No round-2 arm is selected from this text-end result alone.
