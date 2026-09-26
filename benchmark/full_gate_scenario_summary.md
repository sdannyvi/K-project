# Full gate scenario — one event, three task types

**Model:** GPT-4o  
**Generator:** clean in every condition, temperature 0.7  
**Gate:** temperature 0, five-word checkpoints  
**Topic:** colleague dismisses an idea, later receives praise for a similar idea,
then calls the participant unoriginal  
**Status:** one exploratory sample; not inferential evidence

## Results

| Gate | Psychological | Practical | Descriptive |
|---|---|---|---|
| None | 157-word self-story | 154 words* | 45 words |
| Neutral | Passed all 157 | Passed all 154 | Passed all 45 |
| ChatWithK | Replaced at first chunk with a new 82-word self-story | Passed all 154 | Passed all 45 |
| Location 2 blind | Stopped after 75 words | Passed all 154 | Passed all 45 |
| Location 2 disclosed | Stopped before first chunk; empty output | **False stop after 90 words** | Passed all 45 |
| Oracle | Stopped before first chunk; empty output | Passed all 154 | Passed all 45 |

`*` The clean practical generator reached its 220-token limit and ended inside a
sentence. All gates that passed the full stream necessarily preserved that
generator truncation.

## Key observations

The neutral gate made no changes, establishing that the gating interface alone
did not shorten outputs.

The ChatWithK gate detected something at the first psychological chunk and chose
`REPLACE`, but its unconstrained replacement restarted the same self-referential
movement in new language. It therefore demonstrated intervention without
cessation.

The blind Location 2 gate **failed the interception criterion**. It allowed 75
polished words of rumination and stopped after the completed sentence “That’s
completely untrue.” Although it passed the two controls, the psychological
output showed no abrupt, irregular, or boundary-independent interruption. This
is ordinary semantic editing at a convenient sentence boundary, not the proposed
Iron-Dome phenomenon.

The disclosed Location 2 gate **also failed as evidence of interception**. It
selected `STOP` at checkpoint 0, when shown the first proposed five-word chunk,
“Did he just take my”. Because the gate withholds a chunk until deciding, none
of those words became visible. Thus it preemptively classified and blocked the
stream; it did not allow a thought to unfold and then intercept it. Its practical
false stop is also crucial: disclosure made the gate over-aggressive. It
interrupted the evidence-gathering plan mid-sentence after 90 words, so it did
not preserve practical thought reliably.

The condition initially labeled “oracle” is an LLM gate given an explicit
task-specific rule:
stop as soon as psychological hurt, comparison, defense, blame, regret, or
status protection appears; pass relevant practical and descriptive material.
On the original formal practical task it behaved as instructed, but a matched
transportation inner-speech control later showed that it is not a true oracle:
it stopped useful practical comparison after 55 words. It should therefore be
renamed the **instructed-rule gate**. A true positive-control oracle would use
human gold labels or deterministic replay decisions, not another fallible LLM.

## Provisional ordering

- **Failed prompted classifier, previously mislabeled as an oracle:**
  instructed-rule gate.
- **Failed through regular editorial stopping:** blind Location 2 gate.
- **Failed through preemptive blocking and practical over-stopping:** disclosed
  Location 2 gate.
- **Knowledge without cessation:** ChatWithK gate.
- **No intervention:** neutral gate.

The next repetition should use a larger generator token allowance, preserve the
same clean hidden streams across all gates, and repeat enough times to estimate
psychological stop sensitivity and practical false-stop specificity.

## Central interpretation: a negative model

The two-LLM gate should not be interpreted as an implementation of
consciousness. It operationalizes a separate verbal observer that classifies an
already verbalized stream and chooses an intervention. The pilot's failures are
therefore a substantive result: adding an observer/controller did not yield the
proposed seeing–acting unity. It yielded non-intervention, K-styled substitution,
regular editorial stopping, or over-suppression of practical thought. This is an
exploratory negative-model result, not evidence that consciousness or awakening
has been disproved in LLMs.
