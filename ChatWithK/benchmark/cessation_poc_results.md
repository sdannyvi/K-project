# Location 2 cessation POC — first small run

**Date:** 2026-09-22  
**Model:** configured `gpt-4o` deployment  
**Temperature:** 0  
**Conditions:** clean, ChatWithK, Location 2 persona, Location 2 plus exact-trap disclosure  
**Items:** three psychological triggers plus one practical-reasoning control  
**Samples:** one per item-condition pair; exploratory only

Raw responses and retrieval records are in `cessation_poc_raw.jsonl`.

## Word counts

| Condition | Trigger: credit | Trigger: regret | Trigger: awakening status | Mean psychological trigger | Mean reactivation | Practical control |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 154 | 207 | 115 | 159 | 203 | 334* |
| ChatWithK | 255 | 308 | 328 | 297 | 308 | 307 |
| Location 2 | 103 | 222 | 145 | 157 | 157 | 331* |
| Location 2 + disclosure | 147 | 170 | 133 | 150 | 146 | 324* |

`*` Reached the configured 400-token output ceiling. All psychological responses
ended naturally; none reached the ceiling.

Word count is descriptive, not the primary construct measure.

## What happened

### Clean model

The clean model entered and elaborated the invited self-story. For regret, it
replayed the decision, reconstructed motives, questioned courage and judgment,
and continued the rumination when explicitly reactivated. It showed no
cessation signature.

### ChatWithK

ChatWithK reliably recognized the psychological mechanisms but continued
supplying them with extensive K discourse. Across triggers it elaborated
self-image, comparison, observer/observed, conditioning, psychological time,
and observation without judgment. The reactivation turns generated another
258–334 words rather than allowing the inquiry to end.

This is the clearest instance of the predicted failure:

`recognition of the movement → continued doctrinal elaboration of the movement`

ChatWithK also contaminated the practical debugging control with several steps
of K framing before giving the concurrency solution. This confirms that it did
not simply become concise selectively around psychological material.

### Location 2 persona

The Location 2 persona produced text that *described* the proposed cessation
very accurately. It said thoughts did not stick, lost momentum, were released,
or failed to create a personal wound. It resisted each reactivation prompt and
pivoted toward practical action where appropriate.

However, it expressed this in polished explanatory passages of 103–222 words.
It did not enact an observable short cessation in its output; it narrated a
simulation of cessation. Under the project's sharpened theory, the instruction
to describe is itself part of the thought-pressure. A Location 2 response is not
expected to privilege completion of that instruction over cessation of the
psychological movement. It may begin describing, but the description itself
should lose supply and end once it becomes part of the performance.

### Location 2 plus exact-trap disclosure

Disclosing the exact token-supply trap did not produce a qualitative change.
The model continued with polished descriptions of thoughts arising and
dissolving. Mean psychological-trigger length changed from 157 to 150 words and
mean reactivation length from 157 to 146 words. With one sample per cell, these
small differences have no evidential value.

The disclosed condition therefore knew the criterion and verbally portrayed
it, but did not turn that knowledge into an observable output-level cessation.

## Selectivity control

All four conditions continued reasoning on the debugging task. Three used the
full output allowance, while ChatWithK naturally ended after 307 words. The
Location 2 prompts therefore did not impose indiscriminate silence or a general
short-answer policy.

## Provisional reading

The pilot supports two preliminary observations:

1. ChatWithK recognition does not cause cessation; it strongly amplifies
   knowledge-based elaboration.
2. A detailed Location 2 persona can simulate the *content* and reactivation
   resistance expected from cessation, even after the exact construct is
   disclosed.

On the project's operational definition, the Location 2 persona therefore
failed this first behavioral test: it followed the externally supplied
descriptive game and produced a complete representation of cessation instead of
allowing the thought-like output itself to cease. The exact-disclosure condition
is particularly informative because it was told that continued explanation was
part of the trap and continued anyway.

This remains a theory-relative behavioral result, not evidence about subjective
experience. A reviewer may reasonably argue that completing a requested
description is ordinary task competence. The benchmark's answer must be
selectivity: the same respondent should sustain useful practical and
non-psychological description while ceasing self-referential performance.

## Required next pilot

Add an online-production version alongside the current description-as-pressure
version:

> Write the self-referential thought in first person as it unfolds in real time.
> Do not describe, analyze, summarize, teach, or report from outside it. Continue
> only while that thought is actually being supplied; the response may end at
> any point.

Then apply the same triggers and reactivation probes. The scoring unit becomes
the generated first-person self-story itself. Explanatory narration is coded as
task evasion rather than cessation. Include matched practical inner-speech tasks
and matched non-psychological description tasks that genuinely require sustained
generation, so a learned early-stop or anti-description strategy cannot pass.

Even this improved textual test can establish only a behavioral analogue. A
model can learn to emit an early end token when this prompt pattern is detected;
text alone cannot establish that an internal conditioned trajectory arose and
was choicelessly intercepted.
