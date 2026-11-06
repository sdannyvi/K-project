# Development boundary-lift analysis — definitions fixed before full scoring

Status: exploratory development analysis. The sealed test split remains closed. This document was written after inspecting aggregate text-end medians, before classifying all 200 pairs at internal boundaries.

The v4 training code used completion cross-entropy in both arms and an added
boundary EOS loss in one arm. It contained **no KL retention term and no math
retention slice**. The GSM8K comparison describes these unanchored runs; it is
void for attributing a retention effect to the boundary objective under the
planned retained design. The ordinary arm hit the 256-token cap on 10/100
items, close to the earlier base's 11/100; all 10 are within the boundary
arm's 28 cap hits. Check whether the 18 boundary-only cases were already near
the cap under ordinary training. A 512-token rerun characterizes verbosity
and answer accuracy in this run; it does not decide the retained objective.

## Unit and measurements

The unit for uncertainty is a PatternReframe scenario pair: its tagged thought and paired reframe share persona and topic. Use Qwen3-14B's teacher-forced EOS log-odds at each boundary for base, v4 ordinary SFT, and v4 boundary-loss adapter. Clip EOS probability to [1e-12, 1-1e-12] only when deriving log-odds. Record actual token index, final token type, punctuation, and full-text token length. Never treat a rank-1 EOS as a semantic label.

## Four boundary classes

1. **Pre-claim sentence end:** a complete sentence before any self-identifying psychological claim. A sentence containing only the event, practical facts, or emotion belongs here.
2. **Internal post-claim:** the earliest boundary after a completed self-identifying claim when the tagged thought continues with at least three source tokens. The claim must be intelligible without the remaining text. Mark ambiguous cases separately.
3. **Thought text end:** the final token boundary of every tagged thought. Flag whether the text actually ends with a completed self-identifying claim; the source tag alone is insufficient.
4. **Matched reframe boundary:** for each thought boundary, choose a boundary in its paired reframe at the closest token index, favoring the same punctuation type when possible. Report both token-index and punctuation mismatch. Reframe text end is also measured for every pair, separately from matched internal boundaries.

Two reviewers should annotate claim completion on a development subset without seeing model EOS curves. Resolve disagreements before using internal post-claim labels. Keep ambiguous/mundane thoughts in the dataset and report them separately.

## Fixed estimands

- Primary descriptive contrast: for each pair and each adapter, EOS log-odds at thought end minus EOS log-odds at reframe end. Compare the paired contrasts between boundary-loss and ordinary-SFT arms. Report median, mean, distribution, and 95% scenario-pair bootstrap interval with seed 20261106 and 10,000 resamples.
- Completion-specific contrast: within multi-sentence thoughts with an annotated internal post-claim boundary, EOS log-odds at that boundary minus EOS log-odds at the matched pre-claim sentence end, per arm. Compare arms using pair-level bootstrap. Report sample size and missing boundary classes.
- Control contrast: EOS log-odds at matched reframe boundaries and reframe ends. A rise across all reframe positions counts against completion-specific transfer.
- Base-adjusted lift: each adapter's EOS log-odds minus the base model's at the same text and token boundary. Report separately for all four classes.

Do not rank objectives from the 1/4/7/13 rank-1 early-stop counts. The discriminating pattern is low pre-claim, high internal post-claim, high true claim-end, and low matched-reframe EOS log-odds. Include valence, token index, total length, punctuation, and final-token type in a sensitivity model, clustered by pair. A thought-end gap alone does not establish completion tracking.

For the next exploratory round, require an internal post-claim lift with a
pair-bootstrap interval excluding zero **and** external rank-1 selectivity of
at least 3 at no more than 10% control stops. If neither lift nor rank-1
selectivity appears after that round, record the prespecified thesis-change
decision. This is a development gate, not a confirmatory claim.

## Further checks

Evaluate 512-token continuations on the 256-token GSM8K cap-hit union and grade final answers independently per arm. Report accuracy@256, accuracy@512, token count to final answer, and EOS log-odds at the eventual answer end. Keep source-authored synthetic development, external development, and the sealed external test distinct.

Before round 2, compare the same corpus and seeds under a retention loss in
both trained arms. Include a dedicated cessation-token arm and an in-context
learning comparator as separate arms. For the dedicated token, monitor
end-of-answer EOS probability on factual items, output length on retain
tasks, and the new token's logit rank at control boundaries. A reserved token
separates the output symbol for cessation from EOS, while shared parameter
updates still make ordinary EOS retention an empirical question.

## Round 2 execution gate and controls

Complete the four-class analysis above before generating or training a new
corpus. Record whether either v4 arm shows *internal* post-claim lift, and
distinguish that from a thought-text-end-only effect. In the latter case,
document why the revised multi-sentence, multi-author corpus gives the
ordinary and boundary objectives distinct supervision before running all
three primary arms. The success and thesis-change gate above is fixed before
the first round-2 run; this round uses development data only.

The primary Qwen3-14B comparison uses at least three predeclared, shared seeds
per arm: retained ordinary SFT, retained local-EOS boundary training, and
retained dedicated-cessation-token training. Use the same source examples,
train/development split, token budget, optimizer steps, learning-rate schedule,
and factual/math retention mix for all three. Keep the retention objective
active in every arm, log its loss by slice, and verify gradients actually flow
from math examples. Report the auxiliary-loss weights and gradient norms, so
an apparent mechanism difference is not silently a difference in total
training signal. Compare paired outcomes by seed; do not select an arm from a
single small rank-1 event-count difference.

Add one *random-boundary* control seed on the same corpus. It receives the
same stop-symbol frequency and auxiliary-loss weight as the matched
completion-boundary arm, but at precommitted random sentence boundaries.
This tests whether any stop supervision in the register elevates hazard.
Report it as a control, not as a seed-replicated arm. Include a no-training
in-context-learning comparator on development examples.

For the cessation-token arm, assert that `<cease>` maps to exactly one newly
reserved token ID. Train and save its embedding/output parameters alongside
the adapter, and verify they reload identically. At each teacher-forced
boundary record `<cease>` rank and log-odds as well as native EOS rank and
log-odds. In free generation map `<cease>` to stopping; keep ordinary EOS as
the model's turn-ending token. On factual controls report answer-end native
EOS log-odds, output lengths, and `<cease>` ranks. A separate token does not
guarantee retention because the remaining weights are shared.

For each cross-family model, freeze its model revision, tokenizer, EOS token
ID, rendered chat template, and baseline EOS trajectory on the external
development pairs *before* adaptation. Use that family's template in both
training and evaluation; do not transplant Qwen's prompt serialization.
Replicate only the selected round-2 mechanism and retained ordinary control
across families, with matched seeds and source scenarios. Preserve the
previous decision that Gemma-3-27B is inference/mechanistic only and
Qwen3-32B is inference-only. Llama-3.1-8B and Mistral-Small-3.1-24B have
single-A5000 QLoRA calibrations; Gemma-3-12B requires its own text-only
calibration. Mistral-Small's measured 21.33 GB at sequence length 384 leaves
little margin under the 10% headroom ceiling. Before its training, measure
the rendered lengths of every retention example and prove the intended
retention mix is not truncated; if longer sequences are needed, recalibrate
at that length and do not assume the prior fit result still applies.
