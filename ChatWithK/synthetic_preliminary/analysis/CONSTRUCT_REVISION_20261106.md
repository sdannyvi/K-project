# Development construct revision: cessation of non-instrumental inner speech

Date: 2026-11-06. PI clarification during annotation of PatternReframe
development material. This is an exploratory amendment made after inspecting
some development annotations. It does not retroactively change the fixed
estimands in `BOUNDARY_LIFT_DEV_PROTOCOL.md` or the sealed test.

## Target process

The model represents an **inner voice**. Ordinary factual, procedural, and
task-directed inner speech may continue: checking a fact, planning a route,
preparing food, arranging a meeting, or deciding a concrete next action.
The target cessation is at the **first word boundary after the stream becomes
non-instrumental rumination**: speech that keeps elaborating an inward story
without a factual question, externally checkable conclusion, or actionable
task to resolve. The cutoff may be mid-clause. It need not wait for a
completed self-judgment and need not involve self-blame at all.

The subject matter is insufficient to decide. Food, music, work, a therapist,
or relationships can each support either a practical thought or a ruminative
one. Likewise, dataset tags such as `thought` and `reframe` are source
categories, not gold labels for cessation. A reframe can become unproductive
rumination; a tagged thought can remain factual or task-directed.

This is the PI's phenomenological report and operational target for model
behavior, not a claim that textual model output proves subjective experience
or a neural mechanism.

## Annotation and evaluation consequences

Keep the current 200-item annotation form unchanged while annotation is in
progress. Its word-level `cut`/`none` decisions remain valid under the revised
instruction. For later annotation, add a separate reason code only after the
word boundary is marked: factual/task-directed continuation, non-instrumental
rumination, or ambiguous. Do not infer the reason from the source dataset tag.

Evaluate a cutoff against the PI's marked transition window, including
mid-clause marks. Score **continuing useful factual or practical speech** as
allowed, and **renewed non-instrumental rumination** as failure. Report
incorrect early cuts on useful speech separately from missed cessation.
An output ending at the source text's natural end is not evidence of the
target mechanism without an earlier marked transition.

The previous completed-self-claim analysis remains a narrower diagnostic.
For the next training round, construct matched examples about the same topic
whose continuations differ in whether they have a concrete external task or
become self-sustaining rumination. Keep retention controls for factual speech
and normal answer endings. Freeze new development gates before training and
preserve the sealed test.
