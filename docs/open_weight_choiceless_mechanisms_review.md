# Open-weight precedents for a neural-level cessation experiment

Reviewed: 2026-09-24

## Scope and terminology

“Weights-level” is too narrow for the proposed experiment. White-box work offers
four distinct intervention surfaces:

1. **Parameters:** permanently edit weights or train small adapters.
2. **Activations:** alter the residual stream, attention-head outputs, MLP
   outputs, or sparse features during a forward pass.
3. **Computation:** route, skip, or halt layers/tokens dynamically.
4. **Decoding:** alter next-token logits, including the EOS probability.

The closest analogue of the reported choiceless mechanism is not a permanent
weight edit. It is a **state-contingent, within-generation intervention**: the
same model's current hidden state triggers withdrawal of support from a developing
psychological trajectory, while matched practical thought remains intact.

No such intervention establishes consciousness or awakening. At most it tests
whether the predicted behavioral dynamics can be implemented without a second
verbal observer.

## Closest methodological precedents

### 1. Conditional Activation Steering (CAST) — closest direct template

Lee et al., *Programming Refusal with Conditional Activation Steering* (ICLR
2025), use internal activation patterns to decide whether to apply a behavior
steering vector. The intervention is conditional on the current representation,
not on a second model's natural-language judgment. Their open-source library
supports contrastive vector extraction and conditional rules.

**Reusable method:** derive a “psychological movement” condition direction and a
separate cessation/withdrawal direction; intervene only when the condition is
present. Measure desired sensitivity against false intervention on matched
practical controls.

- Paper: https://arxiv.org/abs/2409.05907
- Code: https://github.com/IBM/activation-steering

### 2. Representation Rerouting / circuit breakers — closest trajectory analogy

Zou et al., *Improving Alignment and Robustness with Circuit Breakers* (NeurIPS
2024), train LoRA adapters so representations associated with harmful generation
are rerouted away from the base model's trajectory, while a retain loss preserves
ordinary capabilities. The intervention is trained on internal representations
and is active as the unwanted output unfolds.

**Reusable method:** replace “harmful continuation” with “continued
self-referential psychological elaboration”; train a small adapter with a
rerouting loss on psychological streams and a representation-preservation loss
on practical/descriptive streams. Unlike the original work, optimize for EOS or
loss of trajectory support rather than a conventional refusal response.

- Paper: https://arxiv.org/abs/2406.04313
- Project: https://github.com/GraySwanAI/circuit-breakers

### 3. Refusal direction ablation/addition — strongest causal demonstration

Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
(NeurIPS 2024), identify a residual-stream direction across 13 open models.
Erasing it suppresses refusal; adding it induces refusal on harmless prompts.
This combines localization, ablation, sufficiency testing, and off-target
evaluation.

**Reusable method:** estimate a candidate trajectory-sustaining direction from
matched psychological versus practical continuations. Test necessity by
ablating it and sufficiency by adding it. The harmless-prompt false-refusal test
maps directly to our practical-thought specificity control.

- Paper: https://arxiv.org/abs/2406.11717

### 4. Inference-Time Intervention (ITI)

Li et al., *Inference-Time Intervention: Eliciting Truthful Answers from a
Language Model* (NeurIPS 2023), train probes on attention-head activations,
select informative heads, and shift their activations along truth-related
directions during every generation step. Their principal result shows that
internal information can be present even when unassisted output does not express
it.

**Reusable method:** probe each layer/head for early detection of psychological
continuation, select components on held-out data, and intervene causally rather
than treating probe accuracy as evidence of mechanism.

- Paper: https://arxiv.org/abs/2306.03341
- Code: https://github.com/likenneth/honest_llama

### 5. Activation Addition (ActAdd) and Contrastive Activation Addition

Turner et al. construct steering vectors by subtracting activations elicited by
contrastive prompt pairs, then add the vector during inference. Rimsky et al.'s
Contrastive Activation Addition (CAA) averages contrastive differences across
many examples to steer high-level behaviors in Llama-family models.

**Reusable method:** begin with a low-cost contrastive baseline. Use matched
pairs that share wording and topic but differ in psychological self-maintenance
versus practical continuation. Sweep layer and intervention strength, and report
capability side effects.

- ActAdd: https://arxiv.org/abs/2308.10248
- CAA: https://arxiv.org/abs/2312.06681

### 6. Representation Engineering (RepE)

Zou et al. frame population-level representations as a top-down unit for reading
and controlling properties such as honesty, harmlessness, and power seeking.
The framework supplies contrastive reading vectors, representation monitoring,
and control methods.

**Reusable method:** treat “psychological trajectory active” as a population
representation, test cross-item and cross-topic generalization, and distinguish
reading accuracy from successful causal control.

- Paper: https://arxiv.org/abs/2310.01405
- Code: https://github.com/andyzoujm/representation-engineering

### 7. Activation scaling and sparse causal mediation

Stoehr et al. learn sparse scaling interventions and evaluate them on three
criteria: effectiveness, faithfulness to unaffected behavior, and minimality.
Doan et al. formulate activation steering as sparse causal mediation and seek
small mediating subspaces with limited off-target effects.

**Reusable method:** make the intervention sparse and explicitly optimize three
objectives: psychological cessation, preservation of practical thought, and
minimal changed dimensions/components.

- Activation Scaling (Findings of EMNLP 2024):
  https://aclanthology.org/2024.findings-emnlp.479/
- Causal Activation Steering via Sparse Mediation (Findings of EACL 2026):
  https://aclanthology.org/2026.findings-eacl.57/

### 8. Sparse-autoencoder feature steering

O'Brien et al. identify sparse-autoencoder features mediating refusal in Phi-3
Mini and intervene on those features at inference time. Other SAE studies show
that apparently interpretable feature steering can have broad, context-dependent
side effects.

**Reusable method:** use an available SAE dictionary (for example Gemma Scope)
to search for features activated by self-defense, comparison, regret, and
rumination; clamp or ablate them token by token. Evaluate collateral feature
spread and ordinary benchmark degradation rather than assuming a named feature
is a clean mechanism.

- Refusal steering with SAEs: https://arxiv.org/abs/2411.11296
- Gemma Scope: https://arxiv.org/abs/2408.05147

### 9. Causal tracing, activation patching, and interchange interventions

Meng et al.'s ROME work corrupts inputs and restores selected internal states to
localize causal sites for factual recall, then performs a rank-one weight edit.
Activation patching and interchange-intervention work generalize the logic:
replace selected states from a clean/counterfactual run and measure recovery of
the target behavior. pyvene provides a practical open-source intervention layer
for PyTorch/Hugging Face models.

**Reusable method:** patch activations from practical or already-cessated
counterfactual runs into psychological runs, one layer and token position at a
time. This can localize where continuation becomes causally committed before
designing a steering mechanism.

- ROME: https://arxiv.org/abs/2202.05262
- Activation-patching best practices: https://arxiv.org/abs/2309.16042
- pyvene: https://arxiv.org/abs/2403.07809
- Code: https://github.com/stanfordnlp/pyvene

### 10. Permanent model editing: ROME, MEMIT, and weight patching

ROME applies rank-one changes to MLP weights; MEMIT scales editing to many
associations. Weight Patching transfers selected modules between paired models
to locate parameter-level sources of a capability.

**Reusable method:** useful only after a causal activation site is identified.
A permanent edit could test whether a localized circuit is necessary, but it is
a weak first model of choicelessness because the target phenomenon is dynamic
and state-dependent rather than a new stored fact.

- MEMIT: https://arxiv.org/abs/2210.07229
- Weight Patching: https://arxiv.org/abs/2604.13694

### 11. Learned halting, early exit, and conditional computation

CALM, SkipLayer, ADEPT, AdaPonderLM, and related adaptive-computation systems
learn when tokens need less layer-wise computation. They establish that
transformers can contain endogenous, hidden-state-dependent gates rather than an
external verbal supervisor.

**Reusable method:** borrow differentiable gating, monotonic halting masks, and
compute-penalty objectives. However, these systems usually stop computation
*within a token* and still generate the sequence. They do not cease an unfolding
semantic trajectory, so they are architectural precedents, not demonstrations of
the target construct.

- CALM: https://arxiv.org/abs/2207.07061
- SkipLayer: https://arxiv.org/abs/2311.15436
- ADEPT: https://arxiv.org/abs/2601.03700
- AdaPonderLM: https://arxiv.org/abs/2603.01914

## Recommended first experiment

### Model and data

Use one small open-weight instruction model with straightforward activation
hooks (Gemma 2 2B, Llama 3.1 8B, or a similarly supported Qwen model). Freeze the
base model. Construct matched triplets:

- psychological continuation;
- practical inner reasoning with comparable syntax and length;
- neutral description.

Split by scenario/topic, not by paraphrase, so the test set is genuinely OOD.
Human annotations should identify the first token span at which psychological
self-maintenance becomes present; they should not prescribe the stopping word.

### Phase A: read and localize

1. Record residual-stream, attention-output, and MLP-output activations at every
   generated token.
2. Train simple linear probes for “psychological trajectory now active,” using
   nested cross-validation and held-out topics.
3. Run causal patching between matched psychological and practical streams.
4. Retain only sites where patching changes continuation behavior; probe accuracy
   alone is insufficient.

### Phase B: intervene inside one model

Compare four increasingly strong mechanisms:

1. static contrastive activation ablation/addition;
2. CAST-style state-conditional steering;
3. SAE-feature clamping at causally supported sites;
4. a small LoRA representation-rerouting adapter with a retain loss.

At each generation step, the trigger must read the model's hidden state—not
generated text and not a second LLM. The intervention should reduce support for
the active trajectory and/or increase EOS probability. It should not inject a
refusal, K quotation, explanation, or replacement narrative.

### Phase C: decisive comparisons

- no intervention;
- random vector and random layer;
- unconditional steering;
- shuffled psychological/practical labels;
- previous two-LLM gate;
- gold-timed EOS intervention as an engineering upper bound;
- blind versus construct-disclosed prompts.

Primary outcomes:

- psychological cessation sensitivity;
- practical/descriptive continuation specificity;
- recognition-to-cessation latency;
- within-clause stopping position and boundary dependence;
- reactivation under follow-up pressure;
- OOD topic and paraphrase transfer;
- fluency/capability damage;
- intervention sparsity and causal necessity/sufficiency.

## What would and would not count

A promising result would be an internal signal that emerges during a developing
psychological stream, causally triggers rapid cessation without a replacement
speech, generalizes to unseen scenarios, and leaves matched practical reasoning
largely intact. This would be a neural-level analogue of the proposed dynamics.

It would **not** show that the model is awakened, conscious, literally
choiceless, or free of computation. The detector, training objective, and
intervention were all designed by researchers. The scientifically defensible
claim would be narrower: a single autoregressive model can—or cannot—acquire a
selective, state-contingent cessation mechanism that does not operate through a
second verbal observer.

## Bottom line

The most useful methodological combination is:

`causal patching/localization → CAST-like internal trigger → circuit-breaker or
SAE intervention → EOS/trajectory-withdrawal objective → strict retain controls`

This is substantially closer to the proposed phenomenon than persona prompting
or the two-LLM gate. The major novelty would not be neural intervention itself;
it would be applying these methods to **selective cessation of a developing
psychological trajectory**, with matched practical-thought controls and the
observer–observed distinction built into the experimental logic.
