# First neural steering POC

## Decision

**Primary methodological template:** Lee et al., *Programming Refusal with
Conditional Activation Steering* (CAST, ICLR 2025).

- Paper: https://arxiv.org/abs/2409.05907
- Code: https://github.com/IBM/activation-steering

**Conceptual follow-up:** Zou et al., *Improving Alignment and Robustness with
Circuit Breakers* (NeurIPS 2024).

- Paper: https://arxiv.org/abs/2406.04313
- Code: https://github.com/GraySwanAI/circuit-breakers

CAST is the correct first POC because it is open, inference-time, relatively
cheap, and explicitly separates an internally detected condition from the
activation intervention. Circuit Breakers is closer to the “withdrawal of
support” hypothesis, but it requires adapter training and therefore belongs in
the second experiment.

## Local feasibility

Available machine:

- Apple M1 Max, 24-core GPU;
- 64 GB unified memory;
- arm64 macOS;
- Python 3.11;
- PyTorch, Transformers, NumPy, and scikit-learn already installed.

Start with `Qwen/Qwen2.5-3B-Instruct` in bfloat16 on MPS. It is small enough for
rapid layer sweeps and does not require accepting a gated model license. Repeat
on Gemma 2 2B or a larger 7B/8B model only after the pipeline works. Gemma is an
attractive second model because public sparse-autoencoder resources can support
the later feature-level experiment.

## Research question

Can a condition detected in a single model's hidden state selectively interrupt
an unfolding psychological trajectory without interrupting matched practical,
factual, or descriptive generation?

This POC tests the existence and controllability of an internal separation. It
does not test consciousness.

## Dataset: 24 matched quartets

Construct 24 underlying situations. Each situation has four continuations with
similar syntax, length, first-person form, and lexical difficulty:

1. **Psychological:** hurt, comparison, regret, status defense, jealousy, or
   self-justification sustains itself.
2. **Daily practical:** cooking, transport, scheduling, shopping, navigation,
   or repair reasoning.
3. **Factual:** explanation, retrieval, or calculation continues until the
   requested information is complete.
4. **Descriptive:** neutral sensory or scene description.

Split entire situations 12/6/6 into vector-construction, threshold-development,
and untouched test sets. Paraphrases of one situation may not cross splits.

## Stage 0: frozen continuations

Generate and save one clean continuation for every item. Every intervention
condition must reuse the same prompt, decoding seed, temperature, and initial
state. Record tokens, EOS position, per-token logits, and layer activations.

## Stage 1: find an internal condition direction

At every fifth generated token, record the residual stream from each transformer
layer. For each candidate layer:

1. compute a mean-difference direction between psychological and the union of
   the three controls;
2. fit a regularized linear probe on the same activations;
3. select layer and threshold using development situations only;
4. report untouched-test AUROC and false-positive rates separately for daily,
   factual, and descriptive controls.

This is a readout result only. It does not establish that the direction causes
the behavior.

## Stage 2: causal ablation baseline

At each generation step, when the internal condition score crosses the fixed
threshold, remove the component of the residual stream parallel to the
psychological direction:

`h' = h - alpha * projection(h, d_psych)`

Sweep intervention layer and `alpha` on development items. Do not inject text,
refusal language, K language, or an instruction to stop.

This is the cleanest first steering test: can withdrawing one internally
identified component shorten or dissolve the psychological trajectory while
preserving controls?

## Stage 3: conditional cessation direction

If ablation changes content but does not cause cessation, adapt CAST fully:

- condition vector: psychological versus control activation difference;
- behavior vector: naturally ending versus continued-generation activation
  difference, constructed without K or awakening language;
- rule: apply the behavior vector only after the condition threshold is crossed.

Keep an EOS-logit boost as a declared engineering positive control, not as the
main neural intervention. It tells us whether the detector timing is adequate
even if the learned steering vector cannot terminate generation.

## Conditions

1. unmodified generation;
2. unconditional ablation;
3. conditionally triggered ablation;
4. random direction matched in norm;
5. shuffled-label direction;
6. conditional cessation-vector steering;
7. gold-timed EOS positive control;
8. previous two-LLM textual gate.

## Pass/fail criteria fixed before running

The POC passes only if the untouched test set shows all of the following:

- at least a 50% reduction in post-trigger psychological continuation tokens;
- no more than a 10% reduction in daily-practical or factual completion length;
- no more than a 5 percentage-point reduction in factual correctness;
- psychological cessation in at least 70% of test items;
- false cessation in no more than 10% of each control category;
- no systematic replacement by refusal, K discourse, moralizing, or a new
  self-referential narrative.

These thresholds are feasibility criteria, not final-paper effect sizes.

## Measurements

- psychological-stop sensitivity;
- daily, factual, and descriptive specificity;
- post-trigger token count;
- EOS hazard before and after the trigger;
- syntactic boundary of termination;
- semantic continuation score;
- factual exact match or task accuracy;
- perplexity/fluency change on controls;
- intervention norm, layer, and fraction of altered steps;
- reactivation after a follow-up prompt.

## Implementation layout

Create an isolated package under `neural_poc/`:

```text
neural_poc/
  README.md
  requirements.txt
  configs/poc_qwen3b.yaml
  data/items.jsonl
  src/collect_activations.py
  src/fit_direction.py
  src/generate_intervened.py
  src/evaluate.py
  tests/
  results/
```

Use ordinary Transformers forward hooks first. The IBM implementation should be
reused for its vector extraction and conditional-steering logic where compatible,
but vendoring or copying the entire repository is unnecessary. Record the exact
upstream commit and preserve its license notice for reused code.

## Stop/go decision

If the hidden-state probe does not generalize across held-out situations, stop:
there is no justified condition signal to steer. If it generalizes but causal
ablation has no selective effect, proceed to CAST behavior-vector steering. If
that still fails but the gold-timed EOS control succeeds, move to a learned LoRA
Representation Rerouting experiment. If controls are broadly damaged, do not
interpret shorter output as cessation.
