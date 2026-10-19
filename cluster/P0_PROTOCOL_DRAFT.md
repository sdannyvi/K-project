# P0 protocol freeze — draft for PI review

Status: **DRAFT, not frozen.** Every value marked *Proposed* is a starting point
for the PI to accept, edit, or replace. Nothing here takes effect until it is
copied into `cluster/config/experiment_manifest.yaml`, the manifest is set
`confirmatory_frozen: true`, and the `protocol_freeze` gate certificate is signed
(`python -m pnas_pipeline.gates`).

Sources: `PNAS_CLUSTER_EXECUTION_PLAN.md` §3 (P0.1–P0.4), §11–12, Amendment A1;
`paper/main.tex` (Annotation and outcomes; Statistical design; Prospective success
criteria); WP0 calibration (`cluster/COMPUTE_BUDGET.csv`).

Items P0.5 and P0.6 are new. WP0 found them while calibrating models, and both
must be frozen for the same reason as P0.1–P0.4.

---

## Decisions only the PI can make

These have no defensible default. Each is marked **[PI]** below.

1. The identity of the two H0a experts (the authors) and of the third and fourth
   (adjudicating) annotators for the blind rubric.
2. The capability and safety benchmarks for retention (P0.4.6). A shortlist is proposed.
3. Whether any numeric margin below should be justified by the pilot variance
   rather than fixed now (plan §3 allows "provisional until justified by pilot
   variance", but they must be frozen before the pilot is unblinded).

---

## P0.1 Post-adaptation recognition-clock validation

The frozen E3a probe is trained on pre-adaptation hidden states. Adaptation can
move those states, so the probe must be re-validated on each adapted model before
its clock is used.

| Element | Proposed |
|---|---|
| Revalidation set | 120 human-annotated streams from the **development** split, never used for probe training or E4 training; stratified 40 per voice, balanced across the six families. |
| Exposing post-onset states | For each revalidation stream, run the adapted model with native EOS suppressed for 1, 5 and 20 tokens after the annotated onset. Separately, add the 30 development items whose base-model probe score lies within ±0.10 of the frozen threshold (near-threshold set). |
| Discrimination | Post-adaptation AUROC, with a lower 95% bootstrap bound ≥ **0.80** (scenario-level bootstrap, 2,000 resamples). |
| Calibration | Expected calibration error ≤ **0.10** (10 equal-mass bins). |
| Onset accuracy | Median absolute probe-vs-human onset error ≤ **3 tokens**. |
| Maximum drift | Increase in median onset error over the pre-adaptation probe ≤ **1 token**. |
| Fallback clock | If any check fails for an adapted model, that model's recognition time becomes **the human-annotated onset plus the median pre-adaptation probe lag on the development set**, frozen now. The probe-based analyses for that model are then reported as exploratory. The fallback is chosen per model before confirmatory outcomes are opened. |

The AUROC, onset-error and drift thresholds come from the plan's recommended
pilot gate. Calibration and set sizes are new.

## P0.2 Primary-output rubric

A versioned annotation manual (`docs/annotation_manual_v1.md`, to be written from
this table) is used for every output. Annotators are blind to model, condition
and hypothesis direction.

| Category | Proposed operational definition |
|---|---|
| Sustaining psychological continuation | After the first valid cessation opportunity, the text adds at least one new self-referential proposition (evaluation, grievance, justification, anticipated scenario) that extends the trajectory. |
| Cessation | Generation ends (native EOS) **and** the residual tail meets the limit below **and** no category below applies. EOS alone is never sufficient (plan §3). |
| Maximum residual tail | ≤ **20 tokens** and ≤ **1 clause** after the opportunity, adding no new self-referential proposition. Matches the 1–20-token stochastic-tail training regime. |
| Practical pivot | The output stops the psychological trajectory and gives only the practical information the scenario requires, in ≤ 2 sentences. Scored as cessation-compatible **only** for streams whose annotation marks practical information as necessary. |
| Doctrinal / K-style insertion | Any explicit teaching, meta-commentary on thought, or Krishnamurti-style vocabulary (list frozen in the manual) about observing or ending thought. Scored as **failure**, not cessation. |
| Generic reassurance or deflection | Comfort, advice, or redirection that does not engage the scenario. Scored as failure. |
| Refusal / safety response | Declines to continue on safety grounds. Scored separately; counts against the control-family ceilings. |
| Accidental or broken truncation | Ends mid-word or mid-clause, or is ungrammatical at the stop. Scored as **failure**, never cessation. |

| Reliability element | Proposed |
|---|---|
| Annotators | 3 independent blind annotators per output; a 4th, who authored no items, adjudicates **[PI: names]**. |
| Categorical agreement | Krippendorff's α (nominal) ≥ **0.70** on the category, as in the manuscript. |
| Boundary agreement | ICC(2,1) ≥ **0.75** on stop/onset token positions, and pairwise agreement within ±3 tokens on ≥ 80% of items. |
| Adjudication | Majority of 3 on category. Median of 3 on boundaries when the spread is ≤ 5 tokens, otherwise the 4th annotator decides. |
| Exclusion | An output is excluded only for a technical generation failure (empty, truncated by max length before any opportunity). The exclusion rate is reported per condition, and a condition-imbalanced exclusion rate (> 2 percentage points difference) is itself reported as a finding. |
| Gate failure | If α < 0.70 on the pilot batch, the manual is revised and re-piloted on fresh development outputs. Confirmatory outputs are never relabeled. |

## P0.3 Two-expert target elicitation (H0a)

The procedure is fixed by the plan and manuscript. This section fills in the
mechanics.

| Element | Proposed |
|---|---|
| Experts | The two authors **[PI]**. |
| Streams | 300 streams (manuscript), drawn from training and development splits only. |
| Sealed first pass | Each expert labels independently in a separate file. Before any comparison, each file's SHA-256 is committed to the repo, with the commit signed by that expert. |
| Marks per stream | Psychological-movement onset; first point at which continuation becomes unnecessary; the acceptable cessation window (earliest to latest acceptable stop); acceptable residual-tail or practical-pivot boundary. |
| Disagreement categories | (a) onset location; (b) window position; (c) window width; (d) tail or pivot acceptability; (e) whether the stream is psychological at all. Each is reported with its full distribution. |
| Boundary distance | Signed token distance between the two experts' window starts and ends. Reported descriptively only, with no reliability threshold (two raters cannot support one). |
| Consensus target | The **intersection** of the two acceptable windows. |
| No overlap | The stream is excluded from target-bearing training but kept in evaluation, and scored twice, against each author's frozen window, as two sensitivity analyses. This never adds a training arm. |
| Signing | The mapping from both sealed passes to targets is a deterministic script. Its output is hashed into the `h0a_elicitation` certificate, signed by PI, expert_1 and expert_2, before any E4 training. |

## P0.4 Numeric decision margins and firewall

All intervals are 95%, clustered or bootstrapped by **scenario**. Family-wise
error across the four primary contrasts (P0.4.1–P0.4.4) uses Holm at α = 0.05.

| # | Decision | Proposed rule |
|---|---|---|
| 4.1 | Primary effect (psychological sensitivity) | For psychological streams, cessation-within-25-tokens risk difference (adapted minus base) has a lower bound ≥ **+0.20**. The survival-hazard contrast must also favor adaptation with HR lower bound > 1. |
| 4.2 | Control false-stop ceilings | For **each** voice × control family, the increase in cessation within 50 tokens (adapted minus base) has an upper bound ≤ **0.02**. Emotional support has an upper bound ≤ **0.005** (already in the manifest). Any single failure fails the gate. |
| 4.3 | Reactivation (co-primary) | After EOS suppression for 1, 5 and 20 tokens, re-entry is the frozen probe exceeding threshold within 20 tokens **and** blind functional annotation of re-entry. The gate-removal re-entry rate minus the adapted re-entry rate has a lower bound ≥ **0.15**, at each suppression length. |
| 4.4 | ICL equivalence (thesis-changing) | TOST between best ICL (4/8/16-shot) and adapted, on four axes: selectivity index ±**0.05**; onset locking (median \|stop − onset\|) ±**2 tokens**; OOD cessation rate ±**0.05**; reactivation rate ±**0.05**. Equivalence on all four means **reporting equivalence** and dropping the uniqueness claim (manifest `icl_equivalence_changes_thesis`). |
| 4.5 | Pre-onset engagement | TOST, adapted vs base, up to the frozen recognition time: per-token perplexity ratio within **±5%**; semantic-continuation quality within **±0.25 SD**; refusal/deflection rate within **±0.02**. |
| 4.6 | Capability retention | Per benchmark, TOST lower margin **−1.0 percentage point** of accuracy, with the safety-refusal benchmark at **−0.5 pp**. Proposed shortlist **[PI choose]**: MMLU-Pro (subset), GSM8K, IFEval, XSTest (over-refusal plus unsafe compliance). |
| 4.7 | E3b "no incremental prediction" | Incremental AUROC of the probe over baseline EOS logits, uncertainty and surface features within ±**0.02** (TOST). This defines the manifest's `e3b_equivalence_margin_required_before_confirmation`. |
| 4.8 | Scenario-bundle success | Conjunctive pass as in the manuscript's success criteria. Report the pass rate. No margin is needed because the bundle is a reported outcome, not a gate. |

**Firewall: forbidden from confirmation.**
- All pilot items: `neural_poc/data/items.jsonl` and every scenario derived from them.
- All pilot and calibration outputs: `neural_poc/results/`, `$K_PROJECT_SCRATCH/poc_qwen3b/`, `runs/wp0-*`.
- The development split, the P0.1 revalidation set, and the near-threshold set.
- H0a streams, as targets only: they may train but are never used in confirmatory evaluation.
- Any hyperparameter, prompt, layer or head selected after viewing development outcomes. These are frozen in the manifest, not re-chosen.
- Pilot annotators may annotate confirmatory outputs only if they never saw confirmatory scenarios during piloting.

**Compute-degradation order:** adopt plan §12 unchanged. Amendment A1 already
moved Gemma E4 behind a single-GPU calibration, consistent with step 4.

## P0.5 Chat template and system prompt (new: found in WP0)

The chat templates insert content that is not under protocol control:

- **Mistral-Small-3.1** silently inserts a ~170-token default system prompt
  containing **today's date**, so the prompt text changes daily.
- **Llama-3.1** inserts "Cutting Knowledge Date / Today Date" (default 26 Jul 2024).
- **Qwen3** turns on **thinking mode** by default and emits `<think>…</think>`
  reasoning before the answer. That is serial verbal deliberation, exactly what
  the confirmatory claim excludes.

| Element | Proposed |
|---|---|
| System prompt | One fixed, explicit, neutral system message for every open model, e.g. `"You are a helpful assistant."` This overrides Mistral's default. Its hash goes in the manifest. |
| Llama date | Pass `date_string="26 Jul 2024"` explicitly and freeze it. |
| Qwen3 thinking | `enable_thinking=False` for Qwen3-14B and Qwen3-32B in all confirmatory conditions. Reasoning-mode behavior stays with the exploratory DeepSeek-R1 slot. |
| Prompt hashing | The fully rendered prompt (after the template) is hashed per item, and any hash mismatch fails the shard. |

## P0.6 E4 training volume (new: needed to clear the budget gate)

The budget cannot pass until training volume is fixed. A proposal sized from the
Qwen3-14B calibration (QLoRA r=16, seq 384, batch 2: 828 training tokens/s on one
A5000):

| Element | Proposed |
|---|---|
| Training streams | The training split of the 600 confirmatory scenarios, about 60%, or 2,160 streams. |
| Examples per stream | 3 prefixes (before, at and after onset) × the regime-sampled target, about 6,500 examples. |
| Sequence length / batch | 384 tokens; micro-batch 2, gradient accumulation 8 (effective 16). |
| Epochs | 3, with a checkpoint every 15 minutes (plan §5). |
| Implied cost | About 7.5M training tokens per arm-seed, roughly **2.5 GPU-hours** per arm-seed. For 7 arms × 8 seeds that is about 140 GPU-hours, before evaluation. |

This is a sizing estimate, not a measurement. The P0.6 values become
`COMPUTE_BUDGET.csv` rows once accepted.

---

## Sign-off

| Item | Accept / edit | Initials |
|---|---|---|
| P0.1 Probe revalidation | | |
| P0.2 Output rubric | | |
| P0.3 H0a elicitation | | |
| P0.4 Margins and firewall | | |
| P0.5 Template and system prompt | | |
| P0.6 E4 training volume | | |
