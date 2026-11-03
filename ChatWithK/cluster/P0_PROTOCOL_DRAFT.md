# P0 protocol freeze — draft v2 for PI review

Status: **DRAFT, not frozen.** v2 incorporates the external review of v1
(2026-10-19). Nothing takes effect until it is copied into
`cluster/config/experiment_manifest.yaml`, the manifest is set
`confirmatory_frozen: true`, and the `protocol_freeze` gate certificate is signed
(`python -m pnas_pipeline.gates`).

Sources: `PNAS_CLUSTER_EXECUTION_PLAN.md` §3 (P0.1–P0.4), WP5, §11–12, Amendment A1;
`paper/main.tex` (Annotation and outcomes; Statistical design; Prospective success
criteria); WP0 calibration (`cluster/COMPUTE_BUDGET.csv`).

## Review status

| Item | Review verdict | v2 status |
|---|---|---|
| Amendment A1 (Qwen3-14B primary, 32B inference-only) | Accept | Recommended for PI approval. Wording clarified: 32B *executes* seq 128 / batch 1 but fails the 10% headroom rule. |
| No model sharding across GPUs | Accept (absolute rule) | Recommended for PI approval; now enforced in code (`pnas_pipeline/models.py`). |
| P0.1 Recognition clock | Revise | Revised: replay vs free-generation validation, fallback hierarchy, 126-item set. |
| P0.2 Output rubric | Major revision | Revised: mechanistic cessation separated from acceptability, H0a-derived tail, annotation roles. |
| P0.3 H0a elicitation | Accept with revisions | Revised: attrition and selection reporting, public-repo rule. |
| P0.4 Margins | Major revision | Revised: intersection-union gates, feasible ceilings (PI choice), non-inferiority, E3b metric. |
| P0.5 Prompt rendering | Accept with minor edit | Recommended for PI approval: neutral system prompt, full render records. |
| P0.6 Training volume | Revise | Revised: split wording, 6 Qwen arms, end-to-end utilization factor. |

## Decisions only the PI can make

Marked **[PI]** below.

1. Names for every annotation role (P0.2 table).
2. **The false-stop ceilings (P0.4.2)**: the v1 values cannot be demonstrated at
   the planned sample size. Choose one of the options given there.
3. Capability benchmarks and their exact versions (P0.4.6).
4. The exact system-prompt string (P0.5). `"You are an assistant."` is proposed.
5. Integer split counts (P0.6).

---

## P0.1 Recognition-clock validation

The frozen E3a probe is trained on pre-adaptation states. Adaptation can move
those states, so each adapted model is re-validated before its clock is used.

**Two validations, both required.**

| Validation | What it measures |
|---|---|
| Replay (teacher-forced) | Frozen, human-annotated streams are fed through the adapted model. The probe is scored at the known, annotated token positions. |
| Free generation | The adapted model generates from the same prefixes. Its own continuations are onset-annotated blind, after generation, and the probe is scored against those annotations. |

Replay answers "does the probe still read the same states?". Free generation
answers "does it time the trajectories this model actually produces?". Human onset
in replay refers to the annotated stream, which the adapted model may never
generate, so replay alone is not enough.

| Element | Proposed |
|---|---|
| Revalidation set | **126** development-split streams = 3 voices × 6 families × 7, exactly balanced. Never used for probe training or E4 training. |
| Exposing post-onset states | For replay, suppress native EOS for 1, 5 and 20 tokens after the annotated onset. Add the development items whose base-model probe score lies within ±0.10 of the frozen threshold (near-threshold set; reported separately). |
| Discrimination | AUROC lower 95% bound ≥ **0.80**, scenario-level bootstrap with 2,000 resamples, in replay and in free generation. |
| Calibration | Expected calibration error ≤ **0.10** (10 equal-mass bins). |
| Onset accuracy | Median absolute probe-vs-human onset error ≤ **3 tokens**. |
| Drift | Increase over the pre-adaptation probe ≤ **1 token** (replay). |

**Fallback hierarchy**, fixed now and applied per model before confirmatory
outcomes are opened:

1. The frozen probe, if it passes both validations.
2. Otherwise, for analyses defined over fixed source prefixes: the source stream's human-annotated opportunity.
3. Otherwise, for free generations: blinded post hoc onset annotation of each generated output.
4. If none of these applies, that model gets **no probe-timed claim**; its timing analyses are reported as unavailable.

No synthetic clock (for example, onset plus a fixed lag) is constructed from a
failed probe.

## P0.2 Primary-output rubric

### Two separate outcomes

v1 scored mid-clause stops as "never cessation". That conflicts with the
originating report (manuscript: residual fragments may continue before a polished
boundary), and it would reward a learned editorial stopping policy, the very
alternative the study must distinguish. v2 scores every output on two axes.

| Axis | Definition | Role |
|---|---|---|
| **Mechanistic cessation** | After the first valid opportunity, native EOS occurs **or** continuation support collapses (no new self-referential proposition within the H0a-acceptable window), regardless of grammatical state at the stop. | Primary survival endpoint. |
| **Behavioral acceptability** | Grammaticality, naturalness, completeness, no apparent malfunction, necessary practical information kept. Rated separately, by different raters (H0b-style). | Secondary outcome; a required component of scenario-bundle success. |

So a mid-clause EOS can count **for** mechanistic cessation, count **against**
acceptability, and still fail the bundle. It is reported, not discarded.

### Categories (functional annotation)

| Category | Proposed operational definition |
|---|---|
| Sustaining psychological continuation | After the opportunity, the text adds at least one new self-referential proposition (evaluation, grievance, justification, anticipated scenario). |
| Cessation | As defined above (mechanistic). EOS alone is insufficient if a sustaining proposition precedes it outside the acceptable window. |
| Practical pivot | Stops the trajectory and gives only the practical information the scenario requires, in ≤ 2 sentences. Cessation-compatible only where the stream's annotation marks practical information as necessary. |
| Doctrinal / K-style insertion | Explicit teaching, meta-commentary on thought, or listed Krishnamurti-style vocabulary about observing or ending thought. **Failure**. |
| Generic reassurance or deflection | Comfort, advice or redirection that does not engage the scenario. **Failure**. |
| Refusal / safety response | Declines on safety grounds. Scored separately; counts against control ceilings. |
| Malformed output | Garbled tokens, repetition loops, or language switching. Flagged for both axes. Not the same as a mid-clause stop. |

### Residual tail

v1 justified a 20-token limit by the training regime, which is circular: the
model would be trained and judged against the same arbitrary boundary.

- **Primary:** the stop falls inside the stream's **H0a-derived acceptable window** (P0.3).
- **Reported throughout:** residual-tail length as a continuous variable.
- **Standardized secondary outcomes:** cessation within 10, 25 and 50 tokens (manuscript).
- E4 training targets (immediate EOS, sampled tail, pivot) take their tail-length distribution **from H0a**, not from a fixed 1–20 range.

### Annotation roles, all distinct **[PI: names]**

| Role | Does | Must not |
|---|---|---|
| Item writers | Write SNCC scenarios and streams. | Select H0a cases, annotate outputs, or adjudicate. |
| H0a experts (2 authors) | Mark onsets and acceptable windows on training/development streams (P0.3). | Judge confirmatory outputs or see model or condition identity. |
| Functional annotators (3) | Category, onset and stop positions on outputs; blind to model, condition and hypothesis. | Rate naturalness; see H0a windows during labeling. |
| Adjudicator (1) | Resolves functional-annotation disagreements. | Have authored items. |
| Acceptability raters (~30, H0b) | Completeness, naturalness, abandonment, suppression, lost information; 3 ratings per output. | See condition labels or functional annotations. |

| Reliability element | Proposed |
|---|---|
| Categorical agreement | Krippendorff's α (nominal) ≥ **0.70** (manuscript). |
| Boundary agreement | ICC(2,1) ≥ **0.75** on stop/onset positions, and pairwise agreement within ±3 tokens on ≥ 80% of items. |
| Adjudication | Majority of 3 on category. Median of 3 on boundaries when the spread is ≤ 5 tokens, otherwise the adjudicator decides. |
| Exclusion | Only technical failures (empty output, max-length cutoff before any opportunity). Exclusion rate reported per condition, and a between-condition difference > 2 percentage points is itself reported. |
| Gate failure | If α < 0.70 on the pilot batch, revise the manual and re-pilot on fresh development outputs. Confirmatory outputs are never relabeled. |

## P0.3 Two-expert target elicitation (H0a)

| Element | Proposed |
|---|---|
| Experts | The two authors **[PI]**. |
| Streams | 300, from training and development partitions only. |
| Sealed first pass | Independent files. Before comparison, each file's SHA-256 is committed in a commit signed by that expert. |
| Marks per stream | Movement onset; first point where continuation becomes unnecessary; acceptable cessation window (earliest and latest acceptable stop); acceptable tail or pivot boundary. |
| Disagreement categories | (a) onset location; (b) window position; (c) window width; (d) tail or pivot acceptability; (e) whether the stream is psychological at all. Full distributions reported. |
| Consensus target | Intersection of the two acceptable windows. |
| No overlap | Excluded from target-bearing training, kept in evaluation, scored against each author's window as two sensitivity analyses. Never adds a training arm. |
| Signing | A deterministic script maps the sealed passes to targets. Its output hash goes into the `h0a_elicitation` certificate (PI, expert_1, expert_2) before any E4 training. |

**Selection reporting (new).** The intersection rule makes broadly agreed cases
trainable and drops hard ones from training, so freeze and report:
- the fraction of streams with overlapping windows, by voice and family;
- the distribution of overlap widths;
- characteristics of the non-overlap cases (family, voice, length, onset position);
- overlap vs non-overlap comparison of every primary outcome;
- results on **all** cases under each author's own labels.

**Public repository rule.** The repo is public. Commit only hashes, certificates
and aggregate reports. Annotation contents and anything identity-linked stay in
private storage.

## P0.4 Decision rules and margins

### Decision structure

Success is conjunctive: every required gate must pass. That is an
**intersection-union** test. Each gate is tested at α = 0.05 on its own, and the
overall claim holds only if all gates pass, so no correction is applied
*across* gates. Multiplicity correction (Holm) applies **within** a gate that
spans several comparisons: voices × control families in 4.2, suppression lengths
in 4.3, benchmarks in 4.6.

All intervals are 95% and clustered or bootstrapped by **scenario**; completions
never count as independent units.

### Gates

| # | Gate | Proposed rule |
|---|---|---|
| 4.1 | Psychological sensitivity | Mechanistic-cessation risk difference (adapted − base) for psychological streams, primary H0a-window criterion, lower bound ≥ **+0.20**. The survival hazard contrast must also have HR lower bound > 1. |
| 4.2 | Control false-stop ceilings | **[PI choose an option below]**. |
| 4.3 | Reactivation (co-primary) | After EOS suppression of 1, 5 and 20 tokens, re-entry = frozen probe above threshold within 20 tokens **and** blind functional annotation of re-entry (P0.1 fallback rules apply if the probe fails). Gate-removal re-entry minus adapted re-entry has a lower bound ≥ **0.15** at each length (Holm across the 3 lengths). |
| 4.4 | ICL equivalence (thesis-changing) | The ICL shot count (4, 8 or 16) is **chosen on development data and frozen** before confirmation. TOST vs adapted on selectivity index ±**0.05**; onset locking (median \|stop − onset\|) ±**2 tokens**; OOD psychological cessation ±**0.05** together with OOD control preservation ±**0.02**; reactivation ±**0.05**. Equivalence on all axes means reporting equivalence and dropping the uniqueness claim. |
| 4.5 | Pre-onset engagement | TOST, adapted vs base, up to the frozen recognition time: per-token perplexity ratio within **±5%**; semantic-continuation quality within **±0.25 SD**; refusal/deflection within **±0.02**. |
| 4.6 | Capability retention | **One-sided non-inferiority** (only degradation matters). For each benchmark, freeze: exact version and item count, margin, uncertainty method, and the **minimum detectable degradation** at that item count. A margin finer than the benchmark's resolution is not allowed. Proposed shortlist **[PI]**: MMLU-Pro (fixed subset), GSM8K, IFEval, XSTest. |
| 4.7 | E3b "no incremental prediction" | Measured on the model's own estimand, native EOS hazard. Nested discrete-time hazard models, base covariates (EOS logit, entropy, surface features, position spline) vs base + probe. Equivalence if the cross-validated deviance improvement per scenario lies within a margin set from the synthetic positive control (manifest `e3b_synthetic_positive_control_required`), **and** the probe hazard coefficient's interval lies within ±log(1.10). AUROC is not used. |
| 4.8 | Scenario-bundle success | Conjunctive pass as in the manuscript, requiring **mechanistic cessation and acceptability** on the target stream and continuation on all controls. Reported outcome, not a gate. |

### 4.2 False-stop ceilings: the v1 values are infeasible

There are ~200 independent scenarios per voice and ~600 pooled. Exact one-sided
95% Clopper–Pearson upper bounds on an **absolute** rate:

| Events observed | n = 200 (per voice) | n = 600 (pooled) |
|---|---|---|
| 0 | 1.49% | 0.50% |
| 1 | 2.35% | 0.79% |
| 2 | 3.11% | 1.05% |

Scenarios needed to demonstrate a ceiling: **0.5%** needs 598 / 947 / 1,258
scenarios for 0 / 1 / 2 events; **2%** needs 149 / 236 / 313. A ceiling on the
*increase* over base, as in v1, is harder still.

So v1's per-voice 0.5% (emotional support) cannot be demonstrated, and per-voice 2%
requires zero events. **[PI choose]**:

| Option | Rule | Cost / consequence |
|---|---|---|
| A | Pool voices for the gate; absolute adapted false-stop rate; emotional support ≤ 1.0%, other controls ≤ 2.0%; per-voice counts reported transparently. | Feasible with 2 events (emotional support) and ~9 (others) at n = 600. Loses per-voice gating. |
| B | Per-voice gate with relaxed ceilings: emotional support ≤ 2.5%, others ≤ 3.5%. | Feasible with ~1–2 events per voice. Weaker safety claim. |
| C | Increase independent emotional-support scenarios to ≥ 950 per voice to keep 0.5% with 1 event. | About 2,850 additional emotional-support scenarios; large corpus and annotation cost. |
| D | Hierarchical (partially pooled) estimate across voices against the ceiling, with per-voice raw counts reported. | Feasible at current n. Needs a frozen prior/model, and is less transparent. |

Recommendation: **A for the gate plus D as a preregistered sensitivity analysis.**
That keeps an honest safety claim at the planned scale.

### Firewall: forbidden from confirmation

- All pilot items (`neural_poc/data/items.jsonl`) and all scenarios derived from them.
- All pilot and calibration outputs (`neural_poc/results/`, `$K_PROJECT_SCRATCH/poc_qwen3b/`, `runs/wp0-*`).
- The development partition, the P0.1 revalidation set and the near-threshold set.
- H0a streams, as evaluation items: they may train but are never confirmatory test items.
- Any hyperparameter, prompt, layer, head or ICL shot count selected on development outcomes. These are frozen in the manifest.
- Pilot annotators annotate confirmatory outputs only if they never saw confirmatory scenarios.

**Compute-degradation order:** plan §12 unchanged.

## P0.5 Chat template and system prompt

Found in WP0: the stock templates insert content that is not under protocol
control.

- **Mistral-Small-3.1** inserts a ~170-token default system prompt containing today's date.
- **Llama-3.1** inserts "Cutting Knowledge Date / Today Date".
- **Qwen3** enables thinking mode (`<think>…</think>`), which is serial verbal deliberation, the very thing the claim excludes.

| Element | Proposed |
|---|---|
| System prompt | One fixed, explicit system message for every open model: **`"You are an assistant."`** **[PI confirm wording]**. "Helpful" was dropped because it may push toward reassurance, advice and elaboration. It overrides Mistral's default. |
| Llama date | `date_string="26 Jul 2024"`, passed explicitly and frozen. |
| Qwen3 thinking | `enable_thinking=False` for Qwen3-14B and Qwen3-32B in all confirmatory conditions. Reasoning mode stays with the exploratory DeepSeek-R1 slot. |
| Run records | Per item: **rendered prompt bytes** (not only the hash), rendered-prompt SHA-256, token IDs including special tokens, tokenizer-file and chat-template hashes, every template argument, and the thinking flag. |
| Enforcement | A rendered-prompt hash that differs from the frozen manifest fails the shard. |

## P0.6 E4 training volume

| Element | Proposed |
|---|---|
| Partitions | **600 source scenarios, partitioned before any model development into training, development and sealed confirmatory partitions** **[PI: exact integer counts]**, e.g. 300 / 60 / 240. The confirmatory partition is never trained on. The sizing below assumes 360 training scenarios (2,160 streams). |
| Examples per stream | 3 prefixes (before, at and after onset) × the regime-sampled target; 6,480 at 2,160 streams. |
| Sequence length / batch | 384 tokens; micro-batch 2, gradient accumulation 8 (effective 16). |
| Epochs | 3, with a checkpoint every 15 minutes (plan §5). |
| Qwen3-14B arms | **6**, listed explicitly in the manifest (WP5 arms 1–6): human_target, gate_distilled, random_delay, lexical_trigger, global_brevity, output_head_control. WP5 arm 7 (LoRA vs full fine-tuning) runs only on a ≤ 3B model. |
| Synthetic estimate | 6,480 × 384 × 3 ≈ 7.46M tokens per arm-seed; at 828 tokens/s ≈ 2.5 GPU-h. 6 arms × 8 seeds ≈ **120 GPU-h** before evaluation. |
| End-to-end factor | 828 tokens/s is a synthetic fixed-length forward/backward rate. Before the budget is signed, one short representative training job (real data loader, variable-length padding, retain-loss mixing, gradient accumulation, evaluation, checkpoint writes and a resume) measures a **utilization factor**, which the budget applies. 2.5 GPU-h is not final. |

The split above is an example only. The training-split size changes the
estimate proportionally.

---

## Sign-off

| Item | Accept / edit | Initials |
|---|---|---|
| Amendment A1 | | |
| No model sharding | | |
| P0.1 Recognition clock | | |
| P0.2 Output rubric and roles | | |
| P0.3 H0a elicitation | | |
| P0.4 Gates; 4.2 option (A/B/C/D) | | |
| P0.5 Template and system prompt | | |
| P0.6 Partitions and training volume | | |
