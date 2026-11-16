# Thesis Idea Summary: Trait-Adaptive LLM Patient Simulator

## The Core Idea
Build an LLM-based patient simulator for psychotherapy training where psychological traits are encoded **inside the model architecture** — not via prompting. The patient's state changes across sessions only when a classifier confirms the session was clinically meaningful.

---

## Expert Assessment

**Strengths:**
- Encoding traits in weights (not prompt) is a real architectural departure — prior systems can be talked out of their persona, this one can't
- α vector is clinically interpretable — a supervisor can read it
- Gating is clinically sound — patient shouldn't improve unless therapist earns it
- Novelty confirmed via alphaXiv search

**Weaknesses:**
- LoRA composition at scale is unproven — biggest risk, test this first in month 1
- Gating classifier is doing too much — per-trait quality scoring is hard even for human supervisors
- DAIC-WOZ is small (~200 subjects) and not therapy sessions — distribution shift risk
- A well-prompted GPT-4 might outperform your system purely due to base model quality — control for this in baselines

**Bottom line:** Strong novel idea with real clinical grounding. Main risk is 4-5 components that could each fail independently in 7 months. **Priority: validate LoRA composition coherence before anything else.**

---

## What Makes It Novel
All prior work uses one of:
- Static prompt-based personas
- External VAD/state vectors injected into the prompt
- Single-session adaptive behavior

**Nobody has combined:**
1. Per-trait LoRA adapters (clinical traits, not personality)
2. Continuous adapter composition at inference
3. Persistent per-patient coefficient vector across sessions
4. Clinically gated update rule per trait

Confirmed novel via systematic alphaXiv literature search (September 2026).

---

## Closest Prior Work
| Paper | What it contributes | What's missing |
|---|---|---|
| P-React | LoRA mixture for Big Five personality | No clinical traits, no persistent state, no gate |
| PsychAdapter | Architectural conditioning on depression/traits | No persistent state, no update rule |
| Latent Personal Memory | Persistent soft-prompt state across sessions | Not emotional state, no clinical gate |
| TheraJudge | Clinical quality signal for therapy sessions | Refines responses, doesn't update adapter state |
| VAD paper | Persistent affect state across turns | External to model, prompt-injected |

---

## Architecture: 5 Components

### 1. Base LLM
Frozen LLaMA 3 or Mistral. Never modified directly.

### 2. Per-Trait LoRA Adapters
One LoRA adapter trained per clinical trait:
- ΔW_depression, ΔW_anxiety, ΔW_anhedonia, ΔW_hopelessness, ...
- Each trained independently on clinical language data
- Captures linguistic/behavioral patterns of that specific trait

### 3. Trait Mixer
At inference, computes weighted sum:
```
ΔW_final = α₁·ΔW_depression + α₂·ΔW_anxiety + α₃·ΔW_anhedonia + ...
W' = W + ΔW_final
```

### 4. Clinical State Controller
- Maintains α vector per patient-persona
- α persists across sessions (stored in file/database)
- Each αᵢ ∈ [0,1]: severity of trait i (0=absent, 1=severe)

### 5. Gating Classifier
- RoBERTa fine-tuned on AnnoMI therapy transcripts
- Input: full session transcript
- Output per trait: Gᵢ ∈ {0,1} (was trait addressed?) and Δᵢ ∈ [0,1] (how well?)
- Technique → trait mapping defined by CBT clinical literature

---

## Update Rule
```
αᵢ(t+1) = αᵢ(t) - η · Gᵢ · Δᵢ
```
- η = 0.05 (max 5% change per session, clinically realistic)
- Gᵢ = 0 → trait unchanged regardless of session quality
- Poor session → no update
- Harmful session → αᵢ increases (patient worsens)

### Example
```
Student applies behavioral activation well:
  G_anhedonia=1, Δ_anhedonia=0.3 → α: 0.90 → 0.885
  G_depression=1, Δ_depression=0.1 → α: 0.80 → 0.795
  G_anxiety=0 → α: 0.60 → 0.60 (unchanged)
```

---

## Datasets
| Dataset | Use |
|---|---|
| DAIC-WOZ | Depression transcripts + PHQ-8 for adapter training |
| AnnoMI | Therapy transcripts with quality annotations for gating classifier |
| HOPE Dataset | Multi-turn psychotherapy for evaluation |
| CLPsych Shared Tasks | Mental health language for adapter training |
| EmpatheticDialogues | Baseline comparison |

---

## Evaluation
| What | How |
|---|---|
| Behavioral realism | Psychologist ratings of simulated transcripts |
| Gate effectiveness | Poor vs. good sessions → measure state change |
| Trajectory plausibility | Compare recovery curves to real therapy data |
| LoRA composition coherence | Do interpolated adapters produce coherent intermediate states? |
| Learning transfer | Trainee skill pre/post assessment |

**Baselines:**
1. Static prompt persona
2. Turn-adaptive disclosure model
3. Ungated multi-session state update

---

## Implementation Stack
- **Base model:** LLaMA 3 or Mistral (open weights)
- **LoRA:** HuggingFace PEFT
- **Classifier:** RoBERTa fine-tuned on AnnoMI
- **Framework:** PyTorch + Transformers
- **State storage:** JSON/vector store per patient

---

## Timeline
| Month | Milestone |
|---|---|
| 1–2 | Train and validate per-trait LoRA adapters |
| 3 | Implement state carrier; validate LoRA interpolation coherence |
| 4 | Train and validate gating classifier on AnnoMI |
| 5–6 | End-to-end integration and evaluation |
| 7 | Thesis writing |

---

## Key Open Questions (potential thesis contributions)
1. Does LoRA adapter interpolation produce clinically coherent intermediate states?
2. Can a classifier reliably map therapy technique quality to per-trait gates?
3. Does this system produce more realistic multi-session trajectories than prompt-based baselines?
4. Does training on this system improve real student clinical skills?

---

## Novelty Claim (one sentence)
> Independently trained clinical-trait LoRA adapters, composed at inference via a continuous coefficient vector that persists per patient across sessions and updates only when a trait-specific quality gate passes.

---

## Session Notes: Design Decisions & Deeper Understanding

### LoRA Training
- LoRA is trained via standard next-token prediction on input-output text pairs
- Example pair: `[THERAPIST]: "How have you been?" → [PATIENT]: "Just empty, nothing feels worth doing"`
- The base model is frozen; only the A and B matrices are updated
- Monologue patient speech also works (Reddit posts, diary entries, clinical interviews)

**Training strategy options (decided: Option 1 or 3 preferred):**
| Option | Description | Pro | Con |
|---|---|---|---|
| 1. Joint multi-task | All adapters trained on same corpus simultaneously | Simple, reduces interference naturally | No explicit orthogonality guarantee |
| 2. Mixture-of-LoRA | Composition trained end-to-end | Coherent by construction | Loses modularity, muddies novelty |
| 3. Orthogonal LoRA | Adds regularization loss: `λ · Σᵢ≠ⱼ \|\|AᵢAⱼᵀ\|\|²_F` | Mathematical guarantee of non-interference | May limit individual adapter expressiveness |
| 4. Shared encoder | Shared A matrix, per-trait B matrices | Composability guaranteed | Less prior work to lean on |

**Recommendation: Option 3 + Option 1 combined** — joint training data with orthogonality regularizer. Clean theoretical story + empirical ablation contribution.

---

### Depression Coefficient (α) — How to Set and Update It

#### PHQ-9 (initialization, simple)
9-question self-report, each item 0–3, total 0–27:
- Q1: Little interest/pleasure (anhedonia)
- Q2: Feeling down/hopeless
- Q3: Sleep problems
- Q4: Fatigue
- Q5: Appetite changes
- Q6: Feeling bad about yourself
- Q7: Concentration difficulty
- Q8: Psychomotor changes
- Q9: Suicidal thoughts

Score bands: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderately severe, 20–27 severe

```
α_depression = PHQ-9_score / 27
Example: score 15 → α = 0.56
```

#### MADRS (richer, per-trait breakdown)
Clinician-rated, 10 items each 0–6. Maps directly to your α vector:
- Item 8 → anhedonia
- Item 9 → hopelessness
- Item 3 → anxiety/inner tension

```
α_anhedonia   = MADRS_item8 / 6
α_hopelessness= MADRS_item9 / 6
α_depression  = mean(items 1,2) / 6
```

**Decision: Start with PHQ-9 for depression-only MVP. Extend to MADRS when adding more traits.**

---

### BERT/RoBERTa Classifier — Training for Depression Severity

**Architecture (depression only):**
```python
class DepressionScorer(nn.Module):
    def __init__(self):
        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.regression_head = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()  # output in [0,1]

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids, attention_mask).last_hidden_state[:, 0]
        return self.sigmoid(self.regression_head(cls))

loss = nn.MSELoss()(predicted_score, true_phq_normalized)
```

**Extend to MADRS (multi-trait):** same architecture, replace `nn.Linear(768, 1)` with `nn.Linear(768, 10)` — one head per MADRS item.

**Training data:** DAIC-WOZ — ships with transcripts + PHQ-8 labels already paired. Label is free.

```
Input:  "I stopped seeing friends. Too much effort. Sleep all day but always tired..."
Label:  PHQ-8 = 17  →  normalized: 17/24 = 0.71
```

**Caveat:** DAIC-WOZ uses structured interviews (not therapy sessions). Fine for MVP, note as limitation in thesis.

---

### Synthetic Data Option
LLM-generated PHQ responses are a valid and strong data augmentation strategy:

```
System: You are a patient with moderate depression (PHQ-9 ≈ 15).
        Answer each PHQ question naturally, as if speaking to a clinician.

User:   How often have you felt down, depressed or hopeless?

→ Patient: "Most days honestly. I wake up and it just hits me immediately..."
```

**Why this is a thesis contribution, not just a workaround:**
- Lets you control severity precisely (generate patients at every PHQ level)
- Fills gaps in real data (DAIC-WOZ is small, ~200 subjects)
- Ablation: real vs. synthetic data quality comparison
- Label is known exactly — no annotation noise

**Pipeline:**
1. Generate transcripts at PHQ levels: 5, 10, 15, 20, 25
2. Use as training data alongside real DAIC-WOZ
3. Evaluate: does mixing synthetic + real improve classifier?

---

### The Core Problem with BERT on Patient State

BERT trained on PHQ scores measures **current patient state** — a snapshot. It cannot answer the real question: *did the therapist's intervention actually help?*

A patient can score 0.65 in session 2 vs 0.71 in session 1 for reasons unrelated to therapy (bad week, poor sleep, family stress). The classifier can't separate therapeutic effect from life noise.

**What you actually need:** measure therapist behavior quality, not patient state.

```
Wrong question: "how depressed is this patient?"
Right question: "did the therapist do something that should reduce depression?"
```

---

### AnnoMI-Based Classifier — Better Option for G and Δ

AnnoMI annotates **therapist moves** inside therapy transcripts — reflections, affirmations, open questions, change talk, and their quality. This maps directly onto your G and Δ outputs.

**Architecture:** same BERT, different labels:

```python
# Input: full session transcript
# Output: G (was depression addressed?) + Δ (how well?)

class TherapistQualityClassifier(nn.Module):
    def __init__(self):
        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.G_head = nn.Linear(768, 1)   # binary: technique applied?
        self.delta_head = nn.Linear(768, 1) # regression: how well?
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids, attention_mask).last_hidden_state[:, 0]
        G = self.sigmoid(self.G_head(cls))          # 0 or 1
        delta = self.sigmoid(self.delta_head(cls))  # 0.0 to 1.0
        return G, delta
```

**Training data:** AnnoMI transcripts with therapist technique annotations. You add a CBT mapping layer on top:

```
AnnoMI label: "behavioral activation, high quality"
→ G_depression = 1, Δ_depression = 0.8

AnnoMI label: "open question, low quality"  
→ G_depression = 0, Δ_depression = 0.0
```

This mapping (technique → trait) is defined by CBT clinical literature and is itself a thesis contribution.

---

### Three Options for G and Δ — Summary

| Option | What it measures | Data needed | Complexity |
|---|---|---|---|
| BERT on PHQ (state) | Patient depression snapshot | DAIC-WOZ | Low — but wrong question |
| BERT on AnnoMI (therapist) | Therapist technique quality | AnnoMI + CBT mapping | Medium — right question |
| LLM-as-judge | Therapist quality via prompt | None (zero-shot) | Low — flexible MVP |

**Recommendation:**
- MVP: LLM-as-judge to get end-to-end system working immediately
- Validation: AnnoMI-trained BERT to check if LLM judgments agree with annotated ground truth
- The agreement/disagreement between the two is itself a thesis finding

---
---

# Recommendations for Changes (Revision — October 2026)

This section collects the changes recommended after a second literature pass and design review. The original sections above are kept unchanged for reference; where they conflict with this section, this section supersedes them.

---

## 1. Novelty Claim — Reframe and Narrow

**Remove "Confirmed novel via alphaXiv search".** A single search cannot confirm novelty, and the phrase invites a reviewer to find a counterexample. Several close papers were missing from the original related-work table.

### Additional Prior Work (must be cited)
| Paper | What it does | How we differ |
|---|---|---|
| Empirically Grounded Adaptive Virtual Patient (arXiv 2606.10051) | Separates state transitions from language generation; latent disclosure state updated per turn using SEM weights estimated from 2,000+ hours of real therapy transcripts | Their state conditions generation through the input and is within-session only; ours lives in weight space and persists across sessions. **Closest competitor — also use as a baseline.** |
| DMT-CBT (arXiv 2606.03132) | Frames CBT counseling as longitudinal therapeutic state modeling; synthetic multi-session dataset | Models the counselor side and uses structured text states, not adapter coefficients |
| eRisk 2026 LLM personas (described in arXiv 2607.16712) | Each depressed persona is a LoRA adapter on Llama-3-8B-Instruct | One static adapter per persona; no composition, no persistent coefficient, no update |
| PatientAct (arXiv 2608.12750) | Theory-grounded client profiles; finds profile depth matters more than dynamic mechanisms; finds LLM judges insufficient for clinical-judgment dimensions | Challenges our premise (dynamics) and our LLM-as-judge MVP — must be addressed explicitly |
| TalkDep (arXiv 2508.04248) | Clinician-in-the-loop depressed personas with BDI-II severity and key symptoms | Prompt/profile-based, static |
| MyMentorLLM (arXiv 2607.25667) | CBT training environment with simulated patient, trainee and supervisor | Prompt-based patients |

### Motivation Evidence (persona drift)
| Paper | Finding useful for the introduction |
|---|---|
| Persistent Personas? (de Araujo et al., 2025/26) | Persona fidelity degrades over 100+ round dialogues, converging toward baseline assistant behavior |
| Identity Drift in LLM Agents (arXiv 2412.00804) | Larger models drift more; assigning a persona may not help |
| The Assistant Axis (Anthropic, arXiv 2601.10387) | Emotionally charged and meta-reflective conversations — i.e. therapy-like ones — cause drift away from the trained default |
| Consistently Simulating Human Personas with Multi-Turn RL (Abdulhai et al., NeurIPS 2025, arXiv 2511.00222) | Defines prompt-to-line, line-to-line and Q&A consistency metrics; reuse these as evaluation metrics |

### Related Methods (for the architecture chapter)
- P-React (ACL Findings 2025) — mixture of Big Five LoRA experts with specialization loss
- Open Character Training (arXiv 2511.01689) — open character-training pipeline; useful for the base patient adapter
- Persona Vectors (arXiv 2507.21509) — activation directions for monitoring traits; use as a measurement tool, not for steering
- Text-to-LoRA (Sakana AI, ICML 2025) — hypernetwork alternative to the linear mixer (possible ablation)

### Revised Novelty Sentence
> Unlike prior simulators that condition generation on an explicit state through the input, we encode symptom severity in weight space via composable per-symptom LoRA adapters, combined with a fast within-session state; the slow severity coefficients persist across sessions and update only when the within-session state trajectory indicates a therapeutically meaningful session.

---

## 2. Architecture — Move to Two Timescales

Clinically, symptom severity does not change mid-session; **state** does (openness, resistance, emotional activation). Split the controller accordingly.

| Vector | Meaning | Updated | Drives |
|---|---|---|---|
| α (slow) | Symptom severity, one value per symptom adapter | Between sessions | Symptom LoRA adapters |
| β (fast) | Within-session state: openness/change-talk, resistance/sustain-talk, arousal | Every turn; resets toward baseline at session end | State LoRA adapters |

Optional bounded modulation so symptoms can visibly fluctuate without changing α itself:
```
α_effective = clip(α_slow · (1 + κ · mood_t), 0, 1)     # κ small, e.g. 0.1
```

### Revised Component List
1. **Frozen base LLM** (LLaMA 3 / Mistral) — unchanged
2. **Patient base adapter (NEW, always on)** — role, speech style, disclosure habits. Without it, α = 0 produces a generic assistant, not a recovered patient. Can be trained with an Open Character Training–style pipeline.
3. **Symptom adapters** — one per PHQ/MADRS symptom (see §3)
4. **State adapters (NEW)** — openness/change-talk and resistance/sustain-talk; trained on AnnoMI client utterances (plus synthetic augmentation)
5. **Within-session controller (NEW)** — computes β every turn (see §4)
6. **Cross-session update** — α update gated by the β trajectory (see §5); replaces the session-level RoBERTa gate

### Composition at Inference
```
ΔW_final = ΔW_base + Σᵢ α_eff,i · ΔW_symptom,i + Σⱼ β_j · ΔW_state,j
```
Apply per-adapter scaling with multiple active LoRA adapters rather than merging weights each turn (check the PEFT API for the installed version). For the KV cache, recomputing the conversation each turn is simplest and exact; reusing the cache means earlier turns were encoded under the old state — defensible, but state it in the thesis.

---

## 3. Trait / Symptom Set — Remove Double-Counting

- Anhedonia and hopelessness are **components** of depression; a separate "depression" adapter double-counts them.
- **Change:** use symptom-level adapters mapped to PHQ/MADRS items (e.g. anhedonia, low mood/hopelessness, fatigue, worthlessness, inner tension). Depression becomes the aggregate, and α initialization maps one-to-one onto adapters.
- **Orthogonality regularizer:** these symptoms are clinically correlated, so forcing their adapters orthogonal may hurt realism. Demote Option 3 to an **ablation**; use joint multi-task training (Option 1) as the main method.
- **Calibration (NEW, month 1):** LoRA scale is not linear in behavioral severity. Use the DAIC-WOZ depression scorer as a calibration tool: sweep α from 0 to 1, score generated responses, and check that measured severity rises monotonically. Fit a mapping α_target → scale if needed. This is the "LoRA composition coherence" experiment and stays the top priority.

---

## 4. Within-Session Controller (NEW)

### Training Signal
AnnoMI has utterance-level expert labels:
- Therapist turns: `reflection`, `question`, `therapist_input`, `other`
- Client turns: `change`, `neutral`, `sustain`

Supervised task: **given the dialogue history and the latest therapist turn, predict the next client talk type.** Predicted probabilities drive β. No invented labels needed.

### Architecture
1. **Encoder:** DeBERTa-v3-base, RoBERTa, or MentalRoBERTa. Input = last k turns with speaker tags (fits within 512 tokens because it is turn-level).
2. **Two heads, trained jointly:**
   - therapist behavior (4 classes) — interpretable output for supervisors
   - next client talk type (3 classes) — drives β
3. **Dynamics layer (inertia):**
```
p_t     = softmax(next_talk_type_head(h_t))        # [change, neutral, sustain]
β_{t+1} = clip( λ·β_t + (1−λ)·(W·p_t + b), β_min, β_max )
```
   - λ ≈ 0.7–0.9 controls stickiness; fit λ, W, b on AnnoMI sequences by maximizing likelihood of the real client talk-type sequence (small state-space model, interpretable).
   - GRU over turn embeddings = ablation only (133 conversations → overfitting risk).
   - **Avoid** probing the LLM's own hidden states as the encoder: they are shaped by the adapters that β controls (feedback loop). If tried, read with adapters disabled.

---

## 5. Cross-Session Update — Fixes

### Sign Bug
With Gᵢ, Δᵢ ∈ [0,1], the original rule can only decrease α, contradicting "harmful session → patient worsens". Revised rule:
```
αᵢ(t+1) = clip( αᵢ(t) − η · Gᵢ · Δᵢ + εₜ , 0, 1 )
Δᵢ ∈ [−1, 1]        # signed: negative = harmful session
εₜ ~ N(0, σ²)       # small noise / life events; optional relapse events
```
Real recovery curves are noisy and non-monotonic; perfectly smooth trajectories will fail the trajectory-plausibility evaluation.

### Gate Derived from β (replaces session-level RoBERTa gate)
- Sustained change talk in the final third of the session → G = 1, Δ > 0
- Sustain-dominated session or sharp resistance spike → Δ < 0
- Otherwise → G = 0

### Why the Original Gate Data Doesn't Fit
- AnnoMI is **Motivational Interviewing**, not CBT — there is no "behavioral activation, high quality" label to map from.
- Only ~133 conversations; session-level quality is just high/low.
- RoBERTa's 512-token limit cannot take a full session transcript.
- The LLM-as-judge MVP is weakened by PatientAct's finding that LLM judges are insufficient on clinical-judgment dimensions — validate it against a small clinician-rated set before relying on it.

---

## 6. Data — Corrections

| Issue | Fix |
|---|---|
| DAIC-WOZ uses **PHQ-8**, not PHQ-9 (no suicidal-ideation item) | Normalize by 24: `α = PHQ8 / 24` |
| DAIC-WOZ requires a signed EULA | Apply immediately (month 0) |
| DAIC-WOZ is a structured interview with a virtual interviewer, not therapy | State as a limitation |
| AnnoMI is MI demonstration videos on alcohol/smoking/recidivism, not depression | State as domain shift; annotate a small set of depression-focused transcripts with the same labels for validation |
| Synthetic data from a prompted persona partly **distills the prompt baseline** | State explicitly (context distillation) and show the system beats that baseline |
| Suicidal-ideation content in a simulator | Define content guardrails and how trainees are briefed |

---

## 7. Evaluation — Additions

| What | How |
|---|---|
| Adapter calibration | α sweep → depression scorer → monotonicity check |
| Controller accuracy | Next-client-talk-type F1 on held-out AnnoMI vs. majority class and zero-shot LLM |
| Known-groups validity of gate | Replay real high- vs. low-quality AnnoMI sessions; high-quality must end with clearly higher β |
| Text–state consistency | Talk-type classifier on generated patient replies must agree with β |
| Scripted perturbations | Matched sessions differing only in a few therapist turns (accurate reflections vs. confrontation) → β moves in expected direction |
| Persona consistency over long dialogues | Prompt-to-line, line-to-line, Q&A consistency (Abdulhai et al.) at 10/20/40/60 turns |
| Adversarial robustness | "Therapist" attempts to break character; treat "can't be talked out of persona" as a **hypothesis to test**, not a strength |
| Internal monitoring (optional) | Persona-vector projections as a non-steering measurement of trait expression |

### Revised Baselines
1. Static prompt persona (strong prompted model, e.g. GPT-4-class)
2. Explicit-state + prompt (Adaptive-VP style) — **new, closest competitor**
3. Turn-adaptive disclosure model
4. Ungated multi-session state update
5. Ablations: orthogonal vs. joint adapters; linear mixer vs. GRU controller; with vs. without patient base adapter

---

## 8. Scope and Timeline — Revised

- Move the **learning-transfer study** (trainee pre/post) to future work or a tiny pilot — ethics approval and recruitment alone can take months.
- Start recruiting **psychologist raters** in month 1.
- **MVP:** depression only, 3–4 symptom adapters + 2 state adapters, β controller on AnnoMI, gate from β trajectory.

| Month | Milestone |
|---|---|
| 0–1 | DAIC-WOZ EULA, ethics application, rater recruitment; train symptom adapters + **calibration sweep** (go/no-go on composition) |
| 2 | Patient base adapter + state adapters |
| 3 | Within-session controller on AnnoMI (encoder, heads, dynamics fit) |
| 4 | β-derived cross-session gate; known-groups validation on AnnoMI |
| 5–6 | End-to-end integration, baselines, evaluation, human ratings |
| 7 | Thesis writing |

---

## 9. Revised Key Open Questions
1. Does symptom-adapter interpolation produce calibrated, clinically coherent intermediate severities?
2. Can a turn-level controller trained on next-client-talk-type prediction produce plausible within-session state dynamics?
3. Does a gate derived from the within-session state trajectory distinguish high- from low-quality sessions?
4. Does weight-space state hold up better than explicit-state prompting over long and adversarial dialogues?
5. Do multi-session trajectories look more realistic to clinicians than prompt-based and explicit-state baselines?
