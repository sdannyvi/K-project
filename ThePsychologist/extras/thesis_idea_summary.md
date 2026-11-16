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
