# Critique: Composable-LoRA Patient Simulator

## Overview

Prior work covers three separate areas: LLM patient simulators (Patient-Ψ, Eeyore, AnnaAgent, PatientAct), composable trait LoRAs for general personality (Persona Cartography, Fusian), and prompt-based multi-session client state.
Nobody found so far combines them for clinical simulation.

The proposed novelty is a patient whose symptom severity is stored in the weights of trait-specific LoRA adapters (depression, anxiety, anhedonia, hopelessness). A weighted sum of the adapters produces the patient's behavior. After each session, a classifier-gated rule updates the adapter coefficients, so recovery or worsening persists across sessions. 
The contribution must be shown against prompt-based state tracking, which is the main baseline.

## Notes

1. **LoRA orthogonality may be needed, but it does not secure trait independence.**
   - Methods exist to push adapters toward orthogonality (orthogonality loss, LoRI's frozen random A with sparse B, TIES/DARE at merge time).
   - Reported results show that weight-space orthogonality does not guarantee behavioral disentanglement.
   - Clinical traits also overlap conceptually, so independence may be the wrong goal.
   - Target low, measured cross-trait leakage instead.

2. **After the adapters are trained, a model can learn to choose the alpha of each trait.**
   - A learned policy could replace or complement the fixed update rule.
   - Precedent: Fusian trains a policy network (RL) to compute adapter mixing weights, and found that plain linear interpolation performed worst.
   - Open question: **what reward or supervision defines a "correct" alpha?** Candidates: simulated PHQ-9/GAD-7 targets, or clinician-rated realism.

## Suggested additions by critique claude: ((Havent read it yet))

- **Leakage matrix and dose-response test.** Vary one alpha and measure the effect on all traits. Check for monotonic, separable changes. This is the go/no-go experiment.
- **Baselines.** Prompt-based state (AnnaAgent/PatientAct style), activation steering, a single LoRA conditioned on a trait vector, and a rule-based state machine calibrated to PHQ-9 recovery curves.
- **Fix the update rule.** Use a signed effect (so harmful sessions can worsen the patient), define additive vs. multiplicative updates, and calibrate the rate to clinical recovery trajectories.
- **Training data.** The data must match the target therapy (AnnoMI is motivational interviewing, not CBT). Plan a small clinician-annotated set.
- **Classifier limits.** Handle long transcripts (RoBERTa's 512-token limit) and measure classifier error, since it becomes the patient's ground truth.
- **In-session dynamics.** Alpha is fixed within a session. Decide how resistance, opening up, and alliance are modeled (for example, a trust-gated memory as in PatientAct).
- **Evaluation.** Blinded clinician ratings, consistency across sessions, controllability, and realism of recovery pace.
- **Scope.** Start with 2-3 traits.
- **Safety and ethics.** Clinician review before any training use, and clear labeling as a simulator.
