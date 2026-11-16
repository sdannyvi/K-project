---

**Introduction**
Psychotherapy training lacks scalable, realistic patient simulators. Existing systems either use static prompts or inject emotional state externally — the model is never fully commited 'to the part'. This thesis proposes a system where psychological traits are encoded architecturally inside the model via composable LoRA adapters. The patient's state updates across sessions only when a classifier confirms the session was clinically meaningful.

---

**Method**
Each clinical trait (depression, anxiety, anhedonia, hopelessness) gets its own LoRA adapter trained on clinical data. At inference, a weighted sum of adapters produces the patient's current behavior: `ΔW = α₁·ΔW_dep + α₂·ΔW_anx + ...`. 
A gating classifier reads the session transcript and outputs per-trait quality scores. 
The update rule `αᵢ(t+1) = αᵢ(t) + η · Gᵢ · Δᵢ` shifts only the traits the session meaningfully addressed.

Update Rule — Detail
After each session, a RoBERTa classifier fine-tuned on AnnoMI reads the full transcript and outputs two values per trait: Gᵢ ∈ {0,1} indicating whether the trait was therapeutically addressed, and 
Δᵢ ∈ [0,1] indicating how effectively.
Gᵢ is determined by detecting which CBT technique was used and mapping it to its target traits via clinical literature. 
Δᵢ is determined by the execution quality of that technique. The coefficient αᵢ decreases only when Gᵢ=1, at a rate controlled 
by η=0.05 — meaning a perfect session can reduce a trait's severity by at most 5%, enforcing clinically realistic recovery pace.
A poor or irrelevant session leaves all coefficients unchanged. If a session is actively harmful, αᵢ can increase, making the patient worse — adding clinical realism in both directions.

---

**Steps**
1. Train one LoRA adapter per trait on clinical datasets (DAIC-WOZ, CLPsych)
2. Validate that adapter interpolation produces coherent intermediate states
3. Fine-tune a RoBERTa gating classifier on AnnoMI therapy transcripts
4. Build the state controller — α vector persisted across sessions
5. Connect all components end-to-end
6. Evaluate against static prompt, turn-adaptive, and ungated baselines