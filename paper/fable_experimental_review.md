# Fable 5 High experimental-program review

Review date: 26 September 2026  
Claude chat: `Language model cessation without verbal control`

## Editorial verdict

Fable's candid assessment was **major revision at the design stage; not yet PNAS-competitive**, while describing the program as potentially important and substantially above field standard in its negative-model ladder, reactivation logic, base/instruct decomposition, and confirmatory governance.

The strongest defensible claim is not that the experiment excludes every possible internal detector. It is that weight adaptation can produce **selective, recognition-coupled termination in native single-process dynamics without serial verbal deliberation or an inference-time external controller**.

## Identification gaps it found

1. An adapted network may internalize a distributed detector/actuator; causal localization cannot simultaneously prove that no discrimination mechanism exists.
2. The corpus must separate first-person model voice, second-person/user-directed response, and quoted or reported character voice.
3. EOS is confounded by length, template position, and instruction tuning unless onset positions and hazard terms are explicitly controlled.
4. Silent cessation requires a recognition clock independent of emitted recognition language.
5. In-context demonstrations are a missing alternative to parameter adaptation.
6. Human baselines are needed to validate cessation points and distinguish natural cessation from visibly broken truncation.
7. Weight adaptation might avoid psychological material before onset rather than selectively cease after recognition.

## Must-fix recommendations incorporated into the manuscript

- 4/8/16-shot ICL arm evaluated against native adaptation.
- Three-way voice stratification and voice-specific false-positive ceilings.
- Onset agreement gate, onset-jitter sensitivity, position-matched families, and position splines.
- Frozen pre-adaptation recognition probe used as the silent recognition clock.
- Expert cessation-point and blinded output-perception human studies.
- Pre-onset engagement equivalence tests.
- Contamination and corpus-expansion-source audits.
- ChatWithK demoted to motivating observational evidence.
- Reactivation elevated to a co-primary outcome.
- Human-target versus gate-distillation bridge between E2 and E4.

## Scope warning

Fable recommended moving the full transition-topology program to a second paper and shrinking the model/probe zoo. The manuscript retains the aspirational E5 program because this draft is a research blueprint, but the eventual PNAS submission should lead with a minimal decisive package and treat broader E5 analyses as exploratory unless their results are unusually coherent.

## Second-round rebuttal and approval

After receiving a point-by-point response incorporating the changes above, Fable closed the internalized-classifier, voice, EOS, ICL, human-baseline, avoidance, safety, leakage, distillation, scope, and provenance objections. It judged the program **PNAS-competitive if the preregistered results land** and ended with:

> Conditional on B1--B4 entering the protocol before pilot annotation and before any adapted model is trained: **OK TO PROCEED**.

The four conditions are protocol specifications rather than experimental redesigns:

1. Revalidate the frozen recognition probe after adaptation on held-out annotated sequences exposed under EOS suppression or on near-threshold items; freeze drift tolerances and a fallback clock before training.
2. Freeze the primary-output rubric before confirmatory generation, including maximum residual-tail length and operational distinctions among cessation, sustaining continuation, practical pivot, doctrinal insertion, deflection, and accidental truncation; require blinded annotation and an agreement gate.
3. Add a quantitative agreement/dispersion gate to H0a for expert cessation points and define exclusion or adjudication for streams that fail it.
4. Replace qualitative terms such as “materially lower” and “matches E4” with numeric confidence-bound and TOST margins; enumerate the pilot-to-confirmatory firewall and pre-specify the order in which compute-heavy conditions may be reduced without inspecting outcomes.

Final second-round verdict: **READY FOR PILOT ONLY** until these four specifications are frozen; once they are written into the protocol, pilot work may proceed. A successful pilot followed by the fully preregistered result pattern was explicitly judged a **PNAS-competitive design**.
