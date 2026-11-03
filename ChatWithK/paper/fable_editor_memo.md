# Fable editorial memo and adopted revision plan

## Source

Independent structural critique obtained from Claude Fable 5 High on 26
September 2026. The prompt described the construct, all five intervention
levels, the current pilot results, author positionality, and the intended venue.
Fable did not receive the full manuscript, so its advice is structural rather
than line-level.

## Central diagnosis

The draft contained two competing papers:

1. a supported negative-result paper showing that verbal recognition,
   instruction, serial supervision, and semantic steering do not entail
   selective termination; and
2. a not-yet-supported positive paper claiming that weight adaptation creates
   endogenous selective termination.

The 1/3 LoRA pilot cannot carry the second claim. The paper must either present
the first contribution now, with the LoRA work as a roadmap, or wait for the
preregistered cluster study and make the weight-level result—positive or
negative—the center. It should not submit the hybrid.

## Recommended architecture

1. Introduction: practical recognition–control gap; contemplative provenance
   gets one paragraph, not the opening claim.
2. Construct and data: define psychological trajectory before methods; foreground
   descriptive, fictional, first-person practical, and lexical minimal-pair
   controls.
3. Recognition without cessation: compress prompting, persona/RAG, serial
   supervision, and CAA into confound-establishing baselines.
4. From external actuation to native adaptation: use detector-plus-EOS as the
   feasibility bridge and classifier-plus-truncation baseline; make LoRA the
   main method.
5. Results: weight-level survival curves, selectivity, false positives,
   onset-time dependence, OOD transfer, cross-model/seed replication, and
   reactivation.
6. Discussion: what EOS does not prove, conceptual provenance, and positionality
   safeguards.

## Krishnamurti rule

Use at most three primary quotations and keep them out of Methods and Results.
Each must follow the chain:

> quote → operational construct → measurable prediction → experiment

The manuscript must remain scientifically intact if every Krishnamurti sentence
is removed. The source motivates the question; it does not validate labels or
outcomes.

## Main reviewer risks and design fixes

- **“Refusal training with metaphysical garnish.”** Include a classifier-plus-
  truncation baseline and show what native adaptation buys empirically.
- **Construct invalidity.** Add annotation rules, agreement, adversarial
  descriptions, fiction, therapy-style reports, and first-person planning.
- **Lexical shortcut.** Use vocabulary-matched minimal pairs, onset analyses,
  and length matching.
- **Same-distribution classification.** Add cross-domain, human-written, and
  paraphrase tests, plus cross-model replication.
- **EOS is not cessation.** Concede this explicitly and foreground reactivation.
- **Author bias.** Preregister, blind labels and analyses, and include an
  investigator without the reported phenomenology where possible.

## Venue recommendation

ACL ARR first. The contribution is natively NLP/interpretability, depends on
methodological detail, and can support a rigorous negative result. PNAS is a
possible later target only after a replicated positive mechanistic study and
substantial independently validated human data.
