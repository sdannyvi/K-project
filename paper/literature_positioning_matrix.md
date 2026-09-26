# Literature positioning matrix

Reviewed through 2026-09-26. This is a working map, not a substitute for checking
the final papers and BibTeX before submission.

## The shortest defensible novelty statement

Prior work asks whether models represent mental states, inspect their own
uncertainty, resist irrelevant context, stop redundant reasoning, or express a
steerable refusal. We ask whether recognition of a **developing psychological
trajectory** becomes **selective cessation of that same trajectory**, while
matched practical inner speech remains intact.

## Neighbor comparison

| Literature | What is detected or represented? | What action follows? | Why it does not subsume this paper |
|---|---|---|---|
| Theory of Mind | Beliefs, intentions, emotions, knowledge states | Usually an answer about an agent; increasingly a social plan | Correct mental-state representation need not terminate the model's own developing trajectory |
| LLM metacognition | Confidence, likely correctness, prospective self-prediction, step errors | Abstain, revise, allocate effort, or report confidence | Target is epistemic performance, not selective withdrawal from self-maintaining psychological generation |
| Context distraction | Irrelevant or conflicting information | Preserve task accuracy | Our contaminant is relevant and answer-like; the failure can remain factually and stylistically impressive |
| Overthinking / early stop | Redundant reasoning after answer sufficiency | End chain of thought or emit final answer | Stopping criterion is solution sufficiency/efficiency; our matched streams may be equally unfinished and fluent |
| Refusal research | Harmful or out-of-scope input class | Produce a refusal template or suppress hazardous output | Refusal begins from input classification; our target is trajectory-onset detection after generation has begun |
| Activation steering | Behavior direction in hidden states | Add, remove, or conditionally apply a direction | Supplies method, not the target; our pilots show redirection can occur without cessation |
| Cognitive reframing | Negative interpretation and thinking traps | Replace with empathic or constructive interpretation | Replacement continues psychological production; it is an explicit alternative to cessation |
| Thought suppression / emotion regulation | Unwanted mental content | Effortful inhibition, reappraisal, distraction, or learned regulation | The motivating report denies an experienced selector or counter-thought; empirical controls must test this rather than assume it |
| Nondual-awareness research | Reduced subject–object structuring; non-conceptual reflexivity | Phenomenological or behavioral consequences | Motivates the distinction but does not provide an LLM benchmark or establish a transformer mechanism |
| Present work | Active self-maintaining trajectory plus recognition of its function | Rapid cessation or practical pivot, selectively | Tests whether recognition becomes causally effective in token supply |

## Closest papers and the required citation move

### LLM cognition and behavior

- **ToMBench** (Chen et al., ACL 2024): cite as broad representational ToM
  coverage; contrast with trajectory control.
- **EPITOME** (Jones et al., TACL 2024): cite for human-model experimental
  comparison and mixed pragmatic performance.
- **FANToM** (Kim et al., EMNLP 2023): cite for consistency tests that expose an
  illusory capacity despite locally correct answers.
- **Belief Without Behavior** (2026 preprint): likely the closest framing-level
  neighbor; concede that representation-to-action gaps are already studied, then
  distinguish self-trajectory cessation from coordinated social action.
- **Evidence for Limited Metacognition in LLMs** (2025/2026): cite as a strong
  example of avoiding self-report and requiring strategic use of internal
  information. Our analogue is strategic use through cessation.

### Overthinking and stopping

- **Thinking Past the Answer** (Caldarella et al., 2026): cite for prefix-level
  harmful overthinking and the finding that continued generation can destabilize
  a correct trajectory.
- **ASAG** (Li et al., 2026): cite for hidden attention-state adaptive stopping.
- **REFRAIN** (Sun et al., 2025): cite for a stop discriminator and adaptive
  thresholds.
- **Entropy After `</Think>`** (2025): cite for proxy-model stopping and token
  savings. Contrast its epistemic convergence signal with semantic selectivity.

These papers eliminate any broad novelty claim about “models do not know when to
stop.” Our claim must always include the words **psychological trajectory**,
**matched practical controls**, and **recognition-to-cessation**.

### Mechanistic control

- **CAA** (Rimsky et al., ACL 2024): direct precedent for contrastive residual
  directions. Our paired psychological/event/reframe experiment should be
  framed as an application and limitation test.
- **Refusal is mediated by a single direction** (Arditi et al., NeurIPS 2024):
  establishes low-dimensional causal control and the necessity of capability
  controls.
- **CAST** (Lee et al., ICLR 2025): direct precedent for state-conditional
  steering; our EOS gate is structurally similar and should not be presented as
  a new gating architecture.
- **Circuit Breakers / Representation Rerouting** (Zou et al., NeurIPS 2024):
  closest precedent for training model representations away from a hazardous
  trajectory. Our distinct contribution is the target construct and onset-level
  matched controls.
- **ASRU** (2026 preprint): especially close two-stage recipe—steering prototype
  followed by parameter optimization. It may become the best method to reuse on
  the A5000 cluster.

### Phenomenology and contemplative science

- **Varela (1996)**: use to justify mutually constraining first- and third-person
  inquiry, not to claim privileged introspection.
- **Przyrembel & Singer (2018)**: use as evidence that disciplined
  micro-phenomenological interviews can produce reliable, differentiable
  experiential profiles.
- **Josipovic (2019)**: use for the specific contrast between non-conceptual
  nondual reflexivity and conceptual self-representation.
- **Martin (2020)**: cite as the source of the PNS/Location taxonomy, with an
  explicit caveat that it is qualitative, self-report based, and not universally
  accepted.

## Claims reviewers are likely to reject

Avoid:

- “LLMs cannot be intelligent because they use memory.”
- “Awakened authors can judge intelligence better than ordinary researchers.”
- “Location 2 provides direct access to the mechanism.”
- “No previous work studies stopping, action, metacognition, or internal
  control.”
- “Mid-sentence EOS demonstrates choiceless awareness.”
- “Failure on this benchmark proves absence of consciousness.”

Use instead:

- The authors' reports exposed a candidate distinction that ordinary benchmark
  taxonomies may not emphasize.
- The distinction is translated into observable predictions and evaluated by
  independent methods.
- Current models can represent the distinction while failing to make it causal
  in generation.
- Engineered cessation and phenomenological choicelessness are explicitly
  separated.

## Venue fit

### ACL ARR

Best fit for the first paper because the core artifacts are an NLP benchmark,
model comparison, activation analysis, and training intervention. The Location-2
material should occupy a compact positionality/motivation subsection. Main claims
must be computational and behavioral.

Required before submission:

- full item set frozen before confirmatory evaluation;
- multiple open and closed model families;
- blinded annotation with agreement;
- repeated stochastic trials;
- strong lexical/style controls;
- survival and false-stop analyses;
- cluster-scale native training;
- code, data statement, licenses, compute accounting, limitations, and AI-use
  disclosure.

### PNAS

Potential later target if the work becomes a genuinely interdisciplinary human–
AI study with independent Location/PNS recruitment, ethics approval,
micro-phenomenological interviews, preregistered behavioral predictions, and
physiological or neural convergence. The current model-only evidence is not yet
broad or independent enough for the likely significance threshold.

### Possible intermediate venues

- TACL if the benchmark and mechanistic analysis become deeper than a conference
  paper but remain NLP-centered.
- *Consciousness and Cognition* for a human-first phenomenology study.
- PNAS Nexus Registered Report for a later preregistered human–AI comparison,
  if scope and methods are sufficiently mature.

## Recommended paper sentence

> We do not ask whether a model can describe awareness or decide that reasoning
> is complete. We ask whether recognition of a self-maintaining psychological
> trajectory becomes causally effective as selective withdrawal of further
> token support, while matched practical thought continues.

