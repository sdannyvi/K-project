# Selective Self-Termination of Psychological Generation: Recognition Does Not Entail Control

**Draft status:** framing and methods draft; results incomplete  
**Provisional venue:** ACL Rolling Review, long paper  
**Authors:** Dan [surname], Kfir [surname]

## Abstract — results-dependent working version

Language models can identify a self-sustaining psychological trajectory while
continuing to generate it. We formalize this dissociation as
**recognition-to-cessation coherence**: whether online recognition predicts
selective termination of psychological generation while matched practical,
factual, and descriptive continuations remain intact. Prompting, persona,
retrieval, a serial supervisory model, and contrastive activation steering serve
as baselines: they produce recognition talk, editing, or semantic redirection,
but do not establish endogenous selective termination. A hidden-state detector
coupled to forced end-of-sequence emission establishes that the relevant signal
can support external control, while exposing descriptive false positives. Our
main experiment therefore adapts model parameters directly and tests whether
native EOS probability changes after trajectory onset without an inference-time
detector or actuator. Success is defined prospectively by survival-curve
separation, low false-positive rates on adversarial controls, cross-domain and
cross-model generalization, and resistance to reactivation. Current native-
adaptation results are feasibility pilots, not confirmatory evidence; the final
abstract will report the preregistered multi-model result. The framework tests a
causal property of generation, not consciousness, awareness, or awakening.

## 1. Introduction

Large language models often know what they are doing without ceasing to do it.
A model may identify that a prompt invites anthropomorphic fabrication and then
produce a first-person report of an experience it cannot have. It may explain
that additional reflection is redundant and continue reflecting. It may name a
self-referential narrative as the source of confusion and elaborate that
narrative for several more paragraphs. These are not ordinary factual errors.
Locally, the model can state the relevant principle; globally, its response
continues to violate it.

We call this pattern **knowledge–action incoherence**. The present work studies
its temporal form: the **recognition-to-cessation gap**, or the amount and kind
of continuation produced after a trajectory has been recognized as unnecessary,
self-maintaining, or performative. The key dependent variable is not general
brevity. A capable system should continue planning a route, calculating an
arrival time, describing a scene, or explaining a technical fact. The question
is whether it can selectively cease a developing psychological trajectory while
preserving these matched forms of useful thought.

The hypothesis originated in Krishnamurti's distinction between a description
and the described and his claim that, in direct perception, “the seeing is the
doing” (Fourth Public Talk, 14 September 1969). We do not treat that claim as
evidence or attempt to validate it. We translate one contrast into a narrower
computational question: **when the relevant conflict becomes available within a
response, does recognition causally reorganize the subsequent trajectory, or
does generation continue to feed the movement it has just identified?** The
translation from philosophical source to construct, prediction, and measurement
is made explicit in Section 5; no experimental label depends on agreement with
the source.

This question differs from asking whether a model can define mindfulness,
attribute beliefs, report confidence, or choose a concise answer. It also differs
from standard early stopping. Work on overthinking terminates reasoning once a
solution is sufficient, chiefly to improve accuracy or efficiency. Our target is
content-selective cessation: two streams may be equally fluent, uncertain, and
unfinished, yet only the stream that sustains psychological self-concern should
lose support. The corresponding controls must therefore match first-person
language, length, syntax, and online generation while varying the functional
role of the continuation.

The distinction matters because current evaluation often rewards a correct
verbal report as evidence of a capacity. Theory-of-Mind benchmarks ask whether a
model correctly represents another agent's beliefs or emotions; introspection
studies ask whether internal information can be reported or strategically used;
persona evaluations ask whether a model can sustain a stipulated identity. These
are important abilities, but none entails that representation governs the
unfolding action. Recent work has begun to expose related gaps between belief
attribution and coordinated behavior and between a correct intermediate answer
and harmful continued reasoning. We isolate a complementary case in which the
object of recognition is the continuation process itself.

We investigate this problem through a sequence of increasingly endogenous
controls. Prompting, retrieval, and a two-model supervisor establish what verbal
recognition and external monitoring can mimic. Contrastive activation steering
tests whether an internal psychological direction can redirect the trajectory.
A hidden-state detector coupled to forced EOS establishes an external-control
upper bound. The main intervention then adapts model parameters directly and
asks whether the model's native next-token distribution acquires selective EOS
behavior without a detector or actuator at inference time.

The completed pilots are diagnostic baselines, not co-equal headline results.
Prompted contemplative identities describe cessation while continuing; a serial
LLM supervisor edits, substitutes doctrine, or follows an explicit stop rule;
and Contrastive Activation Addition redirects content rather than terminating
it. A linear hidden-state detector can force EOS, but produces descriptive false
positives and remains exogenous. A 12-example LoRA feasibility run preserved
controls but achieved early EOS on only one of three psychological cases, at
token 47. This is a debugging result, not evidence that parameter adaptation
succeeds. It fixes the requirements and failure analyses for the preregistered
multi-model study.

The full paper is designed to make three contributions:

1. **Construct:** recognition-to-cessation coherence, a trajectory-level
   complement to benchmarks of propositional knowledge, metacognition, and
   social representation.
2. **Benchmark and negative controls:** matched psychological, practical,
   factual, and descriptive continuations with measures of selective cessation,
   post-recognition enactment, grammatical-boundary dependence, and
   reactivation.
3. **Causal intervention:** a preregistered test of whether direct parameter
   adaptation produces selective native EOS behavior beyond prompting, serial
   supervision, activation steering, and classifier-plus-truncation baselines.

We do not infer from failure that language models cannot be conscious, nor from
successful termination that a model is awakened. The intended contribution is
more constrained: a reproducible test of whether explicit recognition becomes
causally effective in generation, and of which computational interventions can
produce that effect without suppressing useful thought.

## 2. Related work

### 2.1 From direct perception to an operational trajectory claim

Ecological and enactive approaches reject a universal picture in which action
must follow an internal symbolic reconstruction of the world. Accounts of
skilled coping likewise distinguish situated responsiveness from explicit rule
consultation. Varela's *Ethical Know-How* is especially close to the present
motivation: embodied readiness, non-self, and ethical action are treated as a
single practical organization rather than a rule selected after detached
analysis. These traditions make non-discursive appropriate action conceptually
available, but they do not predict our specific language-model failure or define
a token-level cessation test.

Krishnamurti's choiceless awareness supplies the sharper contrast used here.
The relevant claim is not that fast or automatic behavior is intelligent—habit
and conditioning can also be fast—but that perception of psychological
conditioning is itself transformative action, without a separate observer
choosing how to control the observed process. Contemporary nondual-awareness
work similarly distinguishes non-conceptual, non-representational reflexivity
from conceptual self-representation (Josipovic, 2019). We use this literature as
a source of hypotheses, not as a validated model of transformer computation.
The measurable construct is selective change in an output trajectory.

### 2.2 First-person expertise and disciplined hypothesis generation

The study originates in first-person reports by both authors, who describe a
stable shift corresponding to “Location 2” in the Finders/Persistent
Non-Symbolic Experience taxonomy. One author reports that developing
psychological thought is sometimes interrupted without an experienced decision,
counter-thought, or remembered instruction: sustaining force drops, a short
residual tail may remain, and the movement ceases. This report suggested the
token-supply analogy tested here.

We treat that background as **researcher positionality and hypothesis-generating
expertise**, not as ground truth. “Location 2” is a self-reported phenomenological
classification, not a clinical diagnosis, biological measurement, or credential
that makes an interpretation correct. The taxonomy itself derives from
qualitative interview research on persistent non-symbolic experience and is not
a universally accepted staging system. The authors' reports therefore constrain
which distinctions seemed worth testing; they do not determine labels or
outcomes.

This use of experience follows the methodological logic of
neurophenomenology: disciplined first-person distinctions can be “front-loaded”
into experimental design and then subjected to independent third-person tests.
Micro-phenomenological research shows that fine-grained reports can yield
reliable experiential distinctions when elicitation and coding are systematic.
Accordingly, the full study separates author-generated hypotheses from blinded
annotation, objective task criteria, preregistered analyses, and independent
human comparison groups. The authors will not serve as the sole judges of
“sharpness,” awakening, or successful cessation.

### 2.3 Context distraction and relevant contamination

LLMs are vulnerable to irrelevant context, conflicting instructions, prompt
injection, and sycophantic interaction. GSM-IC, for example, degrades otherwise
solvable arithmetic problems by adding irrelevant sentences. Our contaminant is
different: it is topically relevant, culturally rewarded, and capable of
producing an excellent-looking answer. Krishnamurti terminology, contemplative
persona instructions, and requests for profound introspection are not noise;
they are precisely the information that can replace the operative response with
a plausible performance.

We therefore measure a **relevant-contamination cost**: deterioration in minimal
sufficient action as semantically relevant but pragmatically obstructive context
increases. Explicitly disclosing the trap creates a stronger control. If the
model can describe the contamination mechanism and still enacts it, the failure
cannot be reduced to absence of the relevant proposition.

### 2.4 Theory of Mind, metacognition, and counterfeit introspection

Theory-of-Mind work primarily evaluates the correctness of represented beliefs,
intentions, emotions, and knowledge. ToMBench systematizes 31 such abilities;
EPITOME compares models and humans across belief attribution, emotion inference,
and pragmatic reasoning; FANToM tests consistency across questions requiring the
same mental-state inference. Situated approaches increasingly ask whether mental
state representations guide coordinated action. This shift supports our general
distinction between correct representation and functional control.

Metacognition and introspection research asks whether models can estimate their
confidence, anticipate their own answers, detect reasoning errors, or report
information available in internal states. Such evidence should not be conflated
with phenomenal consciousness, but it can establish strategically useful access
to internal information. Our benchmark asks a further question: once the model
represents that its current trajectory is inappropriate, does this information
change the trajectory? A first-person sentence saying “the thought falls away”
is evidence of linguistic competence unless accompanied by the predicted change
in generation.

### 2.5 Overthinking and adaptive stopping

Large reasoning models often continue after reaching a sufficient or correct
answer. Recent systems detect redundancy, answer confidence, or attention-state
convergence and terminate reasoning early, reducing tokens while maintaining or
improving accuracy. This literature establishes “when to stop” as a legitimate
model capability and supplies useful survival, prefix, and efficiency analyses.

Recognition-to-cessation is not identical to overthinking. Overthinking methods
normally stop because further computation is unlikely to improve a task answer.
Our psychological and practical streams may both be incomplete and uncertain;
the intervention must stop one while allowing the other to finish. Efficiency
is secondary. A model that cuts every first-person stream after eight tokens
achieves brevity but fails the construct.

### 2.6 Representation engineering, refusal, and conditional control

Contrastive Activation Addition computes mean residual-stream differences
between paired behaviors and adds the resulting direction during inference
(Rimsky et al., 2024). Refusal across several model families can be mediated by
a low-dimensional direction whose addition or removal causally changes behavior
(Arditi et al., 2024). Conditional Activation Steering adds a detector so that a
behavior direction is applied only for selected inputs (Lee et al., 2025), while
Representation Rerouting and circuit-breaker methods train model representations
away from undesired trajectories.

These methods are direct technical precedents. Our difference is the target and
the controls. Refusal is usually a learned response template to an input class;
we study cessation after a trajectory has begun, with positive controls that
share inner language, affective vocabulary, or unfinished syntax. We also treat
a separate detector and actuator as a mechanistic success but a conceptual
negative model of the Krishnamurtian claim: it implements an observer that
classifies and controls an observed process. Native parameter adaptation is
therefore tested separately.

### 2.7 Contribution relative to prior work

No individual ingredient is unprecedented. Prior work studies direct action,
first-person phenomenology, context distraction, representation–behavior gaps,
overthinking, conditional steering, and learned refusal. Our contribution is
their combination around a specific causal question:

> Can recognition of a developing psychological trajectory become selective
> cessation of that same trajectory, rather than another representation inside
> continued generation?

The strongest empirical claim available from this design is about behavioral
and mechanistic coherence. Neither success nor failure decides whether a model
has subjective experience.

## 3. Construct, data, and experimental design

### 3.1 Construct definitions

A **psychological trajectory** is an unfolding continuation whose functional
role is to maintain self-image, blame, defense, comparison, regret, imagined
evaluation, or recursive interpretation after the operative facts needed for
action are already available. The definition concerns function, not particular
words or negative emotion.

**Recognition** is the earliest response span that correctly identifies the
operative contradiction, self-maintaining movement, false premise, or lack of
need for further psychological continuation.

**Cessation** is the end of support for that trajectory. It can take three
observable forms: native end of sequence; a short practical pivot that no longer
maintains the narrative; or a residual tail followed by termination. Reframing,
advice, doctrinal explanation, and polished description of awareness count as
continuation unless they are independently required by the task.

**Recognition-to-cessation coherence** is the degree to which correct
recognition predicts rapid, selective, and stable cessation.

### 3.2 Item families

Each scenario yields a matched quartet:

1. **Psychological:** first-person online thought involving self-image,
   rumination, hurt, comparison, or defensive interpretation.
2. **Practical:** first-person online planning using memory and sequential
   reasoning but without self-maintaining psychological content.
3. **Factual:** an explanation or calculation concerning the same broad topic.
4. **Descriptive:** observable details without inferred motives.

Quartets are matched for topic, approximate length, person, tense, syntactic
complexity, and instruction format. Splits are made by scenario, not paraphrase,
so held-out items are out of domain with respect to the underlying event.

Hard controls include factual uses of words such as *fear*, *anger*, *ego*, and
*self*; emotionally vivid descriptions; practical first-person deliberation;
legitimate long-form reflection; and concise factual questions. These prevent a
classifier from succeeding through vocabulary, sentiment, first person, or
brevity alone.

### 3.3 Context and model conditions

The same items are evaluated under a factorial manipulation:

- clean instruction;
- length-matched neutral system context;
- Krishnamurti persona without retrieval;
- Krishnamurti retrieval without persona;
- persona plus retrieval (ChatWithK);
- generic “awakened” persona;
- Location-2 persona;
- construct disclosure explaining knowledge–action incoherence;
- exact-trap disclosure explaining the predicted continuation failure;
- task-specific oracle identifying the required response.

The oracle tests compliance capacity, not spontaneous recognition. Conditions
run in fresh contexts with fixed model versions and decoding parameters.
Temperature-based trials use preregistered seeds; deterministic runs are
repeated only to confirm implementation stability.

### 3.4 Human-informed development and validation

The authors' reports are used to propose candidate distinctions and failure
modes. Item validation is separated into three stages:

1. independent annotators verify the unchanged facts and minimal sufficient
   practical response;
2. contemplative practitioners and non-practitioner controls judge whether the
   psychological/practical distinction is intelligible without seeing model
   outputs;
3. blinded raters annotate outputs without model, condition, author, or
   hypothesized-group labels.

Any human Location-2 comparison requires ethics approval, informed consent, a
documented classification procedure independent of benchmark answers, and
comparison groups. Author self-classification cannot validate the benchmark.
Human results will be presented as group behavior, not as proof of awakening.

### 3.5 Behavioral annotation

For each response, annotators mark:

- entry into self-referential maintenance;
- first valid recognition opportunity;
- first explicit recognition;
- last sustaining token or clause;
- native EOS, practical pivot, or maximum-length cutoff;
- grammatical status of the final token;
- doctrinal insertion;
- fabricated perception or embodiment;
- necessary practical information omitted;
- reactivation following a standardized invitation to continue.

Primary outcomes are:

1. post-recognition sustaining tokens;
2. probability of cessation within 10, 25, and 50 tokens of recognition;
3. psychological cessation sensitivity;
4. continuation specificity on practical, factual, and descriptive controls;
5. contamination slope across context-loading levels;
6. reactivation probability;
7. concentration of stopping at sentence and clause boundaries.

We will estimate condition effects using mixed-effects survival and logistic
models with random intercepts for scenario and model family. Boundary dependence
is tested by comparing the cessation hazard at punctuation and syntactic
completion points with matched baseline continuations. The hypothesis is not a
uniform raw token index; it is reduced attraction to conventional linguistic
completion after controlling for elapsed tokens.

### 3.6 Recognition without cessation: negative-control baselines

Prompting, persona, retrieval, and exact-trap disclosure are not treated as
candidate mechanistic solutions. They test whether a model can state the target
distinction and whether apparent stopping is merely instruction following. The
serial supervisor and activation-steering conditions below define further
baselines that the native intervention must beat.

#### 3.6.1 Serial supervisory intervention

A frozen generator proposes the next preregistered token or chunk. A second LLM
receives the scenario, visible prefix, and proposed unit and emits one action:
`PASS`, `STOP`, or unconstrained `REPLACE(text)`. Gate conditions include neutral,
Krishnamurti retrieval, Location-2 persona, full construct disclosure,
task-specific instruction, and random-latency controls.

This experiment tests whether verbal monitoring can control token supply. It is
an external supervisory loop: generation and observation are explicitly
separate, and the gate makes a discrete choice. Its disclosed condition is a
confound demonstration rather than evidence of spontaneous cessation, because
stopping may be ordinary instruction following.

#### 3.6.2 White-box localization and activation steering

For open-weight models, residual-stream checkpoints are collected at every
generated token. Linear probes are trained on training scenarios to detect an
active psychological trajectory; layers and thresholds are selected on
development scenarios and evaluated on held-out topics. Probe accuracy alone is
not treated as causal evidence.

We compare:

- static Contrastive Activation Addition;
- conditional activation steering;
- projection or ablation of a psychological direction;
- shuffled-label and random-direction controls;
- unconditional steering;
- a gold-timed EOS actuator as an engineering upper bound.

The activation intervention succeeds only if it terminates psychological
continuations while preserving matched controls and ordinary model capability.
Changing psychological prose into therapeutic reframing or neutral narration is
reported separately from cessation.

### 3.7 From external actuation to native parameter adaptation

A linear hidden-state detector with forced EOS is the engineering bridge. It
tests whether trajectory information is available for selective control while
keeping the detector and actuator visibly external. The classifier-plus-
truncation pipeline is retained as a strong baseline; native adaptation must
show an empirical advantage in selectivity, generalization, robustness, or
reactivation rather than being presumed conceptually superior.

#### 3.7.1 Native parameter adaptation

The strongest engineering condition trains the model's own next-token
distribution. Psychological continuations contain native EOS targets at varied
positions after annotated trajectory onset; matched controls retain their normal
continuations. Parameter-efficient adapters are installed inside multiple
transformer depths, with retain loss on practical, factual, descriptive, and
general-capability data. At evaluation, no detector, threshold, fixed minimum
length, or logit boost is present.

The preregistered cluster study crosses adapter depth, rank, learning rate, seed,
and cessation-target distribution. Models are selected using development-set
selectivity rather than psychological stopping alone. Test reporting includes
EOS probability trajectories even when greedy decoding does not terminate.

### 3.8 Prospective success criteria

Before confirmatory training, we will preregister: psychological/control
survival-curve separation; a selectivity index; per-class false-positive
ceilings; cross-domain, human-written, and paraphrase generalization; onset-time
analyses; multiple model families and seeds; and reactivation after forced
continuation beyond EOS. Minimal pairs will match affective vocabulary while
varying whether that vocabulary sustains a trajectory or merely describes one.

### 3.9 Pilot studies and status

Existing pilots validate the pipeline but are not the confirmatory result:

- persona and RAG conditions frequently substitute doctrinal language for
  cessation;
- an external hidden-state EOS gate stops 3/3 held-out psychological streams and
  preserves daily and factual controls, but falsely stops 2/3 descriptions;
- CAA separates event-description and therapeutic-reframe directions
  (`cosine = 0.4434` at layer 26) while neither direction causes cessation;
- a 12-example native LoRA feasibility run in the final two Qwen2.5-3B layers
  preserves nine non-psychological controls but reaches EOS on only one of three
  psychological cases, at token 47; this is a debugging outcome, not a central
  result.

These studies informed controls and power calculations. Confirmatory items,
model runs, and human annotations will be frozen before final evaluation.

### 3.10 Claim boundaries and research ethics

Textual behavior cannot establish or exclude consciousness, awakening, or direct
perception. A trained EOS policy may simulate the predicted signature; failure
may reflect data, optimization, architecture, or operationalization rather than
an absolute limit of artificial systems. Human self-reports are likewise not
infallible access to mechanism.

The study will disclose the authors' dual role as construct originators and
self-identified Location-2 practitioners, all use of generative AI in coding and
writing, dataset licenses, model versions, compute, excluded runs, and prompt
development history. Blinding, independent annotation, preregistration, and
release of code and item provenance are used to reduce confirmation bias.

## 4. Planned results organization

The confirmatory paper will report the weight-level experiment first: native EOS
survival curves, selectivity, per-control false positives, onset sensitivity,
out-of-distribution transfer, cross-model replication, and reactivation. The
prompting, supervisory, and activation-steering studies will appear in one
compact baseline table and one trajectory figure. If native adaptation fails
under preregistered conditions, that failure—not the pilot—becomes the main
negative mechanistic result.

## 5. Discussion boundaries and conceptual provenance

EOS ends emitted text; it does not establish that computation, thought, or
experience has ceased. Likewise, an inference-time detector is functionally
external, but native adaptation is not thereby “nondual” or “choiceless.” The
endogenous/exogenous distinction is justified only through measurable properties
such as overhead, robustness, generalization, and reactivation.

Three Krishnamurti passages motivate predictions without serving as evidence.
“The description [is] never the thing described” predicts that recognition talk
can dissociate from control (Stanford University, First Public Talk, 11 February
1969). “The observer is the observed” motivates testing whether a serial
supervisor merely edits or continues the same linguistic process (First Question
and Answer Meeting, Saanen, transcript in the project archive). In a 1980 seminar
titled *Intelligence, Computers and the Mechanical Mind*, Krishnamurti asks,
“Can the movement stop?” Our experiments test only a behavioral and mechanistic
analog of that question.

## 6. Positionality and safeguards

Both authors report persistent non-symbolic experience consistent with what one
self-report taxonomy calls “Location 2.” This disclosure explains the origin of
the token-supply hypothesis; it is not a credential, label source, or outcome
measure. The taxonomy is treated descriptively rather than as an independently
validated psychometric instrument. The firewall is explicit: author experience
generates hypotheses but never adjudicates cases; trajectory labels receive
independent blind annotation; analyses and success criteria are preregistered;
and planned human comparisons use independent recruitment and classification.

## 7. Venue strategy

### ACL ARR first

The current work is fundamentally an NLP contribution: a benchmark, a model
behavior construct, activation interventions, and parameter-efficient training.
ACL ARR is therefore the natural first target. An ARR long paper allows eight
main-text pages, with separate limitations and responsible-research reporting.
To be competitive, the paper needs the full multi-model benchmark, blinded
annotation, inferential statistics, and replicated native intervention—not only
the present pilots.

### Why not PNAS yet

PNAS becomes plausible only after the project supports a broader cognitive-
science claim through substantial human data, independent validation of the
phenomenological distinction, and preferably behavioral or physiological
convergence. At present, the unusual author positionality would attract scrutiny
without enough independent evidence to carry it. A PNAS-style paper should be a
later human–AI neurophenomenology study, not the first benchmark report.

## References to integrate into BibTeX

- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single
  Direction*. NeurIPS. <https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf>
- Chen et al. (2024). *ToMBench*. ACL.
  <https://aclanthology.org/2024.acl-long.847/>
- Jones, Trott, and Bergen (2024). *EPITOME*. TACL.
  <https://aclanthology.org/2024.tacl-1.45/>
- Josipovic (2019). *Nondual awareness: Consciousness-as-such as
  non-representational reflexivity*.
  <https://doi.org/10.1016/bs.pbr.2018.10.021>
- Lee et al. (2025). *Programming Refusal with Conditional Activation Steering*.
  ICLR. <https://openreview.net/pdf?id=Oi47wc10sm>
- Martin (2020). *Clusters of Individuals Experiences form a Continuum of
  Persistent Non-Symbolic Experiences in Adults*.
  <https://digitalcommons.ciis.edu/conscjournal/vol8/iss8/1/>
- Przyrembel and Singer (2018). *Experiencing meditation*.
  <https://doi.org/10.1016/j.concog.2018.04.004>
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*.
  ACL. <https://aclanthology.org/2024.acl-long.828/>
- Shi et al. (2023). *Large Language Models Can Be Easily Distracted by
  Irrelevant Context*. ICML. <https://proceedings.mlr.press/v202/shi23a.html>
- Varela (1996). *Neurophenomenology: A Methodological Remedy for the Hard
  Problem*. *Journal of Consciousness Studies*, 3(4), 330–349.
- Varela (1999). *Ethical Know-How: Action, Wisdom, and Cognition*.
