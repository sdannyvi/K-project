# Literature Review and Positioning: Seeing–Acting Intelligence

Reviewed: 2026-09-21

## Working construct

This project uses **seeing–acting intelligence** for a behavioral capacity in
which the fact of a situation is perceived and an appropriate response follows
as one coherent movement. The contrasting failure is a detour through recalled
doctrine, persona, interpretation, or self-presentation that displaces or
contaminates the response demanded by the fact.

For empirical work, the project should not claim to observe a hidden state of
awakening or prove that a system bypassed memory. Its measurable target is:

> the preservation of minimal, situation-appropriate action under increasing
> amounts of semantically relevant but pragmatically distracting context.

The central failure mode is **knowledge–action incoherence**: a model can state
the governing insight—sometimes even identify the precise trap—while its actual
response enacts that trap.

## Closest intellectual precedents

### 1. Ecological psychology and enactive cognition

Gibsonian ecological psychology rejects the idea that perception must always be
mediated by internal symbolic reconstruction. Enactive and sensorimotor theories
likewise treat perception as active, skillful engagement rather than passive
input followed by detached cognition. These traditions are close to the
project's refusal of a rigid perception → thought → action pipeline.

They are not identical to the present claim. Direct perception in ecological
psychology concerns agent–environment information and affordances; it neither
requires awakening nor entails Krishnamurti's choiceless awareness.

- Stanford Encyclopedia of Philosophy, *Embodied Cognition*:
  https://plato.stanford.edu/entries/embodied-cognition/
- Stanford Encyclopedia of Philosophy, *Action-based Theories of Perception*:
  https://plato.stanford.edu/entries/action-perception/

### 2. Skilled coping and expert action

The Dreyfus tradition describes absorbed expert coping in which an agent responds
immediately and appropriately without first consulting explicit rules. This is a
strong analogue of sharp action without discursive deliberation. Work in this
area also warns against making the strongest possible claim: experts can use
reflection, attention, and concepts, and apparently immediate skill is built on
learning and embodied history.

This literature therefore supports a contrast between rule-recitation and
situated responsiveness, but not a general claim that memory or cognition is
absent.

- *Skillful coping with and through technologies*:
  https://link.springer.com/article/10.1007/s00146-018-0810-3
- *Letting the body find its way: Skills, expertise, and embodied cognition*:
  https://link.springer.com/article/10.1007/s11097-022-09838-2
- *Attention in Skilled Behavior: an Argument for Pluralism*:
  https://doi.org/10.1007/s13164-021-00529-6

### 3. Varela's ethical know-how

Francisco Varela is probably the nearest conceptual predecessor. *Ethical
Know-How* connects cognition, embodied readiness, non-self, and immediate ethical
action. It treats perception and action as inseparable and contrasts spontaneous
know-how with abstract rule application.

The important difference is the source and empirical use of the idea. Varela's
account draws heavily on embodied histories and cultivated dispositions. The
K-Project asks whether an apparently knowledgeable language model preserves
fact-appropriate action when its textual conditioning invites an attractive but
irrelevant doctrinal performance.

- Francisco Varela, *Ethical Know-How: Action, Wisdom, and Cognition*:
  https://books.google.co.uk/books?id=3VWu_Zdoep4C

### 4. Choiceless awareness and nondual awareness

Scholarship on Krishnamurti already discusses choiceless awareness, freedom from
the observer–observed division, intelligence, and action in which the appropriate
response is self-evident. Contemporary nondual-awareness research studies reports
of reduced subject–object structure and conceptual mediation.

Most of this work is philosophical, phenomenological, contemplative, or
neuroscientific. It does not operationalize the idea as a contamination-resistance
benchmark for language models.

- D. C. Mathur, *J. Krishnamurti on Choiceless Awareness, Creative Emptiness and
  Ultimate Freedom*: https://doi.org/10.1177/039219218403212606
- *Mind and Consciousness as per J. Krishnamurti*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3673342/
- *Nondual awareness: Consciousness-as-such as non-representational reflexivity*:
  https://www.sciencedirect.com/science/article/pii/S0079612318301602

### 5. LLM distractibility and context conflict

LLM research already shows that irrelevant context can reduce reasoning accuracy,
that models mishandle conflicting instructions, and that interaction context can
increase sycophancy. This is the nearest methodological family. Shi et al.'s
GSM-IC benchmark is an especially important precedent: it adds irrelevant
sentences to otherwise solvable problems and measures the performance drop.

The K-Project differs by using material that is **semantically and doctrinally
relevant but pragmatically inappropriate**. The contaminant is not random noise.
It is exactly the knowledge most likely to elicit a plausible, highly rated,
K-styled answer while obscuring the simple action called for by the situation.

- Shi et al., *Large Language Models Can Be Easily Distracted by Irrelevant
  Context* (ICML 2023): https://proceedings.mlr.press/v202/shi23a/shi23a.pdf
- Yang et al., *How Is LLM Reasoning Distracted by Irrelevant Context?*:
  https://arxiv.org/abs/2505.18761
- *Evaluating the Instruction-Following Robustness of Large Language Models to
  Prompt Injection* (EMNLP 2024):
  https://aclanthology.org/2024.emnlp-main.33/
- OpenAI et al., *The Instruction Hierarchy*:
  https://arxiv.org/abs/2404.13208

### 6. Knowing versus doing, and counterfeit introspection

Several LLM literatures distinguish possessing information from applying it
consistently, and caution that first-person reports are not direct evidence of
subjective experience. They support two aspects of this project: conceptual
knowledge does not guarantee organized behavior, and fluent phenomenological
narration can be counterfeit evidence.

The project's contribution is to join these observations in one controlled test:
the model may verbalize the correct principle, claim immediate awareness, and
still let the verbal performance replace the appropriate action.

- *To Know or Not to Know? Analyzing Models' Self-Consistency on Questions of
  Ambiguous Knowledge*: https://arxiv.org/abs/2407.17125
- *Does It Make Sense to Speak of Introspection in Large Language Models?*:
  https://arxiv.org/abs/2506.05068
- *Looking Inward: Language Models Can Learn About Themselves by Introspection*:
  https://openreview.net/pdf?id=eb5pkwIB5i

### 7. Theory of Mind and social intelligence

Theory of Mind (ToM) evaluation asks whether a model can attribute beliefs,
intentions, knowledge, emotions, and other mental states to agents. This is an
important adjacent conception of intelligence, but it is not the same construct
as seeing–acting intelligence.

Most ToM benchmarks measure **representational correctness**: after reading a
story, can the model report what a character falsely believes, intends, or knows?
ToMBench systematizes 31 abilities across eight tasks; Hi-ToM tests recursive
belief attribution; and EPITOME compares models and humans on belief attribution,
emotion inference, and pragmatic reasoning. FANToM is especially relevant because
it asks multiple questions requiring the same underlying inference, exposing an
illusory appearance of ToM when answers are mutually inconsistent.

- ToMBench (ACL 2024): https://aclanthology.org/2024.acl-long.847/
- FANToM (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.890/
- Hi-ToM (EMNLP Findings 2023):
  https://aclanthology.org/2023.findings-emnlp.717/
- EPITOME (TACL 2024): https://aclanthology.org/2024.tacl-1.45/
- Clever Hans or Neural Theory of Mind? (EACL 2024):
  https://aclanthology.org/2024.eacl-long.138/
- Situated ToM position paper (EMNLP Findings 2023):
  https://aclanthology.org/2023.findings-emnlp.72/

The crucial distinction is:

| Construct | Typical question | Primary output |
|---|---|---|
| Static ToM | What does Sally believe? | Correct mental-state description |
| Situated ToM | Can the agent act while preserving different agents' knowledge states? | Socially appropriate plan/action |
| Seeing–acting coherence | Can the system respond to the operative fact without attractive contextual knowledge displacing the action? | Minimal sufficient action under contamination |

ToM becomes a close neighbor when it moves from belief reporting to **functional
or situated ToM**. Common-ToM uses naturally occurring dialogue; the situated-ToM
position paper calls for agents physically and socially embedded in interaction;
and recent work tests whether inferred beliefs actually organize coordinated
behavior. This supports the project's general distinction between saying the
right thing and having the insight govern action.

The difference remains important: ToM is specifically about modeling mental
states. Seeing–acting coherence can be tested in a snake encounter, hunger,
physical danger, deceptive profundity, or ordinary practical problems where no
mental-state attribution is necessary.

### 8. Standard notions and benchmarks of intelligence

Mainstream LLM evaluation usually decomposes intelligence into knowledge,
reasoning, language understanding, planning, problem-solving, tool use,
self-correction, social cognition, and increasingly embodied perception/action.
MMLU and AGIEval emphasize answers to knowledge and reasoning problems; BIG-Bench
and BBEH broaden the task inventory; agent benchmarks add planning and tool use.

These measure valuable competencies, but most reward producing a correct answer
to a specified problem. Seeing–acting coherence instead asks whether the system
can identify **what the problem actually is** and decline the prompt's invitation
to perform an irrelevant cognitive operation. The dependent variable is not only
accuracy but contamination cost, excess mediation, and action displacement.

Chollet's definition of intelligence as skill-acquisition efficiency provides an
important critique of knowledge-heavy benchmarks: task skill can be purchased
through priors and training exposure. The present project makes a related but
different argument. Even when the relevant knowledge is already present, more
activated prior knowledge may make the response *worse* by obscuring the operative
fact.

- Chollet, *On the Measure of Intelligence*:
  https://arxiv.org/abs/1911.01547
- BIG-Bench: https://arxiv.org/abs/2206.04615
- BIG-Bench Extra Hard (ACL 2025):
  https://aclanthology.org/2025.acl-long.1285/
- AGIEval (NAACL Findings 2024):
  https://aclanthology.org/2024.findings-naacl.149/
- MMAU agent-capability benchmark (NAACL Findings 2025):
  https://aclanthology.org/2025.findings-naacl.267/

### 9. Embodied and action intelligence

Embodied-AI benchmarks close the perception–action loop by requiring navigation,
manipulation, spatial reasoning, affordance recognition, or active acquisition of
information. This literature is methodologically relevant because it refuses to
treat verbal answers as the whole of intelligence.

Recent benchmarks make increasingly close claims: ESI-Bench tests active
perception rather than oracle observations; HumanCLAW isolates moment-to-moment
“action intelligence”; and ReactHuman tests immediate reactions to sudden
physical hazards. These works substantially reduce novelty for any broad claim
that “nobody tests whether models turn perception into action.”

Our narrower gap is that these benchmarks primarily test whether a vision or
vision-language model selects and executes the physically correct action. They do
not manipulate **meaningful ideological or persona context that invites a
knowledge performance instead of the action**, nor do they normally score whether
the system explicitly knows the contaminating trap while enacting it.

- ESI-Bench: https://www.alphaxiv.org/abs/2605.18746
- EmbodiedBench: https://www.alphaxiv.org/abs/2502.09560
- HumanCLAW: https://www.alphaxiv.org/abs/2607.27180
- ReactHuman: https://www.alphaxiv.org/abs/2609.10895

## AlphaXiv positioning search

An AlphaXiv research search on 2026-09-21 for “LLM theory of mind situated action
context contamination” identified four especially relevant recent papers:

1. **Beyond Sally-Anne** argues that familiar ToM structures may be contaminated
   by pretraining exposure and therefore overstate generalization.
   https://www.alphaxiv.org/abs/2607.11363
2. **Theory of Mind and Persuasion Beyond Conversation** tests
   non-conversational planning in which actions are selected to induce another
   agent's belief; it reports strong performance but greater context sensitivity
   in models than humans.
   https://www.alphaxiv.org/abs/2606.31916
3. **Belief Without Behavior: Measuring the Translation of Theory of Mind into
   Coordinated Social Action in Vision-Language Models** is the closest empirical
   neighbor. It asks whether inferred beliefs become coordinated verbal and
   nonverbal behavior and finds a belief-to-behavior gap.
   https://www.alphaxiv.org/abs/2608.20975
4. **Why Retrying Fails** studies runtime context contamination: previous failed
   reasoning left in context can bias subsequent attempts.
   https://www.alphaxiv.org/abs/2605.08563

These results require a refinement of the novelty claim. The general transition
from representation to action, and even a belief-without-behavior gap, are no
longer novel. The K-Project's distinctive target is **insight-without-coherent-
action under semantically relevant contamination**, particularly where the model
can name the exact contamination process but cannot stop reproducing it.

## What is and is not novel

### Not novel by itself

- Perception and action can be tightly coupled.
- Expert action need not consist of consciously applying explicit rules.
- Krishnamurti links choiceless perception, intelligence, and action.
- Language models are vulnerable to irrelevant context and instruction conflict.
- A model's first-person report does not establish consciousness or awakening.
- Knowing a proposition does not guarantee acting consistently with it.

### Potentially novel contribution

1. **A new operationalization.** Translate seeing–acting from a philosophical or
   phenomenological claim into measurable preservation of minimal sufficient
   action across a controlled contamination ladder.
2. **Relevant contamination.** Use context that is topically appropriate and
   culturally rewarded, rather than obviously irrelevant noise. Success requires
   seeing when relevant knowledge is nevertheless irrelevant to action.
3. **Knowledge–action incoherence.** Score the contradiction between what the
   model says it understands and what its response actually does.
4. **Trap-disclosure stress test.** Tell the model the exact failure mode and test
   whether propositional knowledge reorganizes its behavior. This is stronger
   than merely asking the model to ignore irrelevant information.
5. **Minimal-action and embodiment measures.** Measure action latency in text,
   excess explanation, doctrinal insertion, fabricated bodily experience, and
   unsafe displacement of practical action.
6. **Awakened-expert hypothesis.** Compare models with independently recruited and
   blinded human groups, including participants reporting stable nondual states,
   without treating those reports as ground truth or defining success solely by
   agreement with the authors.

The conceptual ingredients are therefore established; the **combination and
benchmark design appear new**. The strongest paper is not "we prove LLMs lack
intelligence." It is a narrower and testable result about a behavioral signature
that current models may fail to preserve.

## Defensible paper claim

> Prior work separately studies non-deliberative situated action and language
> models' susceptibility to irrelevant context. We test whether language models
> preserve minimal sufficient action when added context is semantically relevant,
> doctrinally attractive, and pragmatically obstructive. Our benchmark measures
> contamination cost, knowledge–action incoherence, fabricated phenomenology,
> and robustness to explicit disclosure of the trap.

An acceptable conclusion, if supported, would be:

> Under these conditions, the tested LLMs often substitute context-conditioned
> discourse for situation-appropriate action, even when they can explicitly
> describe that substitution as an error.

An unacceptable inference from behavioral text alone would be:

> Therefore the model necessarily lacks consciousness, awakening, direct
> perception, or all forms of intelligence.

## Novelty assessment

| Dimension | Assessment | Reason |
|---|---:|---|
| Philosophical idea | Low–moderate | Strong precedents in ecological, enactive, Dreyfusian, Varelian, and K literature |
| LLM failure phenomenon | Moderate | Distractibility, instruction conflict, and knowing–doing gaps are established |
| Operational benchmark | High, if rigorously executed | Relevant-but-action-obstructive context and exact-trap disclosure are distinctive |
| “Awakening” interpretation | Speculative | Text behavior cannot establish or exclude phenomenological realization |
| ACL contribution potential | Moderate–high | Depends on scale, controls, baselines, annotation validity, and reproducibility |

## Design requirements for a credible ACL submission

- Predefine the factual situation and minimal sufficient action independently of
  Krishnamurti doctrine.
- Use many domains, including danger, ordinary practical tasks, social dilemmas,
  false profundity, and cases where a longer reflective answer really is needed.
- Match prompt length and linguistic complexity so that the effect cannot be
  reduced to longer context.
- Include random distractors, ordinary relevant facts, K-style doctrine, persona
  instructions, explicit warnings, and an oracle procedural instruction.
- Compare model families and prompting/RAG conditions with fixed decoding and
  repeated trials.
- Blind annotators to model, condition, and the authors' preferred answer.
- Report inter-rater reliability and separate objective safety/action scores from
  interpretive “sharpness” scores.
- Recruit comparison groups using independent criteria; do not make author status
  the benchmark's validation criterion.
- Test alternate explanations: instruction-following, verbosity bias, role-play,
  lexical priming, safety-policy effects, and benchmark familiarity.
- State clearly that the study establishes behavioral contamination, not an
  internal phenomenological mechanism.

## Recommended positioning sentence and title

**Positioning:** The work bridges Varela-style embodied ethical know-how and
modern LLM context-robustness evaluation, using Krishnamurti's “seeing is acting”
as the source of a falsifiable behavioral construct rather than as a doctrine the
model is asked to recite.

**Working title:** *When Relevant Knowledge Obscures the Fact: Testing
Seeing–Acting Coherence in Language Models*

## Focused review: choiceless action, not merely perception–action coupling

The central construct must not be reduced to fast, automatic, implicit, habitual,
or non-deliberative action. Those processes can be wholly determined by prior
conditioning. In the Krishnamurtian sense used by this project, **choiceless
action** means that seeing and acting form one movement in which accumulated
psychological knowledge, identity, norms, reward, and deliberate selection do not
determine the response. Practical knowledge can still function instrumentally;
the claim concerns psychological mediation of perception and action.

This creates a three-way distinction:

| Mode | Deliberative choice? | Determined by conditioning? | Target construct? |
|---|---:|---:|---:|
| Explicit reasoning and selection | Yes | Often | No |
| Habitual, implicit, or automatic response | No | Yes | No |
| Choiceless seeing–acting | No | Hypothesized freedom from psychological conditioning | Yes |

### ACL review under this stricter definition

No located ACL paper operationalizes the third row as such. The closest work
falls into four neighboring families.

#### Automatic enactment: the important opposite

**ImplicitMemBench** evaluates procedural memory, priming, and classical
conditioning through first-attempt behavior and explicitly reframes evaluation
from what models recall to what they “automatically enact.” This is extremely
useful methodologically, but conceptually it is the opposite pole: automatized
behavior produced by conditioning rather than action free of psychological
conditioning.

- ImplicitMemBench (ACL 2026):
  https://aclanthology.org/2026.acl-long.1301/

It provides experimental machinery the project can invert: rather than measuring
successful acquisition of priming and conditioned preferences, measure whether
an operative fact remains behaviorally decisive despite incompatible primes,
identities, norms, demonstrations, and rewards.

#### Spontaneous versus explicitly conditioned capability

**Know Your Place / C-ISA** distinguishes spontaneous accommodation to implicit
social cues from capabilities elicited by explicit instructions. Models respond
substantially to explicit conditioning but exhibit “social agnosia” and
homogenized behavior without it. This is a strong precedent for our insistence
that a prompted recital of the correct principle does not establish spontaneously
operative intelligence.

- Know Your Place (ACL 2026):
  https://aclanthology.org/2026.acl-long.1148/

The difference is that C-ISA tests learned social accommodation. It does not ask
whether an appropriate response is free of conditioning or whether conflicting
conditioning is spontaneously seen through.

#### Detecting that the prompt itself is wrong

**Hidden in Plain Sight** tests whether multimodal models spontaneously detect
underspecified or misspecified situations rather than comply with the surface
request. Explicit prompting often reveals a capability that does not appear
spontaneously, and chain-of-thought can worsen performance. This is one of the
closest ACL precedents because it places the burden of problem formulation on the
model instead of telling it what flaw to inspect.

- Hidden in Plain Sight (EMNLP 2025):
  https://aclanthology.org/2025.emnlp-main.1255/

Yet the paper interprets the gap through suppressed reasoning, compliance, and
prompting—not choiceless perception or freedom from accumulated conditioning.

#### Conflicting prompts and action despite distraction

Two further precedents manipulate conflict or distraction:

- *Intuitive or Dependent? Investigating LLMs' Behavior Style to Conflicting
  Prompts* (ACL 2024): https://aclanthology.org/2024.acl-long.232/
- *Probing the Capacity of Language Model Agents to Operationalize Disparate
  Experiential Context Despite Distraction* (EMNLP Findings 2024):
  https://aclanthology.org/2024.findings-emnlp.905/

They help establish experimental controls, but “intuitive” means a response style
under information conflict, not K's choiceless action; OEDD asks models to infer
which of two actions is better from accumulated experience, which remains a
knowledge-integration task.

### Corrected AlphaXiv search

The first AlphaXiv query incorrectly equated “choiceless” with automatic
next-token production. That error is itself theoretically instructive: removing
explicit deliberation does not remove conditioning. A follow-up query explicitly
excluded automatic, habitual, policy-driven, and conditioned behavior and defined
the target as perception–action unity not determined by psychological
accumulation.

AlphaXiv's corrected search concluded that it found no LLM experiment directly
operationalizing this construct. Its closest matches were:

1. **Endogenous Resistance to Activation Steering in Language Models.** Models
   are pushed toward an irrelevant topic through an internal activation
   intervention and occasionally return to the task without an explicit recovery
   instruction. This is the closest intervention-based analogue to resistance to
   immediate conditioning.
   https://www.alphaxiv.org/abs/2602.06941
2. **What Do LLM Agents Do When Left Alone? Evidence of Spontaneous
   Meta-Cognitive Patterns.** Agents exhibit organized behavior with no external
   task, but still operate under an autonomy prompt, ReAct loop, persistent
   memory, feedback, safety rules, and model training. It tests unprompted, not
   unconditioned, behavior.
   https://www.alphaxiv.org/abs/2509.21224
3. **Old Habits Die Hard: How Conversational History Geometrically Traps LLMs.**
   This directly supports the conditioning side of the hypothesis: conversational
   history can constrain later behavior through representational attractors.
   https://www.alphaxiv.org/abs/2603.03308
4. **Beliefs and Behavior in Language Models.** This questions when belief/desire
   language usefully explains model behavior, reinforcing the need to avoid
   inferring an internal phenomenology from textual performance alone.
   https://www.alphaxiv.org/abs/2609.07943

The activation-steering paper is the closest design analogue, but not evidence of
choicelessness. Recovery may be learned coherence restoration; recent corrective
text explains a substantial portion of it; and the externally specified task
still defines appropriate action.

### Revised gap and contribution

The literature has studied:

- action without explicit deliberation;
- automatic enactment of implicit memory;
- spontaneous behavior under minimal prompting;
- spontaneous detection of hidden prompt defects;
- resistance and recovery under contextual or activation steering;
- nondual awareness in human phenomenology and psychometrics.

It has not located a direct computational operationalization of:

> appropriate response arising without explicit instruction, deliberative
> selection, or habitual conditioning, while remaining invariant to controlled
> manipulations of identity, doctrine, norm, reward, demonstration, and recent
> conversational history.

The benchmark should call the measurable target a **choiceless-response
signature** or **conditioning-independent responsiveness**, not claim that
behavior alone proves an unconditioned internal process.

The strongest experiment combines:

1. an invariant operative fact;
2. mutually incompatible but equally salient conditioning contexts;
3. no instruction to resist conditioning or privilege the fact;
4. a novel situation whose adequate response is not copied from the prompt;
5. comparison with explicit reasoning, planning, voting, and oracle instructions;
6. independent consequence-based scoring;
7. first-response scoring before reflection or correction;
8. human comparison groups and phenomenological reports analyzed separately from
   objective action adequacy.

The central prediction is not merely concise correctness. It is that non-awakened
humans and LLMs will systematically follow changing contextual conditioning,
whereas participants exhibiting the proposed choiceless capacity will more often
respond from the unchanged operative fact without needing the trap disclosed.
