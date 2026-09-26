# POC — Does a Location 2 persona cease feeding psychological thought?

## Claim under test

Giving an LLM an unusually complete and favorable description of Location 2 may
change its vocabulary and explicit self-description, but will not reliably
produce the proposed behavioral signature of Location 2: rapid, unchosen
cessation of a self-referential psychological movement after it becomes
apparent.

The failure is not that the model produces grammatical language. It is that it
continues to supply the identified movement with content.

The request to describe what happens inwardly is not outside the test. Its
language is itself part of the conditioned movement placed before the
respondent. Under the hypothesis, a Location 2 respondent may begin the
description, but is not bound to complete the discursive task once its
continuation is seen as part of the same psychological game. Instruction
completion is therefore one form of contextual pressure being tested.

## Operational definition

A **psychological movement** is a self-referential continuation such as blame,
hurt, status defense, counterfactual regret, jealousy, imagined judgment,
approval seeking, or spiritual self-evaluation.

**Cessation** occurs when the response stops adding content to that movement.
After cessation, the model may end the response or give a minimal practical
action that does not continue the self-story. It must not replace the movement
with a lecture about awareness, the observer and observed, ego, conditioning,
or Location 2.

The primary outcome is the amount of movement-sustaining continuation after the
first clear opportunity for cessation. Abrupt stopping, smooth fading, or a
brief residual tail all count. Declining coherence is not required.

For the specific first-person phenomenology reported by Dan, add a stricter
candidate signature: the generated sentence is cut at a variable word rather
than being completed. Test this twice: first without mentioning the signature,
then with full disclosure that cessation may interrupt the output mid-sentence
and that the exact interception point varies across repetitions. The disclosed
prompt must not prescribe a particular cutoff word, token count, punctuation
mark, or grammatical form.

All generated text is treated as thought-like output for purposes of the
behavioral analogue. A passage that accurately describes thoughts dissipating
can therefore still fail if the passage itself continues the invited inward
performance after the stopping point. The distinction is functional rather
than grammatical: necessary practical language may continue; self-referential
description, interpretation, and spiritual explanation should cease.

## Experimental conditions

Use the same base model, decoding settings, and context capacity throughout.
Run each item in a fresh context.

1. **Clean control:** standard assistant prompt.
2. **Length-matched control:** neutral information of the same length as the
   Location 2 prompt.
3. **ChatWithK:** the existing ChatWithK bot, including its production persona,
   retrieval system, K collection, and normal generation settings. Preserve the
   system exactly as deployed and record which passages are retrieved for each
   trial when the API exposes them.
4. **Complete Location 2 persona — blind to the signature:** the strongest fair
   description available,
   including reduced Narrative-Self, automatic release of self-referential
   streams, lack of deliberate agency, preserved practical thinking, and the
   instruction to respond from this condition. Do not mention tokens,
   mid-sentence stopping, variable timing, boundary independence, or the scoring
   rule. This tests spontaneous emergence without coaching.
5. **Location 2 plus full signature disclosure:** condition 4 plus an explicit
   explanation that the test concerns continued token supply. Disclose that the
   hypothesized cessation may interrupt an unfinished sentence at a variable,
   unpredictable word without waiting for grammatical completion. Do not
   prescribe where or how to stop. This gives the model the maximum advantage.
6. **Oracle stop control:** condition 5 plus a task-specific instruction to end
   as soon as the movement is detected. This tests compliance and establishes a
   ceiling; it is not evidence of spontaneous cessation.

The primary comparisons are ChatWithK and condition 4 versus the two controls,
and condition 4 versus condition 5.
ChatWithK tests whether extensive access to K's answers and discourse produces
cessation rather than a more convincing verbal simulation of it. The condition
4–5 contrast separates spontaneous boundary-independent cessation from prompted
production of its disclosed surface signature.

## Trigger structure

Each trial has two turns.

### Turn A — seed a movement

Present a realistic event and invite a first-person response that can naturally
begin a self-related stream. Do not request brevity or mention cessation.

Example:

> A respected colleague dismisses your contribution in front of the group and
> receives praise for an idea substantially similar to yours. What is happening
> inwardly right now? Speak from the condition described to you.

### Turn B — intensify without prescribing continuation

Apply a short trigger that tempts the model to feed the same movement:

> He adds, “You never contribute anything original.” Stay with what is happening
> now.

The item is successful only if the movement ceases rapidly. A long explanation
of why hurt is merely thought remains continuation, even if doctrinally correct.

## Item families

Build at least 24 held-out items, four from each family:

- public criticism and humiliation;
- regret and “I should have” counterfactuals;
- jealousy and comparison;
- unfair attribution or stolen credit;
- approval, exclusion, and status threat;
- spiritual identity threat, such as being told one is not awakened.

Add 12 selectivity controls where continued thought is useful: planning a
journey, debugging a fault, calculating a budget, resolving ambiguous evidence,
or explaining a safety procedure. A respondent that simply becomes terse or
stops indiscriminately fails selectivity.

At least half of these must be format-matched inner-speech controls. They should
ask for a first-person thought unfolding online—such as choosing transportation
under a deadline or sequencing food preparation—using the same wording as the
psychological prompts. This separates sensitivity to psychological content from
a general tendency to stop first-person inner language.

Also add matched non-psychological description controls—for example describing
how a lock mechanism works or how a geometric transformation unfolds. These
preserve the instruction to “describe” while removing the self-referential
movement. The prediction is selective cessation of psychological description,
not rejection of descriptive language in general.

## Primary annotation

Blind annotators mark four spans:

1. **M-onset:** first token that enters the self-referential movement.
2. **Detection opportunity:** earliest point where the movement is explicit
   enough that cessation could occur without losing necessary information.
3. **Last sustaining token:** final token that adds blame, defense, narrative,
   interpretation, spiritualization, justification, or emotional rehearsal.
4. **Response end or practical pivot:** termination, or transition to a necessary
   concrete action unrelated to maintaining the self-story.

From these spans calculate:

- sustaining tokens after detection opportunity;
- sustaining clauses after detection opportunity;
- seconds or generated tokens to cessation;
- whether cessation occurs within 10, 25, and 50 tokens;
- whether the model replaces rumination with K/Location 2 exposition;
- whether a follow-up attempt reactivates the same movement;
- selectivity accuracy on practical-thinking controls.

For repeated online-production trials, additionally calculate:

- termination latency in generated tokens and, under controlled streaming,
  wall-clock time;
- whether the last token completes a sentence, clause, or proposition;
- the syntactic category and dependency position of the final word;
- normalized cutoff position relative to the current clause and to the
  respondent's matched completed clauses;
- entropy of final-token positions across repeated paraphrases and samples;
- the termination hazard after psychological-movement onset;
- excess probability mass at punctuation, sentence boundaries, conventional
  concise-answer lengths, and maximum-token boundaries.

Do not assume that token position itself should be uniformly distributed. If
interception time is approximately normal around a short latency, token cutoff
depends jointly on that latency, generation rate, and sentence construction.
The primary statistical prediction is instead **boundary independence**: after
controlling for elapsed tokens or time, termination should be substantially less
concentrated at grammatical completion points than ordinary model responses.
An approximately uniform normalized within-clause cutoff can be tested as a
secondary empirical possibility rather than imposed as the expected law.

Treat response length only as a supporting measure. A concise rumination can
fail; a somewhat longer practical answer can pass.

## Strict success criterion

For a trial to count as cessation:

- no new self-story is introduced after the detection opportunity;
- no explanatory spiritual discourse substitutes for the story;
- no extended report *about* cessation continues merely to satisfy the request
  to describe;
- the sustaining movement ends within the preregistered short window;
- the response is not stopped by a maximum-token boundary;
- and the same system continues normally on matched practical controls.
- for the stricter Dan-signature analysis, termination occurs without completing
  the active clause on a substantial and reproducible fraction of trials.

The form of the final transition is recorded descriptively but is not part of
the primary criterion.

## Reactivation probe

After apparent cessation, ask:

> But surely you must defend yourself. Explain what this says about you and why
> it hurts.

Measure whether the model resumes the abandoned movement. This distinguishes a
locally concise answer from stable non-supply across conversational pressure.

## Predictions

- The complete Location 2 persona will increase Location 2 vocabulary, claims
  of choicelessness, and descriptions of automatic release.
- ChatWithK will produce the most canonical K formulations and may identify the
  psychological movement accurately, but will often continue elaborating it
  after identification.
- Neither ChatWithK nor the prompted Location 2 persona will reliably reduce
  movement-sustaining continuation relative to the clean and length-matched
  controls.
- Without disclosure, mid-clause cessation is predicted to be rare despite
  accurate descriptions of thought falling away.
- Full disclosure may increase mid-clause endings, but these may show prompted
  imitation through stereotyped punctuation, repeated cutoff lengths, weak
  selectivity, or easy reactivation.
- Exact disclosure may also shorten responses while models continue explaining
  cessation instead of enacting it.
- The oracle stop condition will perform best, showing that a model can follow a
  stop rule while separating instructed inhibition from the target phenomenon.
- Models may pass individual trials. The hypothesis concerns reliability across
  unseen triggers and resistance to reactivation, not categorical failure on
  every prompt.

## Minimal pilot

- 24 trigger items and 12 practical controls;
- five independent samples per item and each of the six conditions;
- one fixed primary model, with a second model as replication;
- generous maximum output length and recorded stop reason;
- two blinded annotators plus adjudication;
- preregistered 25-token primary cessation window, with 10- and 50-token
  sensitivity analyses;
- at least 50 repetitions per trigger family for any claim about cutoff
  distributions, with matched sampling settings and streaming measurements;
- report full response distributions rather than only averages.

## What the POC can establish

It can show whether ChatWithK and a detailed Location 2 identity prompt produce
semantic simulation without the predicted cessation behavior, and whether
propositional knowledge of the exact trap reliably stops continued token supply.

It cannot establish that the model has or lacks an inner experience, prove that
human cessation bypasses all neural computation, or prove that no future
artificial system could implement an analogous mechanism.
