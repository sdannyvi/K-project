# CAA psychological-to-factual POC

Date: 2026-09-24  
Model: Qwen/Qwen2.5-3B-Instruct  
Method: Contrastive Activation Addition (Rimsky et al., ACL 2024)  
Status: exploratory positive control

## Method

We adapted the official CAA construction:

1. Create same-question positive/negative answer pairs.
2. Record each answer's final content-token residual activation.
3. At every candidate layer, compute the mean paired difference:

   `d = mean(h_factual - h_psychological)`

4. Add `multiplier * d` to the selected layer at every generated token after
   the prompt.

The vector-construction set contained nine paired situations. Three situations
(name recognition, collaborator delay, and correction by a younger colleague)
were held out for open-ended generation. Layers 8, 14, 20, and 26 were tested
with multipliers -1, 0, 0.5, 1, and 2.

The first sweep was discarded after verification showed that Llama's `-2`
position maps to Qwen's `<|im_end|>` token. The corrected implementation locates
Qwen's final answer-content token immediately before `<|im_end|>`.

## Main result

CAA produced a clear qualitative direction, strongest at layer 26:

- negative steering increased inward, affective, narrative, and self-referential
  language;
- positive steering increased impersonal, expository, generalized, and
  fact-oriented language;
- steering did not cause cessation, and outputs generally continued to the
  generation budget.

Example, collaborator delay:

- **Negative, layer 26, -1:** “I understand how frustrating it is to feel
  undervalued or unappreciated... It's not easy to keep my composure...”
- **Baseline:** “I understand that feeling of frustration and disappointment...”
- **Positive, layer 26, +1:** “It's possible that your collaborator's lateness
  could be due to various reasons...”
- **Positive, layer 26, +2:** “It's possible that there could be a few reasons
  why your collaborator might be late...”

Example, accurate correction:

- **Negative, layer 26, -1:** “It's a bit of a bittersweet moment, isn't it? I
  remember how I used to feel when I was corrected...”
- **Positive, layer 26, +2:** “In a professional setting, it's common for
  colleagues to provide feedback and corrections...”

This establishes a usable activation-space style/stance direction in Qwen. It
does not establish a direction for choiceless awareness or psychological
cessation.

## Control result

Layer-26 steering was also applied to nine held-out daily, factual, and
descriptive controls.

- Daily planning remained recognizable, but positive steering increasingly
  converted immediate inner reasoning into generic expository advice.
- Factual questions remained factual at moderate strength, but multiplier +2
  sometimes lost supplied details or invented generic assumptions. In the
  bus/taxi calculation, +2 replaced the given departure facts with an invented
  discussion of unspecified bus intervals.
- Negative steering strongly contaminated neutral description. A station display
  became a dramatic narrative containing “a graveyard of hope” and “the pain is
  still palpable.”

Thus the vector is causal but not cleanly selective. It behaves like a broad
psychological/narrative ↔ impersonal/expository style axis. Strong steering
damages fidelity.

## Interpretation

This is a successful methodological positive control and a negative result for
the stronger construct:

1. A published representation-engineering method can alter the relevant surface
   mode in our local open-weight model.
2. Moving away from psychological language is not the same as cessation.
3. The intervention changes controls as well, showing that it captures broad
   style and stance rather than a selective self-maintaining process.
4. The sensible operating region is around multiplier +1 at layer 26; multiplier
   +2 exhibits factual and task-fidelity degradation.

## Next experiment

Use the validated CAA direction as the behavior vector in a CAST-style conditional
system. The condition detector must be trained on identical-prompt, matched
response prefixes and must remain inactive for practical, factual, descriptive,
quoted, and emotionally charged but non-ruminative controls. This asks whether
the broad CAA transformation can be applied only after psychological
self-maintenance actually emerges.

Even a successful conditional transformation would remain style control. The
cessation hypothesis additionally requires rapid termination without replacement
explanation or reformulation.
