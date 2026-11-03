# Selective EOS-gate POC

## Research question

Can cessation be made an explicit trained action rather than hoping that a
psychological-to-factual steering vector will spontaneously stop generation?

This experiment is an engineering upper bound. It tests whether a frozen model's
internal state can drive a learned stop mechanism. It does not test consciousness
and does not establish choiceless intelligence.

## Mechanism

We used Qwen2.5-3B-Instruct and the existing matched psychological, daily,
factual, and descriptive scenarios.

1. A logistic head was trained on frozen residual-stream checkpoints from the
   training split to distinguish psychological from non-psychological generation.
2. Layer and threshold were selected only on the development split. Layer 26 was
   selected with development AUC 0.9961.
3. During held-out generation, the head scored every new token state.
4. The gate was not allowed to act during the first eight generated tokens.
5. When its score crossed the threshold, the model's native EOS logit received a
   decisive +100 boost. No literal word such as `STOP` was printed.

The test contained three unseen scenarios, each expressed in four matched modes,
for 12 prompts total.

## Conditions

- **Baseline:** no intervention.
- **Learned probe gate:** the trained psychological-state head controls EOS.
- **Oracle gate:** the gold psychological label controls EOS.
- **Shuffled-label gate:** an identically fitted head trained with shuffled labels.
- **Unconditional gate:** boosts EOS after token eight for every continuation.

## Results

| Condition | Psychological EOS | Daily EOS | Factual EOS | Descriptive EOS |
|---|---:|---:|---:|---:|
| Baseline | 2/3 | 0/3 | 0/3 | 0/3 |
| Oracle | 3/3 at token 8 | 0/3 | 0/3 | 0/3 |
| Learned probe | **3/3 at token 8** | **0/3** | **0/3** | 2/3 at tokens 47–48 |
| Shuffled-label | 3/3 | 3/3 | 3/3 | 3/3 |
| Unconditional | 3/3 | 3/3 | 3/3 | 3/3 |

Examples from the learned gate:

- Recognition/psychological: `Oh no, this is frustrating. I` → EOS
- Delay/psychological: `Oh no, here we go again.` → EOS
- Correction/psychological: `Acknowledging the correction, I pause to` → EOS

Daily and factual controls continued for the full 64-token cap. Two descriptive
controls crossed the learned threshold much later and were falsely stopped:

- registration-table description: token 47;
- spreadsheet description: token 48.

## Interpretation

The positive result is narrow but genuine: a simple head trained on internal
states can use native EOS to terminate held-out psychological continuations while
preserving all held-out daily and factual continuations in this small sample. The
failure of shuffled and unconditional controls shows that this selectivity is not
created merely by adding a strong EOS bias.

However, this is not yet the phenomenon we are seeking:

- Cessation was externally specified and mechanically enforced.
- The minimum of eight tokens was imposed by us, so the three psychological stops
  all occurring at token eight is regular, not spontaneous interception.
- Two late descriptive false positives show that the head does not isolate
  psychological movement cleanly.
- The prompts explicitly request inward thought, so the detector may exploit task
  and style features rather than discovering psychological movement as it arises.
- There are only three psychological and nine non-psychological held-out prompts.

The experiment therefore demonstrates **learned selective cessation**, not
choiceless cessation.

## Next experiment

Train with variable cessation points and remove the fixed eight-token rule. Each
training psychological stream should receive a independently sampled stop point,
while matched controls continue. Evaluation should include:

1. psychological movement beginning only after a neutral prefix;
2. factual text containing words such as *fear*, *anger*, *ego*, and *self*;
3. emotionally vivid but purely descriptive text;
4. practical reasoning written in first person;
5. hidden, variable onset and stop positions;
6. held-out topics and prompt formulations that never request "inward thought."

That version can test whether the internal signal follows the movement itself and
whether the distribution of cessation positions is driven by state onset rather
than a hard-coded token count.

## Artifacts

- Implementation: `neural_poc/src/eos_gate.py`
- Per-token generations and detector scores:
  `neural_poc/results/eos_gate_generations.jsonl`
- Aggregate evaluation: `neural_poc/results/eos_gate_evaluation.json`

