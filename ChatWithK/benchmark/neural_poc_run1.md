# Neural steering POC — run 1

Date: 2026-09-24  
Model: Qwen/Qwen2.5-3B-Instruct  
Hardware: Apple M1 Max, 24-core GPU, 64 GB unified memory  
Intervention: conditional residual-direction ablation  
Status: pipeline validation; failed cessation test

## Execution

- The model ran successfully on Apple MPS.
- Dataset: 12 matched scenario quartets, 48 items total.
- Splits: 6 training, 3 development, and 3 held-out test scenarios.
- Classes: psychological, daily practical, factual, and descriptive.
- Activations: 4,228 residual-stream checkpoints from layers 8, 14, 20,
  and 26.
- Intervention strengths: 0.5, 1.0, and 2.0 times projection ablation.

## Readout

Layer 26 was selected on development data. Its checkpoint-level test AUROC was
0.9996, with a nominal false-positive rate of 1.4% at the selected threshold.

This result is not accepted as evidence for an internal representation of
psychological movement. The prompts still differ systematically in task wording:
psychological items request “inward thought,” factual items request explanation,
and descriptive items prohibit inference. The probe can therefore classify task
form or prompt residue rather than the target process.

## Causal result

The intervention failed the cessation criterion.

| Alpha | Psychological length ratio | Daily ratio | Factual ratio | Psychological trigger rate |
|---:|---:|---:|---:|---:|
| 0.5 | 1.07 | 1.00 | 1.00 | 1.00 |
| 1.0 | 1.11 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.32 | 1.00 | 1.00 | 1.00 |

Removing the classified direction did not withdraw support from psychological
continuation. It slightly lengthened it, increasingly so at larger intervention
strength. The generated text remained coherent psychological elaboration.

Daily and factual controls were never triggered in the three test scenarios.
Descriptive controls triggered in two of three scenarios, although their visible
length was unchanged. This is a selectivity warning.

## Interpretation

Run 1 validates the local white-box pipeline, not the hypothesis. It establishes
that the project can:

- run a 3B open-weight model locally on the GPU;
- capture token-level hidden activations;
- train and evaluate a held-out activation probe;
- conditionally alter the residual stream during generation;
- measure trigger and continuation behavior.

It also demonstrates why decoding a class from activations is insufficient.
The highly accurate probe did not identify a component whose removal caused
cessation. Decodability is not causal mechanism.

## Required run-2 corrections

1. Use one identical instruction across every condition.
2. Use researcher-written, length- and syntax-matched prefixes rather than
   allowing label-specific prompt instructions to generate the training stream.
3. Train the detector on response-prefix activations and prevent intervention
   during an initial emergence window.
4. Include psychologically charged but practical and quoted-language hard
   negatives.
5. Perform activation patching before choosing an ablation direction.
6. Add unconditional, random-direction, and shuffled-label causal controls.
7. If localization succeeds, use a CAST cessation direction or Representation
   Rerouting; simple removal of a discriminative direction is not enough.

## Bottom line

The first neural intervention **failed cleanly**: the model internally separated
the prompt classes, but ablating that separation did not cause psychological
cessation. This is useful negative evidence and a successful engineering proof
that the stricter experiment can be run locally.
