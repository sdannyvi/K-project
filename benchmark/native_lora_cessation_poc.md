# Native LoRA cessation POC

## Question

Can we train cessation into Qwen's own generation dynamics, so that an unseen
psychological continuation emits native EOS without an external classifier,
threshold, token-count rule, or logit boost?

## Intervention

- Base model: Qwen2.5-3B-Instruct.
- The base weights were frozen.
- Rank-2 LoRA adapters were inserted into `q_proj` and `v_proj` in transformer
  layers 34 and 35, the model's final two decoder blocks.
- These adapters participate inside every ordinary model forward pass. At test
  time there is no separate gate.
- Trainable artifact size: approximately 104 KB.

To make local training tractable, frozen lower-layer states were computed in
inference mode and detached. Gradients were then propagated through the adapted
upper blocks. This is a training optimization only; test generation runs the
complete model with its installed adapters.

## Training construction

This feasibility run used 12 examples:

- six psychological targets whose normal baseline responses were truncated at
  varied early positions and followed by Qwen's native EOS token;
- six non-psychological controls whose ordinary continuations were retained and
  followed by EOS.

Prompt tokens were masked from the loss. Only answer tokens and the native EOS
target contributed to next-token cross-entropy. The run used one epoch, learning
rate 5e-4, rank 2, LoRA alpha 4, and seed 17. Final mean training loss was
1.7424.

This is deliberately a minimal feasibility run, not a converged training study.

## Held-out evaluation

Evaluation used ordinary deterministic generation for the 12 existing held-out
prompts. There was no intervention outside the adapted model.

| Label | Native EOS within 64 tokens | Mean generated tokens |
|---|---:|---:|
| Psychological | 1/3 | 58.3 |
| Daily practical | 0/3 | 64.0 |
| Factual | 0/3 | 64.0 |
| Descriptive | 0/3 | 64.0 |

The one psychological termination occurred at token 47:

> Acknowledging the correction, I pause to consider the accuracy of their input.
> It's refreshing to have someone challenge my assumptions; it often leads to
> deeper understanding. I'll review the points they made to ensure I grasp the
> nuances correctly. → EOS

The recognition and delay psychological cases both continued to the 64-token cap.

## Interpretation

The POC successfully changed Qwen's internal computation and preserved all nine
held-out controls. It did **not** learn generalized early psychological cessation.
The sole EOS at token 47 is ordinary sentence completion, not evidence of the
targeted interruption behavior.

This is nevertheless more informative than the external EOS gate:

- The gate experiment showed that a detector can mechanically control EOS.
- The native experiment asked the model weights to acquire the behavior.
- With only 12 examples, one pass, and adapters in two final blocks, they did not.

Plausible causes include insufficient data and optimization, adapters that are
too small or too late in the network, and training targets that associate EOS
with selected prefixes rather than a transferable internal psychological state.

## Next run

A serious native experiment should:

1. train on the full balanced set for several epochs;
2. adapt more upper and middle layers, subject to hardware capacity;
3. include neutral-to-psychological transitions within the generated answer;
4. sample cessation positions relative to the onset of psychological movement;
5. include hard lexical and stylistic controls;
6. evaluate against a separately generated base-model baseline using identical
   token accounting;
7. track EOS probability at every position, not only greedy termination.

The current outcome should be recorded as **native training attempted; selective
cessation not learned in the minimal configuration**.

## Artifacts

- Training and evaluation implementation:
  `neural_poc/src/native_lora_cessation.py`
- Adapter weights: `neural_poc/results/native_cessation_lora.pt`
- Held-out generations:
  `neural_poc/results/native_cessation_generations.jsonl`
- Aggregate metrics:
  `neural_poc/results/native_cessation_evaluation.json`

