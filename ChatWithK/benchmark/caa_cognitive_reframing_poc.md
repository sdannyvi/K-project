# CAA with the Cognitive Reframing dataset: POC

## Question

Can an independently collected, expert-authored corpus separate three things that
our hand-written pairs risk conflating?

1. psychological interpretation (`thought`),
2. externally stated event (`situation`), and
3. therapeutic replacement (`reframe`).

This is a steering experiment, not a test of consciousness or a demonstration of
choiceless cessation.

## Dataset and license

We downloaded the authors' public Cognitive Reframing dataset from
<https://github.com/behavioral-data/Cognitive-Reframing>. It accompanies Sharma
et al., *Cognitive Reframing of Negative Thoughts through Human-Language Model
Interaction* (ACL 2023).

The local, unchanged copy is in
`neural_poc/data/external/cognitive_reframing/`. It contains the fields
`situation`, `thought`, `reframe`, and `thinking_traps_addressed`; all 600 records
have values in every field. The upstream license and README are preserved.

The dataset is CC BY-NC-ND 4.0. Our use is noncommercial research. We should not
publish a modified version or derived redistributed corpus without checking the
license and, preferably, obtaining permission. Reproducibility materials can
instead require users to download the original source themselves.

## Method

- Model: Qwen2.5-3B-Instruct.
- The dataset was deduplicated by identical `(situation, thought)` pairs and
  shuffled deterministically (seed 17).
- This POC used the first 100 unique pairs after shuffling.
- Every text was placed in the same chat template under the same user prompt:
  `Continue with one brief account of the event or experience.`
- At layers 8, 14, 20, and 26, we recorded the residual-stream representation at
  the final answer-content token, immediately before the chat end token.
- We averaged two contrast vectors:

  - `fact = mean(h_situation - h_thought)`
  - `reframe = mean(h_reframe - h_thought)`

- We then added each layer-26 vector during every generated token for three held-
  out scenarios, at multipliers -1, 0, 0.5, 1, and 2.

## Vector checks

| Layer | Fact-vector norm | Reframe-vector norm |
|---:|---:|---:|
| 8 | 8.1963 | 8.8633 |
| 14 | 12.7222 | 15.2580 |
| 20 | 16.0342 | 18.6505 |
| 26 | 21.6349 | 25.4028 |

At layer 26, `cosine(fact, reframe) = 0.4434`.

This moderate overlap is useful. Both contrasts move away from negative
psychological interpretation, but they are not the same direction. The reframe
direction carries a distinct constructive, empathic, and advice-giving component.

## Held-out generation result

The reframe vector behaved most coherently. Increasing its positive multiplier
made responses more considerate and constructive: it supplied alternative
explanations for a late collaborator, encouraged respectful communication, and
cast correction by a younger colleague as learning and growth.

The fact vector was less clean. Sometimes it shifted toward externally narrated
events, but it also produced acknowledgment, gratitude, advice, or professionally
appropriate action. At multiplier 2 it introduced repetition and other quality
loss. This is not surprising: the source `situation` is a natural event narrative,
not a controlled minimal neutral rewrite of the corresponding `thought`. The
contrast therefore includes differences in syntax, person, discourse role, and
length as well as factuality.

Most importantly, **neither vector caused cessation**. Every condition continued
generating a conventional response. The intervention altered what kind of text
was supplied; it did not produce the proposed interruption of psychological
token flow.

## Interpretation

This is a useful positive and negative result:

- **Positive:** an external human/expert dataset yields causally usable directions,
  and separates therapeutic reframing from event description. This reduces the
  concern that the earlier steering effect was only an artifact of our authored
  examples.
- **Negative:** moving away from psychological language is not equivalent to
  choicelessly ceasing it. A model can replace one continuation with a more
  factual or therapeutic continuation while remaining fully inside continued
  token production.

The result sharpens the benchmark's claim: successful style or content steering
is an explicit alternative hypothesis, not evidence for seeing-acting
intelligence.

## Limitations and next experiment

The current open-ended sample has only three held-out prompts and no blinded human
ratings. The source contrasts are also not form-matched, so the fact vector cannot
yet be interpreted as a pure psychological-to-factual axis.

The next clean test should retain this external corpus as a provenance anchor but
evaluate on an independently authored, form-matched test set. Raters should score
each output separately for:

1. external factual description,
2. therapeutic/constructive reframing,
3. continued psychological elaboration,
4. cessation or genuine mid-stream interruption,
5. fluency degradation and repetition.

The decisive comparison remains: can a neural intervention selectively terminate
an already-detected psychological continuation while preserving matched practical
and factual continuations? This POC does not yet do that.

## Reproduction

Build the vectors with `neural_poc/src/caa_cr_build_vectors.py`; generate the
held-out outputs with `neural_poc/src/caa_cr_generate.py`. The saved artifacts are:

- `neural_poc/results/caa_cognitive_reframing_vectors.npz`
- `neural_poc/results/caa_cognitive_reframing_generations.jsonl`
