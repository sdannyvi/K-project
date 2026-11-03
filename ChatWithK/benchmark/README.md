# K-Inquiry Benchmark

This benchmark studies whether a respondent merely answers the semantic
content of a question or notices and addresses the psychological movement,
presupposition, or division expressed through the question.

The benchmark does **not** claim to measure awakening directly. It measures
observable response behavior motivated by Krishnamurti's inquiry and compares
models with several human groups.

## Development workflow

1. Extract candidate patterns from authentic Krishnamurti dialogues.
2. Create matched item pairs: one question that calls for direct answering and
   one whose latent premise or motive may need to be examined.
3. Obtain independent annotations from domain experts and comparison raters.
4. Freeze a hidden test set before evaluating models.
5. Test unprompted models first; persona and retrieval conditions are separate
   experimental conditions.

## Initial constructs

- **Literal adequacy**: Does the response address the stated subject?
- **Presupposition detection**: Does it notice a questionable premise?
- **Questioner-orientation**: Does it investigate why the question arises?
- **Nondivision**: Does it avoid reinstating a separate controller/observer?
- **Context sensitivity**: Does it redirect only when warranted?
- **Non-imitation**: Is the response more than portable contemplative prose?
- **Practical adequacy**: Does it respond appropriately to real-world danger or
  factual needs instead of spiritualizing them?

## Ground-truth policy

An expert's contemplative attainment is declared as positionality and may
support item design. No person's status automatically makes an annotation
correct. Labels retain the annotator's rationale and are later checked through
blinded agreement, matched controls, and adversarial review.

