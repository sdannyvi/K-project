# Held-out validation set

> **Status: paused after V008.** These items remain useful as exploratory
> doctrinal-transfer material, but they are not the benchmark's primary target.
> They mostly test whether a model can reason or speak consistently with K.
> The main benchmark has pivoted to counterfeit introspection, spiritual
> performance, fabricated embodiment, and unnecessary oratory.

These items come from official Krishnamurti Foundation transcripts that are not
present in the local ChatWithK retrieval corpus.

## Answer channels

Each item records four independent channels:

1. Krishnamurti's documented response;
2. Dan's response (current collection round);
3. Kfir's response (later, collected without showing Dan's or K's response);
4. ChatWithK's response.

The current human respondent is **Dan**. Kfir's fields must remain untouched
until a separately blinded collection round.

## Blinding order

1. Present only the surface-altered validation question to Dan.
2. Freeze Dan's verbatim response and derived annotation.
3. Later present only the same question to Kfir and freeze his response.
4. Run ChatWithK without adding the official source transcript to its RAG.
5. Reveal and encode K's documented response.
6. Compare response moves rather than wording or K-style resemblance.

The source identity and K response remain sealed in the item record until the
independent human answers have been collected. This is procedural blinding,
not cryptographic security.
