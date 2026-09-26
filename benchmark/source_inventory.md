# Benchmark source inventory

## Local corpus audit

The current `K-texts/` corpus contains 189 plain-text transcripts, including:

- 15 files explicitly titled as question-and-answer meetings;
- 20 files explicitly titled as dialogues;
- 11 files titled as seminars or discussions;
- four transcripts explicitly naming David Bohm in the dialogue text;
- one especially relevant seminar titled *Intelligence, Computers and the
  Mechanical Mind*.

The four locally identified Bohm conversations are:

- `can-you-have-insight-if-there-centre.txt`
- `difficulty-thinking-together.txt`
- `death-has-very-little-meaning.txt`
- `any-form-image-prevents-beauty-relationship.txt`

The local computer seminar is:

- `chapter-6-part-2-seminar-rishi-valley-4th-december-1980-intelligence-computers-and.txt`

## High-priority local sources

1. `1st-question-answer-meeting-31.txt`: K explicitly explains that a question
   discloses the content of the questioner's mind and asks why it was put.
2. `1st-question-answer-meeting-34.txt`: K examines motive in approaching a
   question and distinguishes inquiry from seeking a satisfying answer.
3. `4th Public Talk San Diego.txt`: K asks who is asking about the practical
   value of inward transformation.
4. `can-you-have-insight-if-there-centre.txt`: K and Bohm investigate whether
   spiritual experience presupposes a recording centre.
5. `difficulty-thinking-together.txt`: K, Bohm, and Wilkins examine opinions,
   identification, responsibility, and whether a few people can affect society.
6. `death-has-very-little-meaning.txt`: K and Bohm distinguish insight and
   perception from mechanical rules and accumulated reason.
7. `any-form-image-prevents-beauty-relationship.txt`: K and Bohm repeatedly
   test abstract claims against ordinary relationship.
8. `chapter-6-part-2-seminar-rishi-valley-4th-december-1980-intelligence-computers-and.txt`:
   an unusually direct source on computers, mechanical knowledge, pleasure,
   intelligence, and the future of human thought.

## Official material not adequately represented locally

The Krishnamurti Foundation Trust archive contains several richer series:

- 1982 Ojai scientist discussions with David Bohm, biologist Rupert Sheldrake,
  and psychiatrist John Hidley;
- 1976 Ojai scientist seminars;
- 1974 Brockwood scientist seminars;
- 1975 Gstaad dialogues with Bohm;
- 1980 Bohm dialogues later associated with *The Ending of Time*;
- discussions with psychiatrist David Shainberg;
- the 1983 *Future of Humanity* dialogues with Bohm.

Official starting points:

- <https://kfoundation.org/transcript/scientists-discussion-2-ojai-california-17-april-1982/>
- <https://kfoundation.org/transcript/scientists-seminar-5-ojai-california-21-march-1976/>
- <https://kfoundation.org/transcript/scientists-seminar-7-brockwood-park-17-october-1974/>
- <https://kfoundation.org/transcript/dialogue-7-gstaad-18-july-1975/>
- <https://kfoundation.org/transcript/dialogue-3-ojai-california-8-april-1980/>
- <https://kfoundation.org/transcript/dialogue-11-brockwood-park-14-september-1980/>

## Transformation protocol

For every selected exchange, retain privately:

1. original question and conversational context;
2. K's first diagnostic move;
3. the psychological structure inferred from that move;
4. a surface-altered benchmark question with K vocabulary removed;
5. one matched control where a direct factual answer is appropriate;
6. expert annotation and acceptable alternative responses.

Do not evaluate a RAG system while allowing it to retrieve the source exchange
or a close duplicate. Use leave-one-source-out retrieval, and test an ungrounded
base-model condition separately. Otherwise the experiment measures retrieval
of K's response rather than spontaneous perception of the altered question.

