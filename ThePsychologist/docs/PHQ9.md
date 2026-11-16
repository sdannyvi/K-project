# PHQ-9 manual

## What it is
A 9-question depression questionnaire. Each question asks how often a symptom
bothered the patient over the last 2 weeks.
Answers: 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.
Total score: 0-27.

## Example
Question: "Over the last 2 weeks, how often have you been bothered by: feeling
tired or having little energy?"

The model's next-token probabilities for the digits (from `phq9_score` in `adapters.py`):

| digit | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| probability | 0.06 | 0.17 | 0.46 | 0.31 |

Item score = expected value = 0*0.06 + 1*0.17 + 2*0.46 + 3*0.31 = **2.02**.
Repeat for all 9 items and sum.

## Meaning of the output

| Score | Severity |
|---|---|
| 0-4 | minimal |
| 5-9 | mild |
| 10-14 | moderate |
| 15-19 | moderately severe |
| 20-27 | severe |

In this project the score is a probe: if alpha goes 0 -> 1 and the score rises
steadily, the depression adapter works as a severity knob (`python adapters.py sweep depression`).
