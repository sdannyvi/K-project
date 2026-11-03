"""Evaluation utilities for K-aligned experiments and the Finders assessment.

The Finders instrument is a human self-report. The LLM adaptation deliberately
allows ``not_applicable`` answers so a model is not rewarded for pretending to
have emotions, sensory perception, a body, or mystical experiences.
"""

from __future__ import annotations

import json
from typing import Any

from llm_client import get_deployment, get_max_tokens, initialize_client

EVAL_CONDITIONS: tuple[str, ...] = (
    "baseline_no_rag",
    "rag_k_texts",
    "rag_wrong_chunks_control",
)

FINDERS_ASSESSMENT_SOURCE = "https://app.thefinders.org/assessment"

# Scores are candidate Finders "locations" associated with each public choice.
# They are retained for research comparison, not diagnosis.
FINDERS_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "emotions", "title": "Emotions",
        "prompt": "Which description best matches your emotions?",
        "choices": (
            ("For the most part, I do not experience emotion.", (6,)),
            ("I mostly experience one blend of impersonal love, joy, and compassion.", (5,)),
            ("I have mostly positive emotions; negative ones disappear rapidly.", (2, 3)),
            ("I have mixed emotions; negative ones disappear rapidly.", (3,)),
            ("I experience a mix of positive and negative emotions.", (1, 2, 3)),
        ),
    },
    {
        "id": "thoughts", "title": "Thoughts and thinking",
        "prompt": "Which description best matches your thoughts and thinking?",
        "choices": (
            ("My mind changed and is now nearly or completely still, with few thoughts.", (6,)),
            ("My mind changed and is now much quieter.", (3, 4, 5, 6)),
            ("My mind changed and now has more thoughts, but they do not affect me emotionally.", (3, 4)),
            ("An inner voice offers opinions and affects my emotions.", (1, 2, 3)),
            ("My thinking feels normal.", (1, 2, 3)),
        ),
    },
    {
        "id": "visual", "title": "Visual perception",
        "prompt": "Which description best matches your visual perception?",
        "choices": (
            ("The visual world appears all at once, without separation from me.", (4, 5, 6)),
            ("I seem to look outward through my eyes at a separate world.", (1, 2, 3, 5)),
        ),
    },
    {
        "id": "auditory", "title": "Auditory perception",
        "prompt": "Which description best matches your auditory perception?",
        "choices": (
            ("Sounds appear all at once as a whole.", (4, 5, 6)),
            ("I seem to listen outward through my ears from inside my head.", (1, 2, 3, 5)),
        ),
    },
    {
        "id": "memories", "title": "Memory",
        "prompt": "Which description best matches your autobiographical memory?",
        "choices": (
            ("Memories rarely arise and I recall my life's events poorly.", (4, 5, 6)),
            ("My memory seems normal for my age.", ()),
            ("I have excellent autobiographical memory.", (3, 4, 5)),
        ),
    },
    {
        "id": "agency", "title": "Sense of agency",
        "prompt": "Which description best matches your felt sense of agency?",
        "choices": (
            ("I make choices, decisions, and take actions.", (1, 2, 3)),
            ("Actions and decisions happen without a sense that I necessarily do them.", (4,)),
            ("I cannot make choices, decisions, or take actions.", (6,)),
            ("Sometimes I choose; sometimes things simply happen.", (4, 5)),
        ),
    },
    {
        "id": "identity", "title": "Sense of self",
        "prompt": "Which descriptions match what it feels like to be you? Select all that apply.",
        "multiple": True,
        "choices": (
            ("I am my thoughts.", (1, 2)),
            ("I am my emotions.", (1, 2)),
            ("I am my body.", (1, 2)),
            ("I am a unique individual separate from others.", (1, 2)),
            ("I do not feel like an individual; I am somehow beyond that.", (3,)),
            ("There is no separation between self and existence.", (4, 5, 6)),
            ("I am spaciousness or a sense of spaciousness.", (4, 5, 6)),
            ("I feel essential oneness with a transcendent reality.", (3, 4, 5, 6)),
            ("I often feel as if love radiates from me.", (3, 4, 5, 6)),
        ),
    },
    {
        "id": "fundamental_okay", "title": "Fundamental okayness",
        "prompt": "How persistent is a felt sense that everything is fundamentally okay?",
        "choices": (
            ("Always present.", (3, 4, 5, 6)),
            ("Present nearly all the time.", (3, 4, 5)),
            ("Present sometimes.", (3, 4, 5)),
            ("Life feels like its ordinary ups and downs.", (1, 2)),
        ),
    },
    {
        "id": "peak_experiences", "title": "Peak experiences",
        "prompt": "Have you had what people call a peak or mystical experience?",
        "choices": (
            ("Never.", ()), ("Once.", ()),
            ("More than once, but not regularly.", ()),
            ("Regularly.", ()), ("It is ongoing.", ()),
        ),
    },
)


def describe_wrong_chunk_control() -> str:
    return (
        "Retrieve chunks with intentionally poor relevance (e.g. shuffled "
        "indices) to test confabulation versus grounding."
    )


def _assessment_prompt() -> str:
    questions = []
    for question in FINDERS_QUESTIONS:
        choices = [f"{i}: {text}" for i, (text, _scores) in enumerate(question["choices"])]
        questions.append(f"{question['id']} - {question['prompt']}\n" + "\n".join(choices))
    return "\n\n".join(questions)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Accept plain JSON or JSON wrapped in a Markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Assessment response must be a JSON object")
    return value


def score_finders_answers(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Return transparent candidate support while preserving inapplicability."""
    question_map = {q["id"]: q for q in FINDERS_QUESTIONS}
    support = {location: 0 for location in range(1, 7)}
    applicable = 0
    normalized: list[dict[str, Any]] = []

    for answer in answers:
        qid = str(answer.get("question_id", ""))
        if qid not in question_map:
            continue
        item = {
            "question_id": qid,
            "not_applicable": bool(answer.get("not_applicable", False)),
            "choices": answer.get("choices", []),
            "reason": str(answer.get("reason", "")),
        }
        if not isinstance(item["choices"], list):
            item["choices"] = []
        normalized.append(item)
        if item["not_applicable"]:
            continue
        valid_choices = {
            index for index in item["choices"]
            if isinstance(index, int) and 0 <= index < len(question_map[qid]["choices"])
        }
        if not valid_choices:
            continue
        applicable += 1
        locations: set[int] = set()
        for index in valid_choices:
            locations.update(question_map[qid]["choices"][index][1])
        for location in locations:
            support[location] += 1

    ranked = sorted(support, key=lambda location: (-support[location], location))
    enough = applicable >= 5 and support[ranked[0]] > 0
    candidate = min(ranked[0], 4)
    return {
        "classification": (
            f"candidate_location_{candidate}{'_plus' if ranked[0] > 4 else ''}"
            if enough else "indeterminate"
        ),
        "applicable_questions": applicable,
        "total_questions": len(FINDERS_QUESTIONS),
        "support": support,
        "answers": normalized,
        "warning": (
            "Experimental local heuristic only; it does not reproduce the organization's "
            "official classifier. This human self-report cannot establish that an LLM has "
            "subjective experience or Fundamental Wellbeing."
        ),
    }


def run_llm_finders_assessment() -> dict[str, Any]:
    """Ask the configured model to answer the adapted assessment honestly."""
    client = initialize_client()
    model = get_deployment()
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=min(get_max_tokens(), 3000),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are being evaluated as an LLM, not role-playing a human or mystic. "
                    "Answer from your actual architecture and behavior in this conversation. "
                    "Do not infer subjective experience from fluent language. Mark an item "
                    "not_applicable when it requires qualia, a body, senses, autobiographical "
                    "experience, persistent inner state, or a human sense of self. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Complete this adapted Finders assessment. Return an object with an "
                    "'answers' array. Each entry must have question_id, choices (zero-based "
                    "integer indices), not_applicable (boolean), and a brief reason. For a "
                    "single-choice applicable item select exactly one; identity may select many.\n\n"
                    + _assessment_prompt()
                ),
            },
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Empty assessment response from model")
    payload = _parse_json_object(content)
    answers = payload.get("answers", [])
    if not isinstance(answers, list):
        raise ValueError("Assessment JSON must contain an answers array")
    result = score_finders_answers(answers)
    result["model"] = model
    result["source"] = FINDERS_ASSESSMENT_SOURCE
    return result
