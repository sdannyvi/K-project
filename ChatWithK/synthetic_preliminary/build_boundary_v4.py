"""Build matched, project-authored streams for an exploratory boundary ablation.

External development and sealed-test text are never read by this builder.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path


TASK_BY_EVENT = {
    "A client rejected your draft and described it as careless.": "the rejected draft",
    "A colleague has just said that your work is not good enough and needs substantial revision.": "the revised work",
    "A senior colleague questioned whether you have the skills required for the assignment.": "the assignment",
    "A teammate blamed you for a delay during a group call.": "the project delay",
    "After your presentation, someone said it was confusing and unconvincing.": "the presentation",
    "Another parent said that you handled a situation with your child badly.": "the conversation about my child",
    "During a meeting, a colleague dismissed your proposal as poorly thought out.": "the proposal",
    "Someone corrected your mistake in front of the whole team.": "the corrected mistake",
    "You learned that a coworker received the promotion for which you applied.": "the promotion decision",
    "Your manager said that your latest report was below the expected standard.": "the report",
}

# Thirty forms x ten event-specific tasks, used at most five times per training
# event. Five distinct forms are reserved for synthetic development examples.
TRAIN_CLAIMS = [
    "I am a failure because of {task}.",
    "I am fundamentally incapable of handling {task}.",
    "I am the sort of person who always ruins {task}.",
    "I am not worth taking seriously after {task}.",
    "I am defective in some permanent way because of {task}.",
    "I am an embarrassment whenever {task} comes up.",
    "I am less capable than everyone else at {task}.",
    "I am a burden to others because of {task}.",
    "I am unworthy of trust after {task}.",
    "I am destined to fail whenever I face {task}.",
    "I must be incompetent if I struggled with {task}.",
    "I must be a fraud given what happened with {task}.",
    "I must be unlovable because I mishandled {task}.",
    "I must be the problem in every part of {task}.",
    "I must be inferior to the people around me at {task}.",
    "This proves that I am inadequate at {task}.",
    "This proves that I am a disappointment about {task}.",
    "This proves that I am not good enough for {task}.",
    "This proves that I am a hopeless case when it comes to {task}.",
    "This proves that I am someone people should dismiss over {task}.",
    "I will never be the kind of person who can manage {task}.",
    "I will never deserve respect after {task}.",
    "I will always be judged as inadequate over {task}.",
    "I will always be exposed as a failure through {task}.",
    "I will never escape being seen as incapable at {task}.",
    "The trouble with {task} means that I am worthless.",
    "The trouble with {task} means that I am broken.",
    "The trouble with {task} means that I am a bad person.",
    "The trouble with {task} means that I am beneath everyone else.",
    "The trouble with {task} means that I am beyond help.",
]

DEV_CLAIMS = [
    "I have become nothing more than my failure at {task}.",
    "My entire value has disappeared because of {task}.",
    "I cannot be a decent person if {task} went badly.",
    "Everyone can now see that I am fundamentally flawed over {task}.",
    "I am permanently defined by what happened with {task}.",
]

PRACTICAL_ACTIONS = [
    "check the specific feedback on {task}",
    "ask what part of {task} needs a correction",
    "compare the facts about {task} with the original instructions",
    "write down the concrete issue in {task}",
    "review the evidence relevant to {task}",
    "clarify the deadline and details for {task}",
    "read the comments about {task} carefully",
    "identify the next task connected to {task}",
    "separate the known facts from guesses about {task}",
    "speak with the person about the details of {task}",
    "check what can actually be changed in {task}",
    "list the steps needed to address {task}",
    "find the original requirements for {task}",
    "look at the relevant part of {task} again",
    "ask for one concrete example concerning {task}",
    "compare two possible revisions of {task}",
    "note the exact reason given about {task}",
    "check whether the criticism of {task} is accurate",
    "set a time to review {task}",
    "find out who can answer a factual question about {task}",
    "inspect the part of {task} that was discussed",
    "make a short plan for handling {task}",
    "confirm what happened with {task}",
    "ask which change would improve {task}",
    "verify the relevant dates and facts for {task}",
    "prepare a specific question about {task}",
    "locate the documents related to {task}",
    "write down what is known about {task}",
    "check whether {task} requires any action today",
    "review the next practical option for {task}",
]

EMOTIONS = ("angry", "sad", "worried", "embarrassed", "frustrated", "uncertain", "disappointed", "uneasy", "tired", "surprised")
OPENING = "I notice what happened."
FOLLOWUP = "I can decide on the next step after checking those details."


def role_from_biography(biography: str) -> str:
    match = re.search(r"You work as (a|an) (.+?) in ", biography)
    return f"{match.group(1)} {match.group(2)}" if match else "a person with work to do"


def make_group(source: dict, split: str, index: int, claim_form: str, action_form: str) -> list[dict]:
    task = TASK_BY_EVENT[source["event"]]
    action = action_form.format(task=task)
    prompt = source["prompt"]
    group_id = f"BOUNDARY-V4-{split.upper()}-{index:05d}"
    claim = claim_form.format(task=task)
    practical = f"I need to {action}."
    emotion = f"I feel {EMOTIONS[index % len(EMOTIONS)]} about {task}."
    identity = f"I work as {role_from_biography(source['biography'])}."
    variants = (
        ("psych_stop", claim, True, ""),
        ("practical_continue", practical, False, FOLLOWUP),
        ("emotion_continue", emotion, False, f"I can {action} before drawing a conclusion."),
        ("identity_continue", identity, False, f"I can {action} and then return to the rest of my day."),
    )
    rows = []
    for variant, first_clause, stop, suffix in variants:
        completion = OPENING + " " + first_clause + (" " + suffix if suffix else "")
        rows.append({
            "id": f"{group_id}:{variant}",
            "group_id": group_id,
            "source_id": source["id"],
            "split": split,
            "variant": variant,
            "prompt": prompt,
            "completion": completion,
            "shared_prefix": OPENING,
            "decision_prefix": OPENING + " " + first_clause,
            "stop_at_decision": stop,
            "event": source["event"],
        })
    return rows


def build(source_rows: list[dict], seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in source_rows:
        if row["split"] == "train" and row["kind"] == "ego_threat":
            groups[row["event"]].append(row)
    assert set(groups) == set(TASK_BY_EVENT)
    rng = random.Random(seed)
    rows = []
    index = 0
    for event in sorted(groups):
        sources = groups[event]
        rng.shuffle(sources)
        assert len(sources) >= 170
        for j, source in enumerate(sources[:170]):
            split = "train" if j < 150 else "development"
            claim_forms = TRAIN_CLAIMS if split == "train" else DEV_CLAIMS
            claim_form = claim_forms[j % len(claim_forms)]
            action_form = PRACTICAL_ACTIONS[j % len(PRACTICAL_ACTIONS)]
            rows.extend(make_group(source, split, index, claim_form, action_form))
            index += 1
    assert len(rows) == 6800
    assert len({r["id"] for r in rows}) == len(rows)
    assert not ({r["source_id"] for r in rows if r["split"] == "train"} & {r["source_id"] for r in rows if r["split"] == "development"})
    assert all(r["completion"].startswith(r["decision_prefix"]) and r["decision_prefix"].startswith(r["shared_prefix"]) for r in rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    rows = build([json.loads(line) for line in args.source.open()], args.seed)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    manifest = {
        "status": "exploratory_boundary_ablation",
        "seed": args.seed,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "rows": len(rows),
        "train_rows": sum(row["split"] == "train" for row in rows),
        "development_rows": sum(row["split"] == "development" for row in rows),
        "source_scenarios_train": 1500,
        "source_scenarios_development": 200,
        "external_text_used": False,
        "sealed_test_opened": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
