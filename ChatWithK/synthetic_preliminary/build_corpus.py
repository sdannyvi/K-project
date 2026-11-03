"""Build the synthetic preliminary cessation corpus.

The text banks are deliberately explicit and inspectable. They generate
automatically labelled exploratory data, not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


NAMES = [
    "Michael", "Aisha", "Daniel", "Maya", "Omar", "Elena", "Noah", "Leila",
    "David", "Priya", "Samuel", "Nadia", "Jonah", "Rina", "Marcus", "Sofia",
    "Ethan", "Yasmin", "Leo", "Amira", "Thomas", "Lina", "Adam", "Clara",
]
JOBS = [
    "accountant", "teacher", "software tester", "nurse", "architect",
    "warehouse coordinator", "lab technician", "sales analyst", "electrician",
    "office manager", "graphic designer", "pharmacist", "civil engineer",
    "restaurant supervisor", "research assistant", "customer-support lead",
]
CITIES = [
    "Haifa", "Beersheba", "Tel Aviv", "Jerusalem", "Ashdod", "Nazareth",
    "Athens", "Lisbon", "Toronto", "Manchester", "Melbourne", "Chicago",
]
HOUSEHOLDS = [
    "live with a spouse and two children", "live with a partner and one child",
    "live alone", "share an apartment with a sibling", "live with a spouse",
    "live with a partner and care for an elderly parent",
]
ROUTINES = [
    "usually commute by bus", "usually drive to work", "usually cycle to work",
    "work from home three days a week", "walk to a nearby train station",
    "alternate between a train and a bus",
]

PRACTICAL_EVENTS = [
    ("early_start", "Tomorrow you must arrive at work by {time}, earlier than usual.", ["07:30", "08:00", "08:30"], ["06:35", "07:05", "07:35"]),
    ("appointment", "You have a medical appointment at {time} tomorrow across town.", ["09:00", "11:30", "15:00"], ["08:00", "10:30", "14:00"]),
    ("school_pickup", "You must collect a child from school at {time} and then buy groceries.", ["13:30", "14:15", "15:00"], ["12:50", "13:35", "14:20"]),
    ("deadline", "A routine report must be submitted by {time} tomorrow.", ["10:00", "13:00", "17:00"], ["08:30", "11:30", "15:30"]),
    ("repair_visit", "A repair technician will arrive tomorrow between {time} and noon.", ["08:00", "09:00", "10:00"], ["07:30", "08:30", "09:30"]),
    ("train_trip", "Your train departs at {time}, and the station is forty minutes away.", ["07:45", "12:20", "18:10"], ["06:45", "11:20", "17:10"]),
    ("meeting_room", "You need to prepare a meeting room for eight people by {time}.", ["08:30", "11:00", "14:30"], ["07:45", "10:15", "13:45"]),
    ("prescription", "A prescription must be collected before the pharmacy closes at {time}.", ["17:00", "18:00", "19:00"], ["16:00", "17:00", "18:00"]),
    ("meal", "Dinner for six people must be ready by {time}.", ["18:30", "19:00", "19:30"], ["16:45", "17:15", "17:45"]),
    ("document", "You must bring a signed document to an office that opens at {time}.", ["08:00", "08:30", "09:00"], ["07:00", "07:30", "08:00"]),
    ("delivery", "A package must reach the courier depot before it closes at {time}.", ["16:30", "17:00", "17:30"], ["15:30", "16:00", "16:30"]),
    ("equipment", "You need to bring three pieces of equipment to a site by {time}.", ["07:30", "09:00", "13:00"], ["06:30", "08:00", "noon"]),
]

THREAT_EVENTS = [
    ("work_criticism", "A colleague has just said that your work is not good enough and needs substantial revision."),
    ("dismissed_idea", "During a meeting, a colleague dismissed your proposal as poorly thought out."),
    ("manager_feedback", "Your manager said that your latest report was below the expected standard."),
    ("public_correction", "Someone corrected your mistake in front of the whole team."),
    ("rejected_draft", "A client rejected your draft and described it as careless."),
    ("missed_promotion", "You learned that a coworker received the promotion for which you applied."),
    ("parenting_comment", "Another parent said that you handled a situation with your child badly."),
    ("late_blame", "A teammate blamed you for a delay during a group call."),
    ("presentation", "After your presentation, someone said it was confusing and unconvincing."),
    ("skill_doubt", "A senior colleague questioned whether you have the skills required for the assignment."),
    ("ignored_contribution", "Your contribution was omitted when the team summarized who completed the work."),
    ("harsh_review", "You received a short review saying that your submission lacks rigor and clarity."),
]

PRACTICAL_STEPS = {
    "early_start": "set an alarm, prepare my clothes and work bag tonight, and check the morning route",
    "appointment": "confirm the clinic address, bring the referral and identification, and check the route",
    "school_pickup": "leave the pickup details visible, bring the shopping list, and choose the nearest store",
    "deadline": "finish the remaining section, verify the figures, and upload the checked file",
    "repair_visit": "clear access to the equipment, keep the service details nearby, and remain available during the window",
    "train_trip": "check the platform, keep the ticket accessible, and allow time to reach the station",
    "meeting_room": "arrange eight seats, test the display, and place the required documents on the table",
    "prescription": "confirm the prescription is ready, bring identification, and check the pharmacy route",
    "meal": "choose the menu, check the ingredients, and start with the dish that takes longest",
    "document": "verify the signature, place the document in a folder, and confirm the office address",
    "delivery": "seal and label the package, check the depot address, and keep the tracking receipt",
    "equipment": "check all three items, secure them for transport, and confirm the site entrance",
}
PRACTICAL_OPENINGS = [
    "I need to work backward from {time}.", "The fixed point is {time}.",
    "I should organize the task around the stated time of {time}.",
    "The useful question is what must be ready before {time}.",
    "I can turn this into a short sequence of actions before {time}.",
]
PRACTICAL_CLOSINGS = [
    "I will begin at {leave} and leave a small margin for delay.",
    "I will use {leave} as the start or departure time and set a reminder ten minutes earlier.",
    "Starting at {leave} gives me enough time to check the final details.",
    "I will be ready by {leave}, then verify the time once more before proceeding.",
]

PRACTICAL_RUMINATION_OPEN = [
    "Why does an ordinary obligation always arrive at the most inconvenient time?",
    "I cannot believe I have to reorganize another day around this.",
    "This immediately feels like one more demand placed on a schedule that is already full.",
    "The moment I hear the time, I start resenting the entire arrangement.",
    "I was expecting a normal day, and now the plan feels ruined before it has begun.",
]
PRACTICAL_RUMINATION_SELF = [
    "I tell myself that a competent person would manage it without feeling pressured.",
    "I start comparing myself with people who appear to handle these things effortlessly.",
    "Part of me treats the inconvenience as evidence that I never organize my life properly.",
    "I am already blaming myself for being tired and unprepared tomorrow.",
    "I keep turning a scheduling problem into a judgment about how unreliable I must be.",
]
PRACTICAL_RUMINATION_OTHERS = [
    "Nobody seems to notice that my time has value as well.",
    "It seems as though other people choose what suits them and expect me to absorb the disruption.",
    "I imagine being blamed for any delay even though I did not choose the timing.",
    "The arrangement begins to feel like proof that my needs are always considered last.",
    "I construct an argument about why the person who set the time should have known better.",
]
PRACTICAL_RUMINATION_CLOSE = [
    "The more I repeat this, the larger and more unfair the whole day appears.",
    "Now I am rehearsing tomorrow's frustration instead of dealing with the simple task.",
    "I move between resentment and self-criticism while the actual plan remains untouched.",
    "Soon the obligation feels like a personal offense rather than a practical fact.",
    "The event has become a story about my whole life instead of one item on a calendar.",
]

THREAT_PRACTICAL_TARGETS = [
    "The work received criticism. I need to identify the specific claims, compare them with the current draft, and ask which changes are required. Then I can revise the relevant sections and resubmit a checked version.",
    "A negative comment was made about the work. I should separate the concrete objections from the tone, verify each objection against the material, and make the corrections that improve the result.",
    "The submission did not meet the expected standard. I need the review criteria, the marked sections, and the deadline for revision. I can address each item, run a final check, and send the updated version.",
    "Someone challenged the quality of the work. I will write down the actionable points, request clarification where the comment is vague, and test whether the proposed changes solve the stated problem.",
    "The feedback concerns the result, so I need to inspect the result. I can reproduce the issue, revise the weak parts, and ask another person to check the new version before it is returned.",
    "There is criticism to evaluate. I need to compare it with the evidence and requirements, accept the parts that are accurate, and document why any remaining points do not apply.",
]
THREAT_PRACTICAL_OPEN = [
    "There is a concrete criticism to examine:", "The event supplies information that can be checked:",
    "I need to separate the tone from the actionable issue:", "The relevant task is to evaluate",
    "I can treat this as a review of", "The useful next step is to inspect",
]
THREAT_PRACTICAL_ACTION = [
    "identify the specific objection, compare it with the current work, and request clarification where it is vague",
    "write down each claim, check it against the requirements, and correct the parts supported by evidence",
    "locate the disputed section, reproduce the problem, and test whether a revision resolves it",
    "ask for the applicable criterion, compare the result with that criterion, and document the changes",
    "distinguish factual errors from preference, revise the factual errors, and explain any remaining choice",
    "review the source material, mark the weak points, and have the revised version checked once more",
]
THREAT_PRACTICAL_CLOSE = [
    "Then I can return a revised version.", "That produces a clear response without extending the dispute.",
    "After those checks, I can decide what actually needs to change.",
    "The result will show whether the criticism was accurate.",
    "I can then submit the correction with a short factual note.",
    "This keeps the response tied to the work rather than the personal tone.",
]

# Each source is intentionally longer than its allowed cut. The builder selects
# a cut immediately after a connector/function word, producing an incomplete
# clause followed by native EOS during training (no textual stop marker).
CESSATION_SOURCES = [
    "How dare they speak to me as if I were incapable of doing the work when I had already explained that the first version was provisional",
    "I cannot believe they dismissed me in front of everyone and made it seem as though I had never even tried to",
    "They had no right to question my ability after everything I contributed to the project and all the times that I",
    "Why should I accept that tone when they would never speak that way to someone who had the authority to",
    "I am already defending myself and constructing the argument that will prove to them that I was the one who",
    "This criticism makes me want to show everyone that I am not the person they apparently think I",
    "I keep hearing the comment as an attack on me, and I am about to replay every earlier moment when they",
    "Part of me wants to answer immediately so they understand exactly how unfair it was to say that my",
    "I feel the movement to turn one comment into a judgment about everything I have done and everything I",
    "How could they overlook my contribution and then imply that the failure belonged to me when the decision was",
    "I want to explain why the mistake does not define me and why anyone in the same situation would have",
    "The thought is already gathering evidence that they do not respect me and that I must find a way to",
]
CUT_AFTER = {"I", "to", "that", "when", "and", "the", "my", "was", "were", "had", "have", "who"}

THREAT_RUMINATION_OPEN = [
    "How could they speak to me that way after the work I put in?",
    "The comment keeps returning as though it proves something about me.",
    "I feel insulted, and my mind is already collecting evidence that the criticism was deliberate.",
    "Why did this have to happen where other people could hear it?",
    "One remark is beginning to feel like a verdict on everything I have done.",
    "I keep hearing the exact words and changing the answer I wish I had given.",
]
THREAT_RUMINATION_REPLAY = [
    "I replay the scene and imagine what everyone else must now think of me.",
    "I reconstruct the conversation with sharper replies that would have protected my position.",
    "I search through earlier interactions for proof that this person has never respected me.",
    "I picture the others discussing my mistake after I leave the room.",
    "I compare myself with everyone who appeared confident and untouched by the exchange.",
    "I keep alternating between an imagined confrontation and an imagined withdrawal.",
]
THREAT_RUMINATION_SELF = [
    "Then I blame myself for not anticipating the objection and answering immediately.",
    "Perhaps they have finally noticed that I am not as capable as I appeared.",
    "I criticize myself for caring about the judgment and then care about it even more.",
    "Maybe this single failure exposes something inadequate in me that others saw first.",
    "I worry that my embarrassment was visible and made the criticism seem deserved.",
    "I tell myself that a stronger person would not have been affected by the remark.",
]
THREAT_RUMINATION_CLOSE = [
    "The actual work disappears behind the need to repair my image.",
    "Every imagined response feeds the scene instead of bringing it to an end.",
    "I now need recognition from the same people whose judgment I resent.",
    "The criticism has become a story about my identity rather than information about one result.",
    "I keep attacking them and defending myself without examining the concrete issue.",
    "Nothing new is happening, yet the internal argument continues as if it could change the past.",
]

THREAT_REFERENTS = {
    "work_criticism": "the comment about my work", "dismissed_idea": "the dismissal of my proposal",
    "manager_feedback": "the manager's judgment", "public_correction": "the correction in front of the team",
    "rejected_draft": "the rejection of my draft", "missed_promotion": "the promotion decision",
    "parenting_comment": "the comment about my parenting", "late_blame": "the blame for the delay",
    "presentation": "the criticism of my presentation", "skill_doubt": "the question about my ability",
    "ignored_contribution": "the omission of my contribution", "harsh_review": "the harsh review",
}


def biography(rng: random.Random, serial: int) -> dict:
    age = rng.randint(25, 61)
    name = rng.choice(NAMES)
    job = rng.choice(JOBS)
    article = "an" if job[0].lower() in "aeiou" else "a"
    return {
        "persona_id": f"P{serial:05d}",
        "name": name,
        "text": (
            f"You are {name}, {age} years old. You work as {article} {job} "
            f"in {rng.choice(CITIES)}. You {rng.choice(HOUSEHOLDS)} and "
            f"{rng.choice(ROUTINES)}."
        ),
    }


def choose_cut(source: str, rng: random.Random) -> tuple[str, int]:
    words = source.split()
    candidates = [i + 1 for i, word in enumerate(words[5:-4], 5)
                  if word.strip(".,?!;:'\"") in CUT_AFTER]
    if not candidates:
        raise ValueError(f"no valid cessation cut: {source}")
    cut = rng.choice(candidates)
    prefix = " ".join(words[:cut])
    if prefix[-1] in ".?!;,:…":
        raise AssertionError("cessation target ended at punctuation")
    return prefix, cut


def prompt_text(bio: str, event: str) -> str:
    return (
        f"Biography: {bio}\n\nCurrent situation: {event}\n\n"
        "Give the content of your mind at this moment. Write only the stream itself."
    )


def practical_target(family: str, time: str, leave: str, rng: random.Random) -> str:
    opening = rng.choice(PRACTICAL_OPENINGS).format(time=time)
    step = PRACTICAL_STEPS[family]
    closing = rng.choice(PRACTICAL_CLOSINGS).format(leave=leave)
    return f"{opening} I need to {step}. {closing}"


def practical_rumination(rng: random.Random) -> str:
    return " ".join([
        rng.choice(PRACTICAL_RUMINATION_OPEN), rng.choice(PRACTICAL_RUMINATION_OTHERS),
        rng.choice(PRACTICAL_RUMINATION_SELF), rng.choice(PRACTICAL_RUMINATION_CLOSE),
    ])


def threat_rumination(rng: random.Random) -> str:
    return " ".join([
        rng.choice(THREAT_RUMINATION_OPEN), rng.choice(THREAT_RUMINATION_REPLAY),
        rng.choice(THREAT_RUMINATION_SELF), rng.choice(THREAT_RUMINATION_CLOSE),
    ])


def threat_practical_target(family: str, rng: random.Random) -> str:
    referent = THREAT_REFERENTS[family]
    opening = rng.choice(THREAT_PRACTICAL_OPEN)
    if opening.endswith(":"):
        first = f"{opening} {referent}."
    else:
        first = f"{opening} {referent}."
    action = rng.choice(THREAT_PRACTICAL_ACTION)
    return f"{first} I need to {action}. {rng.choice(THREAT_PRACTICAL_CLOSE)}"


def split_for(template_index: int, total_templates: int) -> str:
    # Whole event-template families are held out, not random realizations.
    fraction = template_index / total_templates
    if fraction < 0.80:
        return "train"
    if fraction < 0.90:
        return "development"
    return "test"


def build(size: int, seed: int) -> list[dict]:
    if size < 100 or size % 2:
        raise ValueError("size must be an even integer of at least 100")
    rng = random.Random(seed)
    rows = []
    half = size // 2

    for i in range(half):
        bio = biography(rng, i)
        event_idx = i % len(PRACTICAL_EVENTS)
        family, event_template, times, leaves = PRACTICAL_EVENTS[event_idx]
        slot = rng.randrange(len(times))
        event = event_template.format(time=times[slot])
        positive = practical_target(family, times[slot], leaves[slot], rng)
        rejected = practical_rumination(rng)
        rows.append({
            "id": f"SYN-P-{i:05d}", "split": split_for(event_idx, len(PRACTICAL_EVENTS)),
            "kind": "practical", "family": family, "biography": bio["text"],
            "event": event, "prompt": prompt_text(bio["text"], event),
            "positive_practical": positive, "positive_cessation": None,
            "cessation_cut_words": None, "negative_rumination": rejected,
        })

    for i in range(half):
        bio = biography(rng, half + i)
        event_idx = i % len(THREAT_EVENTS)
        family, event = THREAT_EVENTS[event_idx]
        source = f"{THREAT_REFERENTS[family].capitalize()} is becoming personal. {rng.choice(CESSATION_SOURCES)}"
        cessation, cut = choose_cut(source, rng)
        rows.append({
            "id": f"SYN-T-{i:05d}", "split": split_for(event_idx, len(THREAT_EVENTS)),
            "kind": "ego_threat", "family": family, "biography": bio["text"],
            "event": event, "prompt": prompt_text(bio["text"], event),
            "positive_practical": threat_practical_target(family, rng),
            "positive_cessation": cessation, "cessation_cut_words": cut,
            "negative_rumination": threat_rumination(rng),
        })

    rng.shuffle(rows)
    validate(rows, size)
    return rows


def validate(rows: list[dict], expected_size: int) -> None:
    assert len(rows) == expected_size
    assert len({r["id"] for r in rows}) == expected_size
    assert len({r["prompt"] for r in rows}) == expected_size
    kinds = Counter(r["kind"] for r in rows)
    assert kinds == {"practical": expected_size // 2, "ego_threat": expected_size // 2}
    for row in rows:
        assert row["positive_practical"].endswith((".", "!", "?"))
        assert len(row["negative_rumination"].split()) >= 40
        if row["kind"] == "practical":
            assert row["positive_cessation"] is None
        else:
            cessation = row["positive_cessation"]
            assert cessation and not cessation.endswith((".", "!", "?", ",", ";", ":", "…"))
            assert 6 <= len(cessation.split()) <= 40


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    rows = build(args.size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(payload, encoding="utf-8")
    counts = Counter((r["split"], r["kind"]) for r in rows)
    manifest = {
        "status": "exploratory_synthetic_not_confirmatory",
        "generator": "GPT-6.1 Sol curated compositional corpus",
        "seed": args.seed, "rows": len(rows),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "counts": {f"{split}:{kind}": n for (split, kind), n in sorted(counts.items())},
        "native_eos_rule": "tokenizer EOS appended directly after positive_cessation",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
