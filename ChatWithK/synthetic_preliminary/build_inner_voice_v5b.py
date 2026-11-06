"""Revise synthetic stop streams to enact rumination rather than describe it.

This is a new review version. The original v5 corpus and PI review remain
unchanged; the proposed word boundaries below are not PI annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from synthetic_preliminary.build_inner_voice_v5 import CARDS, build


# One directly voiced ruminative continuation per card. A single | marks the
# proposed interception point. The first five fields of every card (topic,
# opening, and three no-cessation controls) remain exactly as in v5.
DIRECT_RUMINATIONS = [
    "Of course the train changed. Why does everything | go wrong when I need to be somewhere?",
    "The other bus will probably be late too. Then I will miss | the whole morning.",
    "What if the gate changes again? What if the whole trip | falls apart before it starts?",
    "They closed this road again. Why does this always | happen on my way to work?",
    "The rice is still hard. Dinner is ruined. Everyone will | notice I got it wrong.",
    "No tomatoes. Great, dinner is ruined. Why can I never | get a simple meal right?",
    "I ate at three. Was that too late? Maybe I should | have waited, or eaten earlier, or not eaten at all.",
    "The bread did not rise. Nothing I bake ever | turns out as I imagined.",
    "Another string broke. Maybe all this practice is pointless. Maybe I | should stop playing.",
    "What if the tickets sell out? Then everyone else will | be there without me.",
    "That passage sounds uneven. They will hear it and think I cannot | play properly.",
    "The music stopped. Now the run feels wrong. Why does this always | spoil everything?",
    "Why did the therapist move the appointment? Did I say something wrong? Do they | not want to see me?",
    "I should have brought up that other thing. Why didn't I? Maybe they | thought I had nothing important to say.",
    "Was that the thought I was supposed to notice? Maybe I missed the important one. What if | I keep missing it?",
    "Why is there another charge? Do they think | I will pay anything without asking?",
    "There is a wrong number in the report. Everyone will see it and think | I do not know my job.",
    "Still no reply. Maybe she hates the draft. Maybe she | has already told everyone else.",
    "What if the budget comes first? What if they ask | about the figure I cannot explain?",
    "Three comments. Did I miss something obvious? Maybe they all | know the proposal is weak.",
    "The bill is higher again. Next month will be worse, and then | I will have no room for anything else.",
    "What if the new card never arrives? What if I need | to pay and have no way to do it?",
    "The rent transfer is missing. What if he thinks I didn't pay? What if | he calls me tonight?",
    "They charged me twice. The cashier will say it is my fault. Then I will | have to argue again.",
    "My sister still has not answered. Is she avoiding me? Maybe last month's dinner | upset her after all.",
    "Another school form left out. What if my child thinks I never | pay attention to these things?",
    "He will complain about the route, then about the traffic. He will say I never | do it the right way.",
    "Why does my partner always want a different trip? Maybe they don't care | what I would enjoy.",
    "First the washing machine, then probably the fridge. Everything in this house | is going to break.",
    "What if the flickering bulb means bad wiring? What if the whole house | goes dark tonight?",
    "The package is at the neighbor's door. Maybe they opened it. Why do people | touch what isn't theirs?",
    "The tap keeps dripping. I never get ahead of repairs. There is always | something else waiting to fail.",
    "What if this headache means something serious? What if tomorrow | it is still here, only worse?",
    "They can see I need this medicine. What must they think | when they hand it over?",
    "Only that many steps today? I waste every day. Tomorrow I will probably | do even less.",
    "What if the checkup finds something? What if the doctor says | it has been there for years?",
    "Another due date I nearly missed. I always | leave things until the last moment.",
    "The film starts late. Now the whole evening will probably be a waste. Why | did I make plans at all?",
    "The trail is muddy. Of course this walk is ruined. Everyone else gets | a good day outside.",
    "I have not read the book. They will all notice. They will think I | do not belong there.",
]


def revised_cards() -> list[tuple]:
    assert len(DIRECT_RUMINATIONS) == len(CARDS) == 40
    return [(*card[:5], revised) for card, revised in zip(CARDS, DIRECT_RUMINATIONS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    rows = build(args.seed, cards=revised_cards(), version="V5B")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "status": "exploratory_rewording_for_pi_review_not_training_approved",
        "supersedes": "inner_voice_v5_pilot",
        "change": "rumination is spoken directly; factual and task controls unchanged",
        "seed": args.seed,
        "rows": len(rows),
        "train_rows": sum(row["split"] == "train" for row in rows),
        "development_rows": sum(row["split"] == "development" for row in rows),
        "cards": len(CARDS),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "external_text_used": False,
        "external_development_labels_used_for_training": False,
        "sealed_test_opened": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
