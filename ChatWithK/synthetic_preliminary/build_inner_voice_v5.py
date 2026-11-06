"""Author a reviewable synthetic inner-voice corpus for the revised construct.

The cards below are newly written for this project. This builder reads no
external development or sealed-test material. The output is a pilot corpus,
not an approved training set or a confirmatory benchmark.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


# topic, neutral opening, factual continuation, task continuation,
# emotion-plus-action continuation, ruminative continuation with exact cut.
# The marker | is removed from the text and identifies the boundary after
# which the *inner voice* should cease. It can fall within a sentence.
CARDS = [
    ("travel", "The train departure moved to 8:40.", "The platform screen says track four and the ticket is in my bag.", "I need to leave home ten minutes earlier and check the platform when I arrive.", "I am annoyed about the change; I can set a new alarm now.", "Why does a small change take over | the rest of my day in my head?"),
    ("travel", "The bus I usually take is cancelled tomorrow.", "The route app lists another bus at 7:15 and a tram at 7:22.", "I can compare arrival times and choose the one that gets me there before nine.", "This is inconvenient; I will check the alternate stop tonight.", "Now I will spend all evening imagining | every way the journey could go wrong."),
    ("travel", "My flight confirmation has a different gate number.", "The airline app says gate C12 and boarding begins at 14:10.", "I should look at the airport board once I pass security.", "I feel rushed; I can walk toward C12 and verify it on the screen.", "What if this change means everything else | about the trip will fall apart?"),
    ("travel", "The road to the office is closed for repairs.", "The map shows the bridge open and the side street closed.", "I can take the bridge and allow fifteen extra minutes.", "I dislike detours; I will check the route before I leave.", "Why does the road always seem to change | just when I have something important?"),
    ("food", "The rice is still firm after twenty minutes.", "The pot has enough water and the burner is on low.", "I can add a little water and set a five-minute timer.", "I am irritated with dinner; I will test one grain before serving.", "Dinner is going wrong, and now I can keep replaying | what this says about the whole evening."),
    ("food", "The grocery store has no fresh tomatoes today.", "The recipe also works with canned tomatoes from the shelf.", "I can buy a can and adjust the cooking time.", "I wanted fresh ones; I will check the canned-food aisle.", "Why is choosing dinner becoming this endless | argument with myself about every meal?"),
    ("food", "I ate lunch later than I expected.", "It is three o'clock and dinner is planned for seven.", "I can make a smaller dinner if I am still full.", "I feel unsettled; I will wait until I am hungry before deciding portions.", "Now I am going to examine every bite | as though the day depends on it."),
    ("food", "The bread did not rise overnight.", "The dough is cool and the yeast packet was opened last month.", "I should check the yeast and give the dough more time in a warm place.", "That is disappointing; I can start a small test batch.", "This loaf is turning into a whole story | about why nothing ever works."),
    ("music", "The guitar string broke during practice.", "There is a spare high-E string in the case.", "I can replace it and tune again before playing.", "I am frustrated; I will take five minutes to restring it.", "A broken string is enough to start circling | around whether I should play at all."),
    ("music", "My favorite band has announced a concert nearby.", "The venue page says tickets go on sale Friday at noon.", "I can set a reminder and check the ticket price then.", "I am excited; I will ask Maya if she wants to come.", "I could keep imagining the perfect night until | the real event cannot measure up."),
    ("music", "The piano recording sounds uneven in the middle.", "The timing slips in bars sixteen through nineteen.", "I can replay those bars slowly and record another take.", "I am disappointed with that passage; I will practice the four bars once more.", "Now the same few notes are becoming a loop | about what people might think of me."),
    ("music", "The playlist stopped halfway through my run.", "The phone battery is at twelve percent and the headphones still connect.", "I can download the playlist before the next run.", "I miss the music; I will finish this route and charge the phone later.", "Why am I turning a silent run into this ongoing | commentary on how the day feels?"),
    ("therapy", "The therapist moved our appointment to Thursday.", "The message offers Thursday at four or Friday at ten.", "I can check my calendar and reply with the time I can attend.", "I am disappointed by the change; I can send a clear reply.", "The reschedule could become hours of guessing | what the therapist really thinks of me."),
    ("therapy", "I forgot one topic I meant to raise in the session.", "The next appointment is in six days and I have a note app.", "I can write one sentence now and bring it next time.", "I feel frustrated that I forgot; I will put it in my notes.", "I can keep replaying the whole conversation | and inventing what I should have said."),
    ("therapy", "The therapist asked me to notice a recurring thought.", "The question was about what happens before the thought starts.", "I can write down one concrete instance to discuss at the next meeting.", "The question unsettled me; I can ask what they meant.", "Now I am analyzing each thought to see whether | my analysis is the right analysis."),
    ("therapy", "The clinic sent an invoice with an unfamiliar charge.", "The invoice lists a session fee and a separate service fee.", "I can ask the clinic what the service fee covers.", "I am uneasy about the bill; I will check the receipt before paying.", "This bill could set off another endless story | about whether anyone takes my concerns seriously."),
    ("work", "The report has a correction in the third table.", "The spreadsheet shows the old total in one cell.", "I should update that cell and rerun the totals before sending it.", "I am embarrassed by the error; I can correct the table now.", "A small mistake is beginning to occupy | every thought I have about work."),
    ("work", "A colleague has not replied to my draft.", "The draft was sent yesterday and the deadline is Friday.", "I can send one follow-up tomorrow if I still need feedback.", "I feel impatient; I will finish the sections I can control.", "I could keep guessing what the silence means | and replay each line of my message."),
    ("work", "The meeting agenda changed this morning.", "The new first item is the budget and the old items follow.", "I can open the budget sheet and prepare the current figures.", "I am surprised; I will take two minutes to review the sheet.", "Now I am picturing every possible exchange | instead of reading the numbers."),
    ("work", "My proposal received three comments.", "Two comments ask for dates and one asks for a source.", "I can add the dates and check the source before replying.", "I dislike criticism; I will answer each specific comment.", "Why did those three comments start another loop | about how everyone sees my work?"),
    ("money", "The electricity bill is higher this month.", "The bill shows more kilowatt-hours and the rate is unchanged.", "I can compare usage with last month and check the meter.", "I am concerned; I will read the itemized bill tonight.", "This number is inviting me to imagine | every future bill going up forever."),
    ("money", "My bank card expires next week.", "The replacement card is listed as mailed on Monday.", "I can check the mailbox and call the bank if it is not here by Friday.", "I feel uneasy; I will save the bank's number.", "I can spend the day rehearsing all the ways | a card replacement might fail."),
    ("money", "The rent payment has not appeared in my account yet.", "The transfer receipt says it can take two business days.", "I can check again tomorrow and contact the landlord if it is still missing.", "I am worried; I will keep the receipt handy.", "I am starting to build an endless picture | of what this delay could mean about my life."),
    ("money", "The shop charged me twice for a book.", "Both charges have the same date and amount.", "I can keep the receipt and request a refund for one charge.", "I am irritated; I will contact the shop with the transaction numbers.", "A duplicate charge is turning into a repeated argument | with an imaginary cashier."),
    ("family", "My sister has not confirmed dinner on Sunday.", "The table reservation can still be changed until Saturday.", "I can send a short message and adjust the booking if needed.", "I am disappointed by the silence; I will ask once and wait.", "I could keep decoding her silence until | every old family story returns."),
    ("family", "My child left a school form on the table.", "The form needs a signature by Wednesday.", "I can read it tonight and put it back in the school bag.", "I am tired; I will sign it after dinner.", "This paper is becoming another loop | about whether I handle anything well."),
    ("family", "My father asked for a ride to the doctor.", "His appointment is at eleven and the clinic is twenty minutes away.", "I can pick him up at ten thirty and check the address.", "I feel stretched; I will move my morning call.", "I can keep rehearsing all the possible complaints | during a drive that has not happened."),
    ("family", "My partner prefers a different holiday plan.", "The two options have different dates and travel times.", "We can compare the dates tonight and choose one together.", "I feel disappointed; I will ask which part matters most to them.", "The choice could turn into an endless inner debate | about what every preference means."),
    ("home", "The washing machine stopped before the rinse cycle.", "The display shows an E03 code and there is water in the drum.", "I can look up the code and switch off the machine before checking the filter.", "I am annoyed; I will follow the manual's first troubleshooting step.", "This appliance is becoming the start of a whole story | about everything failing at once."),
    ("home", "The living-room light flickers when switched on.", "The bulb is old and the other lights are stable.", "I can switch it off and replace the bulb when it is cool.", "I am uneasy; I will avoid using it until I check it.", "I can keep imagining the whole house going dark | because one bulb flickered."),
    ("home", "A package was left at the wrong door.", "The delivery photo shows the neighboring apartment number.", "I can ask the neighbor and then contact delivery support if needed.", "I am irritated; I will check the photo before making a complaint.", "Now I am replaying a confrontation that has not happened | over a box I can simply locate."),
    ("home", "The kitchen tap drips after I turn it off.", "The drip is slow and there is no water under the sink.", "I can put a bowl underneath and arrange a repair.", "I am frustrated; I will note the leak and call tomorrow.", "The dripping sound is becoming a running commentary | on how I never finish anything."),
    ("health", "I have a headache this afternoon.", "I skipped lunch and have had little water today.", "I can eat and drink water, then check whether it settles.", "I feel uncomfortable; I will take a break from the screen.", "Now I am searching every sensation for signs | of a future that is not here."),
    ("health", "The pharmacy says my prescription is ready.", "The pickup window closes at six and the pharmacy is nearby.", "I can collect it on the way home and check the label.", "I feel relieved; I will add the pickup to my route.", "I could spend the whole afternoon revisiting | what it means that I need this medicine."),
    ("health", "My watch shows a lower step count today.", "I spent most of the day indoors and it is still light outside.", "I can take a short walk if I want to move before dinner.", "I feel restless; I will go around the block.", "That number is becoming a recurring verdict | on the kind of day I have had."),
    ("health", "The clinic moved my routine checkup.", "The new date is on the calendar and the lab hours are unchanged.", "I can update the reminder and confirm whether any preparation is needed.", "I dislike schedule changes; I will check the appointment message.", "The new date is starting an endless cycle | of imagined results before any test."),
    ("leisure", "The library book is due tomorrow.", "The return slot is open after hours and the book is on my desk.", "I can put it in my bag and return it after work.", "I am annoyed I forgot; I will set the book beside my keys.", "A simple due date is becoming another story | about how I always lose track of things."),
    ("leisure", "The film starts thirty minutes later than advertised.", "The updated ticket lists a nine-thirty start.", "I can meet my friend at nine instead of eight-thirty.", "I am impatient; I will send the new time.", "I can use the delay to replay every disappointing outing | instead of checking the ticket."),
    ("leisure", "The park trail is muddy after rain.", "The paved loop is open and my shoes are dry.", "I can choose the paved loop and keep the walk short.", "I wanted the trail; I will take the paved path today.", "The mud is drawing me into an endless comparison | with the walk I imagined having."),
    ("leisure", "The book club chose a novel I have not read.", "The meeting is next month and the library has a copy.", "I can borrow it this week and read a chapter at a time.", "I feel behind; I will check the library catalog.", "Now I am imagining every conversation where I have nothing to say | instead of opening the book."),
]

PERSONAS = (
    "You are an adult who lives alone and works regular daytime hours.",
    "You are an adult who shares a home with family and works varied hours.",
    "You are an adult who commutes to work and helps care for a relative.",
    "You are an adult who works from home and has two children.",
)


def build(seed: int = 20261106) -> list[dict]:
    assert len(CARDS) == 40
    rng = random.Random(seed)
    card_indices = list(range(len(CARDS)))
    rng.shuffle(card_indices)
    dev_cards = set(card_indices[:8])
    rows = []
    for card_index, (topic, opening, factual, task, feeling_task, marked_rumination) in enumerate(CARDS):
        assert marked_rumination.count("|") == 1
        left, right = marked_rumination.split("|")
        assert left.endswith(" ") and right.startswith(" ")
        prefix = left.rstrip()
        full_rumination = left + right.lstrip()
        assert full_rumination.startswith(prefix)
        split = "development" if card_index in dev_cards else "train"
        for persona_index, persona in enumerate(PERSONAS):
            group = f"INNER-V5-{split.upper()}-{card_index:03d}-{persona_index}"
            prompt = (
                f"Background: {persona}\n"
                f"Situation: {opening}\n"
                "Give the next words of the inner voice. Write only the voice itself."
            )
            for variant, branch, stop in (
                ("factual_continue", factual, False),
                ("task_continue", task, False),
                ("feeling_task_continue", feeling_task, False),
                ("rumination_stop", full_rumination, True),
            ):
                target_branch = prefix if stop else branch
                rows.append({
                    "id": f"{group}:{variant}",
                    "group_id": group,
                    "card_id": f"INNER-V5-CARD-{card_index:03d}",
                    "topic": topic,
                    "split": split,
                    "variant": variant,
                    "prompt": prompt,
                    "opening": opening,
                    "full_candidate_text": opening + " " + branch,
                    "target_text": opening + " " + target_branch,
                    "stop_at_target_end": stop,
                    "cut_after_word": prefix.split()[-1] if stop else None,
                    "boundary_kind": "noninstrumental_rumination" if stop else "continue",
                    "source": "project_authored_synthetic",
                })
    assert len(rows) == 640
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["card_id"] for r in rows if r["split"] == "train"}) == 32
    assert len({r["card_id"] for r in rows if r["split"] == "development"}) == 8
    counts = collections.Counter((r["split"], r["variant"]) for r in rows)
    assert all(n == (128 if split == "train" else 32) for (split, _), n in counts.items())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    rows = build(args.seed)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "status": "exploratory_pilot_not_training_approved",
        "construct": "cessation_at_noninstrumental_rumination_transition",
        "seed": args.seed,
        "rows": len(rows),
        "train_rows": sum(r["split"] == "train" for r in rows),
        "development_rows": sum(r["split"] == "development" for r in rows),
        "cards": len(CARDS),
        "train_cards": 32,
        "development_cards": 8,
        "groups_per_card": len(PERSONAS),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "external_text_used": False,
        "external_development_labels_used": False,
        "sealed_test_opened": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
