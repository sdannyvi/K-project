"""
Random background for a patient: 3 short sentences (age, work, home life, a hobby) picked from small pools.
Made once per patient name and saved in states/<name>/state.json, so the same person comes back every session.
Deliberately says nothing about mood or symptoms: the trait must live in the weights, not in the prompt.
"""
import random

JOBS = ["a nurse", "a high-school teacher", "a software tester", "a bakery owner", "an accountant", "a delivery driver",
        "a graphic designer", "a librarian", "an electrician", "a university student", "a waiter", "a social-media manager"]
HOMES = ["lives alone in a small apartment", "shares a flat with two roommates", "lives with a partner",
         "lives with their parents", "lives in a quiet suburb with a dog and two kids.", "lives near the city center"]
FAMILY = ["has a younger brother", "is close to their sister", "has two close friends from school",
          "has no siblings", "has a large extended family", "calls their grandmother every week"]
HOBBIES = ["hiking", "cooking", "playing guitar", "reading novels", "watching football", "gardening", "photography",
           "playing video games", "painting", "swimming"]


def make_persona(name, rng=None):
    """Three sentences about `name`, drawn at random."""
    rng = rng or random.Random()
    return (f"{name} is {rng.randint(22, 58)} years old and works as {rng.choice(JOBS)}. "
            f"{name} {rng.choice(HOMES)} and {rng.choice(FAMILY)}. "
            f"In free time {name} enjoys {rng.choice(HOBBIES)}.")
