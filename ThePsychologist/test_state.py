"""Tests for state.py. Run: python test_state.py (or pytest)."""
import tempfile
from pathlib import Path

import state

state.TRAITS = ("depression", "anxiety")  # the tests need two traits; real traits come from config.py

START = state.initial_state({"depression": 0.8, "anxiety": 0.5})
NO_GATE = {t: 0 for t in state.TRAITS}


def test_perfect_session_reduces_by_at_most_eta():
    """A perfect session lowers severity by exactly ETA of its current value."""
    new = state.update(START, {"depression": 1}, {"depression": 1.0})
    assert abs(new["depression"] - 0.8 * (1 - state.ETA)) < 1e-9  # exactly 5% less
    assert new["anxiety"] == START["anxiety"]  # untouched trait


def test_irrelevant_session_changes_nothing():
    """With every gate at 0, beta is unchanged."""
    assert state.update(START, NO_GATE, {"depression": 1.0}) == START


def test_harmful_session_increases_severity():
    """Negative effect makes the trait worse."""
    new = state.update(START, {"depression": 1}, {"depression": -1.0})
    assert new["depression"] > START["depression"]


def test_values_stay_in_unit_interval():
    """Beta is clipped at 1 from above and 0 from below."""
    high = state.initial_state({"depression": 1.0})
    assert state.update(high, {"depression": 1}, {"depression": -1.0})["depression"] == 1.0
    low = state.initial_state()
    assert state.update(low, {"depression": 1}, {"depression": 1.0})["depression"] == 0.0


def test_invalid_inputs_raise():
    """Bad gate, effect or trait name raises ValueError."""
    for bad in (lambda: state.update(START, {"depression": 2}, {}),
                lambda: state.update(START, {"depression": 1}, {"depression": 1.5}),
                lambda: state.initial_state({"mania": 0.3})):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_save_load_roundtrip():
    """Saving then loading returns the same beta vector."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "beta.json"
        state.save(START, p)
        assert state.load(p) == START


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
