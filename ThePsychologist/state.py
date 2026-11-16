"""
Patient state controller: one severity coefficient (beta, 0-1, linear in PHQ; see calibration.py) per clinical trait.

Pure Python, no ML dependencies. The beta vector is what later gets converted
to beta and injected into the model (as adapter mixing weights) or into the prompt (baseline).
"""
import json
from pathlib import Path

from config import TRAITS

# Maximum relative change per session. A perfect session lowers a trait's
# severity by at most 5% of its current value (clinically paced recovery).
ETA = 0.05


def initial_state(severities: dict[str, float] | None = None) -> dict[str, float]:
    """
    Return the starting beta vector. Traits not given start at 0 (absent).
    """
    severities = severities or {}
    unknown = set(severities) - set(TRAITS)
    if unknown:
        raise ValueError(f"Unknown traits: {unknown}")
    return {t: _clip(severities.get(t, 0.0)) for t in TRAITS}


def update(beta: dict[str, float],
           gate: dict[str, int],
           effect: dict[str, float],
           eta: float = ETA) -> dict[str, float]:
    """
    Apply one session's outcome to the beta vector.

    gate[t]   in {0, 1}   : was trait t therapeutically addressed? -->(G)
    effect[t] in [-1, 1]  : how well? Negative = harmful session -->(s).

    Rule: beta_t <- clip(beta_t * (1 - eta * G_t * s_t), 0, 1)

    Design choices:
    - Multiplicative, so "at most eta" really means eta of the *current* severity.
    - Signed effect, so a harmful session can make the patient worse.
    - G = 0 leaves the trait untouched (irrelevant sessions change nothing).
    - A trait at exactly 0 cannot get worse; the update is relative to severity.
    """
    new = {}
    for t in TRAITS:
        g, s = gate.get(t, 0), effect.get(t, 0.0)
        if g not in (0, 1):
            raise ValueError(f"gate[{t}] must be 0 or 1, got {g}")
        if not -1.0 <= s <= 1.0:
            raise ValueError(f"effect[{t}] must be in [-1, 1], got {s}")
        new[t] = _clip(beta[t] * (1 - eta * g * s))
    return new


def save(beta: dict[str, float], path: str | Path) -> None:
    """Persist beta between sessions."""
    Path(path).write_text(json.dumps(beta, indent=2))


def load(path: str | Path) -> dict[str, float]:
    """Load a previously saved beta vector."""
    return json.loads(Path(path).read_text())


def _clip(x: float) -> float:
    """Clamp x to [0, 1]."""
    return max(0.0, min(1.0, x))
