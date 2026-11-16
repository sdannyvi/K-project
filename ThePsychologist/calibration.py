"""
NOT run directly: helper module, executed by `python adapters.py calibrate <trait>` (and read by patient.py).

Beta -> alpha calibration. The raw adapter scale alpha is very uneven (PHQ jumps early, then saturates).
Beta is a severity knob in [0, 1] that is linear in PHQ: PHQ(alpha(beta)) = lo + beta * (hi - lo).
Built by inverting the measured PHQ(alpha) curve (`python adapters.py calibrate <trait>`).
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")  # save to file only, no display needed on the GPU box
import matplotlib.pyplot as plt

from config import CALIB_FRACTION


def build(alphas, phq):
    """Calibration from a measured sweep: monotone PHQ(alpha) for alpha >= 0, plus the beta=0 / beta=1 PHQ."""
    # negative alphas are dropped: beta only covers the healthy -> severe direction
    pts = sorted((a, p) for a, p in zip(alphas, phq) if a >= 0)
    a = np.array([x for x, _ in pts])
    p = np.array([y for _, y in pts])
    peak = int(np.argmax(p)) + 1  # keep only the climb up to the PHQ peak: points after it are a plateau or a dip
    a, p = a[:peak], np.maximum.accumulate(p[:peak])  # remaining small dips are flattened to stay monotone
    # beta=0 is the alpha=0 baseline; beta=1 stops short of the plateau (CALIB_FRACTION) where the curve is flat
    lo, hi = p[0], p[0] + CALIB_FRACTION * (p[-1] - p[0])
    return {"alphas": a.tolist(), "phq": p.tolist(), "phq_lo": float(lo), "phq_hi": float(hi)}


def beta_to_alpha(beta, cal):
    """Alpha whose measured PHQ equals the linear target for this beta."""
    # the PHQ we want at this beta is a straight line; interp over the measured curve read backwards gives alpha
    beta = min(max(beta, 0.0), 1.0)
    target = cal["phq_lo"] + beta * (cal["phq_hi"] - cal["phq_lo"])
    p, a = np.array(cal["phq"]), np.array(cal["alphas"])
    keep = np.concatenate(([True], np.diff(p) > 1e-9))  # np.interp needs strictly increasing x; plateau points dropped
    return float(np.interp(target, p[keep], a[keep]))


def beta_to_phq(beta, cal):
    """The linear PHQ target for this beta (what the patient should score)."""
    return cal["phq_lo"] + min(max(beta, 0.0), 1.0) * (cal["phq_hi"] - cal["phq_lo"])


def save(cal, path):
    path.write_text(json.dumps(cal, indent=2))


def load(path):
    """The saved calibration, or None if `calibrate` was never run (callers then treat beta as raw alpha)."""
    return json.loads(path.read_text()) if path.exists() else None


def plot(cal, path, check=None):
    """Save two panels. Left: alpha and beta against PHQ (beta is straight by construction, alpha bends and flattens).
    Right (if `check` = (betas, measured PHQs) is given): the real test, PHQ re-measured at alpha(beta); should be a line."""
    p, a = np.array(cal["phq"]), np.array(cal["alphas"])
    beta = np.clip((p - cal["phq_lo"]) / (cal["phq_hi"] - cal["phq_lo"]), 0, 1)  # beta = inverse of the target line
    fig, axes = plt.subplots(1, 2 if check else 1, figsize=(11 if check else 6, 4), squeeze=False)
    ax = axes[0, 0]
    ax.plot(p, a, "o-", label="alpha (raw LoRA scale)")
    ax.plot(p, beta, "s-", label="beta (calibrated)")
    ax.set_xlabel("PHQ-9")
    ax.set_ylabel("alpha / beta")
    ax.legend()
    ax.grid(alpha=0.3)
    if check:
        ax = axes[0, 1]
        betas, measured = check
        ax.plot([0, 1], [cal["phq_lo"], cal["phq_hi"]], "--", color="gray", label="ideal (linear)")
        ax.plot(betas, measured, "o-", label="measured at alpha(beta)")
        ax.set_xlabel("beta")
        ax.set_ylabel("PHQ-9")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
