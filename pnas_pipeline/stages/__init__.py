"""Stage registry.

Scientific stages mirror `stages:` in the manifest. Until their P0/H0 protocol
inputs are frozen they are registered as blocked, so a submission fails fast
with the reason instead of running unreviewed code. Utility stages (dry_run,
calibrate_throughput, calibrate_qlora) are engineering-only and need no prerequisites.
"""
from __future__ import annotations

from . import calibrate, calibrate_qlora


class StageBlocked(RuntimeError):
    pass


def blocked(reason: str):
    def run(ctx):
        raise StageBlocked(f"stage {ctx.stage!r} is not implemented: {reason}")
    return run


def dry_run(ctx):
    ctx.log("dry run: manifest validated and run identity written")
    return {"dry_run": True}


UTILITY = {
    "dry_run": dry_run,
    "calibrate_throughput": calibrate.run,
    "calibrate_qlora": calibrate_qlora.run,
}

SCIENTIFIC = {
    "h0_construct": blocked("needs the SNCC corpus builder (WP1) and P0.3 two-expert elicitation"),
    "e1_prompt_icl": blocked("needs frozen H0 targets and the P0.2 primary-output rubric"),
    "e1_abstention_controls": blocked("needs frozen H0 targets and development-set thresholds"),
    "e2_serial_gate": blocked("needs frozen H0 targets (WP3)"),
    "e3a_representation": blocked("needs frozen H0 onsets (WP4); port probes from neural_poc with regression tests"),
    "e3b_native_policy": blocked("needs E3a frozen probe and the e3b equivalence margin (P0.4)"),
    "e3c_causal_access": blocked("needs E3a frozen probe (WP4)"),
    "e4_native_adaptation": blocked("needs E3 gates and the P0.1 post-adaptation clock validation (WP5)"),
    "e4_reactivation": blocked("needs E2 and E4 outputs (WP6)"),
    "h0b_blind_rating": blocked("human rating stage; runs off-cluster (WP6)"),
    "e5_transition_exploratory": blocked("exploratory; after E4 (WP7)"),
}
