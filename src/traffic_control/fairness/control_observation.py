"""Explicit controller memory for a new, opt-in observation schema.

The legacy observation omitted the pending transition target and the clearance
empty timer. Both affect the next state even when phase and phase age agree.
These local controller variables need no future demand or remote observations.
This removes those aliases; it does not make aggregate traffic fully Markov.
"""

SCHEMAS = ("legacy-v2", "controller-memory-v1")
FEATURES = ("committed_EW", "committed_NS", "committed_PED",
            "clearance_empty_known", "clearance_empty_elapsed")


def controller_memory(signal, now):
    """Return dimensionless local features, with a distinct absent-timer flag."""
    empty_known = signal.stage == "PED_CLEARANCE" and signal.empty_since is not None
    elapsed = max(0., now - signal.empty_since) if empty_known else 0.
    return ([float(signal.target == phase) for phase in ("EW", "NS", "PED")]
            + [float(empty_known), elapsed / 30.]), list(FEATURES)
