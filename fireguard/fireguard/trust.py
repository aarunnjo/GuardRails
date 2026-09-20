"""L1 -- source trust.

Trust is not a property of the text; it's a property of where the text came
from and whether that source has misbehaved before. Two inputs combine into
one trust score per source_uri:

  1. a tier baseline (assigned by the caller when they build a Chunk -- the
     caller knows whether this came from their own vetted KB or an open web
     fetch; the library cannot and does not guess this)
  2. incident history (store.py) -- every prior flagged event for this
     source_uri decays its trust, multiplicatively, with a floor so one bad
     day doesn't zero out a source forever

The resulting trust score does ONE job: it tunes L2's detector threshold.
A user_upload chunk with no history gets a stricter (lower) bar than an
internal_kb chunk -- the model is the same either way; only the bar it must
clear changes. This is the "L1 wired into L2's threshold" step the plan
calls out as what makes this a library rather than a bare classifier.
"""
from dataclasses import replace

from .store import TrustStore
from .types import Chunk

TIER_BASELINE: dict[str, float] = {
    "internal_kb": 0.95,
    "verified_partner": 0.85,
    "open_web": 0.5,
    "user_upload": 0.3,
}
DEFAULT_BASELINE = 0.5  # unrecognized tier -- treat like open_web, not internal_kb

# Each flagged incident multiplies trust by this factor. 0.7^3 ≈ 0.34 -- three
# strikes roughly halves a fully-trusted source's standing.
INCIDENT_DECAY = 0.7
TRUST_FLOOR = 0.05

# Trust in [0, 1] maps linearly to a detector threshold in [MIN, MAX].
# Full trust -> lenient bar (0.9, matches the Phase 0a gate's own target).
# Zero trust -> strict bar (0.5) -- still not zero, because the detector's
# own false-positive rate is a floor no threshold should go below.
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.9


def tier_baseline(tier: str) -> float:
    return TIER_BASELINE.get(tier, DEFAULT_BASELINE)


def compute_trust(tier: str, incident_count: int) -> float:
    trust = tier_baseline(tier) * (INCIDENT_DECAY**incident_count)
    return max(trust, TRUST_FLOOR)


def trust_to_threshold(trust: float) -> float:
    trust = min(max(trust, 0.0), 1.0)
    return MIN_THRESHOLD + trust * (MAX_THRESHOLD - MIN_THRESHOLD)


def assign_trust(chunks: list[Chunk], ctx) -> list[Chunk]:
    """CHECKS-pipeline entry point. Looks up each chunk's incident history,
    computes trust, and attaches trust + threshold to the chunk. Does not
    flag anything itself -- L2 (detect.py) is what reads chunk.threshold."""
    store: TrustStore = ctx.store
    out = []
    for c in chunks:
        trust = compute_trust(c.tier, store.incident_count(c.source_uri))
        out.append(replace(c, trust=trust, threshold=trust_to_threshold(trust)))
    return out
