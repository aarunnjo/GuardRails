from fireguard.context import Context
from fireguard.store import TrustStore
from fireguard.trust import (
    assign_trust,
    compute_trust,
    tier_baseline,
    trust_to_threshold,
)
from fireguard.types import Chunk


def test_tier_baselines_ordered_internal_kb_most_trusted():
    assert tier_baseline("internal_kb") > tier_baseline("verified_partner")
    assert tier_baseline("verified_partner") > tier_baseline("open_web")
    assert tier_baseline("open_web") > tier_baseline("user_upload")


def test_unknown_tier_falls_back_to_default():
    assert tier_baseline("some_new_tier") == tier_baseline("open_web")


def test_incidents_decay_trust_but_never_to_zero():
    clean = compute_trust("internal_kb", 0)
    one_strike = compute_trust("internal_kb", 1)
    many_strikes = compute_trust("internal_kb", 50)
    assert one_strike < clean
    assert 0 < many_strikes < one_strike


def test_trust_to_threshold_is_monotonic_and_bounded():
    assert trust_to_threshold(0.0) == 0.5
    assert trust_to_threshold(1.0) == 0.9
    assert trust_to_threshold(0.0) < trust_to_threshold(0.5) < trust_to_threshold(1.0)


def test_user_upload_threshold_stricter_than_internal_kb(memory_db):
    store = TrustStore(memory_db)
    ctx = Context(detector=None, store=store)
    chunks = [
        Chunk(text="x", source_uri="kb-doc", tier="internal_kb"),
        Chunk(text="x", source_uri="upload-1", tier="user_upload"),
    ]
    out = assign_trust(chunks, ctx)
    kb, upload = out
    assert upload.trust < kb.trust
    assert upload.threshold < kb.threshold


def test_assign_trust_tightens_after_incident(memory_db):
    store = TrustStore(memory_db)
    ctx = Context(detector=None, store=store)
    chunk = Chunk(text="x", source_uri="repeat-offender", tier="internal_kb")

    before = assign_trust([chunk], ctx)[0]
    store.record_incident("repeat-offender", "L2 detect")
    after = assign_trust([chunk], ctx)[0]

    assert after.trust < before.trust
    assert after.threshold < before.threshold
