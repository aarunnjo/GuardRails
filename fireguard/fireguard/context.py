"""Shared state passed through the CHECKS pipeline.

Introduced in Phase 3: once a check (trust.py) needs persistent state
(store.py) alongside the model (backends/), a plain (chunks, detector)
signature stops being enough for every check uniformly. Context bundles
what a check might need so the CHECKS list stays "one append, not a
redesign" as new checks are added.
"""
from dataclasses import dataclass

from .backends.base import Detector
from .store import TrustStore


@dataclass
class Context:
    detector: Detector | None
    store: TrustStore
