"""The Firewall class -- the library's three entry points.

Phase 1: everything was stubbed. Phase 2 (this file, now) wires in the
first real check -- normalize.py -- leaving trust and detection stubbed
until Phase 3.
"""
from .backends.base import Detector
from .normalize import normalize
from .types import AdmitVerdict, Chunk, RetrievalSet, ScanResult, VerifyResult


def _apply_trust_stub(chunks: list[Chunk], detector: Detector | None) -> list[Chunk]:
    return chunks  # Phase 3 replaces this with trust.py's tier lookup + threshold tuning


def _per_chunk_detect_stub(chunks: list[Chunk], detector: Detector | None) -> list[Chunk]:
    return chunks  # wired to a real Detector once one is promoted from spike/detect.py


CHECKS = [normalize, _apply_trust_stub, _per_chunk_detect_stub]


class Firewall:
    def __init__(self, detector: Detector | None = None):
        self._detector = detector

    def admit(self, chunk: Chunk) -> AdmitVerdict:
        return AdmitVerdict(blocked=False)

    def scan(self, retrieval_set: RetrievalSet) -> ScanResult:
        chunks = retrieval_set.chunks
        for check in CHECKS:
            chunks = check(chunks, self._detector)
        return ScanResult(verdict="allow", approved_chunks=chunks)

    def verify(self, query: str, answer: str, sources: list[Chunk]) -> VerifyResult:
        return VerifyResult(verdict="allow")
