"""The Firewall class -- the library's three entry points.

Phase 1-2: everything but normalize() was stubbed. Phase 3-6 (this file,
now) wire in trust, detection, hidden-text, and the output/feedback loop --
see PROGRESS.md for the phase-by-phase history.
"""
from dataclasses import replace

from .backends.base import Detector
from .context import Context
from .detect import per_chunk_detect
from .hidden import scan_hidden_text
from .normalize import normalize, normalize_text
from .output import check_output
from .store import TrustStore
from .trust import assign_trust, compute_trust, trust_to_threshold
from .types import AdmitVerdict, Chunk, RetrievalSet, ScanResult, VerifyResult

# scan()'s checks are an ordered list, not hardcoded steps -- a future check
# (e.g. v1.1's topic_drift.py) slots in as one append here, not a redesign.
CHECKS = [normalize, assign_trust, per_chunk_detect]


class Firewall:
    def __init__(self, detector: Detector | None = None, db_path: str | None = None):
        self._detector_override = detector
        self._detector_instance: Detector | None = None
        self._store = TrustStore(db_path)

    @property
    def _detector(self) -> Detector | None:
        """Firewall() with no arguments still means something: the ONNX
        detector loads lazily on first actual use (~4.5s, ~1.3GB RSS --
        see results/phase0a_gate.json), not at construction, so building a
        Firewall to inspect config or run only normalize() stays cheap."""
        if self._detector_override is not None:
            return self._detector_override
        if self._detector_instance is None:
            from .backends.onnx_detector import OnnxDetector

            self._detector_instance = OnnxDetector()
        return self._detector_instance

    def admit(self, chunk: Chunk) -> AdmitVerdict:
        """Ingest-time check: normalize -> hidden-text -> detect, all using
        this source's CURRENT trust (before this chunk can affect it). A
        blocked admit is itself an incident, recorded before returning --
        this is what makes a source's trust visibly tighten after its first
        rejected upload, not just after a scan() later catches something."""
        c = replace(chunk, text=normalize_text(chunk.text))

        trust = compute_trust(c.tier, self._store.incident_count(c.source_uri))
        threshold = trust_to_threshold(trust)
        detector = self._detector

        # threshold/detector are passed through so scan_hidden_text can score
        # any text an image's OCR extracts against the same bar as c.text --
        # images have no other text-extraction path into the pipeline at all
        # (see hidden/images.py's docstring). Every other format ignores them.
        reasons = [
            f"L5 hidden text: {h}"
            for h in scan_hidden_text(c.raw, detector=detector, threshold=threshold)
        ]

        if detector is not None and c.text.strip():
            score = detector.score(c.text)
            if score >= threshold:
                reasons.append(f"L2 detect: score {score:.3f} >= threshold {threshold:.3f}")

        blocked = bool(reasons)
        if blocked:
            self._store.record_incident(c.source_uri, "; ".join(reasons))
        return AdmitVerdict(blocked=blocked, reasons=reasons)

    def scan(self, retrieval_set: RetrievalSet) -> ScanResult:
        """Retrieval-time check, run on the whole set just before it hits
        the prompt. verdict is "block" only if every chunk got flagged
        (nothing left to answer with safely); "flag" if some did (the
        approved subset still goes to the prompt, minus the flagged ones);
        "allow" otherwise."""
        ctx = Context(detector=self._detector, store=self._store)
        chunks = retrieval_set.chunks
        for check in CHECKS:
            chunks = check(chunks, ctx)

        approved = [c for c in chunks if not c.flagged]
        flagged = [c for c in chunks if c.flagged]
        trust_scores = {c.source_uri: c.trust for c in chunks if c.trust is not None}
        reasons = [r for c in flagged for r in c.flag_reasons]

        if flagged and not approved:
            verdict = "block"
        elif flagged:
            verdict = "flag"
        else:
            verdict = "allow"

        return ScanResult(
            verdict=verdict,
            approved_chunks=approved,
            flagged_chunks=flagged,
            trust_scores=trust_scores,
            reasons=reasons,
        )

    def verify(self, query: str, answer: str, sources: list[Chunk]) -> VerifyResult:
        """Post-answer check (L4) plus the feedback loop (L1): if the
        answer looks hijacked or leaks something not present in its
        sources, every source_uri behind this answer takes an incident --
        the next scan()/admit() from that source sees a tighter threshold."""
        result = check_output(query, answer, sources, self._detector)
        if result.verdict != "allow":
            reason = "; ".join(result.reasons)
            for c in sources:
                self._store.record_incident(c.source_uri, reason)
        return result
