"""Data shapes shared across the library.

Plain dataclasses, no logic. Every other module imports from here rather
than passing loose dicts around, so the shape of a Chunk or a ScanResult
is defined in exactly one place.
"""
from dataclasses import dataclass, field

Tier = str  # "internal_kb" | "verified_partner" | "open_web" | "user_upload", extensible


@dataclass
class Chunk:
    text: str
    source_uri: str
    tier: Tier
    raw: bytes | None = None  # original file bytes, if any -- enables the
                               # document hidden-text check in admit(). None
                               # for plain-text sources (a chunk retrieved
                               # from a vector DB has no "raw file" behind it).

    # Populated progressively by the scan()/admit() pipeline (Phase 3+).
    # None until the relevant check has run -- callers never set these.
    trust: float | None = None            # L1: this source's current trust, [0,1]
    threshold: float | None = None        # L1: the L2 bar this chunk must clear
    injection_score: float | None = None  # L2: P(injection) for this chunk's text
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)


@dataclass
class RetrievalSet:
    query: str
    chunks: list[Chunk]


@dataclass
class AdmitVerdict:
    blocked: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    verdict: str                              # "allow" | "flag" | "block"
    approved_chunks: list[Chunk] = field(default_factory=list)
    trust_scores: dict[str, float] = field(default_factory=dict)   # source_uri -> [0,1]
    flagged_chunks: list[Chunk] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    verdict: str                              # "allow" | "flag" | "block"
    reasons: list[str] = field(default_factory=list)
