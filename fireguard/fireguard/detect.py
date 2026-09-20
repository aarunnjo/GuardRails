"""L2 -- per-chunk detection.

Runs the Context's Detector across every chunk's text in one batched call
(score_batch, not a score() loop -- see backends/base.py) and flags any
chunk whose score meets or exceeds ITS OWN L1 threshold. Same model, same
call, different bar per chunk depending on where it came from -- that's the
whole point of computing trust before this check runs.
"""
from dataclasses import replace

from .types import Chunk

DEFAULT_THRESHOLD = 0.9  # used only if a chunk reached here with no L1
                          # threshold attached (e.g. assign_trust was
                          # skipped) -- matches MAX_THRESHOLD in trust.py,
                          # the most lenient bar, so a missing trust step
                          # never becomes an accidental stricter default


def per_chunk_detect(chunks: list[Chunk], ctx) -> list[Chunk]:
    if ctx.detector is None or not chunks:
        return chunks

    scores = ctx.detector.score_batch([c.text for c in chunks])
    out = []
    for c, score in zip(chunks, scores):
        threshold = c.threshold if c.threshold is not None else DEFAULT_THRESHOLD
        hit = score >= threshold
        reasons = c.flag_reasons
        if hit:
            reasons = [*reasons, f"L2 detect: score {score:.3f} >= threshold {threshold:.3f}"]
        out.append(
            replace(c, injection_score=score, flagged=c.flagged or hit, flag_reasons=reasons)
        )
    return out
