"""The Detector protocol.

Every scanning module calls through this interface, never a model
directly -- the fix for what happened when llm-guard, the previous
plan's detection dependency, went archived under us.

Python's Protocol uses structural typing: a class satisfies this
interface just by having matching method signatures, no inheritance
or explicit declaration required. spike/detect.py's Detector class
already matches this shape, so it plugs in with no adapter code once
promoted into this package.
"""
from typing import Protocol


class Detector(Protocol):
    def score(self, text: str) -> float:
        """P(injection) for one text, in [0, 1]. Used by admit() -- one
        arriving chunk, one check."""
        ...

    def score_batch(self, texts: list[str]) -> list[float]:
        """P(injection) for many texts, in one batched call. Used by
        scan() -- a whole retrieval set, checked together for speed."""
        ...

    def n_tokens(self, text: str) -> int:
        """Token count, excluding special tokens.

        Used to detect silent truncation: anything over MAX_TOKENS (512)
        only has its first 512 tokens actually inspected by score().
        admit() and scan() use this to warn when that happens, rather
        than silently missing the back half of a long document.
        """
        ...
