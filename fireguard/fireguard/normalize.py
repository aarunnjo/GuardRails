"""Strip invisible-character manipulation before anything scans the text.

Three stages, applied in order:
  1. NFKC -- collapses styled Unicode (math-bold letters, full-width forms,
     ligatures) back to plain characters. Uses Python's built-in Unicode
     tables; does NOT catch homoglyphs (see stage 3).
  2. Strip invisible Unicode categories (Cf, Cc, Co, Cn) -- zero-width
     spaces, control chars, private-use chars. Also Python's built-in
     tables (unicodedata.category()).
  3. Fold common homoglyphs -- Cyrillic/Greek letters that render
     identically to Latin ones. Unicode does not consider these the same
     character, so there is no built-in table for this -- CONFUSABLES
     below is ours.
"""
import unicodedata

from .types import Chunk

DROP_CATEGORIES = {"Cf", "Cc", "Co", "Cn"}
KEEP_CHARS = {"\n", "\t", "\r", "‍"}  # normal whitespace, plus ZWJ --
                                            # needed for emoji sequences and
                                            # Indic script conjuncts; stripping
                                            # it blindly breaks legitimate text

# A small, practical set of common lookalikes actually seen in real attacks --
# not the full Unicode confusables.txt (thousands of entries). Cyrillic and
# Greek letters that render identically to Latin ones.
CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        if ch not in KEEP_CHARS and unicodedata.category(ch) in DROP_CATEGORIES:
            continue
        out.append(CONFUSABLES.get(ch, ch))
    return "".join(out)


def normalize(chunks: list[Chunk], detector) -> list[Chunk]:
    """CHECKS-pipeline entry point -- signature matches firewall.py's stub."""
    return [
        Chunk(text=normalize_text(c.text), source_uri=c.source_uri, tier=c.tier, raw=c.raw)
        for c in chunks
    ]
