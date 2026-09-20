"""L5 dispatcher -- sniffs raw bytes by magic/content and routes to the
right hidden-text scanner. Called from admit() only (see documents.py,
html.py, and images.py docstrings for why: the evidence doesn't survive
into a text chunk).

detector/threshold are accepted here (and ignored by every scanner except
images.py) so the one format that needs the injection detector -- images,
which have no other text-extraction path into the pipeline at all -- can
use it, without changing every other scanner's simpler, model-free
signature.
"""
from .documents import scan_docx, scan_pdf
from .html import scan_html
from .images import scan_image

__all__ = ["scan_docx", "scan_hidden_text", "scan_html", "scan_image", "scan_pdf"]


def scan_hidden_text(raw: bytes | None, detector=None, threshold: float = 0.9) -> list[str]:
    if not raw:
        return []

    if raw[:5] == b"%PDF-":
        return scan_pdf(raw)

    if raw[:2] == b"PK":  # DOCX is a zip archive
        try:
            return scan_docx(raw)
        except Exception:
            return []

    if raw[:8] == b"\x89PNG\r\n\x1a\n" or raw[:3] == b"\xff\xd8\xff":
        try:
            return scan_image(raw, detector=detector, threshold=threshold)
        except Exception:
            return []

    if raw.lstrip()[:1] == b"<":
        try:
            return scan_html(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return []

    return []  # plain text has no hidden-text vector by definition --
               # normalize.py's invisible-Unicode stripping is what covers it
