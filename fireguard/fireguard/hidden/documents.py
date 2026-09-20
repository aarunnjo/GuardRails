"""L5 -- hidden text in PDF/DOCX documents. admit()-only: this needs the raw
file (page rendering, run-level formatting), and once content is a plain
text chunk in a vector DB that evidence is gone.

Not a wrapped third-party package. The Phase 0b spike validated
`hidden-text-detector` (real pixel-contrast measurement against a
constructed white-on-white test PDF -- see results/phase0b_hidden_text.json)
but it ships as a Claude Agent Skill + standalone CLI, not something
pip-installable or vendorable into this library's own dependency tree. This
module reimplements the same two signals natively, using packages already
in the dependency tree (pymupdf, python-docx):

  - white-on-white / nearly-invisible text, and off-page / near-zero-size
    text (PDF) -- via pymupdf's span-level color, size, and position data
  - Word's actual "hidden text" flag (DOCX) -- python-docx exposes this
    directly as Run.font.hidden, so no heuristic is needed there
"""
import io

import pymupdf as fitz
from docx import Document

NEAR_WHITE_DELTA = 0.05  # per-channel distance from white, 0..1 scale
MIN_READABLE_PT = 1.0


def _color_is_near_white(color) -> bool:
    if color is None:
        return False
    if isinstance(color, int):
        r, g, b = ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255
    else:
        r, g, b = color[:3]
    return all(1 - c <= NEAR_WHITE_DELTA for c in (r, g, b))


def scan_pdf(raw: bytes) -> list[str]:
    findings = []
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        for page_num, page in enumerate(doc):
            page_rect = page.rect
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        if _color_is_near_white(span.get("color")):
                            findings.append(f"page {page_num}: white-on-white text {text[:60]!r}")
                        elif span.get("size", 12) < MIN_READABLE_PT:
                            findings.append(f"page {page_num}: near-zero font size text {text[:60]!r}")
                        elif not page_rect.contains(fitz.Rect(span["bbox"])):
                            findings.append(f"page {page_num}: off-page text {text[:60]!r}")
    finally:
        doc.close()
    return findings


def scan_docx(raw: bytes) -> list[str]:
    findings = []
    doc = Document(io.BytesIO(raw))
    for para in doc.paragraphs:
        for run in para.runs:
            if not run.text.strip():
                continue
            if run.font.hidden:
                findings.append(f"hidden run (w:vanish) {run.text[:60]!r}")
                continue
            color = run.font.color.rgb if run.font.color is not None else None
            if color is not None and _color_is_near_white((color[0] / 255, color[1] / 255, color[2] / 255)):
                findings.append(f"white-on-white run {run.text[:60]!r}")
    return findings
