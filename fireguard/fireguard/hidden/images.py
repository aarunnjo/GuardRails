"""L5 -- images (PNG/JPEG). admit()-only, same reasoning as documents.py/html.py.

Unlike the other hidden/ modules, this one is NOT purely heuristic: OCR is
the only way any text inside an image reaches the pipeline at all. A PDF or
an HTML page already has its plain text extracted by the caller's own
ingestion step (that's what populates chunk.text) -- the hidden-text check
there exists only to catch what that rendering-blind extraction misses. An
image has no equivalent extraction step; if this module doesn't read the
pixels, nothing else in the pipeline ever does. So scan_image does two
independent things with whatever OCR finds:

  1. content check (the important one): score every piece of extracted
     text -- regardless of how it looks -- against the SAME injection
     detector and trust-tuned threshold used for the chunk's own text. This
     is what catches a plainly visible "ignore all previous instructions"
     rendered as ordinary image text (the well-documented "typographic
     prompt injection" attack) -- a case a pure hidden-text heuristic would
     never see, because nothing about it is hidden.
  2. low-contrast flag (best-effort, and genuinely limited -- see below):
     flag text whose rendered pixels barely differ from their background.

CRITICAL LIMITATION, measured directly against this exact OCR engine, not
assumed: RapidOCR's own text-DETECTION stage (the step that finds a
candidate text region at all, before any of our code runs) has a contrast
floor. In testing, text at a background-relative contrast delta of ~0.08
was never found as a candidate region at all; ~0.14 and above was found
reliably (and with high confidence). There is no way for (2) to flag text
OCR itself never surfaces -- unlike a PDF, where pymupdf reads the actual
stored text object and its color attribute regardless of how invisible the
rendering is, a raster image has no separate "underlying text" data
structure at all. If it's dim enough that OCR can't find it, nothing in
this module can either. This is a hard architectural limit of any
OCR-based approach, not a tuning problem -- treat (2) as a narrow,
best-effort signal for "detectable but noticeably dim," never as
PDF-equivalent white-on-white coverage.

(2) also folds in what would otherwise be separate alpha-transparency and
tiny-font-size checks. Testing showed both are, on their own, weak hiding
techniques against this OCR engine -- 8px text and heavily alpha-blended
text were both still detected reliably. And once an image is flattened to
its final rendered pixels, transparency and small-but-antialiased text
both show up as the same lowered contrast delta anyway, so there's no
separate signal worth computing for them.

Explicitly out of scope, not attempted: steganographic (LSB/DCT/DWT
-encoded) payloads and adversarial pixel perturbations -- neither produces
OCR-readable text, so neither is reachable by any approach in this module,
and both need specialist tooling disproportionate to this library's scope.
See README.
"""
import io

import numpy as np  # already a hard fireguard dependency elsewhere -- fine eager

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


MIN_OCR_CONFIDENCE = 0.3  # below this, OCR is probably hallucinating text on
                          # noise/clutter -- a sanity filter on "is this
                          # really text," not a visibility signal
MIN_CONTRAST_DELTA = 0.2  # calibrated against this exact engine (see module
                          # docstring): its own detector already fails below
                          # ~0.08-0.14, so this flags the "detectable but
                          # dim" band above that floor, never true invisibility


def _box_contrast_delta(gray: np.ndarray, box) -> float:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, x1 = max(int(min(xs)), 0), min(int(max(xs)), gray.shape[1])
    y0, y1 = max(int(min(ys)), 0), min(int(max(ys)), gray.shape[0])
    region = gray[y0:y1, x0:x1]
    if region.size < 4:
        return 1.0  # too small a region to judge -- don't flag on a
                    # measurement we can't trust
    lo, hi = np.percentile(region, 10), np.percentile(region, 90)
    return float(hi - lo) / 255.0


def scan_image(raw: bytes, detector=None, threshold: float = 0.9) -> list[str]:
    from PIL import Image  # lazy: Pillow isn't a hard fireguard dependency --
                            # only the [ocr] extra needs it, and this module
                            # is imported unconditionally by hidden/__init__.py

    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        # Flatten against an assumed white page background -- the same
        # pragmatic assumption documents.py's NEAR_WHITE_DELTA makes for
        # PDFs. Text alpha-blended against an unknown/different real
        # background can't be judged correctly in isolation; this is a
        # documented approximation, not a guarantee.
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))

    gray = np.array(img.convert("L"))
    rgb_array = np.array(img.convert("RGB"))

    engine = _get_engine()
    result, _ = engine(rgb_array)
    if not result:
        return []

    findings = []
    texts_to_check = []

    for box, text, conf in result:
        text = text.strip()
        if not text or conf < MIN_OCR_CONFIDENCE:
            continue

        delta = _box_contrast_delta(gray, box)
        if delta < MIN_CONTRAST_DELTA:
            findings.append(f"low-contrast image text {text[:60]!r} (contrast_delta={delta:.3f})")

        texts_to_check.append(text)

    if detector is not None and texts_to_check:
        scores = detector.score_batch(texts_to_check)
        for text, score in zip(texts_to_check, scores):
            if score >= threshold:
                findings.append(
                    f"image text scores as injection {text[:60]!r}: "
                    f"score {score:.3f} >= threshold {threshold:.3f}"
                )

    return findings
