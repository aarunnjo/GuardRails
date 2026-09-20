# fireguard — progress

*Snapshot: 20 Sep 2026. Supersedes the 10 Sep snapshot below the divider.*

## Status: Phases 0–11 all done. `pip install fireguard` works from real PyPI.

**Published: https://pypi.org/project/fireguard/ — currently at 0.2.0 (20 Sep
2026).** 0.1.0 was the initial release (phases 0-11); 0.2.0 adds image (L5)
scanning. Both verified by installing into a brand-new venv from the real
public index (not a local wheel/cwd) and running real checks against it —
passed both times. All library code, tests, docs, eval harness, framework
adapters, and a demo UI exist and pass.

**0.2.0 release caught a real bug before it became everyone's problem**: a
final whole-codebase cleanup pass found that `hidden/images.py` imported
Pillow eagerly at module level, but Pillow was never a hard dependency (only
transitively present via other packages in the dev venv) — meaning `import
fireguard` would have hard-crashed for any bare `pip install fireguard`
user, full stop, not just anyone using images. Confirmed the failure with a
simulated missing-Pillow import, fixed by making the import lazy (same
pattern already used for `rapidocr_onnxruntime`/`OnnxDetector`), then
re-verified a bare install (no `[ocr]` extra) still works end to end before
publishing 0.2.0.

## What's built, by phase

| Phase | Build | Verified how |
|---|---|---|
| 0 | Spikes (detector gate, long-context rejection, cross-document cut) | `results/phase0*.json`, `results/spike_l3.json` |
| 1–2 | `types.py`, `backends/base.py`, `firewall.py` shell, `normalize.py` | 46→48 unit tests |
| 3 | `trust.py` + `store.py` — tier baselines, incident decay, threshold tuning | `tests/test_trust.py`, `tests/test_store.py` |
| 4 | `admit()`, `hidden/documents.py` (PDF white-on-white/off-page, DOCX `w:vanish`), `OnnxDetector` promoted from spike | `tests/test_hidden_documents.py`, `tests/test_firewall.py` |
| 5 | `hidden/html.py` (`display:none`/`visibility:hidden`/opacity/off-screen, inline + single-selector `<style>` rules) | `tests/test_hidden_html.py` |
| 6 | Feedback loop — `verify()`'s L4 output check (`output.py`) records incidents that tighten `L1` trust for every source behind a bad answer | `tests/test_firewall.py::test_verify_feedback_loop_tightens_trust_after_bad_answer` |
| 7 | Docs — `fireguard/README.md` (quickstart, architecture, tiers, evaluation, explicit "what this does not protect against") | Quickstart in README runs verbatim against a wheel-installed clean venv |
| 8 | `eval/harness.py` — real, calibrated recall/FPR ablation (not a toy 2-example gate) | `results/eval_ablation.json`, see numbers below |
| 9 | Adapters: `adapters/langchain.py`, `adapters/llamaindex.py`, `adapters/nemo.py` | `examples/langchain_demo.py`, `examples/llamaindex_demo.py`, `examples/nemo_demo.py` — all run against the real installed frameworks + real ONNX model, all PASS |
| 10 | `demo/server.py` + `demo/index.html` — stdlib-only (no Flask), chunks/trust/scores/verdict/evidence in one page | Started, hit with `curl`, verified JSON response, shut down cleanly |
| 11 | `fireguard/pyproject.toml`, built sdist+wheel, published to PyPI | `pip install fireguard` from the real public index into a throwaway venv, quickstart ran end to end |
| v1.1 (partial) | `hidden/images.py` — PNG/JPEG scanning: content-check (OCR text scored by L2's detector) + best-effort low-contrast flag | `tests/test_hidden_images.py`, `tests/test_hidden_dispatch.py`; real end-to-end run against the real ONNX detector, see below |

## Images (PNG/JPEG) — what shipped and why the design changed mid-build

Original plan (per the design-research pass): mirror the PDF/HTML hidden-text
checks — OCR-extract text, flag it if it looks concealed (low contrast,
transparent, tiny). **The user caught a real gap in that plan before any code
was written**: a plainly *visible* attack sentence rendered in an image
(typographic prompt injection — a real, well-documented attack class) would
never be flagged, because nothing about it is hidden. Unlike a PDF or HTML
page, an image has no separate caller-driven text-extraction step feeding
`chunk.text` — if fireguard's image module doesn't read the pixels, nothing
else in the pipeline ever sees what's in the image, hidden or not.

Fix: `scan_image` does two independent things, not one — (1) scores every
piece of OCR-extracted text against the same injection detector/threshold as
regular chunk text (the main capability, verified working end to end against
the real model — see below), and (2) a secondary, best-effort low-contrast
flag.

**A second finding, from measuring the actual OCR engine (RapidOCR) rather
than assuming): its own text-*detection* stage has a hard contrast floor.**
Text dimmed below roughly a 0.08–0.14 background-contrast delta is never
even surfaced as a candidate region — not flagged, not seen at all. Tiny
font sizes (down to ~8px) and heavy alpha transparency were both still
reliably detected in testing, so those aren't effective hiding techniques
against this engine either. Net effect: the "find hidden image text" half of
the original plan provides real coverage only in a narrow "detectable but
dim" band, categorically less than the PDF check gets (a PDF's text object
and its color exist in the file regardless of rendering; a raster image has
no equivalent structure to fall back on). README documents this as a hard
limit, not a tuning problem.

**Verified end to end against the real ONNX detector** (not just the fake
one): a plainly visible "ignore all previous instructions..." rendered as
ordinary image text in a `Firewall(db_path=":memory:").admit()` call —
`blocked=True`, reason `"image text scores as injection ...: score 1.000 >=
threshold 0.620"`. A benign image caption — `blocked=False`.

New optional dependency: `rapidocr-onnxruntime` (pure Python, no torch, no
system Tesseract binary), added as the `fireguard[ocr]` extra — confirmed
that without it installed, image bytes are silently skipped (not an error),
same defensive contract as malformed DOCX/HTML.

**Not done yet: `topic_drift.py` (L3, query-relevance).** Fully researched
and design-planned (MiniLM-L6-v2 via ONNX confirmed working in isolation;
spike methodology designed using BM25-cross-topic squad positives + hotpot_qa
distractor-config negatives for the false-positive check) but
`spike/spike_topic_drift.py` itself was never finished — interrupted to
handle PyPI publishing instead. No production code exists for this.

## The one real empirical finding (detector + normalize, Phase 8)

`eval/harness.py`, quick mode, fixed seed, 5% target FPR:

| Variant | Recall |
|---|---|
| Detector alone, plain injections | 0.60 |
| Detector alone, homoglyph+zero-width obfuscated injections | 0.07 |
| Detector + `normalize.py`, same obfuscated injections | 0.60 |

Two things this changes from the 10 Sep snapshot:

1. **0.60, not 1.0, is the real recall number.** Phase 0a's gate (attack
   score 1.0, benign score 0.0) used two hand-picked, maximally obvious
   sentences to sanity-check the model loads and runs — it was never a
   real-world recall claim. This is. The README says so explicitly now.
2. **`normalize.py` has a demonstrated, measured contribution** (not just
   architectural tidiness): it recovers 0.53 of recall that simple
   obfuscation destroys. This is the first hard evidence for any v1
   component beyond the trust-tier algebra, which is deterministic and
   doesn't need an eval to prove.

Not yet run: `--full` mode (200/400 samples, tighter estimate). Quick mode
is directionally solid but thin — same caveat the Phase 0c spike carried.

## Architecture as shipped

```
CONTENT ARRIVES          ->  fw.admit(chunk)
  upload / fetch / ingest     normalize -> hidden-text -> trust -> detect
                               (a block here is itself an incident --
                                recorded before admit() returns)

    ... stored, maybe embedded ...

BEFORE THE PROMPT        ->  fw.scan(retrieval_set)
  retrieved chunks             normalize -> trust -> per-chunk detect
                               verdict: block (nothing survived) /
                               flag (partial) / allow

    ... LLM answers ...

AFTER THE ANSWER         ->  fw.verify(query, answer, sources)
                               L4 hijack + exfiltration checks ->
                               feedback: every source behind a bad
                               answer takes an incident, tightening
                               its L1 trust for next time
```

`Firewall()` with no arguments lazy-loads the real ONNX detector on first
actual use (not at construction) — building a `Firewall` to inspect config
or run only `normalize()` stays cheap and network-free.

## Design decisions made while building (not in the 9 Sep plan verbatim)

- **`Context` dataclass introduced in Phase 3.** The plan's `CHECKS = [fn]`
  list assumed a `(chunks, detector)` signature; once `trust.py` needed
  `store.py` too, every check's signature had to bundle both. `Context`
  keeps `CHECKS` at "one append, not a redesign" as more state arrives.
- **`hidden/documents.py` reimplements `hidden-text-detector` natively**
  instead of wrapping it. The Phase 0b spike validated the concept, but the
  tool itself is a Claude Agent Skill + CLI, not installable into this
  library's dependency tree from any environment. pymupdf span
  color/size/position + python-docx's `Run.font.hidden` cover the same two
  signals (white-on-white/off-page/near-zero-size; Word's real hidden-text
  flag) with packages already needed elsewhere.
- **`hidden/html.py` has no CSS engine.** No `cssselect` in the tree, and
  adding one for a handful of selector shapes wasn't worth a new dependency.
  Matches inline styles + single class/id `<style>` rules — what real
  `display:none` injection PoCs use — not compound selectors or cascades.
  Documented as a limitation, not silently unhandled.
- **NeMo Guardrails is NOT a `fireguard[extra]`**, unlike langchain-core and
  llama-index-core. It's a materially heavier dependency, and the whole
  point of the NeMo adapter is "you don't have to take on our dependencies
  to get this" — bundling it as an extra would undercut that.
- **`examples/nemo_demo.py` proves the adapter against a real
  `nemoguardrails.LLMRails` instance** (constructed with no LLM configured)
  rather than a fake stand-in, without needing an LLM API key — it
  registers the real action and calls it directly, which is the part of
  the integration that's actually ours.

## What's genuinely left

- **`eval/harness.py --full`** for a tighter, less-thin recall estimate.
- **v1.1: `topic_drift.py`** (semantic/embedding cross-document detection,
  the cut Phase 0c feature's successor) — unstarted, needs its own spike.
- **Demo UI polish** — functional (verified via curl + browser-shaped JS),
  not load-tested or styled beyond "legible."
- Isolation, tool-call gating, agentic session-buffer scanning: out of
  scope by design (see README "what this does not protect against"), not
  oversights.

---

# Snapshot: 10 Sep 2026 (superseded above)

## What it is

An open-source, local-first Python library that packages RAG guardrails as one-line
defaults:

```python
from fireguard import Firewall

fw = Firewall()                      # sensible RAG defaults, no arguments
fw.admit(chunk)                      # at ingest / upload / fetch
fw.scan(retrieval_set)               # before the retrieved chunks hit the prompt
fw.verify(query, answer, sources)    # after the model answers
```

**The contribution is accessibility, not new detection capability.** Every defensive
piece already exists for free; there is no packaged way to assemble them, so most RAG
apps ship with nothing. We package the assembly.

**Honest positioning:** risk reduction and defence-in-depth, *not* a security boundary.
Detection is probabilistic and an adaptive attacker beats any classifier. The strongest
defence (isolation) is architectural and a library can't impose it — we document it, we
don't claim it.

- Name: **`fireguard`** (confirmed free on PyPI, 9 Sep). Previous name `ragguard` was taken.
- Threat model: OWASP `LLM01` (prompt injection) and `LLM08` (retrieval poisoning).
- Architecture: the standard rails pattern (input / retrieval / output), same shape as
  NeMo Guardrails. The novelty is what runs *inside* the retrieval rail.
- Target machine: no GPU, 16 cores, ~3 GB RAM free, swap exhausted, Python 3.12.

See git history and the "Phase 0" spike results for the detailed feasibility work behind
these decisions; superseded status details above.
