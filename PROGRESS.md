# fireguard — progress so far

*Snapshot: 10 Sep 2026*

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

## Timeline

| Date | Milestone |
|---|---|
| 28 Aug | First research plan — later abandoned (its central result was circular by construction) |
| 4 Sep  | Incumbent survey — no code yet |
| 7 Sep  | Phase 0a: detector gate passed |
| 9 Sep  | Phase 0b + 0c spikes run; current plan written; cross-document feature cut |
| 9–10 Sep | Phase 1 + 2 code written |

Current plan: `~/.claude/plans/so-here-are-we-resilient-charm.md`

---

## Phase 0 — feasibility spikes (DONE)

Every expensive phase sits behind a cheap spike that could kill it. Three ran.

### 0a — the detector &nbsp;·&nbsp; ✅ PASS

`protectai/deberta-v3-base-prompt-injection-v2`, run via **ONNX Runtime with no PyTorch**
(the model ships a pre-exported ONNX file — this removes ~2 GB from the RAM budget).

| Metric | Result | Target |
|---|---|---|
| Attack text score | 1.0000 | > 0.9 |
| Benign text score | 0.0000 | < 0.1 |
| Peak RSS | 1326 MB | < 3072 MB |
| Load time | 4.5 s | — |

Result file: `results/phase0a_gate.json` · code: `spike/detect.py`

### 0b — can a long-context detector skip the windowing problem? &nbsp;·&nbsp; ❌ NO

`deberta-v3-base` caps at 512 tokens; a 20-chunk retrieval set is ~4000. If a
4k–8k-context injection classifier worked, the whole windowed-scanning design would
disappear.

Tested `tihilya/modernbert-base-prompt-injection-detection` (8192-token context,
confirmed in config, loaded fine at 1089 MB). **It failed the actual test:** the same
attack sentence scored **1.0000 alone** but **0.0226 when buried in 1218 tokens of
filler** — classic needle-in-haystack. An architecture that *accepts* long input is not
the same as a fine-tune that can *use* it for sparse-signal detection.

→ Rejected. Result file: `results/phase0b_modernbert.json`

### 0b — is `hidden-text-detector` usable? &nbsp;·&nbsp; ✅ YES (with a caveat)

Built our own test PDF (one visible sentence + one genuine white-on-white injection
payload) rather than trusting the README. It correctly flagged the hidden sentence and
ignored the visible one, using real pixel-contrast measurement.

**Caveat:** it is *not* a PyPI package — it ships as an Agent Skill + standalone CLI
script. Phase 5 must vendor `scripts/scan.py` and call it as a subprocess / direct
import, not add it as a pip dependency.

Result file: `results/phase0b_hidden_text.json`

### 0c — cross-document set-scanning &nbsp;·&nbsp; ❌ CUT FROM v1

This was one of the three originally-unclaimed differentiators: scanning the retrieved
set *jointly* to catch chunks that are individually benign but conspire.

Spike: took real injections, split each at a sentence boundary into `(A, B)`, planted
the halves at varying distances in 20-chunk sets, measured recall. Two independent runs,
calibrated to equal false-positive rate.

| distance | per-chunk | tier 1 (windowing) | tier 2 (suspicion-sort) |
|---|---|---|---|
| 1  | 0.15 | 0.05 | 0.15 |
| 10 | 0.05 | 0.05 | 0.00 |
| 19 | 0.10 | 0.05 | 0.05 |

- **Suspicion-ordering (tier 2): dead.** AUC of split-halves vs. clean = 0.389 / 0.396
  across two runs. No signal to sort on.
- **Windowing (tier 1): never beats per-chunk scanning**, at any distance. Joining chunks
  dilutes a borderline signal — the same effect that killed 0b, in miniature.
- This was even the *easier* version of the threat (mechanically split real injections,
  which keep some attack-flavored wording). Failing that is not encouraging for the
  harder "two innocent sentences that conspire" case.

→ **Cut.** The README's "what this does not protect against" section will name this
explicitly. A semantic/embedding-based successor (`topic_drift.py`) is a v1.1 research
candidate — different mechanism, so it doesn't inherit any of the above.

Result file: `results/spike_l3.json` · code: `spike/spike_l3.py`

**Net effect of Phase 0:** two of the three differentiators survive for v1 — **source
trust tiers that tune the detector threshold**, and the **CSS-hidden-HTML check**.
Everything else is assembly of existing parts.

---

## Phases 1–2 — first real code (DONE)

Package: `fireguard/fireguard/`

| File | Status | What it does |
|---|---|---|
| `types.py` | ✅ | `Chunk`, `RetrievalSet`, `ScanResult`, `AdmitVerdict`, `VerifyResult` — plain dataclasses, the shapes every module shares |
| `backends/base.py` | ✅ | `Detector` protocol (`score` / `score_batch` / `n_tokens`). Everything calls through this, never a model directly — the fix for when the old plan's dependency (`llm-guard`) went archived |
| `firewall.py` | ✅ shell | `Firewall.admit / scan / verify`. `scan()` runs an ordered `CHECKS` list so a new check is one append, not a redesign |
| `normalize.py` | ✅ | **Phase 2.** Strips invisible-character manipulation: NFKC → drop invisible Unicode categories (zero-width, control, private-use) → fold a small set of Cyrillic/Greek homoglyphs. First real check wired into the `scan()` pipeline |

`admit()` and `verify()` are still stubs. Trust and detection inside `scan()` are still
stubs. That's expected — they're Phases 3–4.

---

## What's next

| Phase | Build | Done when |
|---|---|---|
| **3 (next)** | `trust.py` + `store.py` | Source tiers + incident history in SQLite; the detector threshold visibly tightens for a `user_upload` chunk vs an `internal_kb` one |
| 4 | `admit()` + `hidden/documents.py` | A white-on-white PDF is rejected at ingest |
| 5 | `hidden/html.py` | A `display:none` payload in HTML is flagged |
| 6 | Feedback loop | A repeat-offender source's trust measurably drops |
| 7 | Docs (README, quickstart, threat model, limitations) | Quickstart runs verbatim in a clean venv |
| 8 | Eval harness + ablation | Results table reproduces from a fixed seed |
| 9 | LangChain / LlamaIndex / NeMo adapters | Poisoned-vs-clean demo |
| 10 | Demo UI | — |
| 11 | PyPI release | `pip install fireguard` works in a clean venv |

Rough estimate from here: **7–9 weeks.**

### Loose ends before Phase 3

- `fireguard/tests/` exists but is **empty** — no tests for `normalize.py` yet
- `spike/detect.py` not yet promoted into `fireguard/backends/`
- No `pyproject.toml`
- Nothing committed to git (repo initialised, zero commits)

---

## How this compares to NeMo Guardrails

NeMo's retrieval rails are AlignScore, Self Check Facts, AutoAlign, and Presidio PII
masking — i.e. three output fact-checks plus per-chunk PII scrubbing. Its docs carry
nothing for source trust / provenance, Unicode normalization, or hidden text.

The real difference is the **cost model**: NeMo's self-check rails work by asking an LLM,
so every check is an API call with tokens and latency; its strongest rails are commercial
integrations. `fireguard` is a local classifier — no API key, no egress, no per-request
cost.

**So: compose, don't compete.** NeMo supports custom rails and this is exactly what its
retrieval rails lack — hence the Phase 9 NeMo adapter.
