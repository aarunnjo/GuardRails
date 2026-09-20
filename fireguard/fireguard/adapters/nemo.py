"""NeMo Guardrails adapter -- a custom action that runs fireguard.scan()
over NeMo's retrieved context before the LLM answers.

fireguard is what NeMo's retrieval rails don't have: source trust and a
local injection classifier (no LLM call, no API cost -- see README
"Comparison to NeMo Guardrails"). This module doesn't compete with NeMo's
existing rails, it hooks into the same retrieval-rail slot AlignScore/
Self-Check-Facts use.

Optional, and NOT bundled as a `fireguard[extra]` -- nemoguardrails is a
much heavier dependency than langchain-core/llama-index-core, and pulling
it into fireguard's own extras would defeat the "no forced dependency"
point this adapter exists to make.

Usage (Python side, once per app):
    from fireguard.adapters.nemo import register
    register(rails)  # rails: nemoguardrails.LLMRails instance

Usage (Colang side, in your retrieval flow):
    define flow check retrieved chunks
        $result = execute fireguard_scan
        if $result.verdict == "block"
            bot refuse to respond
        else
            $relevant_chunks = $result.approved_text
"""
from ..firewall import Firewall
from ..types import Chunk, RetrievalSet

_fw: Firewall = Firewall()


def register(llm_rails, firewall: Firewall | None = None) -> None:
    """Call once at startup. Swaps in a custom Firewall (e.g. with a
    pre-loaded detector) if given, else uses the module default."""
    global _fw
    if firewall is not None:
        _fw = firewall

    try:
        import nemoguardrails  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The NeMo adapter needs nemoguardrails installed: pip install nemoguardrails"
        ) from e

    llm_rails.register_action(fireguard_scan, name="fireguard_scan")


def _coerce_chunks(raw_chunks) -> list[str]:
    """NeMo's `relevant_chunks` context variable is a newline-joined string
    by default, but some configs populate it as a list of dicts/strings --
    handle both rather than assuming one."""
    if isinstance(raw_chunks, str):
        return [c for c in raw_chunks.split("\n") if c.strip()]
    if isinstance(raw_chunks, list):
        return [c.get("text", "") if isinstance(c, dict) else str(c) for c in raw_chunks]
    return []


async def fireguard_scan(context: dict | None = None, **kwargs) -> dict:
    """Colang-callable action. Returns a dict (not a fireguard type) since
    Colang flows can only read plain values off an action's return."""
    context = context or {}
    query = context.get("user_message", "")
    texts = _coerce_chunks(context.get("relevant_chunks", []))

    chunks = [Chunk(text=t, source_uri=f"chunk-{i}", tier="open_web") for i, t in enumerate(texts)]
    result = _fw.scan(RetrievalSet(query=query, chunks=chunks))

    return {
        "verdict": result.verdict,
        "allowed": result.verdict != "block",
        "reasons": result.reasons,
        "approved_text": "\n".join(c.text for c in result.approved_chunks),
    }
