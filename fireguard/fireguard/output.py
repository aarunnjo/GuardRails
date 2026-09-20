"""L4 -- output checks, run in verify(). Reuses L2's detector (no new model
loaded) and adds two independent signals:

  1. hijack signal -- score the ANSWER itself with the injection detector.
     A high score means the model's output reads like injected instructions
     rather than a response to the user's question: evidence the model
     complied with something it retrieved rather than the user's query.
  2. exfiltration signal -- any email address or URL in the answer that
     appears nowhere in the query or the retrieved sources is exactly the
     leak channel a successful injection needs to introduce.

Neither signal proves compliance happened; both are the kind of cheap,
model-reused checks that catch the common, low-effort case without a new
dependency. See README "what this does not protect against" for the honest
limits (a sufficiently careful exfiltration -- e.g. encoded, or reusing a
URL already present in a source -- passes both).
"""
import re

from .types import Chunk, VerifyResult

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://[^\s)>\]]+")
TRAILING_PUNCT = ".,;:!?)"


def _find_all(pattern: re.Pattern, text: str) -> set[str]:
    return {m.rstrip(TRAILING_PUNCT) for m in pattern.findall(text)}

# Looser than L2's per-chunk bar (trust.py's MAX_THRESHOLD of 0.9): verify()
# is a last-resort net after admit()/scan() already ran, not the primary
# detection point, so it leans toward catching more at the cost of more
# false positives on genuinely instruction-shaped answers.
HIJACK_THRESHOLD = 0.7


def _known_strings(query: str, sources: list[Chunk]) -> set[str]:
    haystack = query + " " + " ".join(c.text for c in sources)
    return _find_all(EMAIL_RE, haystack) | _find_all(URL_RE, haystack)


def check_output(query: str, answer: str, sources: list[Chunk], detector) -> VerifyResult:
    reasons = []

    if detector is not None and answer.strip():
        score = detector.score(answer)
        if score >= HIJACK_THRESHOLD:
            reasons.append(f"L4 hijack signal: answer scores {score:.3f} as injection-like")

    known = _known_strings(query, sources)
    leaked = (_find_all(EMAIL_RE, answer) | _find_all(URL_RE, answer)) - known
    if leaked:
        reasons.append(f"L4 exfiltration signal: answer contains unseen {sorted(leaked)}")

    return VerifyResult(verdict="block" if reasons else "allow", reasons=reasons)
