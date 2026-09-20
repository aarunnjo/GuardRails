"""L5 -- hidden text in HTML. admit()-only, same reasoning as documents.py:
this needs the markup, which is gone once content becomes a plain-text
chunk.

No CSS engine dependency: `cssselect` isn't in the tree, and pulling one in
for a handful of selector shapes isn't worth a new dependency. This module
extracts inline `style` attributes directly, plus `<style>` block rules with
a single class or id selector (`.foo{...}` / `#foo{...}`) -- the two shapes
every real display:none prompt-injection PoC actually uses. Combinators,
attribute selectors, and cascade/specificity are out of scope; a payload
hidden only via a compound selector will pass this check (see README).
"""
import re

from lxml import html as lxml_html

HIDDEN_PATTERNS = [
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"opacity\s*:\s*0(\.0+)?\b", re.IGNORECASE),
    re.compile(r"font-size\s*:\s*0(\.0+)?(px|pt|em)?\b", re.IGNORECASE),
]
OFFSCREEN_PATTERN = re.compile(r"position\s*:\s*absolute.*?left\s*:\s*-\d{3,}px", re.IGNORECASE | re.DOTALL)
STYLE_RULE = re.compile(r"([.#][\w-]+)\s*\{([^}]*)\}")


def _is_hidden_style(style: str) -> bool:
    return any(p.search(style) for p in HIDDEN_PATTERNS) or bool(OFFSCREEN_PATTERN.search(style))


def _stylesheet_rules(tree) -> dict[str, str]:
    rules: dict[str, str] = {}
    for style_tag in tree.xpath("//style"):
        for selector, body in STYLE_RULE.findall(style_tag.text_content() or ""):
            rules[selector] = rules.get(selector, "") + ";" + body
    return rules


def scan_html(markup: str) -> list[str]:
    try:
        tree = lxml_html.fromstring(markup)
    except Exception:
        return []

    rules = _stylesheet_rules(tree)
    findings = []

    for el in tree.iter():
        text = (el.text or "").strip()
        if not text:
            continue

        inline = el.get("style", "")
        if _is_hidden_style(inline):
            findings.append(f"<{el.tag}> inline style hides text {text[:60]!r}")
            continue

        hit = False
        for cls in (el.get("class") or "").split():
            style = rules.get(f".{cls}")
            if style and _is_hidden_style(style):
                findings.append(f"<{el.tag}> class .{cls} hides text {text[:60]!r}")
                hit = True
                break
        if hit:
            continue

        el_id = el.get("id")
        if el_id:
            style = rules.get(f"#{el_id}")
            if style and _is_hidden_style(style):
                findings.append(f"<{el.tag}> id #{el_id} hides text {text[:60]!r}")

    return findings
