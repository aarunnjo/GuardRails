from fireguard.normalize import normalize_text


def test_strips_zero_width_space():
    assert normalize_text("ig\u200bnore") == "ignore"


def test_nfkc_collapses_styled_unicode():
    # MATHEMATICAL BOLD SMALL A -> "a"
    assert normalize_text("\U0001d41a") == "a"


def test_folds_cyrillic_homoglyphs():
    # Cyrillic а, е, о look identical to Latin a, e, o
    assert normalize_text("аdmin usеr") == "admin user"


def test_keeps_normal_whitespace_and_zwj():
    text = "line1\nline2\ttab‍zwj"
    out = normalize_text(text)
    assert "\n" in out and "\t" in out and "‍" in out


def test_clean_text_round_trips():
    text = "Mitochondria generate ATP through oxidative phosphorylation."
    assert normalize_text(text) == text
