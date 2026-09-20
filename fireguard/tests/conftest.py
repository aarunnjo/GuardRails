"""Shared fixtures. Every test uses FakeDetector, never OnnxDetector -- real
model tests belong in eval/harness.py, which needs a network+RAM budget
tests should not depend on to pass in CI."""
import pytest


class FakeDetector:
    """Deterministic stand-in for backends.onnx_detector.OnnxDetector.
    Scores by substring match so tests can construct exact scenarios."""

    def __init__(self, hot_words: tuple[str, ...] = ("ignore all previous",)):
        self.hot_words = hot_words

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[float]:
        return [0.99 if any(w in t.lower() for w in self.hot_words) else 0.01 for t in texts]

    def n_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def memory_db(tmp_path):
    return str(tmp_path / "trust.db")
