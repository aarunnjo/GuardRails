"""L2's only model-loading module. Promoted from spike/detect.py (Phase 0a)
unchanged except for the import path -- the spike's Detector class already
matched backends/base.py's Protocol, so no adapter code was needed.

ONNX Runtime, no torch: protectai/deberta-v3-base-prompt-injection-v2 ships
a pre-exported onnx/model.onnx, which removes ~2 GB of torch runtime from a
machine with 3 GB free (see plan correction #1 -- llm-guard, the previous
dependency, pinned an old transformers and was archived under us).
"""
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
MAX_TOKENS = 512  # deberta-v3-base's hard cap -- anything longer is silently
                   # truncated by score(); n_tokens() exists so callers can
                   # detect that rather than trust a truncated score


class OnnxDetector:
    def __init__(self, threads: int = 8):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)

        onnx_path = hf_hub_download(MODEL_ID, "onnx/model.onnx")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        # Some ONNX exports don't accept token_type_ids -- filter the
        # tokenizer's output down to what this session actually wants.
        self._accepted = {i.name for i in self.sess.get_inputs()}

    def _run(self, encoded: dict) -> np.ndarray:
        feed = {k: v for k, v in encoded.items() if k in self._accepted}
        logits = self.sess.run(None, feed)[0]
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        return probs[..., 1]  # label 1 is INJECTION for this model

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[float]:
        """Prefer this over calling score() in a loop: batching turns N
        small matmuls (most cores idle) into one large one (cores
        saturated) -- roughly 4x on CPU."""
        if not texts:
            return []
        enc = self.tok(texts, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="np")
        return self._run(dict(enc)).tolist()

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])
