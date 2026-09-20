from .base import Detector

__all__ = ["Detector"]

# OnnxDetector is deliberately not imported here: it pulls in onnxruntime +
# transformers + a model download, and most of the package (types, trust,
# store, normalize) has nothing to do with any of that. Import it directly
# from fireguard.backends.onnx_detector, or just construct Firewall() with
# no detector -- it lazy-loads OnnxDetector on first use.
