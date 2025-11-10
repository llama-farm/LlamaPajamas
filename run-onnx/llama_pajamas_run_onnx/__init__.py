"""ONNX Runtime backend for edge deployment.

Supports CPU, AMD GPU, ARM processors, NVIDIA Jetson, and more.
Optimized for vision and speech models with INT8/INT4 quantization.
"""

from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend
from llama_pajamas_run_onnx.backends.speech import ONNXSpeechBackend

__all__ = [
    "ONNXVisionBackend",
    "ONNXSpeechBackend",
]

__version__ = "0.1.0"
