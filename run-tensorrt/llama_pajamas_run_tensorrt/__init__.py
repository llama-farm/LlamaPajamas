"""TensorRT runtime for Llama-Pajamas.

Optimized inference on NVIDIA GPUs using TensorRT.

Supported:
- LLM inference (INT8, FP16, INT4 quantization)
- Vision models (YOLO, ViT, CLIP)
- High-performance batch processing
- Dynamic shape optimization
"""

from .backends import TensorRTLLMBackend, TensorRTVisionBackend

__version__ = "0.1.0"

__all__ = [
    "TensorRTLLMBackend",
    "TensorRTVisionBackend",
]
