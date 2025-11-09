"""Llama-Pajamas Runtime - CoreML Backend (Apple Silicon Multi-Modal).

Supports:
- Vision: Object detection (YOLO), classification (ViT), embeddings (CLIP)
- Speech: Speech-to-Text (Whisper), Text-to-Speech (FastSpeech2)
- Optimized for Apple Neural Engine (ANE) on Mac M1/M2/M3 and iOS
"""

from .backends.vision import CoreMLVisionBackend
from .backends.speech_stt import CoreMLSTTBackend
from .backends.speech_tts import CoreMLTTSBackend

__version__ = "0.2.0"

__all__ = [
    "CoreMLVisionBackend",
    "CoreMLSTTBackend",
    "CoreMLTTSBackend",
]
