"""CoreML backend implementations for vision and speech."""

from .vision import CoreMLVisionBackend
from .speech_stt import CoreMLSTTBackend
from .speech_tts import CoreMLTTSBackend

__all__ = [
    "CoreMLVisionBackend",
    "CoreMLSTTBackend",
    "CoreMLTTSBackend",
]
