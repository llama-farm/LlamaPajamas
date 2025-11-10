"""CoreML backend implementations for vision and speech."""

from .vision import CoreMLVisionBackend
from .stt import CoreMLSTTBackend
from .speech_tts import CoreMLTTSBackend  # Stub for future CoreML TTS
from .tts_system import SystemTTSBackend  # Working system TTS

__all__ = [
    "CoreMLVisionBackend",
    "CoreMLSTTBackend",
    "CoreMLTTSBackend",
    "SystemTTSBackend",
]
