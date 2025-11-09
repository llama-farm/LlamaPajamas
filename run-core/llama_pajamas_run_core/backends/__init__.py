"""Backend interfaces for LLM, vision, and speech inference."""

from .base import Backend
from .vision_base import (
    VisionBackend,
    DetectionResult,
    ClassificationResult,
    BoundingBox,
)
from .speech_base import (
    SpeechBackend,
    STTBackend,
    TTSBackend,
    TranscriptionResult,
    TranscriptionSegment,
)

__all__ = [
    # LLM backend
    "Backend",
    # Vision backend
    "VisionBackend",
    "DetectionResult",
    "ClassificationResult",
    "BoundingBox",
    # Speech backends
    "SpeechBackend",
    "STTBackend",
    "TTSBackend",
    "TranscriptionResult",
    "TranscriptionSegment",
]
