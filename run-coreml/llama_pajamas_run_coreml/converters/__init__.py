"""CoreML model converters (PyTorch → CoreML)."""

from .base import CoreMLConverter, CoreMLConverterConfig
from .vision import (
    YOLOv8Converter,
    CLIPConverter,
    ViTConverter,
    convert_yolo,
    convert_clip,
    convert_vit,
)

# TODO: Week 5-6 - Speech converters
# from .speech import convert_whisper, convert_fastspeech2

__all__ = [
    # Base classes
    "CoreMLConverter",
    "CoreMLConverterConfig",
    # Vision converters
    "YOLOv8Converter",
    "CLIPConverter",
    "ViTConverter",
    "convert_yolo",
    "convert_clip",
    "convert_vit",
]
