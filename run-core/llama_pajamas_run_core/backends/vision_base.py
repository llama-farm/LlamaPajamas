"""Base backend interface for vision model inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Union
import numpy as np

# Type alias for image inputs (allow both PIL and numpy)
ImageInput = Any  # PIL.Image.Image or np.ndarray


@dataclass
class BoundingBox:
    """Bounding box coordinates (normalized or absolute)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class DetectionResult:
    """Object detection result."""

    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox.to_dict(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


@dataclass
class ClassificationResult:
    """Image classification result."""

    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


class VisionBackend(ABC):
    """Abstract base class for vision model inference.

    Supports three types of vision tasks:
    - Object detection (YOLO, Detr, etc.)
    - Image classification (ViT, ResNet, etc.)
    - Image embeddings (CLIP, DINO, etc.)
    """

    @abstractmethod
    def load_model(self, model_path: str, model_type: str, **kwargs) -> None:
        """Load vision model.

        Args:
            model_path: Path to model file (.mlpackage, .engine, .onnx, etc.)
            model_type: One of: 'detection', 'classification', 'embedding'
            **kwargs: Backend-specific parameters
                - class_names: List[str] for detection/classification
                - input_size: Tuple[int, int] for resizing
                - confidence_threshold: float for detection
                - iou_threshold: float for NMS
        """
        pass

    @abstractmethod
    def detect(
        self,
        image: ImageInput,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> List[DetectionResult]:
        """Perform object detection.

        Args:
            image: PIL Image or numpy array (HWC, RGB, uint8 or float32)
            confidence_threshold: Minimum confidence to keep detection
            iou_threshold: IoU threshold for Non-Maximum Suppression
            **kwargs: Backend-specific parameters

        Returns:
            List of detection results sorted by confidence (descending)

        Raises:
            RuntimeError: If model is not loaded or wrong model type
        """
        pass

    @abstractmethod
    def classify(self, image: ImageInput, top_k: int = 5) -> List[ClassificationResult]:
        """Perform image classification.

        Args:
            image: PIL Image or numpy array (HWC, RGB, uint8 or float32)
            top_k: Number of top predictions to return

        Returns:
            List of classification results sorted by confidence (descending)

        Raises:
            RuntimeError: If model is not loaded or wrong model type
        """
        pass

    @abstractmethod
    def embed(self, image: ImageInput) -> np.ndarray:
        """Generate image embeddings.

        Args:
            image: PIL Image or numpy array (HWC, RGB, uint8 or float32)

        Returns:
            Embedding vector (e.g., 512-D for CLIP, 768-D for ViT)
            Shape: (embedding_dim,)

        Raises:
            RuntimeError: If model is not loaded or wrong model type
        """
        pass

    @abstractmethod
    def batch_detect(
        self,
        images: List[ImageInput],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> List[List[DetectionResult]]:
        """Batch object detection.

        Args:
            images: List of PIL Images or numpy arrays
            confidence_threshold: Minimum confidence to keep detection
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detection results per image
        """
        pass

    @abstractmethod
    def batch_classify(
        self, images: List[ImageInput], top_k: int = 5
    ) -> List[List[ClassificationResult]]:
        """Batch image classification.

        Args:
            images: List of PIL Images or numpy arrays
            top_k: Number of top predictions per image

        Returns:
            List of classification results per image
        """
        pass

    @abstractmethod
    def batch_embed(self, images: List[ImageInput]) -> np.ndarray:
        """Batch image embeddings.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            Embedding matrix of shape (batch_size, embedding_dim)
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources (GPU/ANE memory)."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get current model type ('detection', 'classification', or 'embedding')."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
