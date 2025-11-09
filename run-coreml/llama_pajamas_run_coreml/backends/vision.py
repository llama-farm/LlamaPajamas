"""CoreML vision backend for Apple Silicon.

Supports:
- Object detection (YOLO-v8, etc.)
- Image classification (ViT, ResNet, etc.)
- Image embeddings (CLIP, DINO, etc.)

Optimized for Apple Neural Engine (ANE).
"""

import logging
from pathlib import Path
from typing import List, Optional, Any, Union, Tuple
import numpy as np
from PIL import Image

from llama_pajamas_run_core.backends import (
    VisionBackend,
    DetectionResult,
    ClassificationResult,
    BoundingBox,
)

# Import shared vision utilities from core
from llama_pajamas_run_core.utils import (
    preprocess_image,
    postprocess_yolo_detections,
    postprocess_vit_classification,
    postprocess_clip_embedding,
    get_coco_class_names,
    get_model_info,
)

logger = logging.getLogger(__name__)


class CoreMLVisionBackend(VisionBackend):
    """CoreML implementation of VisionBackend.

    Uses coremltools for model loading and inference on Apple Silicon.
    Automatically leverages Apple Neural Engine (ANE) when available.
    """

    def __init__(self):
        """Initialize CoreML vision backend."""
        self.model = None
        self._model_path: Optional[Path] = None
        self._model_type: Optional[str] = None
        self._class_names: Optional[List[str]] = None
        self._input_size: Optional[Tuple[int, int]] = None
        self._conf_threshold: float = 0.5
        self._iou_threshold: float = 0.45

        # Check CoreML availability
        try:
            import coremltools as ct  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "coremltools not installed. Install with: "
                "pip install coremltools or uv add coremltools"
            )

    def load_model(self, model_path: str, model_type: str, **kwargs) -> None:
        """Load CoreML vision model.

        Args:
            model_path: Path to .mlpackage or .mlmodel file
            model_type: One of: 'detection', 'classification', 'embedding'
            **kwargs:
                - class_names: List[str] for detection/classification
                - input_size: Tuple[int, int] for resizing (default: model input size)
                - confidence_threshold: float (default: 0.5)
                - iou_threshold: float (default: 0.45)
        """
        import coremltools as ct

        if model_type not in ["detection", "classification", "embedding"]:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                "Must be 'detection', 'classification', or 'embedding'"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading CoreML model: {model_path}")
        logger.info(f"Model type: {model_type}")

        # Load CoreML model with compute units set to ALL (ANE + GPU + CPU)
        self.model = ct.models.MLModel(
            str(model_path),
            compute_units=ct.ComputeUnit.ALL,  # Use ANE when possible
        )

        self._model_path = model_path
        self._model_type = model_type
        self._class_names = kwargs.get("class_names")
        self._input_size = kwargs.get("input_size")
        self._conf_threshold = kwargs.get("confidence_threshold", 0.25)
        self._iou_threshold = kwargs.get("iou_threshold", 0.7)

        # Auto-detect input size and class names from model if not provided
        if self._input_size is None and model_type != "embedding":
            # Try to get input size from model spec
            spec = self.model.get_spec()
            if len(spec.description.input) > 0:
                input_desc = spec.description.input[0]
                if hasattr(input_desc.type, "imageType"):
                    self._input_size = (
                        input_desc.type.imageType.width,
                        input_desc.type.imageType.height,
                    )

        # Set default class names based on model type
        if self._class_names is None:
            if model_type == "detection":
                self._class_names = get_coco_class_names()
                logger.info(f"   Using default COCO class names (80 classes)")
            elif model_type == "classification":
                # Will use class indices if labels not provided
                self._class_names = None

        logger.info(f"✅ CoreML model loaded successfully")
        logger.info(f"   Compute units: ALL (ANE + GPU + CPU)")
        if self._input_size:
            logger.info(f"   Input size: {self._input_size}")

    def detect(
        self,
        image: Any,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> List[DetectionResult]:
        """Perform object detection using YOLO-v8 or similar models.

        Args:
            image: PIL Image, file path, or numpy array
            confidence_threshold: Minimum confidence (default: 0.5, use model default if None)
            iou_threshold: IoU threshold for NMS (default: 0.45, usually handled by model)

        Returns:
            List of DetectionResult objects sorted by confidence (descending)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._model_type != "detection":
            raise RuntimeError(
                f"Model type is '{self._model_type}', not 'detection'. "
                "Load a detection model first."
            )

        # Use model default or provided threshold
        conf_threshold = confidence_threshold or self._conf_threshold
        iou_threshold = iou_threshold or self._iou_threshold

        # 1. Preprocess image - CoreML expects PIL.Image
        target_size = self._input_size or (640, 640)
        preprocessed = preprocess_image(
            image=image,
            target_size=target_size,
            normalize=False,  # CoreML ImageType handles normalization
        )

        # 2. Run CoreML inference
        output = self.model.predict({"image": preprocessed})

        # 3. Extract outputs (YOLO-v8 format: coordinates + confidence)
        # Output keys vary by model, try common variants
        if "coordinates" in output and "confidence" in output:
            coordinates = output["coordinates"]
            confidences = output["confidence"]
        elif "var_1387" in output and "var_1388" in output:
            # CoreML auto-generated names
            coordinates = output["var_1387"]
            confidences = output["var_1388"]
        else:
            # Try first two outputs
            output_keys = list(output.keys())
            if len(output_keys) >= 2:
                coordinates = output[output_keys[0]]
                confidences = output[output_keys[1]]
            else:
                raise RuntimeError(
                    f"Unexpected output format. Keys: {output_keys}. "
                    "Expected 'coordinates' and 'confidence' (or similar)."
                )

        # 4. Postprocess detections using shared utility
        detections = postprocess_yolo_detections(
            coordinates=coordinates,
            confidences=confidences,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=self._class_names,
        )

        # Sort by confidence (descending)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(f"Detected {len(detections)} objects (conf >= {conf_threshold})")
        return detections

    def classify(self, image: Any, top_k: int = 5) -> List[ClassificationResult]:
        """Perform image classification using ViT, ResNet, or similar models.

        Args:
            image: PIL Image, file path, or numpy array
            top_k: Number of top predictions to return (default: 5)

        Returns:
            List of ClassificationResult objects sorted by confidence (descending)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._model_type != "classification":
            raise RuntimeError(
                f"Model type is '{self._model_type}', not 'classification'. "
                "Load a classification model first."
            )

        # 1. Preprocess image - CoreML expects PIL.Image
        target_size = self._input_size or (224, 224)
        preprocessed = preprocess_image(
            image=image,
            target_size=target_size,
            normalize=False,  # CoreML ImageType handles normalization
        )

        # 2. Run CoreML inference
        output = self.model.predict({"image": preprocessed})

        # 3. Extract logits (try common output names)
        if "logits" in output:
            logits = output["logits"]
        elif "output" in output:
            logits = output["output"]
        elif "classLabel_probs" in output:
            # Some models output probabilities directly
            logits = output["classLabel_probs"]
        else:
            # Use first output
            output_keys = list(output.keys())
            if len(output_keys) > 0:
                logits = output[output_keys[0]]
            else:
                raise RuntimeError(
                    f"Unexpected output format. Keys: {output_keys}. "
                    "Expected 'logits' or 'output'."
                )

        # 4. Postprocess classification using shared utility
        predictions = postprocess_vit_classification(
            logits=logits,
            class_names=self._class_names,
            top_k=top_k,
        )

        logger.debug(f"Top-{top_k} classifications: {[p.class_name for p in predictions]}")
        return predictions

    def embed(self, image: Any) -> np.ndarray:
        """Generate image embeddings using CLIP, DINO, or similar models.

        Args:
            image: PIL Image, file path, or numpy array

        Returns:
            Embedding vector (e.g., 512-D for CLIP, 768-D for ViT)
            Shape: (embedding_dim,)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._model_type != "embedding":
            raise RuntimeError(
                f"Model type is '{self._model_type}', not 'embedding'. "
                "Load an embedding model first."
            )

        # 1. Preprocess image - CoreML expects PIL.Image
        target_size = self._input_size or (224, 224)
        preprocessed = preprocess_image(
            image=image,
            target_size=target_size,
            normalize=False,  # CoreML ImageType handles normalization
        )

        # 2. Run CoreML inference
        output = self.model.predict({"image": preprocessed})

        # 3. Extract embedding (try common output names)
        if "embedding" in output:
            embedding = output["embedding"]
        elif "output" in output:
            embedding = output["output"]
        elif "pooler_output" in output:
            # CLIP models
            embedding = output["pooler_output"]
        else:
            # Use first output
            output_keys = list(output.keys())
            if len(output_keys) > 0:
                embedding = output[output_keys[0]]
            else:
                raise RuntimeError(
                    f"Unexpected output format. Keys: {output_keys}. "
                    "Expected 'embedding' or 'output'."
                )

        # 4. Postprocess embedding using shared utility (L2 normalization)
        embedding = postprocess_clip_embedding(
            embedding=embedding,
            normalize=True,
        )

        logger.debug(f"Generated embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        return embedding

    def batch_detect(
        self,
        images: List[Any],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> List[List[DetectionResult]]:
        """Batch object detection."""
        # Simple implementation: loop over images
        return [
            self.detect(img, confidence_threshold, iou_threshold, **kwargs)
            for img in images
        ]

    def batch_classify(
        self, images: List[Any], top_k: int = 5
    ) -> List[List[ClassificationResult]]:
        """Batch image classification."""
        return [self.classify(img, top_k) for img in images]

    def batch_embed(self, images: List[Any]) -> np.ndarray:
        """Batch image embeddings."""
        embeddings = [self.embed(img) for img in images]
        return np.stack(embeddings, axis=0)

    def unload(self) -> None:
        """Unload model and free resources."""
        if self.model is not None:
            logger.info(f"Unloading CoreML model: {self._model_path}")
            self.model = None
            self._model_path = None
            self._model_type = None
            self._class_names = None

    @property
    def model_type(self) -> str:
        """Get current model type."""
        if self._model_type is None:
            raise RuntimeError("Model not loaded")
        return self._model_type

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
