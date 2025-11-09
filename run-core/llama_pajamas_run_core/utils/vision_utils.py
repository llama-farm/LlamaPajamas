"""Vision preprocessing and postprocessing utilities for all runtimes.

Provides reusable functions for YOLO detection, CLIP embeddings, and ViT classification.
Shared across CoreML, ONNX, TensorRT, and other vision backends.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

from ..backends.vision_base import DetectionResult, ClassificationResult, BoundingBox


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_image(
    image: Union[str, Path, Image.Image],
    target_size: Tuple[int, int],
    normalize: bool = False,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Image.Image:
    """
    Preprocess image for vision model inference.

    Args:
        image: Input image (path or PIL Image)
        target_size: (width, height) to resize to
        normalize: Whether to normalize (some backends handle this internally)
        mean: Mean values for normalization (if normalize=True)
        std: Std values for normalization (if normalize=True)

    Returns:
        Preprocessed PIL Image ready for inference
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')

    # Resize
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Note: CoreML ImageType handles normalization automatically
    # Only apply manual normalization if needed for non-ImageType inputs
    if normalize and mean and std:
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - np.array(mean)) / np.array(std)
        img = Image.fromarray((img_array * 255).astype(np.uint8))

    return img


# ============================================================================
# YOLO Postprocessing
# ============================================================================

def postprocess_yolo_detections(
    coordinates: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    class_names: Optional[List[str]] = None,
) -> List[DetectionResult]:
    """
    Postprocess YOLO detection outputs.

    Args:
        coordinates: (N, 4) array of [x, y, width, height] (relative to image size)
        confidences: (N, 80) array of class confidences
        conf_threshold: Confidence threshold for filtering
        iou_threshold: IoU threshold for NMS (usually handled by model)
        class_names: List of class names (COCO dataset)

    Returns:
        List of DetectionResult objects sorted by confidence
    """
    # Default COCO class names if not provided
    if class_names is None:
        class_names = get_coco_class_names()

    detections = []

    for i in range(len(coordinates)):
        # Get max confidence and class
        class_conf = confidences[i]
        class_id = int(np.argmax(class_conf))
        confidence = float(class_conf[class_id])

        # Filter by confidence threshold
        if confidence < conf_threshold:
            continue

        # Get bbox (convert from [x, y, w, h] to [x1, y1, x2, y2])
        bbox_xywh = coordinates[i]  # [x, y, w, h]
        x, y, w, h = bbox_xywh
        bbox = BoundingBox(
            x1=float(x),
            y1=float(y),
            x2=float(x + w),
            y2=float(y + h),
        )

        detection = DetectionResult(
            bbox=bbox,
            class_id=class_id,
            class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
            confidence=confidence,
        )
        detections.append(detection)

    return detections


def get_coco_class_names() -> List[str]:
    """Get COCO dataset class names (80 classes)."""
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]


# ============================================================================
# CLIP Postprocessing
# ============================================================================

def postprocess_clip_embedding(
    embedding: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Postprocess CLIP embedding output.

    Args:
        embedding: Raw embedding from model (1, 768) or (768,)
        normalize: Whether to L2-normalize the embedding

    Returns:
        Processed embedding (768,)
    """
    # Flatten if needed
    if embedding.ndim > 1:
        embedding = embedding.flatten()

    # L2 normalize
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    return embedding


def compute_cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding

    Returns:
        Cosine similarity score [-1, 1]
    """
    # Ensure flattened
    emb1 = embedding1.flatten()
    emb2 = embedding2.flatten()

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)


# ============================================================================
# ViT Postprocessing
# ============================================================================

def postprocess_vit_classification(
    logits: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[ClassificationResult]:
    """
    Postprocess ViT classification output.

    Args:
        logits: Raw logits from model (1, 1000) or (1000,)
        class_names: List of class names (ImageNet-1k)
        top_k: Number of top predictions to return

    Returns:
        List of ClassificationResult objects sorted by confidence
    """
    # Flatten if needed
    if logits.ndim > 1:
        logits = logits.flatten()

    # Apply softmax
    probs = softmax(logits)

    # Get top-k indices
    top_k_indices = np.argsort(probs)[-top_k:][::-1]

    # Build predictions
    predictions = []
    for idx in top_k_indices:
        pred = ClassificationResult(
            class_id=int(idx),
            class_name=class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}",
            confidence=float(probs[idx]),
        )
        predictions.append(pred)

    return predictions


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to logits."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


# ============================================================================
# Batch Processing
# ============================================================================

def batch_preprocess_images(
    image_paths: List[Union[str, Path]],
    target_size: Tuple[int, int],
) -> List[Image.Image]:
    """
    Batch preprocess multiple images.

    Args:
        image_paths: List of image paths
        target_size: (width, height) to resize to

    Returns:
        List of preprocessed PIL Images
    """
    return [preprocess_image(img_path, target_size) for img_path in image_paths]


# ============================================================================
# Model Info
# ============================================================================

def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get model information and recommended settings.

    Args:
        model_type: One of "yolo", "clip", "vit"

    Returns:
        Dictionary with model info
    """
    model_specs = {
        "yolo": {
            "input_size": (640, 640),
            "output_format": "coordinates + confidences",
            "classes": 80,
            "class_names": get_coco_class_names(),
            "conf_threshold": 0.25,
            "iou_threshold": 0.7,
        },
        "clip": {
            "input_size": (224, 224),
            "output_format": "embedding",
            "embedding_dim": 768,
            "normalize": True,
        },
        "vit": {
            "input_size": (224, 224),
            "output_format": "logits",
            "classes": 1000,
            "top_k": 5,
        },
    }

    return model_specs.get(model_type.lower(), {})
