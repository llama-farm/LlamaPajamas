"""Utility functions for CoreML runtime.

Imports shared vision utilities from core package.
"""

from llama_pajamas_run_core.utils import (
    preprocess_image,
    postprocess_yolo_detections,
    postprocess_clip_embedding,
    postprocess_vit_classification,
    compute_cosine_similarity,
    batch_preprocess_images,
    get_model_info,
    get_coco_class_names,
)

from .image_utils import (
    load_image,
    extract_video_frames,
    batch_load_images,
)

__all__ = [
    # Shared vision utilities (from core)
    "preprocess_image",
    "postprocess_yolo_detections",
    "postprocess_clip_embedding",
    "postprocess_vit_classification",
    "compute_cosine_similarity",
    "batch_preprocess_images",
    "get_model_info",
    "get_coco_class_names",
    # CoreML-specific utilities
    "load_image",
    "extract_video_frames",
    "batch_load_images",
]
