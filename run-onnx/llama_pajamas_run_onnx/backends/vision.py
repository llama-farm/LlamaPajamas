"""ONNX Runtime backend for vision models.

Supports:
- Object detection (YOLO)
- Classification (ViT, ResNet)
- Embeddings (CLIP)
- Segmentation (Mask R-CNN)

Optimized for CPU, AMD GPU, ARM processors, NVIDIA Jetson.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXVisionBackend:
    """ONNX Runtime backend for vision models."""

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.model_type: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None

    def load_model(
        self,
        model_path: str,
        model_type: str = "detection",
        providers: Optional[List[str]] = None,
        **kwargs,
    ):
        """Load ONNX model.

        Args:
            model_path: Path to .onnx file
            model_type: One of "detection", "classification", "segmentation", "embedding"
            providers: Execution providers (default: ["CPUExecutionProvider"])
                      Available: "CPUExecutionProvider", "CUDAExecutionProvider",
                                "TensorrtExecutionProvider", "OpenVINOExecutionProvider"
            **kwargs: Additional session options
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.info(f"Loading ONNX model from {model_path}")
        logger.info(f"Using execution providers: {providers}")

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Apply custom options
        if "num_threads" in kwargs:
            sess_options.intra_op_num_threads = kwargs["num_threads"]
        if "enable_profiling" in kwargs:
            sess_options.enable_profiling = kwargs["enable_profiling"]

        # Create inference session
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )

        self.model_type = model_type

        # Get input/output info
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape

        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"Model loaded successfully")
        logger.info(f"Input: {self.input_name} - {self.input_shape}")
        logger.info(f"Outputs: {self.output_names}")

        # Get available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")

    def preprocess_image(
        self, image: Image.Image, target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: PIL Image
            target_size: Target (width, height)

        Returns:
            Preprocessed numpy array
        """
        # Resize
        image = image.resize(target_size, Image.Resampling.BILINEAR)

        # Convert to numpy
        img_array = np.array(image).astype(np.float32)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Run object detection.

        Args:
            image: PIL Image
            confidence_threshold: Minimum confidence score
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detections: [{"bbox": [x, y, w, h], "confidence": float, "class_id": int}, ...]
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Post-process detections (YOLO format)
        detections = self._postprocess_yolo(
            outputs, confidence_threshold, iou_threshold, image.size
        )

        return detections

    def classify(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run image classification.

        Args:
            image: PIL Image
            top_k: Return top K predictions

        Returns:
            List of predictions: [{"class_id": int, "confidence": float}, ...]
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        input_tensor = self.preprocess_image(image, target_size=(224, 224))

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Post-process (softmax + top-k)
        logits = outputs[0][0]
        probs = self._softmax(logits)
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = [
            {"class_id": int(idx), "confidence": float(probs[idx])}
            for idx in top_indices
        ]

        return predictions

    def embed(self, image: Image.Image) -> np.ndarray:
        """Generate image embedding.

        Args:
            image: PIL Image

        Returns:
            Embedding vector (numpy array)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        input_tensor = self.preprocess_image(image, target_size=(224, 224))

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Return embedding (first output)
        embedding = outputs[0][0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _postprocess_yolo(
        self,
        outputs: List[np.ndarray],
        confidence_threshold: float,
        iou_threshold: float,
        original_size: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Post-process YOLO outputs."""
        # YOLO output format: [batch, num_detections, 85] (x, y, w, h, conf, class_scores...)
        predictions = outputs[0][0]

        detections = []
        for pred in predictions:
            x, y, w, h = pred[:4]
            confidence = pred[4]
            class_scores = pred[5:]

            if confidence < confidence_threshold:
                continue

            class_id = int(np.argmax(class_scores))
            class_confidence = class_scores[class_id]

            if class_confidence < confidence_threshold:
                continue

            # Scale to original image size
            orig_w, orig_h = original_size
            x = float(x * orig_w)
            y = float(y * orig_h)
            w = float(w * orig_w)
            h = float(h * orig_h)

            detections.append(
                {
                    "bbox": [x, y, w, h],
                    "confidence": float(confidence * class_confidence),
                    "class_id": class_id,
                }
            )

        # Apply NMS
        detections = self._nms(detections, iou_threshold)

        return detections

    def _nms(
        self, detections: List[Dict[str, Any]], iou_threshold: float
    ) -> List[Dict[str, Any]]:
        """Non-maximum suppression."""
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while len(detections) > 0:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping boxes
            detections = [
                det
                for det in detections
                if self._iou(best["bbox"], det["bbox"]) < iou_threshold
            ]

        return keep

    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

        x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def unload_model(self):
        """Unload model from memory."""
        self.session = None
        logger.info("Model unloaded")
