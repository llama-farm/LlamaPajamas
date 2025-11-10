#!/usr/bin/env python3
"""
Test ONNX Vision backend with quantized YOLO model.

Demonstrates:
- Loading quantized INT8 ONNX model
- CPU inference
- Object detection

Usage:
    uv run python examples/test_onnx_vision.py
"""

import sys
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend


def create_test_image() -> Image.Image:
    """Create a test image with some shapes."""
    img = Image.new("RGB", (640, 640), color=(135, 206, 235))  # Sky blue background

    draw = ImageDraw.Draw(img)

    # Draw some rectangles (simulating objects)
    draw.rectangle([100, 100, 200, 200], fill=(255, 0, 0))  # Red square
    draw.rectangle([300, 200, 450, 350], fill=(0, 255, 0))  # Green rectangle
    draw.ellipse([400, 400, 550, 550], fill=(255, 255, 0))  # Yellow circle

    return img


def main():
    print("=" * 60)
    print("ONNX Vision Backend Test - YOLO INT8")
    print("=" * 60)
    print()

    # Model paths (try both FP32 and INT8)
    fp32_model = Path("../quant/models/yolo-v8n/tensorrt/fp16/yolov8n.onnx")
    int8_model = Path("../quant/models/yolo-v8n/onnx/yolov8n_int8.onnx")

    # Choose which model to test
    if int8_model.exists():
        model_path = int8_model
        model_type = "INT8 Quantized"
    elif fp32_model.exists():
        model_path = fp32_model
        model_type = "FP32"
    else:
        print("❌ No YOLO ONNX model found!")
        print(f"Expected: {fp32_model} or {int8_model}")
        return

    print(f"📦 Loading model: {model_path}")
    print(f"   Type: {model_type}")
    print(f"   Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    print()

    # Initialize backend
    backend = ONNXVisionBackend()

    # Load model (CPU execution)
    start = time.time()
    backend.load_model(
        str(model_path),
        model_type="detection",
        providers=["CPUExecutionProvider"],  # CPU only
        num_threads=4,
    )
    load_time = time.time() - start

    print(f"✅ Model loaded in {load_time*1000:.1f}ms")
    print()

    # Create test image
    print("🖼️  Creating test image...")
    test_image = create_test_image()

    # Run inference
    print("🔍 Running object detection...")
    start = time.time()
    detections = backend.detect(
        test_image, confidence_threshold=0.25, iou_threshold=0.45
    )
    inference_time = time.time() - start

    print(f"✅ Inference complete in {inference_time*1000:.1f}ms")
    print(f"   Found {len(detections)} detections")
    print()

    # Print detections
    if detections:
        print("Detections:")
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["confidence"]
            class_id = det["class_id"]
            print(
                f"  {i+1}. Class {class_id}, Confidence: {conf:.3f}, BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
            )
    else:
        print("No detections (expected - test image has simple shapes, not real objects)")

    print()
    print("=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    print(f"Model loading: {load_time*1000:.1f}ms")
    print(f"Inference:     {inference_time*1000:.1f}ms")
    print(f"FPS (est):     {1/inference_time:.1f}")
    print()
    print("✅ ONNX Vision backend working!")
    print()
    print("💡 This demonstrates ONNX Runtime can run quantized models on CPU.")
    print("   Perfect for edge deployment (Raspberry Pi, Jetson, AMD GPU, etc.)")


if __name__ == "__main__":
    main()
