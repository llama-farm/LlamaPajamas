"""Comprehensive vision test across all models and backends."""

from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend
from llama_pajamas_run_core.utils import get_imagenet_class_names, annotate_image_with_detections
from PIL import Image
import sys

# Test image
IMAGE_PATH = "/Users/robthelen/Downloads/test-images/horse.jpg"

print("\n" + "="*80)
print("COMPREHENSIVE VISION MODEL TEST")
print("="*80)
print(f"Test Image: {IMAGE_PATH}")
print("="*80)

# Load image once
image = Image.open(IMAGE_PATH)
imagenet_names = get_imagenet_class_names()

# ============================================================================
# COREML TESTS
# ============================================================================

print("\n" + "="*80)
print("COREML BACKEND TESTS")
print("="*80)

# Test 1: YOLO-v8n CoreML (Detection Model)
print("\n" + "-"*80)
print("1. YOLO-v8n (CoreML FP16) - Detection Model")
print("-"*80)

try:
    backend = CoreMLVisionBackend()
    backend.load_model("../quant/models/yolo-v8n/coreml/fp16/model.mlpackage", model_type="detection")

    # Test detection
    print("\n[Detection Task]")
    detections = backend.detect(image, confidence_threshold=0.3)
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections[:5], 1):
        print(f"  {i}. {det.class_name}: {det.confidence:.1%}")

    # Save annotated
    annotated = annotate_image_with_detections(image, detections, task_type="detection")
    annotated.save("/tmp/horse_yolo_coreml_detection.jpg")
    print(f"✅ Saved: /tmp/horse_yolo_coreml_detection.jpg")

    # Test localization
    print("\n[Localization Task]")
    if detections:
        top = max(detections, key=lambda d: d.confidence)
        print(f"Main object: {top.class_name} ({top.confidence:.1%})")
        annotated = annotate_image_with_detections(image, [top], task_type="localization")
        annotated.save("/tmp/horse_yolo_coreml_localization.jpg")
        print(f"✅ Saved: /tmp/horse_yolo_coreml_localization.jpg")

    # Test segmentation
    print("\n[Segmentation Task]")
    annotated = annotate_image_with_detections(image, detections, task_type="segmentation")
    annotated.save("/tmp/horse_yolo_coreml_segmentation.jpg")
    print(f"✅ Saved: /tmp/horse_yolo_coreml_segmentation.jpg")

except Exception as e:
    print(f"❌ YOLO CoreML failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: ViT-base CoreML (Classification Model)
print("\n" + "-"*80)
print("2. ViT-base (CoreML FP16) - Classification Model")
print("-"*80)

try:
    backend = CoreMLVisionBackend()
    backend.load_model("../quant/models/vit-base/coreml/fp16/model.mlpackage", model_type="classification")

    # Test classification
    print("\n[Classification Task]")
    predictions = backend.classify(image, top_k=5)
    print(f"Top 5 predictions:")
    for i, pred in enumerate(predictions, 1):
        label = imagenet_names[pred.class_id] if pred.class_id < len(imagenet_names) else pred.class_name
        print(f"  {i}. {label}: {pred.confidence:.1%}")

    # Save annotated
    annotated = annotate_image_with_detections(image, predictions, task_type="classification")
    annotated.save("/tmp/horse_vit_coreml_classification.jpg")
    print(f"✅ Saved: /tmp/horse_vit_coreml_classification.jpg")

except Exception as e:
    print(f"❌ ViT CoreML failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: CLIP-ViT-base CoreML FP16
print("\n" + "-"*80)
print("3. CLIP-ViT-base (CoreML FP16) - Classification Model")
print("-"*80)

try:
    backend = CoreMLVisionBackend()
    backend.load_model("../quant/models/clip-vit-base/coreml/fp16/model.mlpackage", model_type="classification")

    # Test classification
    print("\n[Classification Task]")
    predictions = backend.classify(image, top_k=5)
    print(f"Top 5 predictions:")
    for i, pred in enumerate(predictions, 1):
        label = imagenet_names[pred.class_id] if pred.class_id < len(imagenet_names) else pred.class_name
        print(f"  {i}. {label}: {pred.confidence:.1%}")

    # Save annotated
    annotated = annotate_image_with_detections(image, predictions, task_type="classification")
    annotated.save("/tmp/horse_clip_coreml_fp16_classification.jpg")
    print(f"✅ Saved: /tmp/horse_clip_coreml_fp16_classification.jpg")

except Exception as e:
    print(f"❌ CLIP CoreML FP16 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: CLIP-ViT-base CoreML INT8
print("\n" + "-"*80)
print("4. CLIP-ViT-base (CoreML INT8) - Classification Model")
print("-"*80)

try:
    backend = CoreMLVisionBackend()
    backend.load_model("../quant/models/clip-vit-base/coreml/int8/model.mlpackage", model_type="classification")

    # Test classification
    print("\n[Classification Task]")
    predictions = backend.classify(image, top_k=5)
    print(f"Top 5 predictions:")
    for i, pred in enumerate(predictions, 1):
        label = imagenet_names[pred.class_id] if pred.class_id < len(imagenet_names) else pred.class_name
        print(f"  {i}. {label}: {pred.confidence:.1%}")

    # Save annotated
    annotated = annotate_image_with_detections(image, predictions, task_type="classification")
    annotated.save("/tmp/horse_clip_coreml_int8_classification.jpg")
    print(f"✅ Saved: /tmp/horse_clip_coreml_int8_classification.jpg")

except Exception as e:
    print(f"❌ CLIP CoreML INT8 failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ONNX TESTS
# ============================================================================

print("\n" + "="*80)
print("ONNX BACKEND TESTS")
print("="*80)

# Test 5: YOLO-v8n ONNX INT8
print("\n" + "-"*80)
print("5. YOLO-v8n (ONNX INT8) - Detection Model")
print("-"*80)

try:
    backend = ONNXVisionBackend()
    backend.load_model(
        "../quant/models/yolo-v8n/onnx/yolov8n_int8.onnx",
        model_type="detection",
        providers=["CPUExecutionProvider"],
        num_threads=4,
    )

    # Test detection
    print("\n[Detection Task]")
    detections = backend.detect(image, confidence_threshold=0.3)
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections[:5], 1):
        print(f"  {i}. {det.class_name}: {det.confidence:.1%}")

    # Save annotated
    annotated = annotate_image_with_detections(image, detections, task_type="detection")
    annotated.save("/tmp/horse_yolo_onnx_detection.jpg")
    print(f"✅ Saved: /tmp/horse_yolo_onnx_detection.jpg")

    # Test localization
    print("\n[Localization Task]")
    if detections:
        top = max(detections, key=lambda d: d.confidence)
        print(f"Main object: {top.class_name} ({top.confidence:.1%})")
        annotated = annotate_image_with_detections(image, [top], task_type="localization")
        annotated.save("/tmp/horse_yolo_onnx_localization.jpg")
        print(f"✅ Saved: /tmp/horse_yolo_onnx_localization.jpg")

except Exception as e:
    print(f"❌ YOLO ONNX failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\nGenerated Images:")
print("  CoreML:")
print("    - /tmp/horse_yolo_coreml_detection.jpg")
print("    - /tmp/horse_yolo_coreml_localization.jpg")
print("    - /tmp/horse_yolo_coreml_segmentation.jpg")
print("    - /tmp/horse_vit_coreml_classification.jpg")
print("    - /tmp/horse_clip_coreml_fp16_classification.jpg")
print("    - /tmp/horse_clip_coreml_int8_classification.jpg")
print("  ONNX:")
print("    - /tmp/horse_yolo_onnx_detection.jpg")
print("    - /tmp/horse_yolo_onnx_localization.jpg")
print("="*80)
