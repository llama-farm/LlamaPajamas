"""Test all CoreML vision models with horse image."""

from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
from llama_pajamas_run_core.utils import get_imagenet_class_names, annotate_image_with_detections
from PIL import Image

# Test image
IMAGE_PATH = "/Users/robthelen/Downloads/test-images/horse.jpg"

print("\n" + "="*80)
print("COREML VISION MODEL TEST - Horse Image")
print("="*80)

# Load image once
image = Image.open(IMAGE_PATH)
imagenet_names = get_imagenet_class_names()

# Test 1: YOLO-v8n CoreML (Detection Model)
print("\n" + "="*80)
print("1. YOLO-v8n (CoreML FP16) - Detection Model")
print("="*80)

backend = CoreMLVisionBackend()
backend.load_model("../quant/models/yolo-v8n/coreml/fp16/model.mlpackage", model_type="detection")

print("\n[Detection Task]")
detections = backend.detect(image, confidence_threshold=0.3)
print(f"Found {len(detections)} objects:")
for i, det in enumerate(detections, 1):
    print(f"  {i}. {det.class_name}: {det.confidence:.1%}")
annotated = annotate_image_with_detections(image, detections, task_type="detection")
annotated.save("/tmp/horse_yolo_detection.jpg")
print(f"✅ Saved: /tmp/horse_yolo_detection.jpg")

print("\n[Localization Task]")
if detections:
    top = max(detections, key=lambda d: d.confidence)
    print(f"Main object: {top.class_name} ({top.confidence:.1%})")
    annotated = annotate_image_with_detections(image, [top], task_type="localization")
    annotated.save("/tmp/horse_yolo_localization.jpg")
    print(f"✅ Saved: /tmp/horse_yolo_localization.jpg")

print("\n[Segmentation Task]")
annotated = annotate_image_with_detections(image, detections, task_type="segmentation")
annotated.save("/tmp/horse_yolo_segmentation.jpg")
print(f"✅ Saved: /tmp/horse_yolo_segmentation.jpg")

# Test 2: ViT-base CoreML
print("\n" + "="*80)
print("2. ViT-base (CoreML FP16) - Classification Model")
print("="*80)

backend = CoreMLVisionBackend()
backend.load_model("../quant/models/vit-base/coreml/fp16/model.mlpackage", model_type="classification")

print("\n[Classification Task]")
predictions = backend.classify(image, top_k=5)
print(f"Top 5 predictions:")
for i, pred in enumerate(predictions, 1):
    label = imagenet_names[pred.class_id] if pred.class_id < len(imagenet_names) else pred.class_name
    print(f"  {i}. {label}: {pred.confidence:.1%}")
annotated = annotate_image_with_detections(image, predictions, task_type="classification")
annotated.save("/tmp/horse_vit_classification.jpg")
print(f"✅ Saved: /tmp/horse_vit_classification.jpg")

# Test 3: CLIP-ViT-base CoreML FP16
print("\n" + "="*80)
print("3. CLIP-ViT-base (CoreML FP16)")
print("="*80)

backend = CoreMLVisionBackend()
backend.load_model("../quant/models/clip-vit-base/coreml/fp16/model.mlpackage", model_type="classification")

print("\n[Classification Task]")
predictions = backend.classify(image, top_k=5)
print(f"Top 5 predictions:")
for i, pred in enumerate(predictions, 1):
    label = imagenet_names[pred.class_id] if pred.class_id < len(imagenet_names) else pred.class_name
    print(f"  {i}. {label}: {pred.confidence:.1%}")
annotated = annotate_image_with_detections(image, predictions, task_type="classification")
annotated.save("/tmp/horse_clip_fp16.jpg")
print(f"✅ Saved: /tmp/horse_clip_fp16.jpg")

# Test 4: CLIP-ViT-base CoreML INT8
print("\n" + "="*80)
print("4. CLIP-ViT-base (CoreML INT8)")
print("="*80)

backend = CoreMLVisionBackend()
backend.load_model("../quant/models/clip-vit-base/coreml/int8/model.mlpackage", model_type="classification")

print("\n[Classification Task]")
predictions = backend.classify(image, top_k=5)
print(f"Top 5 predictions:")
for i, pred in enumerate(predictions, 1):
    label = imagenet_names[pred.class_id] if pred.class_id < len(imagenet_names) else pred.class_name
    print(f"  {i}. {label}: {pred.confidence:.1%}")
annotated = annotate_image_with_detections(image, predictions, task_type="classification")
annotated.save("/tmp/horse_clip_int8.jpg")
print(f"✅ Saved: /tmp/horse_clip_int8.jpg")

print("\n" + "="*80)
print("COREML TESTS COMPLETE!")
print("="*80)
print("\nGenerated Images:")
print("  - /tmp/horse_yolo_detection.jpg")
print("  - /tmp/horse_yolo_localization.jpg")
print("  - /tmp/horse_yolo_segmentation.jpg")
print("  - /tmp/horse_vit_classification.jpg")
print("  - /tmp/horse_clip_fp16.jpg")
print("  - /tmp/horse_clip_int8.jpg")
print("="*80)
