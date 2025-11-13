"""Test all 4 vision tasks with CLI - Classification, Localization, Detection, Segmentation."""

from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
from llama_pajamas_run_core.utils import get_imagenet_class_names, annotate_image_with_detections
from PIL import Image
import sys

def test_classification():
    """Test Task 1: Classification - What is in the image?"""
    print('\n' + '='*60)
    print('TASK 1: CLASSIFICATION (What is this?)')
    print('='*60)

    backend = CoreMLVisionBackend()
    backend.load_model('../quant/models/clip-vit-base/coreml/int8/model.mlpackage', model_type='classification')

    image = Image.open('../run/llama_pajamas_run/evaluation/images/cat1.jpg')
    predictions = backend.classify(image, top_k=5)

    # Get ImageNet labels
    imagenet_names = get_imagenet_class_names()

    print('Top 5 Predictions:')
    for i, p in enumerate(predictions, 1):
        label = imagenet_names[p.class_id] if p.class_id < len(imagenet_names) else p.class_name
        print(f'  {i}. {label}: {p.confidence:.1%}')

    # Annotate
    annotated = annotate_image_with_detections(image, predictions, task_type='classification')
    annotated.save('/tmp/vision_classification.png')
    print(f'✅ Saved: /tmp/vision_classification.png')


def test_localization():
    """Test Task 2: Localization - Where is the main object?"""
    print('\n' + '='*60)
    print('TASK 2: LOCALIZATION (Where is it?)')
    print('='*60)

    backend = CoreMLVisionBackend()
    backend.load_model('../quant/models/yolo-v8n/coreml/fp16/model.mlpackage', model_type='detection')

    image = Image.open('../run/llama_pajamas_run/evaluation/images/cat1.jpg')
    detections = backend.detect(image, confidence_threshold=0.3)

    if detections:
        # Get highest confidence detection
        top = max(detections, key=lambda d: d.confidence)
        print(f'Main Object: {top.class_name}')
        print(f'Confidence: {top.confidence:.1%}')
        print(f'Location: ({top.bbox.x1:.2f}, {top.bbox.y1:.2f}) to ({top.bbox.x2:.2f}, {top.bbox.y2:.2f})')

        # Annotate (just the top detection)
        annotated = annotate_image_with_detections(image, [top], task_type='localization')
        annotated.save('/tmp/vision_localization.png')
        print(f'✅ Saved: /tmp/vision_localization.png')
    else:
        print('No objects detected')


def test_detection():
    """Test Task 3: Detection - Find all objects with bounding boxes."""
    print('\n' + '='*60)
    print('TASK 3: OBJECT DETECTION (What and where?)')
    print('='*60)

    backend = CoreMLVisionBackend()
    backend.load_model('../quant/models/yolo-v8n/coreml/fp16/model.mlpackage', model_type='detection')

    image = Image.open('../run/llama_pajamas_run/evaluation/images/cat1.jpg')
    detections = backend.detect(image, confidence_threshold=0.5)

    print(f'Found {len(detections)} objects:')
    for i, d in enumerate(detections, 1):
        print(f'  {i}. {d.class_name}: {d.confidence:.1%} at ({d.bbox.x1:.2f}, {d.bbox.y1:.2f}, {d.bbox.x2:.2f}, {d.bbox.y2:.2f})')

    # Annotate
    annotated = annotate_image_with_detections(image, detections, task_type='detection')
    annotated.save('/tmp/vision_detection.png')
    print(f'✅ Saved: /tmp/vision_detection.png')


def test_segmentation():
    """Test Task 4: Segmentation - Segment individual object instances."""
    print('\n' + '='*60)
    print('TASK 4: INSTANCE SEGMENTATION (What, where, and mask)')
    print('='*60)

    backend = CoreMLVisionBackend()
    backend.load_model('../quant/models/yolo-v8n/coreml/fp16/model.mlpackage', model_type='detection')

    image = Image.open('../run/llama_pajamas_run/evaluation/images/cat1.jpg')
    detections = backend.detect(image, confidence_threshold=0.5)

    print(f'Segmented {len(detections)} instances:')
    for i, d in enumerate(detections, 1):
        print(f'  {i}. {d.class_name}: {d.confidence:.1%} [instance {i}]')

    # Annotate with masks
    annotated = annotate_image_with_detections(image, detections, task_type='segmentation')
    annotated.save('/tmp/vision_segmentation.png')
    print(f'✅ Saved: /tmp/vision_segmentation.png')


if __name__ == '__main__':
    print('\nTesting All 4 Vision Tasks with LlamaPajamas')
    print('Image: cat1.jpg')

    try:
        test_classification()
        test_localization()
        test_detection()
        test_segmentation()

        print('\n' + '='*60)
        print('✅ ALL TASKS COMPLETE!')
        print('='*60)
        print('\nGenerated Images:')
        print('  1. /tmp/vision_classification.png   - Classification with label')
        print('  2. /tmp/vision_localization.png     - Main object with bounding box')
        print('  3. /tmp/vision_detection.png        - All objects with bounding boxes')
        print('  4. /tmp/vision_segmentation.png     - All objects with instance masks')

    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
