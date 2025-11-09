"""Test CoreML VisionBackend with converted models.

Tests all three vision tasks:
- Object detection (YOLO)
- Image classification (ViT)
- Image embeddings (CLIP)
"""

import logging
import time
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_yolo_detection():
    """Test YOLO object detection."""
    from llama_pajamas_run_coreml.backends import CoreMLVisionBackend

    logger.info("=" * 70)
    logger.info("TEST 1: YOLO Object Detection")
    logger.info("=" * 70)

    # Paths
    model_path = Path("/tmp/coreml-test-models/yolo-v8n-coreml/model.mlpackage")
    test_images = list(Path("../run/llama_pajamas_run/evaluation/images").glob("*.jpg"))
    test_images.extend(list(Path("../run/llama_pajamas_run/evaluation/images").glob("*.png")))
    test_images = [p for p in test_images if p.stem not in ["cat", "cat1", "cat2"]][:3]

    if not model_path.exists():
        logger.warning(f"YOLO model not found: {model_path}")
        logger.warning("Skipping YOLO test. Run test_converters.py first.")
        return False

    try:
        # Initialize backend
        backend = CoreMLVisionBackend()

        # Load model
        backend.load_model(
            model_path=str(model_path),
            model_type="detection",
            input_size=(640, 640),
        )

        # Test on images
        for img_path in test_images:
            logger.info(f"\nProcessing: {img_path.name}")

            # Run detection
            start = time.time()
            detections = backend.detect(
                image=str(img_path),
                confidence_threshold=0.25,
            )
            inference_time = (time.time() - start) * 1000

            # Log results
            logger.info(f"  Inference time: {inference_time:.2f}ms")
            logger.info(f"  Detected {len(detections)} objects:")
            for det in detections[:5]:  # Top 5
                logger.info(
                    f"    - {det.class_name}: {det.confidence:.2%} "
                    f"bbox=[{det.bbox.x1:.1f}, {det.bbox.y1:.1f}, {det.bbox.x2:.1f}, {det.bbox.y2:.1f}]"
                )

        # Unload
        backend.unload()
        logger.info("\n✅ YOLO detection test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ YOLO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vit_classification():
    """Test ViT image classification."""
    from llama_pajamas_run_coreml.backends import CoreMLVisionBackend

    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: ViT Image Classification")
    logger.info("=" * 70)

    # Paths
    model_path = Path("/tmp/coreml-test-models/vit-base-coreml/model.mlpackage")
    test_images = list(Path("../run/llama_pajamas_run/evaluation/images").glob("*.jpg"))
    test_images.extend(list(Path("../run/llama_pajamas_run/evaluation/images").glob("*.png")))
    test_images = test_images[:3]

    if not model_path.exists():
        logger.warning(f"ViT model not found: {model_path}")
        logger.warning("Skipping ViT test. Run test_converters.py first.")
        return False

    try:
        # Initialize backend
        backend = CoreMLVisionBackend()

        # Load model
        backend.load_model(
            model_path=str(model_path),
            model_type="classification",
            input_size=(224, 224),
        )

        # Test on images
        for img_path in test_images:
            logger.info(f"\nProcessing: {img_path.name}")

            # Run classification
            start = time.time()
            predictions = backend.classify(
                image=str(img_path),
                top_k=5,
            )
            inference_time = (time.time() - start) * 1000

            # Log results
            logger.info(f"  Inference time: {inference_time:.2f}ms")
            logger.info(f"  Top-5 predictions:")
            for pred in predictions:
                logger.info(f"    - {pred.class_name}: {pred.confidence:.2%}")

        # Unload
        backend.unload()
        logger.info("\n✅ ViT classification test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ ViT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_embeddings():
    """Test CLIP image embeddings."""
    from llama_pajamas_run_coreml.backends import CoreMLVisionBackend
    from llama_pajamas_run_core.utils import compute_cosine_similarity

    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: CLIP Image Embeddings")
    logger.info("=" * 70)

    # Paths
    model_path = Path("/tmp/coreml-test-models/clip-vit-base-coreml/model.mlpackage")
    test_images = list(Path("../run/llama_pajamas_run/evaluation/images").glob("*.jpg"))
    test_images.extend(list(Path("../run/llama_pajamas_run/evaluation/images").glob("*.png")))
    test_images = test_images[:3]

    if not model_path.exists():
        logger.warning(f"CLIP model not found: {model_path}")
        logger.warning("Skipping CLIP test. Run test_converters.py first.")
        return False

    try:
        # Initialize backend
        backend = CoreMLVisionBackend()

        # Load model
        backend.load_model(
            model_path=str(model_path),
            model_type="embedding",
            input_size=(224, 224),
        )

        # Generate embeddings
        embeddings = []
        for img_path in test_images:
            logger.info(f"\nProcessing: {img_path.name}")

            # Generate embedding
            start = time.time()
            embedding = backend.embed(image=str(img_path))
            inference_time = (time.time() - start) * 1000

            embeddings.append(embedding)

            # Log results
            logger.info(f"  Inference time: {inference_time:.2f}ms")
            logger.info(f"  Embedding shape: {embedding.shape}")
            logger.info(f"  Embedding L2 norm: {np.linalg.norm(embedding):.4f}")

        # Compare similarities
        if len(embeddings) > 1:
            logger.info("\nEmbedding similarities (cosine):")
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                    logger.info(
                        f"  {test_images[i].name} <-> {test_images[j].name}: {sim:.4f}"
                    )

        # Unload
        backend.unload()
        logger.info("\n✅ CLIP embedding test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ CLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all vision backend tests."""
    logger.info("=" * 70)
    logger.info("CoreML VisionBackend Tests")
    logger.info("=" * 70)
    logger.info("")

    # Run tests
    results = {
        "yolo": test_yolo_detection(),
        "vit": test_vit_classification(),
        "clip": test_clip_embeddings(),
    }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for model_name, success in results.items():
        if success is None:
            status = "⏭️  SKIPPED"
        elif success:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        logger.info(f"{model_name.upper()}: {status}")

    # Check overall result
    all_tested = [v for v in results.values() if v is not None]
    if all_tested and all(all_tested):
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    elif not all_tested:
        logger.info("\n⚠️  NO TESTS RAN (models not found)")
        return 2
    else:
        logger.info("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
