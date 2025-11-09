"""End-to-end test of CoreML models with evaluation images.

Tests YOLO detection, CLIP embeddings, and ViT classification.
"""

import logging
import time
from pathlib import Path
import numpy as np
from PIL import Image
import coremltools as ct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_image(image_path: Path, target_size: tuple) -> Image.Image:
    """Load and preprocess image for CoreML models."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    # CoreML expects PIL.Image.Image objects
    return img


def test_yolo_detection(model_path: Path, image_paths: list):
    """Test YOLO object detection."""
    logger.info("=" * 70)
    logger.info("YOLO-v8n Detection Test")
    logger.info("=" * 70)

    try:
        # Load model
        logger.info(f"Loading model: {model_path}")
        model = ct.models.MLModel(str(model_path))

        logger.info(f"Model inputs: {model.get_spec().description.input}")
        logger.info(f"Model outputs: {model.get_spec().description.output}")

        # Test on images
        for img_path in image_paths:
            logger.info(f"\nProcessing: {img_path.name}")

            # Preprocess
            start = time.time()
            img = load_and_preprocess_image(img_path, (640, 640))

            # Inference
            output = model.predict({"image": img})
            inference_time = (time.time() - start) * 1000

            # Results
            logger.info(f"  Inference time: {inference_time:.2f}ms")
            logger.info(f"  Output keys: {output.keys()}")
            for key, value in output.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key} shape: {value.shape}")
                    if hasattr(value, 'dtype'):
                        logger.info(f"  {key} dtype: {value.dtype}")

        logger.info("\n✅ YOLO detection test completed")
        return True

    except Exception as e:
        logger.error(f"❌ YOLO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_embeddings(model_path: Path, image_paths: list):
    """Test CLIP embedding generation."""
    logger.info("\n" + "=" * 70)
    logger.info("CLIP-ViT-Base Embedding Test")
    logger.info("=" * 70)

    try:
        # Load model
        logger.info(f"Loading model: {model_path}")
        model = ct.models.MLModel(str(model_path))

        logger.info(f"Model inputs: {model.get_spec().description.input}")
        logger.info(f"Model outputs: {model.get_spec().description.output}")

        embeddings = []

        # Test on images
        for img_path in image_paths:
            logger.info(f"\nProcessing: {img_path.name}")

            # Preprocess
            start = time.time()
            img = load_and_preprocess_image(img_path, (224, 224))

            # Inference
            output = model.predict({"image": img})
            inference_time = (time.time() - start) * 1000

            # Extract embedding
            embedding = output['embedding']
            embeddings.append(embedding)

            # Results
            logger.info(f"  Inference time: {inference_time:.2f}ms")
            logger.info(f"  Embedding shape: {embedding.shape}")
            logger.info(f"  Embedding L2 norm: {np.linalg.norm(embedding):.4f}")
            logger.info(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")

        # Compare similarities
        if len(embeddings) > 1:
            logger.info("\nEmbedding similarities (cosine):")
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine similarity
                    sim = np.dot(embeddings[i].flatten(), embeddings[j].flatten()) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    logger.info(f"  {image_paths[i].name} <-> {image_paths[j].name}: {sim:.4f}")

        logger.info("\n✅ CLIP embedding test completed")
        return True

    except Exception as e:
        logger.error(f"❌ CLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vit_classification(model_path: Path, image_paths: list, labels_file: Path = None):
    """Test ViT image classification."""
    logger.info("\n" + "=" * 70)
    logger.info("ViT-Base Classification Test")
    logger.info("=" * 70)

    try:
        # Load model
        logger.info(f"Loading model: {model_path}")
        model = ct.models.MLModel(str(model_path))

        logger.info(f"Model inputs: {model.get_spec().description.input}")
        logger.info(f"Model outputs: {model.get_spec().description.output}")

        # Load labels
        labels = None
        if labels_file and labels_file.exists():
            import json
            metadata = json.load(open(labels_file))
            labels = metadata.get('labels', [])
            logger.info(f"Loaded {len(labels)} class labels")

        # Test on images
        for img_path in image_paths:
            logger.info(f"\nProcessing: {img_path.name}")

            # Preprocess
            start = time.time()
            img = load_and_preprocess_image(img_path, (224, 224))

            # Inference
            output = model.predict({"image": img})
            inference_time = (time.time() - start) * 1000

            # Extract logits
            logits = output['logits']

            # Apply softmax
            probs = np.exp(logits) / np.sum(np.exp(logits))

            # Get top-5 predictions
            top5_indices = np.argsort(probs.flatten())[-5:][::-1]

            # Results
            logger.info(f"  Inference time: {inference_time:.2f}ms")
            logger.info(f"  Logits shape: {logits.shape}")
            logger.info(f"  Top-5 predictions:")
            for idx in top5_indices:
                prob = probs.flatten()[idx]
                if labels and idx < len(labels):
                    label = labels[idx]
                    logger.info(f"    {idx}: {label} ({prob * 100:.2f}%)")
                else:
                    logger.info(f"    {idx}: class_{idx} ({prob * 100:.2f}%)")

        logger.info("\n✅ ViT classification test completed")
        return True

    except Exception as e:
        logger.error(f"❌ ViT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all model tests."""
    # Paths
    models_dir = Path("/tmp/coreml-test-models")
    images_dir = Path("../run/llama_pajamas_run/evaluation/images")

    # Get image files (exclude videos for now)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    image_paths.sort()

    logger.info(f"Found {len(image_paths)} test images: {[p.name for p in image_paths]}")
    logger.info("")

    # Test each model
    results = {}

    # YOLO
    yolo_model = models_dir / "yolo-v8n-coreml" / "model.mlpackage"
    if yolo_model.exists():
        results['yolo'] = test_yolo_detection(yolo_model, image_paths)
    else:
        logger.warning(f"YOLO model not found: {yolo_model}")
        results['yolo'] = False

    # CLIP
    clip_model = models_dir / "clip-vit-base-coreml" / "model.mlpackage"
    if clip_model.exists():
        results['clip'] = test_clip_embeddings(clip_model, image_paths)
    else:
        logger.warning(f"CLIP model not found: {clip_model}")
        results['clip'] = False

    # ViT
    vit_model = models_dir / "vit-base-coreml" / "model.mlpackage"
    vit_labels = models_dir / "vit-base-coreml" / "conversion_metadata.json"
    if vit_model.exists():
        results['vit'] = test_vit_classification(vit_model, image_paths, vit_labels)
    else:
        logger.warning(f"ViT model not found: {vit_model}")
        results['vit'] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{model_name.upper()}: {status}")

    all_passed = all(results.values())
    logger.info("")
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.info("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
