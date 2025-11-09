"""Extensible evaluation script for CoreML vision models.

Evaluate any CoreML vision model with custom paths and configurations.

Usage:
    # Evaluate single model
    python evaluate_model.py --model /path/to/model.mlpackage --type detection --images /path/to/images

    # Evaluate with custom config
    python evaluate_model.py --model /path/to/model.mlpackage --type classification \\
        --images /path/to/images --top-k 10 --output ./results

    # Evaluate multiple models
    python evaluate_model.py --models-dir /path/to/models --images /path/to/images
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

from benchmark_vision import VisionBenchmark
from evaluate_comprehensive import ComprehensiveEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_single_model(
    model_path: Path,
    model_type: str,
    images_dir: Path,
    output_dir: Path,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate a single CoreML vision model.

    Args:
        model_path: Path to .mlpackage model
        model_type: One of 'detection', 'classification', 'embedding'
        images_dir: Directory containing test images
        output_dir: Directory to save results
        **kwargs: Additional model-specific parameters:
            - conf_threshold: float (detection only)
            - iou_threshold: float (detection only)
            - top_k: int (classification only)

    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 70)
    logger.info(f"Evaluating {model_type} model: {model_path.name}")
    logger.info("=" * 70)

    # Get test images
    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Test images: {len(image_paths)}")

    # Initialize evaluator and benchmark
    evaluator = ComprehensiveEvaluator(output_dir=output_dir)
    benchmark = VisionBenchmark(output_dir=output_dir)

    # Run evaluation based on model type
    if model_type == "detection":
        conf_threshold = kwargs.get("conf_threshold", 0.25)
        iou_threshold = kwargs.get("iou_threshold", 0.7)

        # Performance
        perf_result = benchmark.benchmark_detection(
            model_path=model_path,
            image_paths=image_paths,
            conf_threshold=conf_threshold,
        )

        # Quality
        quality_result = evaluator.evaluate_detection_quality(
            model_path=model_path,
            image_paths=image_paths,
        )

    elif model_type == "classification":
        top_k = kwargs.get("top_k", 5)

        # Performance
        perf_result = benchmark.benchmark_classification(
            model_path=model_path,
            image_paths=image_paths,
            top_k=top_k,
        )

        # Quality
        quality_result = evaluator.evaluate_classification_quality(
            model_path=model_path,
            image_paths=image_paths,
        )

    elif model_type == "embedding":
        # Performance
        perf_result = benchmark.benchmark_embeddings(
            model_path=model_path,
            image_paths=image_paths,
        )

        # Quality
        quality_result = evaluator.evaluate_embedding_quality(
            model_path=model_path,
            image_paths=image_paths,
        )

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Combine results
    results = {
        "model_path": str(model_path),
        "model_type": model_type,
        "num_images": len(image_paths),
        "performance": perf_result.to_dict(),
        "quality": quality_result,
    }

    # Save results
    output_file = output_dir / f"{model_path.stem}_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n📊 Results saved to: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Type: {model_type}")
    logger.info(f"Performance:")
    logger.info(f"  - FPS: {perf_result.fps:.1f}")
    logger.info(f"  - Latency: {perf_result.avg_latency_ms:.1f} ± {perf_result.std_latency_ms:.1f} ms")
    logger.info(f"  - Model size: {perf_result.model_size_mb:.1f} MB")
    logger.info(f"Quality:")

    if model_type == "detection":
        logger.info(f"  - Total detections: {quality_result['total_detections']}")
        logger.info(f"  - Avg confidence: {quality_result['avg_confidence']:.2%}")
        logger.info(f"  - Unique classes: {quality_result['unique_classes_detected']}")
    elif model_type == "classification":
        logger.info(f"  - Avg top-1 confidence: {quality_result['avg_top1_confidence']:.2%}")
        logger.info(f"  - High confidence (>70%): {quality_result['high_confidence_predictions']}/{quality_result['images_processed']}")
    elif model_type == "embedding":
        logger.info(f"  - Embedding dim: {quality_result['embedding_dim']}")
        logger.info(f"  - Avg similarity: {quality_result['avg_similarity']:.3f}")

    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate CoreML vision models with custom configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single YOLO model
  python evaluate_model.py --model /tmp/yolo.mlpackage --type detection \\
      --images ./images --conf-threshold 0.3

  # Evaluate ViT model with top-10 predictions
  python evaluate_model.py --model /tmp/vit.mlpackage --type classification \\
      --images ./images --top-k 10

  # Evaluate CLIP model
  python evaluate_model.py --model /tmp/clip.mlpackage --type embedding \\
      --images ./images

  # Evaluate all models in directory
  python evaluate_model.py --models-dir /tmp/coreml-test-models \\
      --images ./images
        """
    )

    # Input arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Path to single CoreML model (.mlpackage)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["detection", "classification", "embedding"],
        help="Model type (required with --model)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        help="Directory containing multiple models (alternative to --model)"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="Output directory for results (default: ./evaluation_results)"
    )

    # Model-specific parameters
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="IoU threshold for detection (default: 0.7)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k predictions for classification (default: 5)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model and not args.type:
        parser.error("--type is required when using --model")
    if not args.model and not args.models_dir:
        parser.error("Either --model or --models-dir must be specified")

    # Paths
    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        parser.error(f"Images directory not found: {images_dir}")

    # Evaluate single model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            parser.error(f"Model not found: {model_path}")

        results = evaluate_single_model(
            model_path=model_path,
            model_type=args.type,
            images_dir=images_dir,
            output_dir=output_dir,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            top_k=args.top_k,
        )

        logger.info("\n🎉 Evaluation complete!")
        return 0

    # Evaluate multiple models from directory
    if args.models_dir:
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            parser.error(f"Models directory not found: {models_dir}")

        from evaluate_comprehensive import ComprehensiveEvaluator

        evaluator = ComprehensiveEvaluator(output_dir=output_dir)
        results = evaluator.run_comprehensive_evaluation(
            models_dir=models_dir,
            images_dir=images_dir,
        )

        logger.info("\n🎉 Comprehensive evaluation complete!")
        return 0


if __name__ == "__main__":
    exit(main())
