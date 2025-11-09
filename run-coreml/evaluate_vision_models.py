"""Evaluate CoreML vision models from quant/models directory.

This script follows the same pattern as GGUF/MLX evaluations:
- Models are in quant/models/{model-name}/coreml/fp16/
- Results are saved alongside the model
- Comparisons are generated at the top level
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
from llama_pajamas_run_core.utils.vision_utils import compute_cosine_similarity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_models(base_dir: Path) -> List[Dict[str, Any]]:
    """Find all CoreML vision models in quant/models structure."""
    models = []

    for model_dir in base_dir.glob("*/"):
        if not model_dir.is_dir():
            continue

        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Find CoreML models
        for fmt in manifest.get("formats", []):
            if fmt["format"] == "coreml":
                model_path = model_dir / fmt["path"]
                if model_path.exists():
                    models.append({
                        "name": manifest["model_name"],
                        "path": model_path,
                        "task": manifest["task"],
                        "manifest": manifest,
                        "format_info": fmt,
                        "model_dir": model_dir,
                    })

    return models


def evaluate_detection(
    backend: CoreMLVisionBackend,
    model_info: Dict[str, Any],
    images: List[Path],
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Evaluate object detection model."""
    logger.info(f"Evaluating detection model: {model_info['name']}")

    latencies = []
    all_detections = []
    confidence_scores = []
    class_distribution = {}

    # Warmup
    for _ in range(3):
        backend.detect(str(images[0]), confidence_threshold=conf_threshold)

    # Benchmark
    for img_path in images:
        start = time.time()
        detections = backend.detect(str(img_path), confidence_threshold=conf_threshold)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        all_detections.extend(detections)

        for det in detections:
            confidence_scores.append(det.confidence)
            class_name = det.class_name or f"class_{det.class_id}"
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

    return {
        "task": "object_detection",
        "num_images": len(images),
        "performance": {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "fps": 1000 / (sum(latencies) / len(latencies)),
        },
        "quality": {
            "total_detections": len(all_detections),
            "detections_per_image": len(all_detections) / len(images),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "unique_classes": len(class_distribution),
            "class_distribution": class_distribution,
        },
    }


def evaluate_classification(
    backend: CoreMLVisionBackend,
    model_info: Dict[str, Any],
    images: List[Path],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate image classification model."""
    logger.info(f"Evaluating classification model: {model_info['name']}")

    latencies = []
    all_predictions = []
    top1_confidences = []

    # Warmup
    for _ in range(3):
        backend.classify(str(images[0]), top_k=top_k)

    # Benchmark
    for img_path in images:
        start = time.time()
        predictions = backend.classify(str(img_path), top_k=top_k)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        all_predictions.append(predictions)

        if predictions:
            top1_confidences.append(predictions[0].confidence)

    return {
        "task": "image_classification",
        "num_images": len(images),
        "performance": {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "fps": 1000 / (sum(latencies) / len(latencies)),
        },
        "quality": {
            "avg_top1_confidence": sum(top1_confidences) / len(top1_confidences) if top1_confidences else 0,
            "top_k": top_k,
        },
    }


def evaluate_embedding(
    backend: CoreMLVisionBackend,
    model_info: Dict[str, Any],
    images: List[Path],
) -> Dict[str, Any]:
    """Evaluate vision embedding model."""
    logger.info(f"Evaluating embedding model: {model_info['name']}")

    latencies = []
    embeddings = []

    # Warmup
    for _ in range(3):
        backend.embed(str(images[0]))

    # Benchmark
    for img_path in images:
        start = time.time()
        embedding = backend.embed(str(img_path))
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        embeddings.append(embedding)

    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = compute_cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    return {
        "task": "vision_embedding",
        "num_images": len(images),
        "performance": {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "fps": 1000 / (sum(latencies) / len(latencies)),
        },
        "quality": {
            "embedding_dim": len(embeddings[0]),
            "avg_pairwise_similarity": sum(similarities) / len(similarities) if similarities else 0,
            "num_comparisons": len(similarities),
        },
    }


def evaluate_model(
    model_info: Dict[str, Any],
    images_dir: Path,
    conf_threshold: float = 0.25,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a single model and save results."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating: {model_info['name']}")
    logger.info(f"Task: {model_info['task']}")
    logger.info(f"Path: {model_info['path']}")
    logger.info(f"{'='*70}")

    # Load images
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not images:
        logger.error(f"No images found in {images_dir}")
        return {}

    logger.info(f"Found {len(images)} test images")

    # Initialize backend
    backend = CoreMLVisionBackend()
    task_type_map = {
        "object_detection": "detection",
        "image_classification": "classification",
        "vision_embedding": "embedding",
    }
    model_type = task_type_map.get(model_info["task"], "detection")

    try:
        backend.load_model(str(model_info["path"]), model_type=model_type)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {}

    # Run evaluation based on task
    if model_info["task"] == "object_detection":
        results = evaluate_detection(backend, model_info, images, conf_threshold)
    elif model_info["task"] == "image_classification":
        results = evaluate_classification(backend, model_info, images, top_k)
    elif model_info["task"] == "vision_embedding":
        results = evaluate_embedding(backend, model_info, images)
    else:
        logger.error(f"Unknown task: {model_info['task']}")
        return {}

    # Add model metadata
    results["model"] = {
        "name": model_info["name"],
        "format": "coreml",
        "precision": model_info["format_info"]["precision"],
        "size_mb": model_info["format_info"]["file_size_mb"],
        "hardware": model_info["format_info"]["hardware_requirements"],
    }

    # Save evaluation.json
    eval_path = model_info["path"].parent / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation to: {eval_path}")

    return results


def generate_comparison_report(
    all_results: List[Dict[str, Any]],
    models_dir: Path,
) -> None:
    """Generate EVALUATION_REPORT.md for each model."""
    # Group by model
    by_model = {}
    for result in all_results:
        model_name = result["model"]["name"]
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(result)

    # Generate report for each model
    for model_name, results in by_model.items():
        model_dir = models_dir / model_name
        report_path = model_dir / "EVALUATION_REPORT.md"

        with open(report_path, "w") as f:
            f.write(f"# {model_name} Evaluation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Formats Evaluated:** {len(results)}\n\n")

            f.write("## Performance Summary\n\n")
            f.write("| Format | Precision | FPS | Avg Latency (ms) | Size (MB) |\n")
            f.write("|--------|-----------|-----|------------------|----------|\n")

            for result in results:
                perf = result["performance"]
                model = result["model"]
                f.write(f"| {model['format']} | {model['precision']} | "
                       f"{perf['fps']:.1f} | {perf['avg_latency_ms']:.1f} | "
                       f"{model['size_mb']:.1f} |\n")

            f.write("\n## Quality Metrics\n\n")
            for result in results:
                f.write(f"### {result['model']['format']} ({result['model']['precision']})\n\n")

                quality = result.get("quality", {})
                for key, value in quality.items():
                    if isinstance(value, float):
                        f.write(f"- **{key}**: {value:.3f}\n")
                    elif isinstance(value, dict):
                        f.write(f"- **{key}**:\n")
                        for k, v in list(value.items())[:10]:  # Top 10 items
                            f.write(f"  - {k}: {v}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            f.write("---\n\n")
            f.write("*Generated by `evaluate_vision_models.py`*\n")

        logger.info(f"Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoreML vision models from quant/models directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../quant/models",
        help="Base models directory (default: ../quant/models)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="../run/llama_pajamas_run/evaluation/images",
        help="Images directory for evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Evaluate specific model only (e.g., yolo-v8n)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K for classification (default: 5)",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    images_dir = Path(args.images)

    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    # Find models
    logger.info(f"Searching for models in: {models_dir}")
    all_models = find_models(models_dir)

    if args.model:
        all_models = [m for m in all_models if m["name"] == args.model]

    if not all_models:
        logger.error("No models found!")
        sys.exit(1)

    logger.info(f"Found {len(all_models)} models to evaluate")

    # Evaluate all models
    all_results = []
    for model_info in all_models:
        result = evaluate_model(
            model_info,
            images_dir,
            conf_threshold=args.conf_threshold,
            top_k=args.top_k,
        )
        if result:
            all_results.append(result)

    # Generate comparison reports
    if all_results:
        generate_comparison_report(all_results, models_dir)
        logger.info(f"\n{'='*70}")
        logger.info("Evaluation complete!")
        logger.info(f"Evaluated {len(all_results)} models")
        logger.info(f"{'='*70}")
    else:
        logger.warning("No models were successfully evaluated")


if __name__ == "__main__":
    main()
