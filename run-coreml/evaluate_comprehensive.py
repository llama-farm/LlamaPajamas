"""Comprehensive evaluation of CoreML vision models.

Runs complete evaluation on available test images:
- Detection accuracy and performance
- Classification accuracy and performance
- Embedding quality and performance
- Comparative analysis across models
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from llama_pajamas_run_coreml.backends import CoreMLVisionBackend
from llama_pajamas_run_core.utils import compute_cosine_similarity
from benchmark_vision import VisionBenchmark, BenchmarkResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation runner."""

    def __init__(self, output_dir: Path = Path("./evaluation_results")):
        """Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = VisionBenchmark(output_dir=output_dir)

    def evaluate_detection_quality(
        self,
        model_path: Path,
        image_paths: List[Path],
    ) -> Dict[str, Any]:
        """Evaluate detection quality metrics.

        Args:
            model_path: Path to detection model
            image_paths: List of test images

        Returns:
            Detection quality metrics
        """
        logger.info("Evaluating detection quality...")

        backend = CoreMLVisionBackend()
        backend.load_model(str(model_path), model_type="detection")

        # Collect detection statistics
        all_detections = []
        confidence_scores = []
        detections_per_image = []
        class_distribution = {}

        for img_path in image_paths:
            detections = backend.detect(str(img_path), confidence_threshold=0.25)
            all_detections.extend(detections)
            detections_per_image.append(len(detections))

            for det in detections:
                confidence_scores.append(det.confidence)
                class_name = det.class_name
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

        backend.unload()

        # Calculate quality metrics
        metrics = {
            "total_detections": len(all_detections),
            "images_processed": len(image_paths),
            "avg_detections_per_image": float(np.mean(detections_per_image)),
            "std_detections_per_image": float(np.std(detections_per_image)),
            "avg_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            "std_confidence": float(np.std(confidence_scores)) if confidence_scores else 0.0,
            "min_confidence": float(np.min(confidence_scores)) if confidence_scores else 0.0,
            "max_confidence": float(np.max(confidence_scores)) if confidence_scores else 0.0,
            "unique_classes_detected": len(class_distribution),
            "class_distribution": dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
        }

        logger.info(f"  Total detections: {metrics['total_detections']}")
        logger.info(f"  Avg confidence: {metrics['avg_confidence']:.2%}")
        logger.info(f"  Unique classes: {metrics['unique_classes_detected']}")

        return metrics

    def evaluate_classification_quality(
        self,
        model_path: Path,
        image_paths: List[Path],
    ) -> Dict[str, Any]:
        """Evaluate classification quality metrics.

        Args:
            model_path: Path to classification model
            image_paths: List of test images

        Returns:
            Classification quality metrics
        """
        logger.info("Evaluating classification quality...")

        backend = CoreMLVisionBackend()
        backend.load_model(str(model_path), model_type="classification")

        # Collect classification statistics
        top1_confidences = []
        top5_confidences = []
        prediction_entropy = []

        for img_path in image_paths:
            predictions = backend.classify(str(img_path), top_k=5)

            if predictions:
                top1_confidences.append(predictions[0].confidence)

                if len(predictions) >= 5:
                    top5_avg = np.mean([p.confidence for p in predictions[:5]])
                    top5_confidences.append(top5_avg)

                    # Calculate entropy (measure of prediction uncertainty)
                    probs = [p.confidence for p in predictions[:5]]
                    probs = np.array(probs) / np.sum(probs)  # Normalize
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    prediction_entropy.append(entropy)

        backend.unload()

        # Calculate quality metrics
        metrics = {
            "images_processed": len(image_paths),
            "avg_top1_confidence": float(np.mean(top1_confidences)),
            "std_top1_confidence": float(np.std(top1_confidences)),
            "avg_top5_confidence": float(np.mean(top5_confidences)) if top5_confidences else 0.0,
            "avg_prediction_entropy": float(np.mean(prediction_entropy)) if prediction_entropy else 0.0,
            "high_confidence_predictions": sum(1 for c in top1_confidences if c > 0.7),
            "low_confidence_predictions": sum(1 for c in top1_confidences if c < 0.3),
        }

        logger.info(f"  Avg top-1 confidence: {metrics['avg_top1_confidence']:.2%}")
        logger.info(f"  High confidence (>70%): {metrics['high_confidence_predictions']}/{len(image_paths)}")

        return metrics

    def evaluate_embedding_quality(
        self,
        model_path: Path,
        image_paths: List[Path],
    ) -> Dict[str, Any]:
        """Evaluate embedding quality metrics.

        Args:
            model_path: Path to embedding model
            image_paths: List of test images

        Returns:
            Embedding quality metrics
        """
        logger.info("Evaluating embedding quality...")

        backend = CoreMLVisionBackend()
        backend.load_model(str(model_path), model_type="embedding")

        # Generate embeddings
        embeddings = []
        norms = []

        for img_path in image_paths:
            embedding = backend.embed(str(img_path))
            embeddings.append(embedding)
            norms.append(np.linalg.norm(embedding))

        backend.unload()

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # Calculate quality metrics
        metrics = {
            "images_processed": len(image_paths),
            "embedding_dim": embeddings[0].shape[0],
            "avg_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "std_similarity": float(np.std(similarities)) if similarities else 0.0,
            "min_similarity": float(np.min(similarities)) if similarities else 0.0,
            "max_similarity": float(np.max(similarities)) if similarities else 0.0,
            "total_comparisons": len(similarities),
        }

        logger.info(f"  Embedding dim: {metrics['embedding_dim']}")
        logger.info(f"  Avg similarity: {metrics['avg_similarity']:.3f}")
        logger.info(f"  Similarity range: [{metrics['min_similarity']:.3f}, {metrics['max_similarity']:.3f}]")

        return metrics

    def run_comprehensive_evaluation(
        self,
        models_dir: Path,
        images_dir: Path,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on all models.

        Args:
            models_dir: Directory containing CoreML models
            images_dir: Directory containing test images

        Returns:
            Complete evaluation results
        """
        logger.info("=" * 70)
        logger.info("COMPREHENSIVE VISION MODEL EVALUATION")
        logger.info("=" * 70)
        logger.info("")

        # Get test images
        image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_paths = sorted(image_paths)

        logger.info(f"Test images: {len(image_paths)}")
        logger.info(f"Images: {[p.name for p in image_paths]}")
        logger.info("")

        results = {
            "test_images": len(image_paths),
            "image_files": [p.name for p in image_paths],
            "models": {},
        }

        # 1. YOLO Detection Evaluation
        yolo_model = models_dir / "yolo-v8n-coreml" / "model.mlpackage"
        if yolo_model.exists():
            logger.info("\n" + "=" * 70)
            logger.info("1. YOLO DETECTION EVALUATION")
            logger.info("=" * 70)

            # Performance benchmark
            perf_result = self.benchmark.benchmark_detection(
                model_path=yolo_model,
                image_paths=image_paths,
                conf_threshold=0.25,
            )

            # Quality evaluation
            quality_result = self.evaluate_detection_quality(
                model_path=yolo_model,
                image_paths=image_paths,
            )

            results["models"]["yolo"] = {
                "performance": perf_result.to_dict(),
                "quality": quality_result,
            }

        # 2. ViT Classification Evaluation
        vit_model = models_dir / "vit-base-coreml" / "model.mlpackage"
        if vit_model.exists():
            logger.info("\n" + "=" * 70)
            logger.info("2. VIT CLASSIFICATION EVALUATION")
            logger.info("=" * 70)

            # Performance benchmark
            perf_result = self.benchmark.benchmark_classification(
                model_path=vit_model,
                image_paths=image_paths,
                top_k=5,
            )

            # Quality evaluation
            quality_result = self.evaluate_classification_quality(
                model_path=vit_model,
                image_paths=image_paths,
            )

            results["models"]["vit"] = {
                "performance": perf_result.to_dict(),
                "quality": quality_result,
            }

        # 3. CLIP Embedding Evaluation
        clip_model = models_dir / "clip-vit-base-coreml" / "model.mlpackage"
        if clip_model.exists():
            logger.info("\n" + "=" * 70)
            logger.info("3. CLIP EMBEDDING EVALUATION")
            logger.info("=" * 70)

            # Performance benchmark
            perf_result = self.benchmark.benchmark_embeddings(
                model_path=clip_model,
                image_paths=image_paths,
            )

            # Quality evaluation
            quality_result = self.evaluate_embedding_quality(
                model_path=clip_model,
                image_paths=image_paths,
            )

            results["models"]["clip"] = {
                "performance": perf_result.to_dict(),
                "quality": quality_result,
            }

        # Save results
        output_file = self.output_dir / "comprehensive_evaluation.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n📊 Results saved to: {output_file}")

        return results


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Test Images: {results['test_images']}")
    logger.info("")

    for model_name, model_results in results["models"].items():
        perf = model_results["performance"]
        qual = model_results["quality"]

        logger.info(f"{model_name.upper()}:")
        logger.info(f"  Performance:")
        logger.info(f"    - FPS: {perf['fps']:.1f}")
        logger.info(f"    - Latency: {perf['avg_latency_ms']:.1f} ± {perf['std_latency_ms']:.1f} ms")
        logger.info(f"    - Model size: {perf['model_size_mb']:.1f} MB")
        logger.info(f"  Quality:")

        if model_name == "yolo":
            logger.info(f"    - Total detections: {qual['total_detections']}")
            logger.info(f"    - Avg confidence: {qual['avg_confidence']:.2%}")
            logger.info(f"    - Unique classes: {qual['unique_classes_detected']}")
        elif model_name == "vit":
            logger.info(f"    - Avg top-1 confidence: {qual['avg_top1_confidence']:.2%}")
            logger.info(f"    - High confidence (>70%): {qual['high_confidence_predictions']}/{qual['images_processed']}")
        elif model_name == "clip":
            logger.info(f"    - Embedding dim: {qual['embedding_dim']}")
            logger.info(f"    - Avg similarity: {qual['avg_similarity']:.3f}")

        logger.info("")


def main():
    """Run comprehensive evaluation."""
    # Paths
    models_dir = Path("/tmp/coreml-test-models")
    images_dir = Path("../run/llama_pajamas_run/evaluation/images")

    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_comprehensive_evaluation(
        models_dir=models_dir,
        images_dir=images_dir,
    )

    # Print summary
    print_summary(results)

    logger.info("🎉 Comprehensive evaluation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
