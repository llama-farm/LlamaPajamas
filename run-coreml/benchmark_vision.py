"""Vision model benchmarking infrastructure.

Benchmarks CoreML vision models on:
- Detection (YOLO): FPS, latency, detection accuracy
- Classification (ViT): FPS, latency, top-1/top-5 accuracy
- Embeddings (CLIP): FPS, latency, similarity consistency

Similar to LLM benchmarking in llama-pajamas-quant.
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from llama_pajamas_run_coreml.backends import CoreMLVisionBackend
from llama_pajamas_run_core.utils import compute_cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark results for a model."""
    model_name: str
    model_type: str
    model_path: str
    model_size_mb: float

    # Performance metrics
    num_images: int
    total_time_s: float
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float

    # Model-specific metrics
    task_metrics: Dict[str, Any]

    # System info
    compute_units: str = "ALL"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class VisionBenchmark:
    """Vision model benchmark runner."""

    def __init__(self, output_dir: Path = Path("./benchmarks")):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def benchmark_detection(
        self,
        model_path: Path,
        image_paths: List[Path],
        conf_threshold: float = 0.25,
        warmup_runs: int = 3,
    ) -> BenchmarkResult:
        """Benchmark object detection model.

        Args:
            model_path: Path to CoreML detection model
            image_paths: List of test images
            conf_threshold: Confidence threshold for detection
            warmup_runs: Number of warmup runs before benchmarking

        Returns:
            BenchmarkResult with detection-specific metrics
        """
        logger.info("=" * 70)
        logger.info(f"Benchmarking Detection Model: {model_path.name}")
        logger.info("=" * 70)

        # Initialize backend
        backend = CoreMLVisionBackend()
        backend.load_model(
            model_path=str(model_path),
            model_type="detection",
        )

        # Warmup
        logger.info(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            backend.detect(str(image_paths[0]), confidence_threshold=conf_threshold)

        # Benchmark
        logger.info(f"Benchmarking on {len(image_paths)} images...")
        latencies = []
        total_detections = 0
        detections_per_image = []

        start_time = time.time()
        for img_path in image_paths:
            img_start = time.time()
            detections = backend.detect(str(img_path), confidence_threshold=conf_threshold)
            img_latency = (time.time() - img_start) * 1000  # ms

            latencies.append(img_latency)
            total_detections += len(detections)
            detections_per_image.append(len(detections))

        total_time = time.time() - start_time

        # Calculate metrics
        latencies = np.array(latencies)
        avg_detections = np.mean(detections_per_image)

        result = BenchmarkResult(
            model_name=model_path.stem,
            model_type="detection",
            model_path=str(model_path),
            model_size_mb=self._get_model_size_mb(model_path),
            num_images=len(image_paths),
            total_time_s=total_time,
            avg_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            fps=len(image_paths) / total_time,
            task_metrics={
                "total_detections": total_detections,
                "avg_detections_per_image": float(avg_detections),
                "confidence_threshold": conf_threshold,
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Unload
        backend.unload()

        # Log results
        self._log_results(result)
        return result

    def benchmark_classification(
        self,
        model_path: Path,
        image_paths: List[Path],
        top_k: int = 5,
        warmup_runs: int = 3,
    ) -> BenchmarkResult:
        """Benchmark image classification model.

        Args:
            model_path: Path to CoreML classification model
            image_paths: List of test images
            top_k: Number of top predictions
            warmup_runs: Number of warmup runs

        Returns:
            BenchmarkResult with classification-specific metrics
        """
        logger.info("=" * 70)
        logger.info(f"Benchmarking Classification Model: {model_path.name}")
        logger.info("=" * 70)

        # Initialize backend
        backend = CoreMLVisionBackend()
        backend.load_model(
            model_path=str(model_path),
            model_type="classification",
        )

        # Warmup
        logger.info(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            backend.classify(str(image_paths[0]), top_k=top_k)

        # Benchmark
        logger.info(f"Benchmarking on {len(image_paths)} images...")
        latencies = []
        top1_confidences = []
        top5_confidences = []

        start_time = time.time()
        for img_path in image_paths:
            img_start = time.time()
            predictions = backend.classify(str(img_path), top_k=top_k)
            img_latency = (time.time() - img_start) * 1000  # ms

            latencies.append(img_latency)
            if predictions:
                top1_confidences.append(predictions[0].confidence)
                top5_avg = np.mean([p.confidence for p in predictions[:5]])
                top5_confidences.append(top5_avg)

        total_time = time.time() - start_time

        # Calculate metrics
        latencies = np.array(latencies)

        result = BenchmarkResult(
            model_name=model_path.stem,
            model_type="classification",
            model_path=str(model_path),
            model_size_mb=self._get_model_size_mb(model_path),
            num_images=len(image_paths),
            total_time_s=total_time,
            avg_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            fps=len(image_paths) / total_time,
            task_metrics={
                "top_k": top_k,
                "avg_top1_confidence": float(np.mean(top1_confidences)),
                "avg_top5_confidence": float(np.mean(top5_confidences)),
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Unload
        backend.unload()

        # Log results
        self._log_results(result)
        return result

    def benchmark_embeddings(
        self,
        model_path: Path,
        image_paths: List[Path],
        warmup_runs: int = 3,
    ) -> BenchmarkResult:
        """Benchmark image embedding model.

        Args:
            model_path: Path to CoreML embedding model
            image_paths: List of test images
            warmup_runs: Number of warmup runs

        Returns:
            BenchmarkResult with embedding-specific metrics
        """
        logger.info("=" * 70)
        logger.info(f"Benchmarking Embedding Model: {model_path.name}")
        logger.info("=" * 70)

        # Initialize backend
        backend = CoreMLVisionBackend()
        backend.load_model(
            model_path=str(model_path),
            model_type="embedding",
        )

        # Warmup
        logger.info(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            backend.embed(str(image_paths[0]))

        # Benchmark
        logger.info(f"Benchmarking on {len(image_paths)} images...")
        latencies = []
        embeddings = []
        embedding_norms = []

        start_time = time.time()
        for img_path in image_paths:
            img_start = time.time()
            embedding = backend.embed(str(img_path))
            img_latency = (time.time() - img_start) * 1000  # ms

            latencies.append(img_latency)
            embeddings.append(embedding)
            embedding_norms.append(np.linalg.norm(embedding))

        total_time = time.time() - start_time

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # Calculate metrics
        latencies = np.array(latencies)

        result = BenchmarkResult(
            model_name=model_path.stem,
            model_type="embedding",
            model_path=str(model_path),
            model_size_mb=self._get_model_size_mb(model_path),
            num_images=len(image_paths),
            total_time_s=total_time,
            avg_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            fps=len(image_paths) / total_time,
            task_metrics={
                "embedding_dim": embeddings[0].shape[0],
                "avg_norm": float(np.mean(embedding_norms)),
                "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
                "std_similarity": float(np.std(similarities)) if similarities else 0.0,
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Unload
        backend.unload()

        # Log results
        self._log_results(result)
        return result

    def _get_model_size_mb(self, model_path: Path) -> float:
        """Get model size in MB."""
        if not model_path.exists():
            return 0.0

        # For .mlpackage, sum all files
        total_size = 0
        if model_path.is_dir():
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        else:
            total_size = model_path.stat().st_size

        return total_size / (1024 * 1024)  # Convert to MB

    def _log_results(self, result: BenchmarkResult) -> None:
        """Log benchmark results."""
        logger.info("")
        logger.info(f"✅ Benchmark Complete: {result.model_name}")
        logger.info(f"   Model type: {result.model_type}")
        logger.info(f"   Model size: {result.model_size_mb:.1f} MB")
        logger.info(f"   Images: {result.num_images}")
        logger.info(f"   Total time: {result.total_time_s:.2f}s")
        logger.info(f"   Avg latency: {result.avg_latency_ms:.2f} ± {result.std_latency_ms:.2f} ms")
        logger.info(f"   Min/Max: {result.min_latency_ms:.2f} / {result.max_latency_ms:.2f} ms")
        logger.info(f"   FPS: {result.fps:.1f}")
        logger.info(f"   Task metrics: {result.task_metrics}")

    def save_results(self, results: List[BenchmarkResult], filename: str = "benchmark_results.json") -> Path:
        """Save all benchmark results to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        logger.info(f"\n📊 Results saved to: {output_path}")
        return output_path


def main():
    """Run vision model benchmarks."""
    # Paths
    models_dir = Path("/tmp/coreml-test-models")
    images_dir = Path("../run/llama_pajamas_run/evaluation/images")

    # Get test images
    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    image_paths = sorted(image_paths)

    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        return 1

    logger.info(f"Found {len(image_paths)} test images")

    # Initialize benchmark runner
    benchmark = VisionBenchmark(output_dir=Path("./benchmarks"))

    results = []

    # Benchmark YOLO detection
    yolo_model = models_dir / "yolo-v8n-coreml" / "model.mlpackage"
    if yolo_model.exists():
        result = benchmark.benchmark_detection(
            model_path=yolo_model,
            image_paths=image_paths,
            conf_threshold=0.25,
        )
        results.append(result)

    # Benchmark ViT classification
    vit_model = models_dir / "vit-base-coreml" / "model.mlpackage"
    if vit_model.exists():
        result = benchmark.benchmark_classification(
            model_path=vit_model,
            image_paths=image_paths,
            top_k=5,
        )
        results.append(result)

    # Benchmark CLIP embeddings
    clip_model = models_dir / "clip-vit-base-coreml" / "model.mlpackage"
    if clip_model.exists():
        result = benchmark.benchmark_embeddings(
            model_path=clip_model,
            image_paths=image_paths,
        )
        results.append(result)

    # Save results
    if results:
        benchmark.save_results(results)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70)
        for r in results:
            logger.info(f"{r.model_name}: {r.fps:.1f} FPS, {r.avg_latency_ms:.1f}ms avg latency")

        return 0
    else:
        logger.warning("No models found to benchmark")
        return 1


if __name__ == "__main__":
    exit(main())
