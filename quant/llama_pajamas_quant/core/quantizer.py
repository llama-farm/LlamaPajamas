"""Main quantizer orchestrator for dual-format conversion."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .detector import ArchitectureDetector
from .manifest import ManifestGenerator
from ..converters.gguf import GGUFConverter
from ..converters.mlx import MLXConverter

logger = logging.getLogger(__name__)


class Quantizer:
    """Main quantizer that orchestrates architecture-aware multi-format conversion."""

    def __init__(self):
        """Initialize quantizer with architecture detector and converters."""
        self.detector = ArchitectureDetector()
        self.manifest_gen = ManifestGenerator()
        self.gguf_converter = GGUFConverter()
        self.mlx_converter = MLXConverter()

    def convert(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        gguf_precision: str = "Q4_K_M",
        mlx_bits: int = 4,
        mlx_mixed_precision: bool = True,
        benchmark: bool = True,  # Run simple benchmarks by default
    ) -> Dict[str, Any]:
        """Convert model to multiple formats with architecture-aware quantization.

        This is the main entry point for Llama-Pajamas quantization. It:
        1. Detects the model architecture
        2. Converts to requested formats (GGUF, MLX, or both)
        3. Generates a unified manifest.json
        4. Optionally benchmarks quantized models (140 questions across 6 categories)

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory for all artifacts
            formats: List of formats to generate (default: ["gguf", "mlx"])
            gguf_precision: GGUF quantization method (default: Q4_K_M)
            mlx_bits: MLX quantization bits (default: 4)
            mlx_mixed_precision: Use mixed precision for MLX (default: True)
            benchmark: Run comprehensive benchmarks after quantization (default: True)

        Returns:
            Dictionary with conversion results, paths, and optional benchmarks

        Example:
            >>> quantizer = Quantizer()
            >>> result = quantizer.convert(
            ...     "Qwen/Qwen3-8B",
            ...     "./models/qwen3-8b",
            ...     formats=["gguf", "mlx"],
            ...     benchmark=True  # Auto-benchmark both models
            ... )
            >>> print(f"GGUF: {result['gguf']['size_gb']:.2f} GB")
            >>> print(f"MLX: {result['mlx']['size_gb']:.2f} GB")
            >>> print(f"GGUF Accuracy: {result['benchmarks']['gguf']['accuracy']:.1%}")
            >>> print(f"MLX Accuracy: {result['benchmarks']['mlx']['accuracy']:.1%}")
        """
        model_path = str(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["gguf", "mlx"]

        logger.info("=" * 70)
        logger.info("Llama-Pajamas Architecture-Aware Quantization")
        logger.info("=" * 70)
        logger.info(f"Model: {model_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Formats: {', '.join(formats)}")
        logger.info("=" * 70)
        logger.info("")

        # Step 1: Detect architecture
        logger.info("[1/4] Detecting model architecture...")
        arch = self.detector.detect(model_path)
        logger.info(f"Detected: {arch.model_type} ({arch.family.value})")
        logger.info(f"Parameters: {arch.params_total}")
        logger.info(f"Attention: {arch.attention_type.value}")

        if arch.is_gqa:
            logger.info(f"GQA Ratio: {arch.gqa_ratio}:1")
        if arch.is_moe:
            logger.info(f"MoE: {arch.num_experts} experts, {arch.num_experts_active} active")

        logger.info("")

        # Get quantization strategy
        strategy = arch.recommend_quantization()
        logger.info("Architecture-aware strategy:")
        for handler in strategy.get("special_handling", []):
            logger.info(f"  - {handler}")
        logger.info("")

        # Step 2: Convert to requested formats
        results = {}
        format_metadata = []

        if "gguf" in formats:
            logger.info("[2/4] Converting to GGUF format...")
            gguf_result = self.gguf_converter.convert(
                model_path=model_path,
                output_dir=output_dir,
                precision=gguf_precision,
                architecture_info=arch.to_dict(),
            )
            results["gguf"] = gguf_result
            format_metadata.append(gguf_result["metadata"])
            logger.info(f"  GGUF: {gguf_result['size_gb']:.2f} GB")
            logger.info("")

        if "mlx" in formats:
            logger.info("[3/4] Converting to MLX format...")
            mlx_result = self.mlx_converter.convert(
                model_path=model_path,
                output_dir=output_dir,
                quantize=True,
                q_bits=mlx_bits,
                mixed_precision=mlx_mixed_precision,
                architecture_info=arch.to_dict(),
            )
            results["mlx"] = mlx_result
            format_metadata.append(mlx_result["metadata"])
            logger.info(f"  MLX: {mlx_result['size_gb']:.2f} GB")
            logger.info("")

        # Step 3: Generate unified manifest
        logger.info("[4/4] Generating manifest.json...")
        manifest = self.manifest_gen.generate(
            model_id=model_path,
            architecture_info=arch.to_dict(),
            formats=format_metadata,
            output_path=output_dir / "manifest.json",
        )
        logger.info(f"  Manifest: {output_dir / 'manifest.json'}")
        logger.info("")

        # Summary
        logger.info("=" * 70)
        logger.info("Conversion Complete!")
        logger.info("=" * 70)
        total_size = sum(r["size_gb"] for r in results.values())
        logger.info(f"Total size: {total_size:.2f} GB")

        for fmt, result in results.items():
            logger.info(f"{fmt.upper()}: {result['size_gb']:.2f} GB")

        logger.info("")
        logger.info("Compression ratios:")
        baseline_size_gb = arch.params_total_num * 2 / (1024**3)  # BF16/FP16

        for fmt, result in results.items():
            compression = baseline_size_gb / result["size_gb"]
            quality_loss = strategy.get("quality_threshold", 0.05) * 100
            logger.info(f"  {fmt.upper()}: {compression:.1f}x (expected <{quality_loss:.0f}% quality loss)")

        logger.info("=" * 70)
        logger.info("")

        # Optional: Benchmark quantized models
        benchmarks = {}
        if benchmark:
            logger.info("=" * 70)
            logger.info("Running Comprehensive Benchmarks")
            logger.info("=" * 70)
            logger.info("140 questions across 6 categories:")
            logger.info("  - Knowledge (MMLU-style): 25 questions")
            logger.info("  - Common Sense (HellaSwag-style): 20 questions")
            logger.info("  - Math (GSM8K-style): 25 questions")
            logger.info("  - Reasoning (ARC-style): 20 questions")
            logger.info("  - Truthfulness (TruthfulQA-style): 20 questions")
            logger.info("  - Tool Calling (BFCL-style): 30 questions")
            logger.info("")

            from ..simple_benchmarks import test_mlx_model, test_gguf_model, compare_results

            # Benchmark MLX model
            if "mlx" in results:
                mlx_path = results["mlx"]["mlx_dir"]  # Use correct key
                logger.info("Benchmarking MLX model...")
                mlx_benchmark = test_mlx_model(str(mlx_path))
                benchmarks["mlx"] = mlx_benchmark

                # Save benchmark with descriptive name
                import json
                precision_type = "mixed" if mlx_mixed_precision else "pure"
                benchmark_name = f"benchmark_mlx_{mlx_bits}bit-{precision_type}.json"
                benchmark_path = output_dir / benchmark_name
                with open(benchmark_path, 'w') as f:
                    json.dump(mlx_benchmark, f, indent=2)
                logger.info(f"  Saved to: {benchmark_path}")

            # Benchmark GGUF model
            if "gguf" in results:
                gguf_path = results["gguf"]["gguf_path"]  # Use correct key
                logger.info("Benchmarking GGUF model...")
                gguf_benchmark = test_gguf_model(str(gguf_path))
                benchmarks["gguf"] = gguf_benchmark

                # Save benchmark with descriptive name
                import json
                benchmark_name = f"benchmark_gguf_{gguf_precision}.json"
                benchmark_path = output_dir / benchmark_name
                with open(benchmark_path, 'w') as f:
                    json.dump(gguf_benchmark, f, indent=2)
                logger.info(f"  Saved to: {benchmark_path}")

            # Print comparison if both formats benchmarked
            if len(benchmarks) == 2:
                compare_results(benchmarks["mlx"], benchmarks["gguf"])

                # Save combined results
                combined_path = output_dir / "benchmarks_comparison.json"
                with open(combined_path, 'w') as f:
                    json.dump(benchmarks, f, indent=2)
                logger.info(f"  Combined results saved to: {combined_path}")

        return {
            "architecture": arch.to_dict(),
            "strategy": strategy,
            "results": results,
            "manifest": manifest,
            "benchmarks": benchmarks,
            "output_dir": str(output_dir),
        }
