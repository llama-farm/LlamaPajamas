"""Main quantizer orchestrator for dual-format conversion."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .detector import ArchitectureDetector
from .manifest import ManifestGenerator
from ..converters.gguf import GGUFConverter
from ..converters.mlx import MLXConverter
from ..evaluator import ModelEvaluator

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
        evaluate: bool = False,
        judge_model: str = "gpt-5-nano",
    ) -> Dict[str, Any]:
        """Convert model to multiple formats with architecture-aware quantization.

        This is the main entry point for Llama-Pajamas quantization. It:
        1. Detects the model architecture
        2. Converts to requested formats (GGUF, MLX, or both)
        3. Generates a unified manifest.json
        4. Optionally evaluates quantized models using LLM-as-judge

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory for all artifacts
            formats: List of formats to generate (default: ["gguf", "mlx"])
            gguf_precision: GGUF quantization method (default: Q4_K_M)
            mlx_bits: MLX quantization bits (default: 4)
            mlx_mixed_precision: Use mixed precision for MLX (default: True)
            evaluate: Run LLM-as-judge evaluation after quantization (default: False)
            judge_model: OpenAI model to use for evaluation (default: gpt-5-nano)

        Returns:
            Dictionary with conversion results, paths, and optional evaluations

        Example:
            >>> quantizer = Quantizer()
            >>> result = quantizer.convert(
            ...     "Qwen/Qwen3-8B",
            ...     "./models/qwen3-8b",
            ...     formats=["gguf", "mlx"],
            ...     evaluate=True  # Auto-evaluate with GPT-5 nano
            ... )
            >>> print(f"GGUF: {result['gguf']['size_gb']:.2f} GB")
            >>> print(f"MLX: {result['mlx']['size_gb']:.2f} GB")
            >>> print(f"Quality: {result['evaluations']['mlx'].avg_quality:.1f}/10")
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

        # Optional: Evaluate quantized models
        evaluations = {}
        if evaluate:
            logger.info("=" * 70)
            logger.info("Running LLM-as-Judge Evaluation")
            logger.info("=" * 70)
            logger.info(f"Judge Model: {judge_model}")
            logger.info("")

            evaluator = ModelEvaluator(judge_model=judge_model)

            # Evaluate MLX model
            if "mlx" in results:
                mlx_path = results["mlx"]["model_path"]
                mlx_config = {
                    "quantization": "mlx",
                    "bits": mlx_bits,
                    "mixed_precision": mlx_mixed_precision,
                }

                mlx_eval = evaluator.evaluate_mlx(mlx_path, mlx_config)
                evaluations["mlx"] = mlx_eval

                # Save evaluation
                eval_path = output_dir / "evaluation_mlx.json"
                evaluator.save_evaluation(mlx_eval, str(eval_path))

            # Evaluate GGUF model
            if "gguf" in results:
                gguf_path = results["gguf"]["model_path"]
                gguf_config = {
                    "quantization": "gguf",
                    "precision": gguf_precision,
                }

                gguf_eval = evaluator.evaluate_gguf(gguf_path, gguf_config)
                evaluations["gguf"] = gguf_eval

                # Save evaluation
                eval_path = output_dir / "evaluation_gguf.json"
                evaluator.save_evaluation(gguf_eval, str(eval_path))

            # Print comparison if both formats evaluated
            if len(evaluations) > 1:
                logger.info("\n" + "=" * 70)
                logger.info("EVALUATION COMPARISON")
                logger.info("=" * 70)
                for fmt, eval_result in evaluations.items():
                    logger.info(f"\n{fmt.upper()}:")
                    logger.info(f"  Quality:   {eval_result.avg_quality:.1f}/10")
                    logger.info(f"  Accuracy:  {eval_result.avg_accuracy:.1f}/10")
                    logger.info(f"  Coherence: {eval_result.avg_coherence:.1f}/10")
                    logger.info(f"  Speed:     {eval_result.avg_generation_time:.2f}s/prompt")
                logger.info("=" * 70)
                logger.info("")

        return {
            "architecture": arch.to_dict(),
            "strategy": strategy,
            "results": results,
            "manifest": manifest,
            "evaluations": evaluations,
            "output_dir": str(output_dir),
        }
