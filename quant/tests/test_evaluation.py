#!/usr/bin/env python3
"""Test LLM-as-judge evaluation system.

This script tests the evaluation system on existing quantized models.
"""

import sys
from pathlib import Path

from llama_pajamas_quant.evaluator import ModelEvaluator


def test_evaluation():
    """Test evaluation on existing Qwen3-8B models."""
    print("=" * 80)
    print("LLM-as-Judge Evaluation Test")
    print("=" * 80)
    print()

    # Paths to existing quantized models
    models_dir = Path.home() / ".llamafarm" / "models"

    # Try to find Qwen3-8B models
    qwen_mlx = models_dir / "qwen3-8b-mlx"
    qwen_gguf = models_dir / "qwen3-8b.gguf"

    # Alternative: use local models if available
    if not qwen_mlx.exists():
        qwen_mlx = Path("./models/qwen3-8b-mlx")
    if not qwen_gguf.exists():
        qwen_gguf = Path("./models/qwen3-8b.gguf")

    # Check what we have
    print("Looking for models...")
    mlx_available = qwen_mlx.exists()
    gguf_available = qwen_gguf.exists()

    print(f"  MLX:  {'✓' if mlx_available else '✗'} {qwen_mlx}")
    print(f"  GGUF: {'✓' if gguf_available else '✗'} {qwen_gguf}")
    print()

    if not mlx_available and not gguf_available:
        print("❌ No models found. Please run quantization first or update paths.")
        return

    # Create evaluator
    evaluator = ModelEvaluator(judge_model="gpt-5-nano")
    evaluations = []

    # Evaluate MLX
    if mlx_available:
        print("\n" + "=" * 80)
        print("Testing MLX Evaluation")
        print("=" * 80)

        mlx_eval = evaluator.evaluate_mlx(
            model_path=str(qwen_mlx),
            quant_config={
                "quantization": "mlx",
                "bits": 4,
                "mixed_precision": True,
            }
        )
        evaluations.append(("MLX", mlx_eval))

        # Save results
        output_path = "./evaluation_mlx_test.json"
        evaluator.save_evaluation(mlx_eval, output_path)

    # Evaluate GGUF
    if gguf_available:
        print("\n" + "=" * 80)
        print("Testing GGUF Evaluation")
        print("=" * 80)

        gguf_eval = evaluator.evaluate_gguf(
            model_path=str(qwen_gguf),
            quant_config={
                "quantization": "gguf",
                "precision": "Q4_K_M",
            }
        )
        evaluations.append(("GGUF", gguf_eval))

        # Save results
        output_path = "./evaluation_gguf_test.json"
        evaluator.save_evaluation(gguf_eval, output_path)

    # Print comparison
    if len(evaluations) > 1:
        print("\n" + "=" * 80)
        print("EVALUATION COMPARISON")
        print("=" * 80)
        print()
        print(f"{'Format':<10} {'Quality':<10} {'Accuracy':<10} {'Coherence':<10} {'Speed':<10}")
        print("-" * 80)

        for fmt_name, eval_result in evaluations:
            print(
                f"{fmt_name:<10} "
                f"{eval_result.avg_quality:<10.1f} "
                f"{eval_result.avg_accuracy:<10.1f} "
                f"{eval_result.avg_coherence:<10.1f} "
                f"{eval_result.avg_generation_time:<10.2f}s"
            )

        print("=" * 80)
        print()

    print("✅ Evaluation test complete!")
    print()


if __name__ == "__main__":
    try:
        test_evaluation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
