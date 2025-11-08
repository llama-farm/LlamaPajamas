#!/usr/bin/env python3
"""Run evaluation on existing Qwen3-8B GGUF and MLX models."""

import sys
from pathlib import Path

from llama_pajamas_quant.evaluator import ModelEvaluator


def main():
    """Evaluate existing Qwen3-8B models."""
    print("=" * 80)
    print("Qwen3-8B Model Evaluation with GPT-5 Nano")
    print("=" * 80)
    print()

    # Model paths
    gguf_path = Path("models/qwen3-8b/gguf/b968826d9c46dd6066d109eabc6255188de91218_q4_k_m.gguf")
    mlx_path = Path("models/qwen3-8b/mlx")

    # Check existence
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return 1

    if not mlx_path.exists():
        print(f"❌ MLX model not found: {mlx_path}")
        return 1

    print(f"✓ GGUF model: {gguf_path}")
    print(f"✓ MLX model:  {mlx_path}")
    print()

    # Create evaluator
    print("Creating evaluator with GPT-5 nano...")
    evaluator = ModelEvaluator(judge_model="gpt-5-nano")
    print("✓ Evaluator ready")
    print()

    evaluations = {}

    # Evaluate MLX
    print("=" * 80)
    print("Evaluating MLX Model (7 prompts)")
    print("=" * 80)

    mlx_eval = evaluator.evaluate_mlx(
        model_path=str(mlx_path),
        quant_config={
            "quantization": "mlx",
            "bits": 4,
            "mixed_precision": True,
        }
    )
    evaluations["mlx"] = mlx_eval

    # Save MLX results
    mlx_output = "models/qwen3-8b/evaluation_mlx.json"
    evaluator.save_evaluation(mlx_eval, mlx_output)

    # Evaluate GGUF
    print("\n" + "=" * 80)
    print("Evaluating GGUF Model (7 prompts)")
    print("=" * 80)

    gguf_eval = evaluator.evaluate_gguf(
        model_path=str(gguf_path),
        quant_config={
            "quantization": "gguf",
            "precision": "Q4_K_M",
        }
    )
    evaluations["gguf"] = gguf_eval

    # Save GGUF results
    gguf_output = "models/qwen3-8b/evaluation_gguf.json"
    evaluator.save_evaluation(gguf_eval, gguf_output)

    # Print comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Metric':<20} {'MLX 4-bit':<15} {'GGUF Q4_K_M':<15} {'Difference':<15}")
    print("-" * 80)

    metrics = [
        ("Accuracy", "avg_accuracy"),
        ("Coherence", "avg_coherence"),
        ("Relevance", "avg_relevance"),
        ("Overall Quality", "avg_quality"),
        ("Avg Gen Time", "avg_generation_time"),
    ]

    for label, attr in metrics:
        mlx_val = getattr(mlx_eval, attr)
        gguf_val = getattr(gguf_eval, attr)

        if "time" in attr.lower():
            # Time metric (lower is better)
            diff = mlx_val - gguf_val
            diff_str = f"{diff:+.2f}s"
            print(f"{label:<20} {mlx_val:<15.2f} {gguf_val:<15.2f} {diff_str:<15}")
        else:
            # Score metric (higher is better)
            diff = mlx_val - gguf_val
            diff_str = f"{diff:+.1f}"
            print(f"{label:<20} {mlx_val:<15.1f} {gguf_val:<15.1f} {diff_str:<15}")

    print("=" * 80)
    print()

    # Summary
    print("📊 Summary:")
    print()

    if mlx_eval.avg_quality > gguf_eval.avg_quality:
        print(f"  🏆 MLX has higher quality ({mlx_eval.avg_quality:.1f} vs {gguf_eval.avg_quality:.1f})")
    elif gguf_eval.avg_quality > mlx_eval.avg_quality:
        print(f"  🏆 GGUF has higher quality ({gguf_eval.avg_quality:.1f} vs {mlx_eval.avg_quality:.1f})")
    else:
        print(f"  🤝 Both have equal quality ({mlx_eval.avg_quality:.1f})")

    if mlx_eval.avg_generation_time < gguf_eval.avg_generation_time:
        print(f"  ⚡ MLX is faster ({mlx_eval.avg_generation_time:.2f}s vs {gguf_eval.avg_generation_time:.2f}s)")
    else:
        print(f"  ⚡ GGUF is faster ({gguf_eval.avg_generation_time:.2f}s vs {mlx_eval.avg_generation_time:.2f}s)")

    print()
    print(f"📁 Results saved to:")
    print(f"  - {mlx_output}")
    print(f"  - {gguf_output}")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
