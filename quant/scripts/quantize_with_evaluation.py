#!/usr/bin/env python3
"""Example: Quantize a model with automatic LLM-as-judge evaluation.

This example shows how to use the Llama-Pajamas quantization pipeline
with integrated evaluation to compare different quantization configurations.
"""

from llama_pajamas_quant import Quantizer


def example_basic_evaluation():
    """Example 1: Basic quantization with evaluation enabled."""
    print("=" * 80)
    print("Example 1: Basic Quantization with Evaluation")
    print("=" * 80)
    print()

    quantizer = Quantizer()

    # Convert model with evaluation enabled
    # This will automatically run GPT-5 nano evaluation after quantization
    result = quantizer.convert(
        model_path="Qwen/Qwen3-8B",
        output_dir="./output/qwen3-8b",
        formats=["gguf", "mlx"],
        evaluate=True,  # Enable automatic evaluation
        judge_model="gpt-5-nano",  # Use GPT-5 nano for judging
    )

    # Access evaluation results
    if "evaluations" in result and result["evaluations"]:
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        for fmt, evaluation in result["evaluations"].items():
            print(f"\n{fmt.upper()}:")
            print(f"  Quality Score:   {evaluation.avg_quality:.1f}/10")
            print(f"  Accuracy:        {evaluation.avg_accuracy:.1f}/10")
            print(f"  Coherence:       {evaluation.avg_coherence:.1f}/10")
            print(f"  Relevance:       {evaluation.avg_relevance:.1f}/10")
            print(f"  Avg Gen Time:    {evaluation.avg_generation_time:.2f}s")

        print("\n" + "=" * 80)


def example_compare_quantizations():
    """Example 2: Compare different quantization configurations."""
    print("=" * 80)
    print("Example 2: Compare Quantization Configurations")
    print("=" * 80)
    print()

    quantizer = Quantizer()

    # Configuration 1: Aggressive quantization (smaller, potentially lower quality)
    print("\n[1/2] Testing aggressive quantization (Q4_0, 4-bit)...")
    result_aggressive = quantizer.convert(
        model_path="Qwen/Qwen3-8B",
        output_dir="./output/qwen3-8b-aggressive",
        formats=["gguf"],
        gguf_precision="Q4_0",  # Faster but potentially lower quality
        evaluate=True,
    )

    # Configuration 2: Balanced quantization (default Q4_K_M)
    print("\n[2/2] Testing balanced quantization (Q4_K_M, 4-bit mixed)...")
    result_balanced = quantizer.convert(
        model_path="Qwen/Qwen3-8B",
        output_dir="./output/qwen3-8b-balanced",
        formats=["gguf"],
        gguf_precision="Q4_K_M",  # Better quality, slightly slower
        evaluate=True,
    )

    # Compare results
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Config':<20} {'Size (GB)':<12} {'Quality':<10} {'Accuracy':<10} {'Speed':<10}")
    print("-" * 80)

    configs = [
        ("Aggressive (Q4_0)", result_aggressive),
        ("Balanced (Q4_K_M)", result_balanced),
    ]

    for config_name, result in configs:
        size = result["results"]["gguf"]["size_gb"]
        eval_data = result["evaluations"]["gguf"]
        quality = eval_data.avg_quality
        accuracy = eval_data.avg_accuracy
        speed = eval_data.avg_generation_time

        print(
            f"{config_name:<20} "
            f"{size:<12.2f} "
            f"{quality:<10.1f} "
            f"{accuracy:<10.1f} "
            f"{speed:<10.2f}s"
        )

    print("=" * 80)
    print()
    print("💡 Tip: Choose based on your priorities:")
    print("   - Aggressive: Smaller size, faster load times")
    print("   - Balanced: Better quality, slightly larger")
    print()


def example_mlx_only():
    """Example 3: MLX-only quantization with evaluation (Apple Silicon)."""
    print("=" * 80)
    print("Example 3: MLX-Only Quantization (Apple Silicon)")
    print("=" * 80)
    print()

    quantizer = Quantizer()

    result = quantizer.convert(
        model_path="Qwen/Qwen3-8B",
        output_dir="./output/qwen3-8b-mlx",
        formats=["mlx"],  # MLX only
        mlx_bits=4,
        mlx_mixed_precision=True,
        evaluate=True,
    )

    # Print results
    if "evaluations" in result and "mlx" in result["evaluations"]:
        eval_data = result["evaluations"]["mlx"]
        print("\n✅ MLX Evaluation Complete!")
        print(f"   Quality: {eval_data.avg_quality:.1f}/10")
        print(f"   Model size: {result['results']['mlx']['size_gb']:.2f} GB")
        print(f"   Evaluation saved to: {result['output_dir']}/evaluation_mlx.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantization with evaluation examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Which example to run (1=basic, 2=compare, 3=mlx-only)",
    )

    args = parser.parse_args()

    try:
        if args.example == 1:
            example_basic_evaluation()
        elif args.example == 2:
            example_compare_quantizations()
        elif args.example == 3:
            example_mlx_only()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
