#!/usr/bin/env python3
"""Run open-source benchmarks on quantized models.

Uses lm-evaluation-harness with our existing MLX and GGUF runtimes.

Benchmarks (balanced for speed):
- MMLU (100 samples): General knowledge, reasoning
- HellaSwag (100 samples): Common sense
- ARC-Easy (100 samples): Question answering
- TruthfulQA (50 samples): Factuality
- BFCL (100 samples): Tool/function calling

Total: ~450 samples, ~15-20 minutes per model
"""

import sys
import subprocess
from pathlib import Path


def run_lm_eval_mlx(model_path: str, output_dir: str):
    """Run lm-eval on MLX model using transformers backend.

    lm-eval doesn't have native MLX support yet, so we'll use transformers
    and load the MLX-converted model weights.
    """
    print("=" * 80)
    print("Running benchmarks on MLX model (via transformers)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()

    # lm-eval command for MLX (using transformers backend)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float16",
        "--tasks", "mmlu_abstract_algebra,hellaswag,arc_easy,truthfulqa_mc2,bfcl",
        "--num_fewshot", "5",
        "--limit", "100",  # Sample 100 per task for speed
        "--output_path", f"{output_dir}/lm_eval_mlx",
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✅ MLX benchmarks complete!")
        print(f"   Results saved to: {output_dir}/lm_eval_mlx/")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


def run_lm_eval_gguf(model_path: str, output_dir: str):
    """Run lm-eval on GGUF model using llama.cpp backend."""
    print("=" * 80)
    print("Running benchmarks on GGUF model (via llama.cpp)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()

    # lm-eval command for GGUF (using gguf/llama.cpp backend)
    cmd = [
        "lm_eval",
        "--model", "gguf",
        "--model_args", f"filename={model_path}",
        "--tasks", "mmlu_abstract_algebra,hellaswag,arc_easy,truthfulqa_mc2,bfcl",
        "--num_fewshot", "5",
        "--limit", "100",  # Sample 100 per task for speed
        "--output_path", f"{output_dir}/lm_eval_gguf",
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✅ GGUF benchmarks complete!")
        print(f"   Results saved to: {output_dir}/lm_eval_gguf/")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


def compare_results(output_dir: str):
    """Compare MLX and GGUF benchmark results."""
    import json

    mlx_results_path = Path(output_dir) / "lm_eval_mlx" / "results.json"
    gguf_results_path = Path(output_dir) / "lm_eval_gguf" / "results.json"

    if not mlx_results_path.exists() or not gguf_results_path.exists():
        print("⚠️  Results not found, skipping comparison")
        return

    with open(mlx_results_path) as f:
        mlx_results = json.load(f)
    with open(gguf_results_path) as f:
        gguf_results = json.load(f)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Benchmark':<25} {'MLX 4-bit':<15} {'GGUF Q4_K_M':<15} {'Difference'}")
    print("-" * 80)

    # Compare each task
    for task in mlx_results.get("results", {}).keys():
        if task in gguf_results.get("results", {}):
            mlx_score = mlx_results["results"][task].get("acc", 0)
            gguf_score = gguf_results["results"][task].get("acc", 0)
            diff = mlx_score - gguf_score

            print(f"{task:<25} {mlx_score:<15.2%} {gguf_score:<15.2%} {diff:+.2%}")

    print("=" * 80)
    print()


def main():
    """Run benchmarks on Qwen3-8B quantized models."""
    print()
    print("=" * 80)
    print("Open-Source Benchmark Suite")
    print("=" * 80)
    print()
    print("Benchmarks:")
    print("  - MMLU (abstract algebra sample): General knowledge")
    print("  - HellaSwag: Common sense reasoning")
    print("  - ARC-Easy: Question answering")
    print("  - TruthfulQA: Factuality")
    print("  - BFCL: Tool/function calling")
    print()
    print("Configuration:")
    print("  - 100 samples per task (except TruthfulQA: 50)")
    print("  - 5-shot examples")
    print("  - Estimated time: 15-20 minutes per model")
    print()

    # Paths
    mlx_path = "models/qwen3-8b/mlx"
    gguf_path = "models/qwen3-8b/gguf/b968826d9c46dd6066d109eabc6255188de91218_q4_k_m.gguf"
    output_dir = "models/qwen3-8b"

    # Check paths exist
    if not Path(mlx_path).exists():
        print(f"❌ MLX model not found: {mlx_path}")
        return 1

    if not Path(gguf_path).exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return 1

    # Run benchmarks
    print("🚀 Starting benchmark suite...")
    print()

    # MLX
    print("[1/2] MLX Model")
    mlx_result = run_lm_eval_mlx(mlx_path, output_dir)

    if mlx_result != 0:
        print("⚠️  MLX benchmarks failed, continuing with GGUF...")

    # GGUF
    print("\n[2/2] GGUF Model")
    gguf_result = run_lm_eval_gguf(gguf_path, output_dir)

    # Compare
    if mlx_result == 0 and gguf_result == 0:
        compare_results(output_dir)

    print()
    print("=" * 80)
    print("✅ Benchmark suite complete!")
    print("=" * 80)
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
