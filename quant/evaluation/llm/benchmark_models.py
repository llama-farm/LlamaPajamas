#!/usr/bin/env python3
"""Comprehensive benchmark runner for GGUF and MLX models.

Benchmarks both GGUF and MLX quantized models using 140 questions across 6 categories.
Compares accuracy and speed between formats.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Import test prompts from package
from llama_pajamas_quant.simple_benchmarks import TEST_PROMPTS

# Removed duplicate TEST_PROMPTS definition - using package version
# TEST_PROMPTS contains 140 questions across 6 categories:
# - Knowledge (MMLU-style): 25 questions
# - Common Sense (HellaSwag-style): 20 questions
# - Math (GSM8K-style): 25 questions
# - Reasoning (ARC-style): 20 questions
# - Truthfulness (TruthfulQA-style): 20 questions
# - Tool Calling (BFCL-style): 30 questions



# Runtime functions for benchmarking models

def test_mlx_model(model_path: str) -> Dict[str, Any]:
    """Test MLX model on prompts."""
    print(f"\n{'='*80}")
    print(f"Testing MLX Model: {model_path}")
    print(f"{'='*80}\n")

    from mlx_lm import load, generate

    print("Loading model...")
    model, tokenizer = load(model_path)
    print("✓ Model loaded\n")

    results = []
    correct = 0
    total = len(TEST_PROMPTS)

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{total}] {test['category']}: ", end="", flush=True)

        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=test["prompt"],
            max_tokens=50,
            verbose=False
        )
        duration = time.time() - start

        # Extract answer (first 10 chars for matching, but keep full response)
        full_response = response.strip()
        answer = full_response[:10]  # First 10 chars for matching
        is_correct = test["expected"].lower() in answer.lower()

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s) - Got: {answer[:20]}, Expected: {test['expected']}")

        results.append({
            "category": test["category"],
            "correct": is_correct,
            "duration": duration,
            "response": full_response  # Store full response, not truncated
        })

    accuracy = correct / total
    avg_time = sum(r["duration"] for r in results) / total

    print(f"\n{'='*80}")
    print(f"MLX Results: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Average time: {avg_time:.2f}s per question")
    print(f"{'='*80}\n")

    return {
        "format": "mlx",
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "results": results
    }


def test_gguf_model(model_path: str) -> Dict[str, Any]:
    """Test GGUF model on prompts."""
    print(f"\n{'='*80}")
    print(f"Testing GGUF Model: {model_path}")
    print(f"{'='*80}\n")

    from llama_cpp import Llama

    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    print("✓ Model loaded\n")

    results = []
    correct = 0
    total = len(TEST_PROMPTS)

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{total}] {test['category']}: ", end="", flush=True)

        start = time.time()
        output = llm(
            test["prompt"],
            max_tokens=50,
            temperature=0.1,
            stop=["\n\n"]
        )
        duration = time.time() - start

        # Extract answer (first 10 chars for matching, but keep full response)
        full_response = output["choices"][0]["text"].strip()
        answer = full_response[:10]  # First 10 chars for matching
        is_correct = test["expected"].lower() in answer.lower()

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s) - Got: {answer[:20]}, Expected: {test['expected']}")

        results.append({
            "category": test["category"],
            "correct": is_correct,
            "duration": duration,
            "response": full_response  # Store full response, not truncated
        })

    accuracy = correct / total
    avg_time = sum(r["duration"] for r in results) / total

    print(f"\n{'='*80}")
    print(f"GGUF Results: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Average time: {avg_time:.2f}s per question")
    print(f"{'='*80}\n")

    return {
        "format": "gguf",
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "results": results
    }


def compare_results(mlx_result: Dict, gguf_result: Dict, models_dir: Path):
    """Compare MLX and GGUF results."""
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print()

    # Overall metrics
    print(f"{'Metric':<20} {'MLX 4-bit':<15} {'GGUF Q4_K_M':<15} {'Winner'}")
    print("-"*80)

    # Overall Accuracy
    mlx_acc = mlx_result["accuracy"]
    gguf_acc = gguf_result["accuracy"]
    winner = "MLX" if mlx_acc > gguf_acc else "GGUF" if gguf_acc > mlx_acc else "Tie"
    print(f"{'Overall Accuracy':<20} {mlx_acc:<15.1%} {gguf_acc:<15.1%} {winner}")

    # Speed
    mlx_time = mlx_result["avg_time"]
    gguf_time = gguf_result["avg_time"]
    winner = "GGUF" if gguf_time < mlx_time else "MLX" if mlx_time < gguf_time else "Tie"
    print(f"{'Avg Time (s)':<20} {mlx_time:<15.2f} {gguf_time:<15.2f} {winner}")

    print()

    # Category breakdown
    print(f"{'Category Breakdown':<20} {'MLX':<15} {'GGUF':<15} {'Difference'}")
    print("-"*80)

    categories = {}
    for result in mlx_result["results"]:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"mlx_correct": 0, "mlx_total": 0, "gguf_correct": 0, "gguf_total": 0}
        categories[cat]["mlx_total"] += 1
        if result["correct"]:
            categories[cat]["mlx_correct"] += 1

    for result in gguf_result["results"]:
        cat = result["category"]
        categories[cat]["gguf_total"] += 1
        if result["correct"]:
            categories[cat]["gguf_correct"] += 1

    for cat in sorted(categories.keys()):
        mlx_cat_acc = categories[cat]["mlx_correct"] / categories[cat]["mlx_total"]
        gguf_cat_acc = categories[cat]["gguf_correct"] / categories[cat]["gguf_total"]
        diff = mlx_cat_acc - gguf_cat_acc
        print(f"{cat:<20} {mlx_cat_acc:<15.1%} {gguf_cat_acc:<15.1%} {diff:+.1%}")

    print("="*80)
    print()

    # Save results
    results = {
        "mlx": mlx_result,
        "gguf": gguf_result,
        "comparison": {
            "accuracy_diff": mlx_acc - gguf_acc,
            "speed_diff": mlx_time - gguf_time
        }
    }

    output_file = str(models_dir / "simple_benchmark_results.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {output_file}\n")


def main():
    """Run simple benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark LLM models (GGUF and MLX)")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Model directory containing gguf/ and mlx/ subdirectories")
    args = parser.parse_args()

    print()
    print("="*80)
    print("Comprehensive Benchmark Suite (Using Our Runtimes)")
    print("="*80)
    print()
    print(f"Tests: {len(TEST_PROMPTS)} questions across 6 categories")
    print("- Knowledge (MMLU-style): 25 questions")
    print("- Common Sense (HellaSwag-style): 20 questions")
    print("- Math (GSM8K-style): 25 questions")
    print("- Reasoning (ARC-style): 20 questions")
    print("- Truthfulness (TruthfulQA-style): 20 questions")
    print("- Tool Calling (BFCL-style): 30 questions")
    print()
    print("Estimated time: 5-8 minutes total")
    print()

    # Auto-discover models (works with new subdirectory structure)
    models_dir = Path(args.model_dir)

    # Find MLX model (first subdirectory with config.json)
    mlx_subdirs = [d for d in (models_dir / "mlx").iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not mlx_subdirs:
        print(f"❌ No MLX model found in {models_dir / 'mlx'}")
        return 1
    mlx_path = str(mlx_subdirs[0])

    # Find GGUF model (first .gguf file in any subdirectory)
    gguf_files = list((models_dir / "gguf").glob("**/*.gguf"))
    if not gguf_files:
        print(f"❌ No GGUF model found in {models_dir / 'gguf'}")
        return 1
    gguf_path = str(gguf_files[0])

    # Test MLX
    mlx_result = test_mlx_model(mlx_path)

    # Test GGUF
    gguf_result = test_gguf_model(gguf_path)

    # Compare
    compare_results(mlx_result, gguf_result, models_dir)

    print("="*80)
    print("✅ Benchmarks Complete!")
    print("="*80)
    print()


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
