#!/usr/bin/env python3
"""Compare all quantizations across all precision levels."""

import json
from pathlib import Path
from typing import Dict, List

def load_benchmark(path: Path) -> Dict:
    """Load benchmark results from JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def main():
    """Compare all quantization benchmarks."""
    models_dir = Path("models/qwen3-8b")

    # Find all benchmark files
    benchmark_files = list(models_dir.glob("benchmark_*.json"))

    if not benchmark_files:
        print("❌ No benchmark files found in models/qwen3-8b/")
        return 1

    print("=" * 100)
    print("COMPREHENSIVE QUANTIZATION COMPARISON")
    print("=" * 100)
    print()

    # Load all benchmarks
    results = []
    for bench_file in sorted(benchmark_files):
        data = load_benchmark(bench_file)
        if data:
            # Determine format and precision from filename
            name = bench_file.stem.replace("benchmark_", "")
            results.append({
                "name": name,
                "file": bench_file.name,
                "data": data
            })

    if not results:
        print("❌ No valid benchmark data found")
        return 1

    # Overall comparison
    print(f"{'Model':<25} {'Accuracy':<12} {'Speed (s)':<12} {'Correct':<10}")
    print("-" * 100)

    for r in results:
        name = r["name"]
        acc = r["data"]["accuracy"]
        time = r["data"]["avg_time"]
        correct = r["data"]["correct"]
        total = r["data"]["total"]

        print(f"{name:<25} {acc:<12.1%} {time:<12.2f} {correct}/{total}")

    print()

    # Category breakdown
    print("=" * 100)
    print("CATEGORY BREAKDOWN")
    print("=" * 100)
    print()

    # Get all categories from first result
    categories = {}
    for result in results[0]["data"]["results"]:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []

    # Calculate per-category accuracy for each model
    category_stats = {}
    for r in results:
        name = r["name"]
        category_stats[name] = {}

        for cat in categories:
            cat_results = [res for res in r["data"]["results"] if res["category"] == cat]
            if cat_results:
                correct = sum(1 for res in cat_results if res["correct"])
                total = len(cat_results)
                category_stats[name][cat] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total > 0 else 0
                }

    # Print category comparison
    categories_sorted = sorted(categories.keys())

    for cat in categories_sorted:
        print(f"\n{cat.upper().replace('_', ' ')}:")
        print(f"{'Model':<25} {'Accuracy':<12} {'Correct':<10}")
        print("-" * 50)

        for r in results:
            name = r["name"]
            if cat in category_stats[name]:
                stats = category_stats[name][cat]
                acc = stats["accuracy"]
                correct = stats["correct"]
                total = stats["total"]
                print(f"{name:<25} {acc:<12.1%} {correct}/{total}")

    # Size comparison
    print()
    print("=" * 100)
    print("SIZE COMPARISON")
    print("=" * 100)
    print()

    # Get sizes from gguf/ and mlx/ directories
    gguf_dir = models_dir / "gguf"
    mlx_dir = models_dir / "mlx"

    print(f"{'Format':<25} {'Size (GB)':<12} {'Path':<50}")
    print("-" * 100)

    # GGUF sizes
    for precision_dir in sorted(gguf_dir.iterdir()):
        if precision_dir.is_dir():
            gguf_files = list(precision_dir.glob("*.gguf"))
            if gguf_files:
                gguf_file = gguf_files[0]
                size_gb = gguf_file.stat().st_size / (1024**3)
                print(f"{'gguf_' + precision_dir.name:<25} {size_gb:<12.2f} {gguf_file}")

    # MLX sizes
    for quant_dir in sorted(mlx_dir.iterdir()):
        if quant_dir.is_dir() and (quant_dir / "config.json").exists():
            total_size = sum(f.stat().st_size for f in quant_dir.rglob("*") if f.is_file())
            size_gb = total_size / (1024**3)
            print(f"{'mlx_' + quant_dir.name:<25} {size_gb:<12.2f} {quant_dir}")

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    # Find best performers
    best_accuracy = max(results, key=lambda r: r["data"]["accuracy"])
    fastest = min(results, key=lambda r: r["data"]["avg_time"])

    print(f"🏆 Best Accuracy: {best_accuracy['name']} ({best_accuracy['data']['accuracy']:.1%})")
    print(f"⚡ Fastest: {fastest['name']} ({fastest['data']['avg_time']:.2f}s per question)")

    # Calculate size efficiency (accuracy per GB)
    print()
    print("Efficiency Rankings (accuracy per GB):")
    print("-" * 50)

    # This is a simplified version - you'd need to match benchmark files to actual sizes
    # For now, just show the accuracy and speed tradeoffs

    print()
    print("Key Insights:")
    for r in sorted(results, key=lambda x: x["data"]["accuracy"], reverse=True):
        name = r["name"]
        acc = r["data"]["accuracy"]
        time = r["data"]["avg_time"]
        print(f"  • {name}: {acc:.1%} accuracy @ {time:.2f}s/question")

    print()
    print("=" * 100)

    return 0

if __name__ == "__main__":
    exit(main())
