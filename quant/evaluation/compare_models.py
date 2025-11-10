#!/usr/bin/env python3
"""Universal model comparison script - works for LLM, Vision, and Speech models.

Automatically detects model type and compares ALL quantizations/conversions.

Usage:
    # Compare all formats for a specific model
    uv run python evaluation/compare_models.py --model-dir ./models/qwen3-1.7b
    uv run python evaluation/compare_models.py --model-dir ./models/yolo-v8n
    uv run python evaluation/compare_models.py --model-dir ./models/whisper-tiny

    # Compare ALL models in models directory
    uv run python evaluation/compare_models.py

Auto-detects model type by looking for:
    - LLM: simple_benchmark_results.json, gguf/, mlx/ directories
    - Vision: yolo*/vit*/clip* names, coreml/ directories with evaluation.json
    - Speech: whisper* names, coreml/ directories with evaluation.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / 1024:.1f} KB"


def get_directory_size(directory: Path) -> int:
    """Get total size of directory."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def detect_model_type(model_dir: Path) -> str:
    """Auto-detect model type: llm, vision, or speech."""
    model_name = model_dir.name.lower()

    # Check by name first
    if "whisper" in model_name:
        return "speech"
    if any(name in model_name for name in ["yolo", "vit", "clip", "resnet", "efficientnet"]):
        return "vision"

    # Check by directory structure
    if (model_dir / "simple_benchmark_results.json").exists():
        return "llm"
    if (model_dir / "gguf").exists() or (model_dir / "mlx").exists():
        return "llm"

    # Check for vision/speech evaluation files
    coreml_evals = list(model_dir.glob("coreml/*/evaluation.json"))
    if coreml_evals:
        # Read one to determine type
        with open(coreml_evals[0]) as f:
            data = json.load(f)
            if "wer" in data or "rtf" in data:
                return "speech"
            if "fps" in data or "detections" in data:
                return "vision"

    return "unknown"


def compare_llm_models(model_dir: Path) -> List[Dict[str, Any]]:
    """Compare all LLM quantizations."""
    results = []

    # Method 1: simple_benchmark_results.json (from benchmark_models.py)
    benchmark_file = model_dir / "simple_benchmark_results.json"
    if benchmark_file.exists():
        with open(benchmark_file) as f:
            data = json.load(f)

        for format_name in ["gguf", "mlx"]:
            if format_name in data:
                fmt_data = data[format_name]
                model_path = Path(fmt_data["model_path"])

                # Extract precision/bits
                if format_name == "gguf":
                    # Extract from path like "gguf/Q4_K_M/model.gguf"
                    parts = model_path.parts
                    precision = next((p for p in parts if any(q in p for q in ["Q4_K_M", "Q3_K_M", "IQ2_XS", "Q5_K_M"])), "Q4_K_M")
                    variant = precision
                    size = get_directory_size(model_dir / "gguf" / precision) if (model_dir / "gguf" / precision).exists() else 0
                else:  # mlx
                    # Extract from path like "mlx/4bit-mixed/"
                    parts = model_path.parts
                    variant = next((p for p in parts if "bit" in p), "4bit-mixed")
                    size = get_directory_size(model_dir / "mlx" / variant) if (model_dir / "mlx" / variant).exists() else 0

                results.append({
                    "format": format_name.upper(),
                    "variant": variant,
                    "display_name": f"{format_name.upper()} {variant.replace('-mixed', '')}",
                    "accuracy": fmt_data["accuracy"],
                    "correct": fmt_data["correct"],
                    "total": fmt_data["total"],
                    "speed": fmt_data["avg_time"],
                    "size": size,
                    "size_display": format_size(size) if size > 0 else "N/A"
                })

    # Method 2: individual evaluation.json files (from run_eval.py)
    eval_files = list(model_dir.glob("**/evaluation.json"))
    for eval_file in eval_files:
        # Skip if already processed via simple_benchmark_results.json
        relative_path = eval_file.relative_to(model_dir)
        format_type = relative_path.parts[0].lower()  # gguf or mlx
        variant = relative_path.parts[1] if len(relative_path.parts) > 1 else "unknown"

        # Check if already in results
        if any(r["format"].lower() == format_type and r["variant"] == variant for r in results):
            continue

        with open(eval_file) as f:
            data = json.load(f)

        size = get_directory_size(eval_file.parent)

        results.append({
            "format": format_type.upper(),
            "variant": variant,
            "display_name": f"{format_type.upper()} {variant}",
            "accuracy": data["accuracy"],
            "correct": data["correct"],
            "total": data["total"],
            "speed": data["avg_time"],
            "size": size,
            "size_display": format_size(size)
        })

    return results


def compare_vision_models(model_dir: Path) -> List[Dict[str, Any]]:
    """Compare all Vision quantizations."""
    results = []

    # Find all evaluation.json files in coreml/, onnx/, tensorrt/
    eval_files = []
    for backend in ["coreml", "onnx", "tensorrt"]:
        backend_dir = model_dir / backend
        if backend_dir.exists():
            eval_files.extend(list(backend_dir.glob("*/evaluation.json")))

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)

        relative_path = eval_file.relative_to(model_dir)
        backend = relative_path.parts[0]  # coreml, onnx, tensorrt
        precision = relative_path.parts[1]  # float16, int8, fp16, etc.

        # Extract performance metrics
        performance = data.get("performance", {})
        quality = data.get("quality", {})
        model_info = data.get("model", {})

        # Get model size (prefer from model info, fallback to directory size)
        size_mb = model_info.get("size_mb", 0)
        if size_mb > 0:
            size = int(size_mb * 1024 * 1024)  # Convert MB to bytes
        else:
            size = get_directory_size(eval_file.parent)

        results.append({
            "backend": backend,
            "precision": precision,
            "display_name": f"{backend.upper()} {precision.upper()}",
            "fps": performance.get("fps", 0),
            "latency": performance.get("avg_latency_ms", 0),
            "detections": quality.get("total_detections", 0),
            "images": data.get("num_images", 0),
            "size": size,
            "size_display": format_size(size)
        })

    return results


def compare_speech_models(model_dir: Path) -> List[Dict[str, Any]]:
    """Compare all Speech quantizations."""
    results = []

    # Find all evaluation.json files in coreml/, onnx/
    eval_files = []
    for backend in ["coreml", "onnx"]:
        backend_dir = model_dir / backend
        if backend_dir.exists():
            eval_files.extend(list(backend_dir.glob("*/evaluation.json")))

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)

        relative_path = eval_file.relative_to(model_dir)
        backend = relative_path.parts[0]  # coreml, onnx
        precision = relative_path.parts[1]  # float16, int8, etc.

        # Extract performance metrics
        performance = data.get("performance", {})
        model_info = data.get("model", {})

        # Get model size (prefer from model info, fallback to directory size)
        size_mb = model_info.get("size_mb", 0)
        if size_mb > 0:
            size = int(size_mb * 1024 * 1024)  # Convert MB to bytes
        else:
            size = get_directory_size(eval_file.parent)

        results.append({
            "backend": backend,
            "precision": precision,
            "display_name": f"{backend.upper()} {precision.upper()}",
            "wer": performance.get("avg_wer", 0),
            "latency": performance.get("avg_latency_ms", 0),
            "rtf": performance.get("rtf", 0),
            "samples": data.get("num_samples", 0),
            "size": size,
            "size_display": format_size(size)
        })

    return results


def generate_llm_table(results: List[Dict[str, Any]]) -> str:
    """Generate LLM comparison table."""
    if not results:
        return ""

    md = []
    md.append("| Format | Accuracy | Speed (s/q) | Size |")
    md.append("|--------|----------|-------------|------|")

    # Sort by accuracy descending
    results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    for r in results:
        accuracy_pct = r["accuracy"] * 100
        md.append(f"| **{r['display_name']}** | {accuracy_pct:.1f}% ({r['correct']}/{r['total']}) | {r['speed']:.2f}s | {r['size_display']} |")

    md.append("")

    # Winner analysis
    if len(results) >= 2:
        fastest = min(results, key=lambda x: x["speed"])
        most_accurate = max(results, key=lambda x: x["accuracy"])
        smallest = min(results, key=lambda x: x["size"])

        md.append("**Analysis:**")
        md.append(f"- 🏆 **Most Accurate**: {most_accurate['display_name']} ({most_accurate['accuracy']*100:.1f}%)")
        md.append(f"- ⚡ **Fastest**: {fastest['display_name']} ({fastest['speed']:.2f}s/question)")
        md.append(f"- 💾 **Smallest**: {smallest['display_name']} ({smallest['size_display']})")

        # Speed comparison
        if len(results) == 2:
            speed_diff_pct = abs(results[0]["speed"] - results[1]["speed"]) / max(results[0]["speed"], results[1]["speed"]) * 100
            if fastest == results[0]:
                slower = results[1]
            else:
                slower = results[0]
            md.append(f"- {fastest['display_name']} is **{speed_diff_pct:.0f}% faster** than {slower['display_name']}")

        md.append("")

    return "\n".join(md)


def generate_vision_table(results: List[Dict[str, Any]]) -> str:
    """Generate Vision comparison table."""
    if not results:
        return ""

    md = []
    md.append("| Backend | Precision | FPS | Latency (ms) | Detections | Size |")
    md.append("|---------|-----------|-----|--------------|------------|------|")

    # Sort by FPS descending
    results = sorted(results, key=lambda x: x["fps"], reverse=True)

    for r in results:
        md.append(f"| **{r['backend'].upper()}** | {r['precision'].upper()} | {r['fps']:.1f} | {r['latency']:.1f} | {r['detections']} | {r['size_display']} |")

    md.append("")

    # Winner analysis
    if len(results) >= 2:
        fastest = max(results, key=lambda x: x["fps"])
        smallest = min(results, key=lambda x: x["size"])

        md.append("**Analysis:**")
        md.append(f"- ⚡ **Fastest**: {fastest['display_name']} ({fastest['fps']:.1f} FPS)")
        md.append(f"- 💾 **Smallest**: {smallest['display_name']} ({smallest['size_display']})")

        # Speed comparison
        if len(results) == 2:
            fps_diff_pct = abs(results[0]["fps"] - results[1]["fps"]) / max(results[0]["fps"], results[1]["fps"]) * 100
            md.append(f"- {fastest['display_name']} is **{fps_diff_pct:.0f}% faster** ({fastest['fps']:.1f} vs {min(r['fps'] for r in results if r != fastest):.1f} FPS)")

        md.append("")

    return "\n".join(md)


def generate_speech_table(results: List[Dict[str, Any]]) -> str:
    """Generate Speech comparison table."""
    if not results:
        return ""

    md = []
    md.append("| Backend | Precision | WER | Latency (ms) | RTF | Size |")
    md.append("|---------|-----------|-----|--------------|-----|------|")

    # Sort by WER ascending (lower is better)
    results = sorted(results, key=lambda x: x["wer"])

    for r in results:
        wer_pct = r["wer"] * 100
        md.append(f"| **{r['backend'].upper()}** | {r['precision'].upper()} | {wer_pct:.1f}% | {r['latency']:.1f} | {r['rtf']:.3f} | {r['size_display']} |")

    md.append("")

    # Winner analysis
    if len(results) >= 2:
        best_accuracy = min(results, key=lambda x: x["wer"])
        fastest = min(results, key=lambda x: x["latency"])
        smallest = min(results, key=lambda x: x["size"])

        md.append("**Analysis:**")
        md.append(f"- 🎯 **Best Accuracy**: {best_accuracy['display_name']} (WER: {best_accuracy['wer']*100:.1f}%)")
        md.append(f"- ⚡ **Fastest**: {fastest['display_name']} ({fastest['latency']:.1f}ms, RTF: {fastest['rtf']:.3f})")
        md.append(f"- 💾 **Smallest**: {smallest['display_name']} ({smallest['size_display']})")

        # INT8 vs FP16 comparison
        int8 = next((r for r in results if "int8" in r["precision"].lower()), None)
        fp16 = next((r for r in results if "float16" in r["precision"].lower() or "fp16" in r["precision"].lower()), None)

        if int8 and fp16 and fp16["latency"] > 0:
            wer_change = (int8["wer"] - fp16["wer"]) * 100
            latency_improvement = (fp16["latency"] - int8["latency"]) / fp16["latency"] * 100 if fp16["latency"] > 0 else 0
            size_reduction = (fp16["size"] - int8["size"]) / fp16["size"] * 100 if fp16["size"] > 0 else 0

            if int8["wer"] <= fp16["wer"] and int8["latency"] < fp16["latency"]:
                md.append(f"- **Winner**: INT8 is superior - {'Better' if wer_change <= 0 else 'Similar'} accuracy (WER {wer_change:+.1f}%), {latency_improvement:.0f}% faster, {size_reduction:.0f}% smaller")
            else:
                md.append(f"- **Trade-off**: INT8 is {size_reduction:.0f}% smaller but WER {wer_change:+.1f}%")

        md.append("")

    return "\n".join(md)


def generate_comparison_report(model_dir: Path) -> str:
    """Generate universal comparison report."""
    model_type = detect_model_type(model_dir)
    model_name = model_dir.name

    md = []
    md.append(f"# {model_name} - Evaluation Comparison")
    md.append("")
    md.append(f"**Model Type:** {model_type.upper()}")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    if model_type == "llm":
        results = compare_llm_models(model_dir)
        if results:
            md.append("## LLM Performance Comparison")
            md.append("")
            md.append(generate_llm_table(results))
        else:
            md.append("⚠️ No LLM evaluation results found")
            md.append("")

    elif model_type == "vision":
        results = compare_vision_models(model_dir)
        if results:
            md.append("## Vision Model Performance Comparison")
            md.append("")
            md.append(generate_vision_table(results))
        else:
            md.append("⚠️ No Vision evaluation results found")
            md.append("")

    elif model_type == "speech":
        results = compare_speech_models(model_dir)
        if results:
            md.append("## Speech Model Performance Comparison")
            md.append("")
            md.append(generate_speech_table(results))
        else:
            md.append("⚠️ No Speech evaluation results found")
            md.append("")

    else:
        md.append("⚠️ Unknown model type - could not determine comparison format")
        md.append("")

    md.append("---")
    md.append("")
    md.append("*Generated by `evaluation/compare_models.py`*")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Universal model comparison - auto-detects LLM/Vision/Speech and compares all quantizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Model directory to compare (e.g., ./models/qwen3-1.7b)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Root models directory (default: ./models)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: <model-dir>/COMPARISON.md)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Universal Model Comparison Tool")
    print("="*80)
    print()

    # Determine which models to process
    if args.model_dir:
        model_dirs = [args.model_dir]
    else:
        # Process all model directories
        model_dirs = [d for d in args.models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not model_dirs:
        print("❌ No model directories found")
        return 1

    for model_dir in sorted(model_dirs):
        model_type = detect_model_type(model_dir)

        print(f"📊 Comparing: {model_dir.name}")
        print(f"   Type: {model_type.upper()}")

        # Generate comparison
        report = generate_comparison_report(model_dir)

        # Determine output path
        if args.output:
            output_file = args.output
        else:
            output_file = model_dir / "COMPARISON.md"

        # Save report
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"   ✅ Saved to: {output_file}")
        print()

        # Print to console
        print(report)
        print()

    print("="*80)
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
