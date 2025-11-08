#!/usr/bin/env python3
"""Generate markdown comparison report from all evaluation JSONs.

Usage:
    uv run python scripts/compare_evaluations.py --model-dir ./models/qwen3-8b

Scans for all evaluation.json files in:
    - ./models/qwen3-8b/gguf/*/evaluation.json
    - ./models/qwen3-8b/mlx/*/evaluation.json

Generates:
    - ./models/qwen3-8b/EVALUATION_REPORT.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_evaluations(model_dir: Path) -> List[Dict[str, Any]]:
    """Load all evaluation.json files from model directory."""
    evaluations = []

    # Find all evaluation.json files
    eval_files = list(model_dir.glob("**/evaluation.json"))

    if not eval_files:
        print(f"⚠️  No evaluation.json files found in {model_dir}")
        return []

    print(f"Found {len(eval_files)} evaluation file(s):\n")

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)

        # Extract model name from path
        # e.g., models/qwen3-8b/gguf/IQ2_XS/evaluation.json → IQ2_XS (GGUF)
        relative_path = eval_file.relative_to(model_dir)
        parts = relative_path.parts

        if len(parts) >= 3:
            format_type = parts[0].upper()  # gguf or mlx
            model_variant = parts[1]  # IQ2_XS, Q3_K_M, 4bit-mixed, etc.
            display_name = f"{model_variant} ({format_type})"
        else:
            display_name = str(relative_path)

        data["display_name"] = display_name
        data["file_path"] = str(eval_file)
        evaluations.append(data)

        print(f"  ✓ {display_name}: {eval_file}")

    print()
    return evaluations


def generate_markdown_report(evaluations: List[Dict[str, Any]], model_dir: Path) -> str:
    """Generate markdown comparison report."""
    if not evaluations:
        return "# Evaluation Report\n\nNo evaluations found."

    # Sort by accuracy (descending)
    evaluations_sorted = sorted(evaluations, key=lambda x: x["accuracy"], reverse=True)

    md = []
    md.append("# Model Evaluation Report")
    md.append("")
    md.append(f"**Generated:** {evaluations[0].get('timestamp', 'N/A')}")
    md.append(f"**Models Evaluated:** {len(evaluations)}")
    md.append(f"**Questions:** {evaluations[0]['total']}")
    md.append("")

    # Overall comparison table
    md.append("## Overall Performance")
    md.append("")
    md.append("| Model | Format | Accuracy | Correct | Avg Time (s) | Size (approx) |")
    md.append("|-------|--------|----------|---------|--------------|---------------|")

    # Add size estimates (you can make this dynamic later)
    size_map = {
        "IQ2_XS": "3.3 GB",
        "Q3_K_M": "3.8 GB",
        "Q4_K_M": "4.7 GB",
        "2bit-mixed": "2.4 GB",
        "3bit-mixed": "3.2 GB (est)",
        "4bit-mixed": "4.3 GB",
    }

    for eval_data in evaluations_sorted:
        display_name = eval_data["display_name"]
        format_type = eval_data["format"].upper()
        accuracy = eval_data["accuracy"]
        correct = eval_data["correct"]
        total = eval_data["total"]
        avg_time = eval_data["avg_time"]

        # Extract model variant for size lookup
        variant = display_name.split(" (")[0]
        size = size_map.get(variant, "N/A")

        md.append(f"| {display_name} | {format_type} | {accuracy:.1%} | {correct}/{total} | {avg_time:.2f} | {size} |")

    md.append("")

    # Category breakdown
    md.append("## Category Breakdown")
    md.append("")

    # Get all categories
    categories = set()
    for eval_data in evaluations:
        categories.update(eval_data.get("category_stats", {}).keys())

    categories = sorted(categories)

    # Create category table
    header = "| Category |"
    separator = "|----------|"
    for eval_data in evaluations_sorted:
        header += f" {eval_data['display_name']} |"
        separator += "----------|"

    md.append(header)
    md.append(separator)

    for category in categories:
        row = f"| {category.replace('_', ' ').title()} |"
        for eval_data in evaluations_sorted:
            cat_stats = eval_data.get("category_stats", {}).get(category, {})
            if cat_stats:
                cat_acc = cat_stats["accuracy"]
                cat_correct = cat_stats["correct"]
                cat_total = cat_stats["total"]
                row += f" {cat_acc:.1%} ({cat_correct}/{cat_total}) |"
            else:
                row += " N/A |"
        md.append(row)

    md.append("")

    # Key insights
    md.append("## Key Insights")
    md.append("")

    best_model = evaluations_sorted[0]
    md.append(f"- **Best Overall:** {best_model['display_name']} with {best_model['accuracy']:.1%} accuracy")

    fastest_model = min(evaluations, key=lambda x: x["avg_time"])
    md.append(f"- **Fastest:** {fastest_model['display_name']} at {fastest_model['avg_time']:.2f}s per question")

    # Size vs quality analysis
    md.append("")
    md.append("### Size vs Quality Trade-offs")
    md.append("")
    md.append("Smaller models with competitive quality:")
    md.append("")

    for eval_data in evaluations_sorted:
        variant = eval_data['display_name'].split(" (")[0]
        if variant in size_map:
            md.append(f"- **{eval_data['display_name']}**: {size_map[variant]}, {eval_data['accuracy']:.1%} accuracy")

    md.append("")

    # Per-model details
    md.append("## Detailed Results")
    md.append("")

    for eval_data in evaluations_sorted:
        md.append(f"### {eval_data['display_name']}")
        md.append("")
        md.append(f"- **Path:** `{eval_data['file_path']}`")
        md.append(f"- **Accuracy:** {eval_data['accuracy']:.1%}")
        md.append(f"- **Avg Time:** {eval_data['avg_time']:.2f}s")
        md.append("")

        md.append("**Category Performance:**")
        md.append("")

        cat_stats = eval_data.get("category_stats", {})
        for cat in sorted(cat_stats.keys()):
            stats = cat_stats[cat]
            md.append(f"- {cat.replace('_', ' ').title()}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        md.append("")

    # Footer
    md.append("---")
    md.append("")
    md.append("*Generated by `compare_evaluations.py`*")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown comparison report from evaluation JSONs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Model directory (e.g., ./models/qwen3-8b)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown file (default: <model-dir>/EVALUATION_REPORT.md)"
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = args.model_dir / "EVALUATION_REPORT.md"

    print("\n" + "="*80)
    print("Evaluation Comparison Report Generator")
    print("="*80)
    print()

    # Load evaluations
    evaluations = load_evaluations(args.model_dir)

    if not evaluations:
        print("❌ No evaluations found. Please run evaluate_model.py first.")
        return 1

    # Generate report
    print("Generating markdown report...")
    report = generate_markdown_report(evaluations, args.model_dir)

    # Save report
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"✅ Report saved to: {args.output}")
    print()
    print("="*80)
    print()

    # Print summary
    print("Summary:")
    print()
    evaluations_sorted = sorted(evaluations, key=lambda x: x["accuracy"], reverse=True)
    for i, eval_data in enumerate(evaluations_sorted, 1):
        print(f"  {i}. {eval_data['display_name']}: {eval_data['accuracy']:.1%} accuracy, {eval_data['avg_time']:.2f}s avg time")
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
