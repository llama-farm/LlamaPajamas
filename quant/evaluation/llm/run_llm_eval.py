#!/usr/bin/env python3
"""Run LLM evaluation on quantized models.

This script evaluates models from quant/models/ using the standard
LLM evaluation prompts and saves results alongside the models.

Usage:
    # Evaluate specific model
    python run_llm_eval.py --model qwen3-8b --format mlx --variant 4bit-mixed

    # Evaluate all variants of a model
    python run_llm_eval.py --model qwen3-8b --format mlx

    # Evaluate all models
    python run_llm_eval.py --all
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_pajamas_quant.evaluator import ModelEvaluator, EvaluationPrompt


def load_prompts(prompts_file: Path) -> list[EvaluationPrompt]:
    """Load evaluation prompts from JSON file."""
    with open(prompts_file) as f:
        data = json.load(f)

    prompts = []
    for p in data["prompts"]:
        prompts.append(
            EvaluationPrompt(
                prompt=p["prompt"],
                category=p["category"],
                expected_qualities=p["expected_qualities"],
            )
        )
    return prompts


def find_models(models_dir: Path, model_name: str = None, format: str = None):
    """Find models to evaluate."""
    models = []

    if model_name:
        model_dirs = [models_dir / model_name]
    else:
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]

    for model_dir in model_dirs:
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Skip vision models
        if manifest.get("task") in ["object_detection", "image_classification", "vision_embedding"]:
            continue

        # Find format directories (gguf/, mlx/)
        for format_dir in model_dir.iterdir():
            if not format_dir.is_dir():
                continue

            format_name = format_dir.name
            if format and format_name != format:
                continue

            # Skip non-model directories
            if format_name in ["hf_model", "hf_cache", "coreml"]:
                continue

            # Find variant directories (Q4_K_M/, 4bit-mixed/, etc.)
            for variant_dir in format_dir.iterdir():
                if not variant_dir.is_dir():
                    continue

                # Check for model file
                model_file = None
                if format_name == "gguf":
                    model_file = next(variant_dir.glob("*.gguf"), None)
                elif format_name == "mlx":
                    # MLX models are directories
                    model_file = variant_dir

                if model_file:
                    models.append({
                        "model_name": manifest["model_name"],
                        "format": format_name,
                        "variant": variant_dir.name,
                        "path": str(model_file),
                        "output_dir": variant_dir,
                    })

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on quantized models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., qwen3-8b)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["gguf", "mlx"],
        help="Model format to evaluate",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Specific variant (e.g., Q4_K_M, 4bit-mixed)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Custom prompts JSON file (default: llm/prompts.json)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../models",
        help="Models directory (default: ../models)",
    )

    args = parser.parse_args()

    # Load prompts
    eval_dir = Path(__file__).parent
    prompts_file = Path(args.prompts) if args.prompts else eval_dir / "prompts.json"

    if not prompts_file.exists():
        print(f"❌ Prompts file not found: {prompts_file}")
        sys.exit(1)

    print(f"📝 Loading prompts from: {prompts_file}")
    prompts = load_prompts(prompts_file)
    print(f"   Found {len(prompts)} evaluation prompts")

    # Find models
    models_dir = Path(__file__).parent.parent / args.models_dir
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        sys.exit(1)

    print(f"\n🔍 Searching for models in: {models_dir}")
    models = find_models(
        models_dir,
        model_name=args.model,
        format=args.format,
    )

    if args.variant:
        models = [m for m in models if m["variant"] == args.variant]

    if not models:
        print("❌ No models found matching criteria")
        sys.exit(1)

    print(f"   Found {len(models)} model(s) to evaluate\n")

    # Initialize evaluator
    evaluator = ModelEvaluator(prompts=prompts)

    # Evaluate each model
    for i, model_info in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(models)}] Evaluating: {model_info['model_name']}")
        print(f"   Format: {model_info['format']}")
        print(f"   Variant: {model_info['variant']}")
        print(f"   Path: {model_info['path']}")
        print(f"{'='*80}")

        try:
            # Run evaluation
            if model_info["format"] == "mlx":
                evaluation = evaluator.evaluate_mlx(
                    model_path=model_info["path"],
                    quant_config={"variant": model_info["variant"]},
                )
            elif model_info["format"] == "gguf":
                evaluation = evaluator.evaluate_gguf(
                    model_path=model_info["path"],
                    quant_config={"variant": model_info["variant"]},
                )
            else:
                print(f"⚠️  Unsupported format: {model_info['format']}")
                continue

            # Save results
            output_path = model_info["output_dir"] / "llm_evaluation.json"
            evaluator.save_evaluation(evaluation, str(output_path))

        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ Evaluation complete! Evaluated {len(models)} model(s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
