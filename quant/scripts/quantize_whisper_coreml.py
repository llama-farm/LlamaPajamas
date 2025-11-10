#!/usr/bin/env python3
"""
Quantize CoreML Whisper encoder models to INT8 for size/speed optimization.

Usage:
    uv run python scripts/quantize_whisper_coreml.py --model whisper-tiny
    uv run python scripts/quantize_whisper_coreml.py --model all
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Any

try:
    import coremltools as ct
except ImportError:
    print("⚠️  coremltools not found. Installing...")
    import subprocess
    subprocess.run(["uv", "pip", "install", "coremltools"], check=True)
    import coremltools as ct


def quantize_model(
    model_path: Path,
    output_path: Path,
    precision: str = "int8"
) -> Dict[str, Any]:
    """
    Quantize a CoreML Whisper encoder model.

    Args:
        model_path: Path to FP16 encoder.mlpackage
        output_path: Path for quantized output
        precision: Target precision (int8)

    Returns:
        Dict with quantization stats
    """
    print(f"\n📦 Loading model: {model_path.parent.parent.parent.name}/{model_path.name}")

    # Load the FP16 model
    model = ct.models.MLModel(str(model_path))

    # Get original size
    original_size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)

    print(f"   Original size: {original_size_mb:.1f} MB (FP16)")
    print(f"   Quantizing to: {precision.upper()}")

    # Check model type
    spec = model.get_spec()
    is_mlprogram = spec.WhichOneof('Type') == 'mlProgram'

    print(f"   Model type: {'mlprogram' if is_mlprogram else 'unknown'}")

    # Whisper encoders should be mlprogram models
    if is_mlprogram:
        # Use modern optimization API
        print(f"   Using optimize.coreml API")

        if precision == "int8":
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                weight_threshold=512
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        else:
            raise ValueError(f"Precision {precision} not supported for Whisper encoders (use int8)")

        quantized_model = ct.optimize.coreml.linear_quantize_weights(model, config=config)

    else:
        raise ValueError(f"Unexpected model type: {spec.WhichOneof('Type')}")

    # Save quantized model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized_model.save(str(output_path))

    # Get quantized size
    quantized_size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)

    # Calculate compression ratio
    compression = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0

    stats = {
        "original_size_mb": round(original_size_mb, 1),
        "quantized_size_mb": round(quantized_size_mb, 1),
        "compression_ratio": round(compression, 2),
        "size_reduction_pct": round((1 - quantized_size_mb / original_size_mb) * 100, 1)
    }

    print(f"   ✅ Quantized size: {quantized_size_mb:.1f} MB ({stats['size_reduction_pct']}% smaller)")
    print(f"   Compression: {compression:.2f}x")

    return stats


def quantize_whisper_models(model_names: list[str], precision: str = "int8"):
    """Quantize specified Whisper encoder models."""

    models_dir = Path(__file__).parent.parent / "models"

    if "all" in model_names:
        model_names = ["whisper-tiny", "whisper-base", "whisper-small"]

    all_stats = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"🔧 Quantizing: {model_name}")
        print(f"{'='*60}")

        # Paths (Whisper uses float16/encoder.mlpackage)
        fp16_path = models_dir / model_name / "coreml" / "float16" / "encoder.mlpackage"
        output_path = models_dir / model_name / "coreml" / precision / "encoder.mlpackage"

        if not fp16_path.exists():
            print(f"⚠️  FP16 encoder not found: {fp16_path}")
            continue

        # Quantize
        stats = quantize_model(fp16_path, output_path, precision=precision)
        all_stats[model_name] = stats

        # Copy evaluation metadata if it exists
        fp16_eval = fp16_path.parent / "evaluation.json"
        if fp16_eval.exists():
            print(f"   📋 Copying evaluation metadata")
            shutil.copy(fp16_eval, output_path.parent / "evaluation_fp16_reference.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 QUANTIZATION SUMMARY ({precision.upper()})")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Original':<12} {'Quantized':<12} {'Reduction':<12} {'Ratio':<8}")
    print(f"{'-'*60}")

    for model_name, stats in all_stats.items():
        print(f"{model_name:<20} "
              f"{stats['original_size_mb']:.1f} MB{'':<6} "
              f"{stats['quantized_size_mb']:.1f} MB{'':<6} "
              f"{stats['size_reduction_pct']:.1f}%{'':<7} "
              f"{stats['compression_ratio']:.2f}x")

    print(f"\n✅ Quantized {len(all_stats)} Whisper encoders to {precision.upper()}")
    print(f"\n💡 Next: Run evaluation on quantized models:")
    print(f"   cd /Users/robthelen/llama-pajamas/run-coreml")
    print(f"   uv run python ../quant/evaluation/stt/run_eval.py \\")
    print(f"       --models-dir ../quant/models")


def main():
    parser = argparse.ArgumentParser(description="Quantize CoreML Whisper encoder models")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=["whisper-tiny", "whisper-base", "whisper-small", "all"],
        help="Model to quantize (can specify multiple times, or use 'all')"
    )
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["int8"],
        help="Target precision (default: int8)"
    )

    args = parser.parse_args()

    if not args.model:
        args.model = ["all"]

    quantize_whisper_models(args.model, precision=args.precision)


if __name__ == "__main__":
    main()
