#!/usr/bin/env python3
"""
Export vision models to TensorRT for NVIDIA GPU inference.

Converts PyTorch models → ONNX → TensorRT with quantization.

Usage:
    # Export YOLO with INT8
    uv run python scripts/export_tensorrt_vision.py --model yolo-v8n --precision int8

    # Export ViT with FP16
    uv run python scripts/export_tensorrt_vision.py --model vit-base --precision float16

    # Export all models
    uv run python scripts/export_tensorrt_vision.py --model all
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Vision model configurations
VISION_CONFIGS = {
    "yolo-v8n": {
        "source": "ultralytics",
        "model_name": "yolov8n.pt",
        "task": "detection",
        "input_shape": (1, 3, 640, 640),
        "expected_size_mb": 6,
    },
    "vit-base": {
        "source": "huggingface",
        "model_name": "google/vit-base-patch16-224",
        "task": "classification",
        "input_shape": (1, 3, 224, 224),
        "expected_size_mb": 330,
    },
    "clip-vit-base": {
        "source": "huggingface",
        "model_name": "openai/clip-vit-base-patch32",
        "task": "embedding",
        "input_shape": (1, 3, 224, 224),
        "expected_size_mb": 150,
    },
}


def export_via_onnx(
    model_name: str,
    config: Dict[str, Any],
    output_dir: Path,
    precision: str = "float16",
) -> Dict[str, Any]:
    """Export vision model to TensorRT via ONNX.

    Pipeline: PyTorch → ONNX → TensorRT

    Args:
        model_name: Model name
        config: Model configuration
        output_dir: Output directory
        precision: Precision ('float16', 'int8')

    Returns:
        Export metadata
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Exporting {model_name} to TensorRT ({precision.upper()})")
    logger.info(f"{'='*70}")

    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{model_name}.onnx"
    engine_path = output_dir / f"{model_name}-{precision}.engine"

    # Step 1: PyTorch → ONNX
    logger.info("\n📦 Step 1: Export to ONNX")

    if config["source"] == "ultralytics":
        # YOLO export
        logger.info(f"   Using Ultralytics export")
        logger.info(f"""
    # Manual command:
    python -m ultralytics.export \\
        --model {config['model_name']} \\
        --format onnx \\
        --imgsz 640 \\
        --simplify
        """)
    elif config["source"] == "huggingface":
        # HuggingFace export
        logger.info(f"   Using HuggingFace export")
        logger.info(f"""
    # Manual command:
    python -m transformers.onnx \\
        --model {config['model_name']} \\
        --feature default \\
        {onnx_path}
        """)

    # Step 2: ONNX → TensorRT
    logger.info("\n🔧 Step 2: Convert ONNX to TensorRT")

    input_shape = config["input_shape"]
    batch_size, channels, height, width = input_shape

    # Build trtexec command
    trtexec_cmd = f"""
    trtexec \\
        --onnx={onnx_path} \\
        --saveEngine={engine_path} \\
        --explicitBatch \\
        --minShapes=input:{batch_size}x{channels}x{height}x{width} \\
        --optShapes=input:{batch_size}x{channels}x{height}x{width} \\
        --maxShapes=input:{batch_size*4}x{channels}x{height}x{width}
    """

    if precision == "float16":
        trtexec_cmd += " \\\n        --fp16"
    elif precision == "int8":
        trtexec_cmd += " \\\n        --int8 \\\n        --best"

    logger.info(f"   Command:\n{trtexec_cmd}")

    # Save export script
    script_path = output_dir / f"export_{model_name}_{precision}.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# TensorRT export script for " + model_name + "\n\n")
        f.write("# Step 1: Export to ONNX\n")
        if config["source"] == "ultralytics":
            f.write(f"python -m ultralytics.export --model {config['model_name']} --format onnx\n\n")
        else:
            f.write(f"python -m transformers.onnx --model {config['model_name']} --feature default {onnx_path}\n\n")
        f.write("# Step 2: Convert to TensorRT\n")
        f.write(trtexec_cmd + "\n")

    script_path.chmod(0o755)

    logger.info(f"\n✅ Export script saved: {script_path}")

    # Metadata
    metadata = {
        "model_name": model_name,
        "task": config["task"],
        "precision": precision,
        "input_shape": input_shape,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "export_script": str(script_path),
        "status": "script_generated",
    }

    # Save metadata
    metadata_path = output_dir / "export_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export vision models to TensorRT")
    parser.add_argument(
        "--model",
        default="yolo-v8n",
        choices=list(VISION_CONFIGS.keys()) + ["all"],
        help="Model to export"
    )
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["float16", "int8"],
        help="Precision mode"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/{model}/tensorrt/{precision})"
    )

    args = parser.parse_args()

    # Determine models to export
    if args.model == "all":
        models_to_export = list(VISION_CONFIGS.keys())
    else:
        models_to_export = [args.model]

    # Export each model
    all_metadata = []
    for model_name in models_to_export:
        config = VISION_CONFIGS[model_name]

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(__file__).parent.parent / "models" / model_name / "tensorrt" / args.precision

        # Export
        metadata = export_via_onnx(
            model_name=model_name,
            config=config,
            output_dir=output_dir,
            precision=args.precision,
        )
        all_metadata.append(metadata)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 EXPORT SUMMARY")
    logger.info("="*70)
    logger.info(f"Models: {len(all_metadata)}")
    logger.info(f"Precision: {args.precision.upper()}")

    for meta in all_metadata:
        logger.info(f"\n{meta['model_name']}:")
        logger.info(f"  Task: {meta['task']}")
        logger.info(f"  Script: {meta['export_script']}")
        logger.info(f"  Engine: {meta['engine_path']}")

    logger.info("\n" + "="*70)
    logger.info("✅ Export scripts generated")
    logger.info("="*70)
    logger.info("\n💡 Next Steps:")
    logger.info("1. Install requirements: pip install tensorrt onnx ultralytics")
    logger.info("2. Run export scripts: bash <script_path>")
    logger.info("3. Test with TensorRT backend")
    logger.info("")


if __name__ == "__main__":
    main()
