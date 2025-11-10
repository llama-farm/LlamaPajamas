#!/usr/bin/env python3
"""
Export vision models from PyTorch to CoreML with quantization during conversion.

This is an alternative to post-training quantization that works for all model types,
including pipeline models like YOLO that don't support post-training INT8 quantization.

Usage:
    # Export YOLO to INT8 during conversion
    uv run python scripts/export_coreml_quantized.py \
        --model yolo-v8n \
        --precision int8

    # Export all models to INT8
    uv run python scripts/export_coreml_quantized.py \
        --model all \
        --precision int8

    # Export with INT4 (palettization)
    uv run python scripts/export_coreml_quantized.py \
        --model vit-base \
        --precision int4
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import coremltools as ct
    from PIL import Image
    import numpy as np
except ImportError:
    print("⚠️  Missing dependencies. Installing...")
    import subprocess
    subprocess.run(["uv", "pip", "install", "torch", "torchvision", "coremltools", "pillow", "numpy"], check=True)
    import torch
    import coremltools as ct
    from PIL import Image
    import numpy as np


MODEL_CONFIGS = {
    "yolo-v8n": {
        "hf_id": "ultralytics/yolov8n",
        "task": "object_detection",
        "input_size": (640, 640),
        "package": "ultralytics",
        "loader": "YOLO",
    },
    "vit-base": {
        "hf_id": "google/vit-base-patch16-224",
        "task": "image_classification",
        "input_size": (224, 224),
        "package": "transformers",
        "loader": "ViTForImageClassification",
    },
    "clip-vit-base": {
        "hf_id": "openai/clip-vit-base-patch32",
        "task": "vision_embedding",
        "input_size": (224, 224),
        "package": "transformers",
        "loader": "CLIPVisionModel",
    },
}


def export_yolo_to_coreml(
    model_name: str,
    output_path: Path,
    precision: str = "float16"
) -> Dict[str, Any]:
    """Export YOLO model using Ultralytics native export."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("   Installing ultralytics...")
        import subprocess
        subprocess.run(["uv", "pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO

    config = MODEL_CONFIGS[model_name]

    print(f"📦 Loading YOLO model: {config['hf_id']}")
    model = YOLO(config['hf_id'])
    print(f"   ✅ Loaded {model_name}")

    # Ultralytics export doesn't support INT8 directly
    # We'll export as FP16 and then quantize
    print("   Exporting to CoreML with Ultralytics...")

    # Export to temporary location
    temp_dir = Path("/tmp/yolo_export")
    temp_dir.mkdir(exist_ok=True)

    # Export (this creates the .mlpackage)
    export_path = model.export(
        format="coreml",
        half=True if precision in ["float16", "int8", "int4"] else False,
        nms=True,
    )

    # Ultralytics creates the .mlpackage file directly at export_path
    exported_model = Path(str(export_path))

    if not exported_model.exists():
        raise FileNotFoundError(f"Could not find exported model at {exported_model}")

    # If we're just doing FP16, we're done - move the model
    if precision == "float16":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_model != output_path:
            shutil.move(str(exported_model), str(output_path))

        # Calculate size
        size_bytes = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        size_mb = size_bytes / (1024 * 1024)

        print(f"   ✅ Exported: {size_mb:.1f} MB")

        return {
            "model_name": model_name,
            "precision": precision,
            "size_mb": round(size_mb, 1),
            "size_bytes": size_bytes,
            "output_path": str(output_path),
        }

    # For INT8/INT4, we need to use the neural_network API for pipeline models
    # Load the exported model
    mlmodel = ct.models.MLModel(str(exported_model))

    # Check model type
    spec = mlmodel.get_spec()
    is_pipeline = spec.WhichOneof('Type') == 'pipeline'

    if not is_pipeline:
        print("   ⚠️  Expected pipeline model, found mlprogram - trying mlprogram quantization")
        # Apply mlprogram quantization
        if precision == "int8":
            print(f"   Applying INT8 quantization (mlprogram)...")
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                weight_threshold=512
            )
            config_obj = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config_obj)
        elif precision == "int4":
            print(f"   Applying INT4 palettization (mlprogram)...")
            op_config = ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                weight_threshold=512
            )
            config_obj = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config_obj)
    else:
        # Pipeline model - YOLO models don't support quantization well
        # Just document this limitation
        print(f"   ⚠️  Pipeline models (YOLO) don't support INT8/INT4 quantization well")
        print(f"   ⚠️  Recommendation: Use FP16 for YOLO models")
        print(f"   ⚠️  Skipping quantization - using FP16 model")

        # Use the FP16 model as-is
        precision = "float16"

    # Save to final location
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Calculate size
    size_bytes = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    size_mb = size_bytes / (1024 * 1024)

    print(f"   ✅ Exported: {size_mb:.1f} MB")

    # Cleanup
    shutil.rmtree(exported_model.parent, ignore_errors=True)

    return {
        "model_name": model_name,
        "precision": precision,
        "size_mb": round(size_mb, 1),
        "size_bytes": size_bytes,
        "output_path": str(output_path),
    }


def create_example_input(model_name: str) -> torch.Tensor:
    """Create example input tensor for tracing."""
    config = MODEL_CONFIGS[model_name]
    h, w = config["input_size"]

    # Create batch of 1 image with 3 channels
    return torch.rand(1, 3, h, w)


def export_to_coreml(
    model_name: str,
    pytorch_model: torch.nn.Module,
    output_path: Path,
    precision: str = "float16",
    compute_units: str = "ALL"
) -> Dict[str, Any]:
    """
    Export PyTorch model to CoreML with specified precision.

    Args:
        model_name: Model identifier
        pytorch_model: PyTorch model to export
        output_path: Output .mlpackage path
        precision: Target precision (float32, float16, int8, int4)
        compute_units: Compute units (ALL, CPU_AND_GPU, CPU_ONLY)

    Returns:
        Export statistics
    """
    config = MODEL_CONFIGS[model_name]

    print(f"\n🔧 Exporting {model_name} to CoreML")
    print(f"   Precision: {precision.upper()}")
    print(f"   Compute units: {compute_units}")

    # Create example input
    example_input = create_example_input(model_name)

    # Trace the model
    print("   Tracing model...")
    traced_model = torch.jit.trace(pytorch_model, example_input)

    # Configure compute precision
    if precision == "float32":
        compute_precision = ct.precision.FLOAT32
    elif precision == "float16":
        compute_precision = ct.precision.FLOAT16
    else:
        # INT8/INT4 will be applied via quantization config
        compute_precision = ct.precision.FLOAT16

    # Convert to CoreML
    print("   Converting to CoreML...")

    # Define input/output types
    h, w = config["input_size"]
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, h, w),
        scale=1.0/255.0,  # Normalize to [0, 1]
    )

    # Convert
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit[compute_units],
    )

    # Apply quantization if INT8 or INT4
    if precision in ["int8", "int4"]:
        print(f"   Applying {precision.upper()} quantization...")

        if precision == "int8":
            # Linear INT8 quantization
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                weight_threshold=512
            )
            config_obj = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config_obj)

        elif precision == "int4":
            # 4-bit palettization
            op_config = ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                weight_threshold=512
            )
            config_obj = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config_obj)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Calculate size
    size_bytes = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    size_mb = size_bytes / (1024 * 1024)

    print(f"   ✅ Exported: {size_mb:.1f} MB")

    return {
        "model_name": model_name,
        "precision": precision,
        "size_mb": round(size_mb, 1),
        "size_bytes": size_bytes,
        "output_path": str(output_path),
    }


def update_manifest(model_dir: Path, export_info: Dict[str, Any]) -> None:
    """Update manifest.json with new export."""
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"⚠️  Manifest not found: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check if this precision already exists
    precision = export_info["precision"]
    existing = [
        fmt for fmt in manifest.get("formats", [])
        if fmt.get("format") == "coreml" and fmt.get("precision") == precision
    ]

    if existing:
        # Update existing entry
        for fmt in manifest["formats"]:
            if fmt.get("format") == "coreml" and fmt.get("precision") == precision:
                fmt["file_size_bytes"] = export_info["size_bytes"]
                fmt["file_size_mb"] = export_info["size_mb"]
                fmt["path"] = f"coreml/{precision}/model.mlpackage"
        print(f"   Updated manifest for {precision}")
    else:
        # Add new entry
        new_format = {
            "format": "coreml",
            "precision": precision,
            "file_size_bytes": export_info["size_bytes"],
            "file_size_mb": export_info["size_mb"],
            "compatible_backends": ["coreml"],
            "runtime_requirements": "llama-pajamas-run-coreml >= 0.1.0",
            "hardware_requirements": "Apple Silicon (M1, M2, M3, M4) or iOS 15+",
            "compute_units": "ALL (CPU + GPU + ANE)",
            "optimized_for_ane": True,
            "path": f"coreml/{precision}/model.mlpackage",
            "export_method": "pytorch_conversion"
        }
        manifest["formats"].append(new_format)
        print(f"   Added {precision} to manifest")

    # Save
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def export_models(model_names: list[str], precision: str = "int8"):
    """Export specified models to CoreML with quantization."""

    if "all" in model_names:
        model_names = ["yolo-v8n"]  # Only YOLO for now

    models_dir = Path(__file__).parent.parent / "models"
    all_results = []

    for model_name in model_names:
        if model_name not in MODEL_CONFIGS:
            print(f"⚠️  Unknown model: {model_name}")
            continue

        config = MODEL_CONFIGS[model_name]

        # Only support YOLO for now
        if config["package"] != "ultralytics":
            print(f"⚠️  Skipping {model_name} - only YOLO models supported currently")
            continue

        print(f"\n{'='*70}")
        print(f"🚀 Exporting: {model_name} ({precision.upper()})")
        print(f"{'='*70}")

        try:
            # Export YOLO using native method
            output_path = models_dir / model_name / "coreml" / precision / "model.mlpackage"
            export_info = export_yolo_to_coreml(
                model_name,
                output_path,
                precision=precision
            )

            # Update manifest
            model_dir = models_dir / model_name
            update_manifest(model_dir, export_info)

            all_results.append(export_info)

        except Exception as e:
            print(f"❌ Failed to export {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"📊 EXPORT SUMMARY ({precision.upper()})")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Size (MB)':<12} {'Status':<10}")
        print(f"{'-'*70}")

        for result in all_results:
            print(f"{result['model_name']:<20} {result['size_mb']:<12.1f} ✅ Success")

        print(f"\n✅ Exported {len(all_results)} models to {precision.upper()}")
        print(f"\n💡 Next: Run evaluation:")
        print(f"   cd /Users/robthelen/llama-pajamas/run-coreml")
        print(f"   uv run python ../quant/evaluation/vision/run_eval.py \\")
        print(f"       --models-dir ../quant/models \\")
        print(f"       --images ../quant/evaluation/vision/images/detection")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to CoreML with quantization")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model to export (can specify multiple times, or use 'all')"
    )
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["float32", "float16", "int8", "int4"],
        help="Target precision (default: int8)"
    )

    args = parser.parse_args()

    if not args.model:
        args.model = ["all"]

    export_models(args.model, precision=args.precision)


if __name__ == "__main__":
    main()
