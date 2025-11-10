#!/usr/bin/env python3
"""
Generic ONNX export script for vision models.

Supports:
- YOLO (Ultralytics)
- ViT, ResNet, CLIP (HuggingFace/PyTorch)
- Custom PyTorch models

Usage:
    # YOLO export
    uv run python export_onnx_vision.py --model yolo-v8n --output models/yolo-v8n/onnx/

    # HuggingFace model
    uv run python export_onnx_vision.py --model google/vit-base-patch16-224 \\
        --type huggingface --output models/vit-base/onnx/

    # Custom PyTorch model
    uv run python export_onnx_vision.py --model path/to/model.pt \\
        --type pytorch --output models/custom/onnx/
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_yolo_to_onnx(
    model_name: str, output_dir: Path, simplify: bool = True, opset: int = 13
) -> Dict[str, Any]:
    """Export YOLO model to ONNX.

    Args:
        model_name: YOLO model name (e.g., "yolov8n", "yolov8s")
        output_dir: Output directory
        simplify: Simplify ONNX model
        opset: ONNX opset version

    Returns:
        Export metadata
    """
    from ultralytics import YOLO

    logger.info(f"Exporting YOLO model: {model_name}")

    # Load model
    model = YOLO(f"{model_name}.pt")

    # Export to ONNX
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    model.export(format="onnx", simplify=simplify, opset=opset)

    # Move to output directory
    default_onnx = Path(f"{model_name}.onnx")
    if default_onnx.exists():
        default_onnx.rename(onnx_path)

    # Get file size
    size_bytes = onnx_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    logger.info(f"✅ ONNX model exported: {onnx_path} ({size_mb:.1f} MB)")

    return {
        "model_name": model_name,
        "onnx_path": str(onnx_path),
        "size_bytes": size_bytes,
        "opset_version": opset,
        "simplified": simplify,
    }


def export_huggingface_to_onnx(
    model_id: str,
    output_dir: Path,
    task: str = "image-classification",
    opset: int = 13,
) -> Dict[str, Any]:
    """Export HuggingFace model to ONNX.

    Args:
        model_id: HuggingFace model ID (e.g., "google/vit-base-patch16-224")
        output_dir: Output directory
        task: Model task
        opset: ONNX opset version

    Returns:
        Export metadata
    """
    from transformers import AutoModel, AutoFeatureExtractor
    from transformers.onnx import export as onnx_export, FeaturesManager

    logger.info(f"Exporting HuggingFace model: {model_id}")

    # Load model and feature extractor
    model = AutoModel.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # Get model kind
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature=task
    )

    # Create ONNX config
    onnx_config = model_onnx_config(model.config)

    # Export
    onnx_export(
        preprocessor=feature_extractor,
        model=model,
        config=onnx_config,
        opset=opset,
        output=onnx_path,
    )

    # Get file size
    size_bytes = onnx_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    logger.info(f"✅ ONNX model exported: {onnx_path} ({size_mb:.1f} MB)")

    return {
        "model_id": model_id,
        "onnx_path": str(onnx_path),
        "size_bytes": size_bytes,
        "opset_version": opset,
        "task": task,
    }


def export_pytorch_to_onnx(
    model_path: str,
    output_dir: Path,
    input_shape: tuple = (1, 3, 224, 224),
    opset: int = 13,
) -> Dict[str, Any]:
    """Export PyTorch model to ONNX.

    Args:
        model_path: Path to PyTorch model (.pt, .pth)
        output_dir: Output directory
        input_shape: Input tensor shape
        opset: ONNX opset version

    Returns:
        Export metadata
    """
    logger.info(f"Exporting PyTorch model: {model_path}")

    # Load model
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Verify
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Get file size
    size_bytes = onnx_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    logger.info(f"✅ ONNX model exported: {onnx_path} ({size_mb:.1f} MB)")

    return {
        "model_path": model_path,
        "onnx_path": str(onnx_path),
        "size_bytes": size_bytes,
        "opset_version": opset,
        "input_shape": input_shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Export vision models to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'yolov8n', 'google/vit-base-patch16-224', 'model.pt')",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["yolo", "huggingface", "pytorch"],
        default="yolo",
        help="Model type",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--task", type=str, default="image-classification", help="Task (HuggingFace)"
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,224,224",
        help="Input shape for PyTorch (comma-separated)",
    )
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX (YOLO)")

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Export based on type
    if args.type == "yolo":
        metadata = export_yolo_to_onnx(
            args.model, output_dir, simplify=args.simplify, opset=args.opset
        )
    elif args.type == "huggingface":
        metadata = export_huggingface_to_onnx(
            args.model, output_dir, task=args.task, opset=args.opset
        )
    elif args.type == "pytorch":
        input_shape = tuple(map(int, args.input_shape.split(",")))
        metadata = export_pytorch_to_onnx(
            args.model, output_dir, input_shape=input_shape, opset=args.opset
        )
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    logger.info(f"\n📦 Export metadata: {metadata}")


if __name__ == "__main__":
    main()
