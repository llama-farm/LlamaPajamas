#!/usr/bin/env python3
"""
Generic ONNX quantization script.

Supports:
- INT8 dynamic/static quantization
- INT4 quantization (where supported)
- Vision and speech models

Usage:
    # INT8 dynamic quantization (no calibration needed)
    uv run python quantize_onnx.py --model models/yolo-v8n/onnx/yolov8n.onnx \\
        --output models/yolo-v8n/onnx/yolov8n_int8.onnx \\
        --precision int8

    # INT8 static quantization (with calibration data)
    uv run python quantize_onnx.py --model models/yolo-v8n/onnx/yolov8n.onnx \\
        --output models/yolo-v8n/onnx/yolov8n_int8_static.onnx \\
        --precision int8 --mode static --calibration-data calibration/

    # INT4 quantization
    uv run python quantize_onnx.py --model models/whisper-tiny/onnx/encoder.onnx \\
        --output models/whisper-tiny/onnx/encoder_int4.onnx \\
        --precision int4

TODO: Unify with CoreML/TensorRT quantization into single quantize_model.py:
    ./quant/scripts/quantize_model.py --model yolo-v8n --backend onnx --precision int8
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    CalibrationDataReader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyCalibrationDataReader(CalibrationDataReader):
    """Dummy calibration data reader for testing."""

    def __init__(self, input_name: str, input_shape: tuple, num_samples: int = 100):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current_sample = 0

    def get_next(self) -> Optional[Dict[str, Any]]:
        """Get next calibration sample."""
        if self.current_sample >= self.num_samples:
            return None

        import numpy as np

        # Generate random sample
        sample = {self.input_name: np.random.randn(*self.input_shape).astype(np.float32)}
        self.current_sample += 1
        return sample

    def rewind(self):
        """Rewind to start."""
        self.current_sample = 0


def quantize_onnx_dynamic(
    model_path: Path, output_path: Path, weight_type: QuantType = QuantType.QInt8
) -> Dict[str, Any]:
    """Apply dynamic quantization to ONNX model.

    Dynamic quantization quantizes weights ahead of time but activations dynamically.
    No calibration data needed.

    Args:
        model_path: Input ONNX model path
        output_path: Output quantized model path
        weight_type: Weight quantization type (QInt8 or QUInt8)

    Returns:
        Quantization metadata
    """
    logger.info(f"Applying dynamic quantization to {model_path}")

    # Apply quantization
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=weight_type,
    )

    # Get sizes
    original_size = model_path.stat().st_size
    quantized_size = output_path.stat().st_size
    reduction = (1 - quantized_size / original_size) * 100

    original_mb = original_size / (1024 * 1024)
    quantized_mb = quantized_size / (1024 * 1024)

    logger.info(f"✅ Dynamic quantization complete:")
    logger.info(f"   Original: {original_mb:.1f} MB")
    logger.info(f"   Quantized: {quantized_mb:.1f} MB")
    logger.info(f"   Reduction: {reduction:.1f}%")

    return {
        "method": "dynamic",
        "weight_type": str(weight_type),
        "original_size_bytes": original_size,
        "quantized_size_bytes": quantized_size,
        "size_reduction_percent": reduction,
    }


def quantize_onnx_static(
    model_path: Path,
    output_path: Path,
    calibration_reader: CalibrationDataReader,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
) -> Dict[str, Any]:
    """Apply static quantization to ONNX model.

    Static quantization quantizes both weights and activations using calibration data.
    Provides better performance than dynamic but requires calibration.

    Args:
        model_path: Input ONNX model path
        output_path: Output quantized model path
        calibration_reader: Calibration data reader
        weight_type: Weight quantization type
        activation_type: Activation quantization type

    Returns:
        Quantization metadata
    """
    logger.info(f"Applying static quantization to {model_path}")

    # Apply quantization
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        weight_type=weight_type,
        activation_type=activation_type,
    )

    # Get sizes
    original_size = model_path.stat().st_size
    quantized_size = output_path.stat().st_size
    reduction = (1 - quantized_size / original_size) * 100

    original_mb = original_size / (1024 * 1024)
    quantized_mb = quantized_size / (1024 * 1024)

    logger.info(f"✅ Static quantization complete:")
    logger.info(f"   Original: {original_mb:.1f} MB")
    logger.info(f"   Quantized: {quantized_mb:.1f} MB")
    logger.info(f"   Reduction: {reduction:.1f}%")

    return {
        "method": "static",
        "weight_type": str(weight_type),
        "activation_type": str(activation_type),
        "original_size_bytes": original_size,
        "quantized_size_bytes": quantized_size,
        "size_reduction_percent": reduction,
    }


def quantize_onnx_int4(model_path: Path, output_path: Path) -> Dict[str, Any]:
    """Apply INT4 quantization to ONNX model.

    Note: INT4 support is limited in ONNX Runtime. May require custom operators.

    Args:
        model_path: Input ONNX model path
        output_path: Output quantized model path

    Returns:
        Quantization metadata
    """
    logger.warning("INT4 quantization has limited support in ONNX Runtime")
    logger.info(f"Attempting INT4 quantization on {model_path}")

    # For now, fall back to INT8 dynamic
    # TODO: Implement proper INT4 quantization when ONNX Runtime supports it
    logger.warning("Falling back to INT8 dynamic quantization")
    return quantize_onnx_dynamic(model_path, output_path, weight_type=QuantType.QInt8)


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models")
    parser.add_argument(
        "--model", type=str, required=True, help="Input ONNX model path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output quantized ONNX model path"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["int8", "int4"],
        default="int8",
        help="Quantization precision",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode (dynamic=no calibration, static=with calibration)",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to calibration data directory (for static mode)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples (dummy data)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply quantization
    if args.precision == "int4":
        metadata = quantize_onnx_int4(model_path, output_path)
    elif args.mode == "dynamic":
        metadata = quantize_onnx_dynamic(model_path, output_path)
    elif args.mode == "static":
        # Load model to get input shape
        model = onnx.load(str(model_path))
        input_name = model.graph.input[0].name
        input_shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in model.graph.input[0].type.tensor_type.shape.dim
        ]

        # Create calibration reader
        if args.calibration_data:
            logger.warning("Custom calibration data not yet implemented")
            logger.info("Using dummy calibration data")

        calibration_reader = DummyCalibrationDataReader(
            input_name=input_name,
            input_shape=tuple(input_shape),
            num_samples=args.num_calibration_samples,
        )

        metadata = quantize_onnx_static(model_path, output_path, calibration_reader)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info(f"\n📦 Quantization metadata: {metadata}")


if __name__ == "__main__":
    main()
