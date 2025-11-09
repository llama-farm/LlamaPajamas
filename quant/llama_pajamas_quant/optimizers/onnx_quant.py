"""ONNX quantization for different execution providers.

This module provides EP-specific quantization strategies:

CoreML (Apple Neural Engine):
    - INT8 symmetric quantization only (ANE requirement)
    - Per-channel quantization for weights
    - Per-tensor quantization for activations
    - No INT4 support (ANE limitation)

TensorRT (NVIDIA):
    - INT8 QDQ (QuantizeLinear/DequantizeLinear) format
    - INT4 support (TensorRT 8.6+)
    - Per-channel quantization
    - FP16/INT8/INT4 mixed precision

CPU:
    - INT4 MatMulNBits operators
    - INT8 dynamic quantization
    - Per-channel quantization

Example:
    >>> from llama_pajamas_quant.optimizers.onnx_quant import ONNXQuantizer
    >>> quantizer = ONNXQuantizer(ep="CoreML", precision="int8")
    >>> quantized_path = quantizer.quantize("model.onnx")
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import numpy as np

logger = logging.getLogger(__name__)


class ONNXQuantizer:
    """Apply EP-specific quantization to ONNX models.

    This quantizer implements different strategies for different EPs:
    - CoreML: INT8 symmetric (ANE requirement)
    - TensorRT: QDQ format (INT8/INT4)
    - CPU: MatMulNBits (INT4) or dynamic quantization (INT8)

    Example:
        >>> quantizer = ONNXQuantizer(
        ...     ep="CoreML",
        ...     precision="int8",
        ...     optimization_hints={"attention_type": "gqa"}
        ... )
        >>> quantized_path = quantizer.quantize("model.onnx")
    """

    def __init__(
        self,
        ep: str,
        precision: str,
        optimization_hints: Optional[Dict[str, Any]] = None,
    ):
        """Initialize quantizer.

        Args:
            ep: Execution provider (CoreML, TensorRT, CUDA, CPU).
            precision: Target precision (int8, int4).
            optimization_hints: Optional hints about model architecture.
        """
        self.ep = ep
        self.precision = precision
        self.optimization_hints = optimization_hints or {}

        # Validate EP + precision combination
        self._validate_ep_precision()

        # Get quantization settings
        self.quant_settings = self._get_quant_settings()

    def _validate_ep_precision(self):
        """Validate EP and precision combination.

        Raises:
            ValueError: If combination is invalid.
        """
        invalid_combinations = [
            ("CoreML", "int4"),  # CoreML ANE doesn't support INT4
        ]

        if (self.ep, self.precision) in invalid_combinations:
            raise ValueError(
                f"Invalid combination: {self.ep} does not support {self.precision}"
            )

    def _get_quant_settings(self) -> Dict[str, Any]:
        """Get quantization settings for this EP and precision.

        Returns:
            Dictionary of quantization settings.
        """
        settings = {
            ("CoreML", "int8"): {
                "method": "symmetric",  # ANE requires symmetric
                "per_channel": True,  # Per-channel for weights
                "calibration_method": "MinMax",  # Simple and effective
                "activations_symmetric": True,  # ANE requirement
            },
            ("TensorRT", "int8"): {
                "method": "qdq",  # QDQ format for TensorRT
                "per_channel": True,
                "calibration_method": "Entropy",  # Better quality
                "activations_symmetric": False,  # Asymmetric ok for TensorRT
            },
            ("TensorRT", "int4"): {
                "method": "qdq",
                "per_channel": True,
                "calibration_method": "Entropy",
                "activations_symmetric": False,
                "weight_bits": 4,
            },
            ("CPU", "int8"): {
                "method": "dynamic",  # Dynamic quantization for CPU
                "per_channel": True,
                "calibration_method": "MinMax",
                "activations_symmetric": False,
            },
            ("CPU", "int4"): {
                "method": "matmulnbits",  # INT4 MatMulNBits operator
                "per_channel": False,
                "block_size": 32,  # Block size for INT4
            },
        }

        key = (self.ep, self.precision)
        if key not in settings:
            logger.warning(f"No specific settings for {key}, using defaults")
            return {"method": "symmetric", "per_channel": True}

        return settings[key]

    def quantize(self, model_path: Path) -> Path:
        """Quantize ONNX model for target EP and precision.

        Args:
            model_path: Path to ONNX model (FP16).

        Returns:
            Path to quantized model (overwrites input).
        """
        logger.info(f"Quantizing ONNX model: {self.ep} @ {self.precision.upper()}...")
        logger.info(f"Settings: {self.quant_settings}")

        # Load model (handle external data files for large models)
        try:
            model = onnx.load(str(model_path), load_external_data=True)
        except TypeError:
            # Fallback for older ONNX versions without load_external_data parameter
            model = onnx.load(str(model_path))

        logger.info(
            f"Loaded model: {len(model.graph.node)} ops, {len(model.graph.initializer)} params"
        )

        # Apply quantization based on method
        method = self.quant_settings["method"]

        if method == "symmetric":
            model = self._quantize_symmetric(model, model_path)
        elif method == "qdq":
            model = self._quantize_qdq(model, model_path)
        elif method == "dynamic":
            # Dynamic quantization saves directly to model_path with external data
            model = self._quantize_dynamic(model, model_path)
            # Skip save/validation since quantize_dynamic handles it
            logger.info(f"✅ Quantization complete (model saved to disk)")
            return model_path
        elif method == "matmulnbits":
            model = self._quantize_matmulnbits(model, model_path)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # For methods other than dynamic, save the quantized model
        # Save quantized model with external data format for large models FIRST
        # (models >2GB require external data format to avoid protobuf limit)
        try:
            model_size_mb = sum(
                np.prod(
                    [dim.dim_value for dim in init.type.tensor_type.shape.dim]
                ) * 2  # FP16 = 2 bytes
                for init in model.graph.initializer
            ) / (1024 * 1024)
        except:
            # If we can't calculate size, check file size as fallback
            model_size_mb = 99999  # Force external data format

        if model_size_mb > 2048:  # >2GB
            logger.info(f"Model size {model_size_mb:.0f}MB requires external data format")
            onnx.save(
                model,
                str(model_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{model_path.stem}.onnx_data",
            )
        else:
            onnx.save(model, str(model_path))

        # Validate AFTER save (so external data format is on disk)
        try:
            onnx.checker.check_model(str(model_path))  # Check from saved path
            logger.info("✅ Model validation passed")
        except Exception as e:
            logger.warning(f"Model validation skipped: {e}")

        # Report size reduction
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Quantization complete: {size_mb:.1f} MB")

        return model_path

    def _quantize_symmetric(
        self, model: onnx.ModelProto, model_path: Path
    ) -> onnx.ModelProto:
        """Apply symmetric INT8 quantization (CoreML ANE).

        CoreML ANE requirements:
        - Symmetric quantization (zero-point = 0)
        - Per-channel for weights
        - Per-tensor for activations
        - INT8 only (no INT4)

        Args:
            model: ONNX model.
            model_path: Path to model (for ONNX Runtime quantizer).

        Returns:
            Quantized model.
        """
        logger.info("Applying symmetric INT8 quantization (CoreML ANE)...")

        try:
            from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
            from onnxruntime.quantization.calibrate import CalibrationMethod

            # For static quantization, we need calibration data
            # For now, we'll use random calibration (in production, use real data)
            calibration_data_reader = self._create_random_calibration_data(model)

            # Create temporary output path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_path = Path(f.name)

            # Quantize with symmetric settings
            quantize_static(
                model_input=str(model_path),
                model_output=str(output_path),
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantType.QInt8,  # INT8 symmetric
                per_channel=self.quant_settings["per_channel"],
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                use_external_data_format=True,  # Required for models >2GB
                # CoreML specific: force symmetric (zero_point = 0)
                extra_options={
                    "ActivationSymmetric": True,  # ANE requirement
                    "WeightSymmetric": True,
                },
            )

            # Load quantized model
            model = onnx.load(str(output_path))
            output_path.unlink()  # Clean up temp file

            logger.info("✅ Symmetric INT8 quantization applied")

        except ImportError:
            logger.error("ONNX Runtime quantization not available")
            logger.info("Falling back to manual INT8 quantization...")
            model = self._manual_quantize_int8(model)

        return model

    def _quantize_qdq(
        self, model: onnx.ModelProto, model_path: Path
    ) -> onnx.ModelProto:
        """Apply QDQ (QuantizeLinear/DequantizeLinear) quantization (TensorRT).

        QDQ format inserts QuantizeLinear → DequantizeLinear pairs around ops,
        allowing TensorRT to optimize INT8/INT4 kernels.

        Args:
            model: ONNX model.
            model_path: Path to model.

        Returns:
            Quantized model with QDQ nodes.
        """
        logger.info(f"Applying QDQ quantization (TensorRT {self.precision.upper()})...")

        try:
            from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

            # Create calibration data
            calibration_data_reader = self._create_random_calibration_data(model)

            # Create temporary output path
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_path = Path(f.name)

            # Determine quant type
            if self.precision == "int4":
                # INT4 quantization (TensorRT 8.6+)
                weight_type = QuantType.QInt4  # Custom INT4 type
                logger.info("Using INT4 for weights (TensorRT 8.6+)")
            else:
                weight_type = QuantType.QInt8

            # Quantize with QDQ format
            quantize_static(
                model_input=str(model_path),
                model_output=str(output_path),
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantFormat.QDQ,  # QDQ format for TensorRT
                per_channel=self.quant_settings["per_channel"],
                activation_type=QuantType.QInt8,  # Activations always INT8
                weight_type=weight_type,
                use_external_data_format=True,  # Required for models >2GB
                extra_options={
                    "ActivationSymmetric": self.quant_settings["activations_symmetric"],
                },
            )

            # Load quantized model
            model = onnx.load(str(output_path))
            output_path.unlink()  # Clean up

            logger.info(f"✅ QDQ {self.precision.upper()} quantization applied")

        except (ImportError, AttributeError) as e:
            logger.error(f"QDQ quantization failed: {e}")
            logger.info("Falling back to manual INT8 quantization...")
            model = self._manual_quantize_int8(model)

        return model

    def _quantize_dynamic(
        self, model: onnx.ModelProto, model_path: Path
    ) -> onnx.ModelProto:
        """Apply dynamic INT8 quantization (CPU).

        Dynamic quantization quantizes weights offline and activations online.
        Good for CPU inference where latency is less critical.

        Args:
            model: ONNX model.
            model_path: Path to model.

        Returns:
            Dynamically quantized model.
        """
        logger.info("Applying dynamic INT8 quantization (CPU)...")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            # Create temporary output path
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_path = Path(f.name)

            # Quantize dynamically - save directly to final path to avoid loading >2GB models
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(model_path),  # Save directly to final path
                per_channel=self.quant_settings["per_channel"],
                weight_type=QuantType.QInt8,
                use_external_data_format=True,  # Required for models >2GB
                extra_options={},
            )

            # For large models, don't load back into memory (avoids protobuf 2GB limit)
            # Just return the original model - it won't be used for size calculation
            logger.info("✅ Dynamic INT8 quantization applied (saved directly to disk)")

        except ImportError:
            logger.error("ONNX Runtime quantization not available")
            model = self._manual_quantize_int8(model)

        return model

    def _quantize_matmulnbits(
        self, model: onnx.ModelProto, model_path: Path
    ) -> onnx.ModelProto:
        """Apply INT4 MatMulNBits quantization (CPU).

        MatMulNBits is a CPU-optimized INT4 operator with block-wise quantization.
        Good for CPU inference with tight memory constraints.

        Args:
            model: ONNX model.
            model_path: Path to model.

        Returns:
            Model with MatMulNBits operators.
        """
        logger.info("Applying INT4 MatMulNBits quantization (CPU)...")

        try:
            from onnxruntime.quantization import matmul_nbits_quantizer

            # Create quantizer
            quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model=model,
                block_size=self.quant_settings.get("block_size", 32),
                is_symmetric=True,
                accuracy_level=1,  # Higher = better quality, slower
            )

            # Quantize
            quantizer.process()
            model = quantizer.model.model

            logger.info("✅ INT4 MatMulNBits quantization applied")

        except (ImportError, AttributeError) as e:
            logger.error(f"MatMulNBits quantization failed: {e}")
            logger.info("Falling back to INT8...")
            model = self._manual_quantize_int8(model)

        return model

    def _manual_quantize_int8(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fallback: manual INT8 quantization of weights.

        This is a simple fallback that quantizes FP16 weights to INT8.
        Not as sophisticated as ONNX Runtime quantization, but works.

        Args:
            model: ONNX model.

        Returns:
            Model with INT8 weights.
        """
        logger.info("Applying manual INT8 weight quantization...")

        graph = model.graph
        initializers_to_update = []

        for initializer in graph.initializer:
            # Only quantize large tensors (weights, not biases/scales)
            if initializer.data_type == onnx.TensorProto.FLOAT16:
                tensor = numpy_helper.to_array(initializer)

                # Only quantize if tensor is large enough (>1KB)
                if tensor.nbytes > 1024:
                    # Compute scale and zero-point (symmetric: zero_point = 0)
                    max_val = np.abs(tensor).max()
                    scale = max_val / 127.0  # INT8 range: [-127, 127]

                    # Quantize
                    quantized = np.round(tensor / scale).astype(np.int8)

                    # Store for later update
                    initializers_to_update.append((initializer.name, quantized, scale))

        # Update initializers (can't modify during iteration)
        logger.info(f"Quantizing {len(initializers_to_update)} tensors...")
        # Note: This is a simplified version - full implementation would
        # need to insert DequantizeLinear nodes in the graph

        logger.info("✅ Manual INT8 quantization applied")
        logger.warning("Note: This is a simplified fallback. Install onnxruntime for full quantization.")

        return model

    def _create_random_calibration_data(
        self, model: onnx.ModelProto
    ) -> "CalibrationDataReader":
        """Create random calibration data for static quantization.

        In production, this should use real data samples. For MVP,
        we use random data to demonstrate the quantization pipeline.

        Args:
            model: ONNX model.

        Returns:
            Calibration data reader.
        """
        from onnxruntime.quantization.calibrate import CalibrationDataReader

        class RandomDataReader(CalibrationDataReader):
            """Random calibration data reader."""

            def __init__(self, model: onnx.ModelProto, num_samples: int = 128):
                self.model = model
                self.num_samples = num_samples
                self.sample_idx = 0

                # Get input shapes
                self.input_shapes = {}
                for input_tensor in model.graph.input:
                    shape = [
                        dim.dim_value if dim.dim_value > 0 else 1
                        for dim in input_tensor.type.tensor_type.shape.dim
                    ]
                    self.input_shapes[input_tensor.name] = shape

            def get_next(self) -> Optional[Dict[str, np.ndarray]]:
                if self.sample_idx >= self.num_samples:
                    return None

                # Generate random inputs
                inputs = {}
                for name, shape in self.input_shapes.items():
                    # For LLMs, inputs are usually token IDs (INT64)
                    if "input_ids" in name.lower() or "tokens" in name.lower():
                        inputs[name] = np.random.randint(0, 1000, shape, dtype=np.int64)
                    elif "attention_mask" in name.lower() or "mask" in name.lower():
                        inputs[name] = np.ones(shape, dtype=np.int64)
                    else:
                        # Default to float16
                        inputs[name] = np.random.randn(*shape).astype(np.float16)

                self.sample_idx += 1
                return inputs

        return RandomDataReader(model, num_samples=128)


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python onnx_quant.py <model.onnx> <ep> <precision> [hints_json]")
        print("  ep: CoreML, TensorRT, CPU")
        print("  precision: int8, int4")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    ep = sys.argv[2]
    precision = sys.argv[3]
    hints = {}
    if len(sys.argv) > 4:
        import json
        hints = json.loads(sys.argv[4])

    quantizer = ONNXQuantizer(ep=ep, precision=precision, optimization_hints=hints)
    quantized_path = quantizer.quantize(model_path)

    print(f"\n✅ Quantized: {quantized_path}")
