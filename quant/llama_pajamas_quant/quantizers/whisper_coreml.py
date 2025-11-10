"""CoreML Whisper (speech-to-text) model quantization."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def quantize_whisper_coreml(
    model_name: str,
    precision: str,
    output_dir: Path,
    models_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Quantize Whisper encoder to CoreML with specified precision.

    Args:
        model_name: Model name (e.g., 'whisper-tiny', 'whisper-base', 'whisper-small')
        precision: Precision ('int8', 'fp16')
        output_dir: Output directory
        models_base_dir: Base models directory (default: ./models)

    Returns:
        Dict with quantization results
    """
    if models_base_dir is None:
        models_base_dir = Path("./models")

    model_dir = models_base_dir / model_name
    fp16_model_path = model_dir / "coreml" / "float16" / "encoder.mlpackage"

    if not fp16_model_path.exists():
        raise FileNotFoundError(
            f"FP16 model not found at {fp16_model_path}. "
            f"Please export model first: uv run python scripts/export_whisper_coreml.py --model {model_name}"
        )

    # Import CoreML tools
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools not installed. Run: uv pip install coremltools>=7.0")

    logger.info(f"Quantizing {model_name} to {precision.upper()}")
    logger.info(f"Source: {fp16_model_path}")

    # Load model
    model = ct.models.MLModel(str(fp16_model_path))

    # Quantize
    if precision == "int8":
        # Use modern optimization API for Whisper
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            weight_threshold=512
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        quantized_model = ct.optimize.coreml.linear_quantize_weights(model, config=config)
    elif precision == "fp16":
        # Already FP16
        quantized_model = model
    else:
        raise ValueError(f"Unsupported precision for Whisper: {precision} (use 'int8' or 'fp16')")

    # Save quantized model
    output_path = output_dir / "encoder.mlpackage"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized_model.save(str(output_path))

    # Get sizes
    original_size = sum(f.stat().st_size for f in fp16_model_path.rglob('*') if f.is_file())
    quantized_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())

    reduction = (1 - quantized_size / original_size) * 100

    logger.info(f"✅ Quantized model saved: {output_path}")
    logger.info(f"   Original: {original_size / 1024 / 1024:.1f} MB")
    logger.info(f"   Quantized: {quantized_size / 1024 / 1024:.1f} MB")
    logger.info(f"   Reduction: {reduction:.1f}%")

    return {
        "model_name": model_name,
        "precision": precision,
        "output_path": str(output_path),
        "original_size_mb": original_size / 1024 / 1024,
        "quantized_size_mb": quantized_size / 1024 / 1024,
        "reduction_percent": reduction,
    }
