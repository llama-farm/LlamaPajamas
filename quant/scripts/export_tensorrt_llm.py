#!/usr/bin/env python3
"""
Export LLM models to TensorRT for NVIDIA GPU inference.

Converts HuggingFace models → ONNX → TensorRT with quantization.

Usage:
    # Export with INT8 quantization
    uv run python scripts/export_tensorrt_llm.py --model Qwen/Qwen3-8B --dtype int8

    # Export with FP16
    uv run python scripts/export_tensorrt_llm.py --model Qwen/Qwen3-8B --dtype float16

    # Export all models
    uv run python scripts/export_tensorrt_llm.py --model all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model configurations
LLM_CONFIGS = {
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "params": "8B",
        "expected_size_mb": 15000,
        "max_batch_size": 8,
        "max_input_len": 2048,
        "max_output_len": 512,
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "params": "4B",
        "expected_size_mb": 8000,
        "max_batch_size": 16,
        "max_input_len": 2048,
        "max_output_len": 512,
    },
}


def export_to_tensorrt(
    model_name: str,
    hf_model_id: str,
    output_dir: Path,
    dtype: str = "float16",
    **kwargs,
) -> Dict[str, Any]:
    """Export HuggingFace model to TensorRT engine.

    Args:
        model_name: Model name (e.g., 'qwen3-8b')
        hf_model_id: HuggingFace model ID
        output_dir: Output directory for TensorRT engine
        dtype: Quantization dtype ('float16', 'int8', 'int4')
        **kwargs: Additional export parameters

    Returns:
        Dict with export metadata
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Exporting {model_name} to TensorRT ({dtype.upper()})")
    logger.info(f"{'='*70}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement TensorRT-LLM export
    # This requires:
    # 1. Load HuggingFace model
    # 2. Convert to TensorRT-LLM format
    # 3. Build engine with specified dtype
    # 4. Save engine and tokenizer

    # Placeholder implementation
    logger.warning("⚠️  TensorRT-LLM export not yet implemented")
    logger.info("   Follow NVIDIA TensorRT-LLM documentation:")
    logger.info("   https://github.com/NVIDIA/TensorRT-LLM")

    # Manual export command template
    engine_path = output_dir / f"{model_name}-{dtype}.engine"

    logger.info("\n📝 Manual Export Command:")
    logger.info(f"""
    # 1. Install TensorRT-LLM
    pip install tensorrt-llm

    # 2. Convert model
    python -m tensorrt_llm.commands.build \\
        --model_dir {hf_model_id} \\
        --output_dir {output_dir} \\
        --dtype {dtype} \\
        --max_batch_size {kwargs.get('max_batch_size', 8)} \\
        --max_input_len {kwargs.get('max_input_len', 2048)} \\
        --max_output_len {kwargs.get('max_output_len', 512)} \\
        --use_gpt_attention_plugin {dtype} \\
        --use_gemm_plugin {dtype}

    # 3. Output: {engine_path}
    """)

    metadata = {
        "model_name": model_name,
        "hf_model_id": hf_model_id,
        "dtype": dtype,
        "engine_path": str(engine_path),
        "max_batch_size": kwargs.get("max_batch_size", 8),
        "max_input_len": kwargs.get("max_input_len", 2048),
        "max_output_len": kwargs.get("max_output_len", 512),
        "status": "template_generated",
    }

    # Save metadata
    metadata_path = output_dir / "export_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Export template generated: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export LLMs to TensorRT")
    parser.add_argument(
        "--model",
        default="qwen3-8b",
        choices=list(LLM_CONFIGS.keys()) + ["all"],
        help="Model to export"
    )
    parser.add_argument(
        "--dtype",
        default="int8",
        choices=["float16", "int8", "int4"],
        help="Quantization dtype"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/{model_name}/tensorrt/{dtype})"
    )

    args = parser.parse_args()

    # Determine models to export
    if args.model == "all":
        models_to_export = list(LLM_CONFIGS.keys())
    else:
        models_to_export = [args.model]

    # Export each model
    for model_name in models_to_export:
        config = LLM_CONFIGS[model_name]

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(__file__).parent.parent / "models" / model_name / "tensorrt" / args.dtype

        # Export
        metadata = export_to_tensorrt(
            model_name=model_name,
            hf_model_id=config["hf_id"],
            output_dir=output_dir,
            dtype=args.dtype,
            max_batch_size=config["max_batch_size"],
            max_input_len=config["max_input_len"],
            max_output_len=config["max_output_len"],
        )

    logger.info("\n" + "="*70)
    logger.info("✅ Export templates generated for all models")
    logger.info("="*70)
    logger.info("\n💡 Next Steps:")
    logger.info("1. Install TensorRT-LLM: pip install tensorrt-llm")
    logger.info("2. Run the generated export commands")
    logger.info("3. Test inference with TensorRT backend")
    logger.info("")


if __name__ == "__main__":
    main()
