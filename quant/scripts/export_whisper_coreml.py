#!/usr/bin/env python3
"""
Export Whisper STT models to CoreML for Apple Silicon optimization.

Supports multiple Whisper variants:
- whisper-tiny: 39M params, ~75 MB
- whisper-base: 74M params, ~140 MB
- whisper-small: 244M params, ~460 MB

Usage:
    # Export all models to FP16
    uv run python scripts/export_whisper_coreml.py --model all --precision float16

    # Export specific model
    uv run python scripts/export_whisper_coreml.py --model whisper-tiny --precision float16

    # Export with INT8 quantization
    uv run python scripts/export_whisper_coreml.py --model whisper-base --precision int8
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import coremltools as ct
    import numpy as np
except ImportError:
    print("⚠️  Missing dependencies. Installing...")
    subprocess.run(["uv", "pip", "install", "torch", "coremltools", "numpy", "openai-whisper"], check=True)
    import torch
    import coremltools as ct
    import numpy as np


WHISPER_CONFIGS = {
    "whisper-tiny": {
        "hf_id": "openai/whisper-tiny",
        "whisper_name": "tiny",
        "params": "39M",
        "expected_size_mb": 75,
        "input_features": 80,  # Mel filterbank features
        "max_length": 448,
    },
    "whisper-base": {
        "hf_id": "openai/whisper-base",
        "whisper_name": "base",
        "params": "74M",
        "expected_size_mb": 140,
        "input_features": 80,
        "max_length": 448,
    },
    "whisper-small": {
        "hf_id": "openai/whisper-small",
        "whisper_name": "small",
        "params": "244M",
        "expected_size_mb": 460,
        "input_features": 80,
        "max_length": 448,
    },
}


def load_whisper_model(model_name: str):
    """Load Whisper model from OpenAI."""
    try:
        import whisper
    except ImportError:
        print("   Installing openai-whisper...")
        subprocess.run(["uv", "pip", "install", "openai-whisper"], check=True)
        import whisper

    config = WHISPER_CONFIGS[model_name]
    whisper_name = config["whisper_name"]

    print(f"📦 Loading Whisper model: {whisper_name}")
    print(f"   Parameters: {config['params']}")

    model = whisper.load_model(whisper_name)
    model.eval()

    print(f"   ✅ Loaded {model_name}")

    return model


def export_whisper_encoder_to_coreml(
    model_name: str,
    whisper_model,
    output_path: Path,
    precision: str = "float16"
) -> Dict[str, Any]:
    """
    Export Whisper encoder (audio → features) to CoreML.

    The encoder takes mel-spectrogram features and produces audio embeddings.
    """
    config = WHISPER_CONFIGS[model_name]

    print(f"\n🔧 Exporting Whisper Encoder: {model_name} ({precision.upper()})")

    # Get encoder
    encoder = whisper_model.encoder
    encoder.eval()

    # Create example input (mel-spectrogram features)
    # Shape: (batch=1, n_mels=80, n_frames=3000)
    example_input = torch.randn(1, config["input_features"], 3000)

    print("   Tracing encoder model...")
    traced_encoder = torch.jit.trace(encoder, example_input)

    # Configure compute precision
    compute_precision = ct.precision.FLOAT16 if precision in ["float16", "int8", "int4"] else ct.precision.FLOAT32

    print("   Converting to CoreML...")

    # Convert
    mlmodel = ct.convert(
        traced_encoder,
        inputs=[ct.TensorType(name="mel_features", shape=(1, config["input_features"], ct.RangeDim(1, 3000)))],
        outputs=[ct.TensorType(name="audio_features")],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Apply quantization if needed
    if precision == "int8":
        print(f"   Applying INT8 quantization...")
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            weight_threshold=512
        )
        config_obj = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config_obj)

    elif precision == "int4":
        print(f"   Applying INT4 palettization...")
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

    print(f"   ✅ Exported encoder: {size_mb:.1f} MB")

    return {
        "model_name": f"{model_name}-encoder",
        "precision": precision,
        "size_mb": round(size_mb, 1),
        "size_bytes": size_bytes,
        "output_path": str(output_path),
        "component": "encoder"
    }


def export_whisper_decoder_to_coreml(
    model_name: str,
    whisper_model,
    output_path: Path,
    precision: str = "float16"
) -> Dict[str, Any]:
    """
    Export Whisper decoder (features → text) to CoreML.

    The decoder takes audio features and generates text tokens.
    Note: This is more complex due to autoregressive nature.
    For now, we'll focus on the encoder which does most of the heavy lifting.
    """
    print(f"\n⚠️  Decoder export not yet implemented")
    print(f"   Whisper decoder is autoregressive and requires special handling")
    print(f"   For now, we'll use the encoder for feature extraction")
    print(f"   Full transcription will run in Python with the encoder on ANE")

    return None


def create_manifest(
    model_name: str,
    encoder_info: Dict[str, Any],
    decoder_info: Optional[Dict[str, Any]],
    models_dir: Path
) -> None:
    """Create manifest.json for STT model."""
    config = WHISPER_CONFIGS[model_name]

    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0",
        "model_id": config["hf_id"],
        "model_name": model_name,
        "generated_at": "2025-11-09T13:30:00Z",
        "llama_pajamas_version": "0.1.0",
        "task": "speech_to_text",
        "architecture": {
            "model_id": config["hf_id"],
            "model_type": "whisper",
            "family": "whisper",
            "task": "automatic_speech_recognition",
            "params_total": config["params"],
            "input_features": config["input_features"],
            "max_length": config["max_length"],
        },
        "formats": []
    }

    # Add encoder format
    if encoder_info:
        precision = encoder_info["precision"]
        manifest["formats"].append({
            "format": "coreml",
            "component": "encoder",
            "precision": precision,
            "file_size_bytes": encoder_info["size_bytes"],
            "file_size_mb": encoder_info["size_mb"],
            "compatible_backends": ["coreml"],
            "runtime_requirements": "llama-pajamas-run-coreml >= 0.1.0",
            "hardware_requirements": "Apple Silicon (M1, M2, M3, M4) or iOS 15+",
            "compute_units": "ALL (CPU + GPU + ANE)",
            "optimized_for_ane": True,
            "path": f"coreml/{precision}/encoder.mlpackage"
        })

    # Save manifest
    manifest_path = model_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📄 Created manifest: {manifest_path}")


def export_models(model_names: list[str], precision: str = "float16"):
    """Export specified Whisper models to CoreML."""

    if "all" in model_names:
        model_names = list(WHISPER_CONFIGS.keys())

    models_dir = Path(__file__).parent.parent / "models"
    all_results = []

    for model_name in model_names:
        if model_name not in WHISPER_CONFIGS:
            print(f"⚠️  Unknown model: {model_name}")
            continue

        print(f"\n{'='*70}")
        print(f"🚀 Exporting: {model_name} ({precision.upper()})")
        print(f"{'='*70}")

        try:
            # Load Whisper model
            whisper_model = load_whisper_model(model_name)

            # Export encoder
            encoder_path = models_dir / model_name / "coreml" / precision / "encoder.mlpackage"
            encoder_info = export_whisper_encoder_to_coreml(
                model_name,
                whisper_model,
                encoder_path,
                precision=precision
            )

            # Decoder export (not yet implemented)
            decoder_info = None

            # Create manifest
            create_manifest(model_name, encoder_info, decoder_info, models_dir)

            all_results.append(encoder_info)

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
        print(f"{'Model':<25} {'Component':<12} {'Size (MB)':<12} {'Status':<10}")
        print(f"{'-'*70}")

        for result in all_results:
            print(f"{result['model_name']:<25} {result['component']:<12} {result['size_mb']:<12.1f} ✅ Success")

        print(f"\n✅ Exported {len(all_results)} models to {precision.upper()}")
        print(f"\n💡 Next steps:")
        print(f"   1. Create evaluation audio dataset")
        print(f"   2. Implement STT evaluation pipeline")
        print(f"   3. Quantize models to INT8/INT4")
        print(f"   4. Benchmark on real audio samples")


def main():
    parser = argparse.ArgumentParser(description="Export Whisper STT models to CoreML")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=list(WHISPER_CONFIGS.keys()) + ["all"],
        help="Model to export (can specify multiple times, or use 'all')"
    )
    parser.add_argument(
        "--precision",
        default="float16",
        choices=["float32", "float16", "int8", "int4"],
        help="Target precision (default: float16)"
    )

    args = parser.parse_args()

    if not args.model:
        args.model = ["whisper-tiny"]  # Default to smallest model

    export_models(args.model, precision=args.precision)


if __name__ == "__main__":
    main()
