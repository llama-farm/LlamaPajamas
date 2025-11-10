#!/usr/bin/env python3
"""
Generic ONNX export script for speech models.

Supports:
- Whisper (OpenAI)
- Wav2Vec (HuggingFace)
- Custom PyTorch models

Usage:
    # Whisper export
    uv run python export_onnx_speech.py --model whisper-tiny --output models/whisper-tiny/onnx/

    # Wav2Vec export
    uv run python export_onnx_speech.py --model facebook/wav2vec2-base-960h \\
        --type huggingface --output models/wav2vec/onnx/

TODO: Unify with export_onnx_vision.py into single export_model.py script:
    ./quant/scripts/export_model.py --model whisper-tiny --backend onnx --precision fp16
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import onnx
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_whisper_to_onnx(
    model_name: str, output_dir: Path, opset: int = 13
) -> Dict[str, Any]:
    """Export Whisper model to ONNX.

    Args:
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        output_dir: Output directory
        opset: ONNX opset version

    Returns:
        Export metadata
    """
    logger.info(f"Exporting Whisper model: {model_name}")

    # Load model
    model = whisper.load_model(model_name)
    model.eval()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export encoder
    encoder_path = output_dir / f"{model_name}_encoder.onnx"
    encoder_size = export_whisper_encoder(model, encoder_path, opset)

    # Export decoder (optional - complex due to autoregressive nature)
    # For now, we'll use Python decoder with ONNX encoder
    logger.info("Decoder export skipped (using Python decoder with ONNX encoder)")

    total_size_mb = encoder_size / (1024 * 1024)

    logger.info(f"✅ Whisper ONNX exported: {output_dir} ({total_size_mb:.1f} MB)")

    return {
        "model_name": model_name,
        "encoder_path": str(encoder_path),
        "encoder_size_bytes": encoder_size,
        "opset_version": opset,
    }


def export_whisper_encoder(
    model: whisper.Whisper, output_path: Path, opset: int = 13
) -> int:
    """Export Whisper encoder to ONNX.

    Args:
        model: Whisper model
        output_path: Output path for encoder.onnx
        opset: ONNX opset version

    Returns:
        File size in bytes
    """
    logger.info("Exporting Whisper encoder...")

    # Create dummy input (mel spectrogram: [batch, n_mels, seq_len])
    # Whisper uses 80 mel bins, 3000 frames = 30 seconds
    dummy_input = torch.randn(1, 80, 3000)

    # Export encoder
    torch.onnx.export(
        model.encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["mel_spectrogram"],
        output_names=["audio_features"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 2: "seq_len"},
            "audio_features": {0: "batch_size"},
        },
    )

    # Verify
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    logger.info(f"✅ Encoder exported: {output_path} ({size_mb:.1f} MB)")

    return size_bytes


def export_huggingface_speech_to_onnx(
    model_id: str, output_dir: Path, opset: int = 13
) -> Dict[str, Any]:
    """Export HuggingFace speech model to ONNX.

    Args:
        model_id: HuggingFace model ID (e.g., "facebook/wav2vec2-base-960h")
        output_dir: Output directory
        opset: ONNX opset version

    Returns:
        Export metadata
    """
    from transformers import AutoModel, AutoFeatureExtractor
    from transformers.onnx import export as onnx_export, FeaturesManager

    logger.info(f"Exporting HuggingFace speech model: {model_id}")

    # Load model and feature extractor
    model = AutoModel.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # Get model kind
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature="automatic-speech-recognition"
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
    }


def main():
    parser = argparse.ArgumentParser(description="Export speech models to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'whisper-tiny', 'facebook/wav2vec2-base-960h')",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["whisper", "huggingface"],
        default="whisper",
        help="Model type",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for ONNX model"
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Export based on type
    if args.type == "whisper":
        # Extract model size (e.g., "whisper-tiny" -> "tiny")
        model_size = args.model.replace("whisper-", "")
        metadata = export_whisper_to_onnx(model_size, output_dir, opset=args.opset)
    elif args.type == "huggingface":
        metadata = export_huggingface_speech_to_onnx(
            args.model, output_dir, opset=args.opset
        )
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    logger.info(f"\n📦 Export metadata: {metadata}")


if __name__ == "__main__":
    main()
