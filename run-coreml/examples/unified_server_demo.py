#!/usr/bin/env python3
"""
Unified Multi-Modal Server Demo

Starts a complete server with ALL modalities:
- Vision (YOLO, ViT, CLIP) - CoreML + ANE
- STT (Whisper) - CoreML + ANE
- TTS (System voices) - Apple AVFoundation
- LLM (Qwen3) - MLX on Apple Silicon

This demonstrates the complete multi-modal AI platform running on-device.

Usage:
    # Start unified server
    uv run python examples/unified_server_demo.py

    # Server endpoints:
    - POST /v1/images/detect - Object detection
    - POST /v1/audio/transcriptions - Speech-to-text
    - POST /v1/audio/speech - Text-to-speech
    - POST /v1/chat/completions - LLM chat (future)
    - GET /health - Health check
    - GET /v1/models - List models
"""

import logging
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run-core"))

from llama_pajamas_run_coreml.server import start_server

logging.basicConfig(level=logging.INFO)


def main():
    """Start unified multi-modal server."""

    # Paths to models
    base_dir = Path(__file__).parent.parent.parent

    # Vision: YOLO-v8n detection (FP16)
    vision_model = base_dir / "quant/models/yolo-v8n/coreml/fp16/model.mlpackage"

    # STT: Whisper-tiny (INT8 for efficiency)
    stt_model = base_dir / "quant/models/whisper-tiny/coreml/int8/encoder.mlpackage"

    print("\n" + "="*70)
    print("🌟 Unified Multi-Modal Server")
    print("="*70)
    print("\nStarting server with ALL modalities:")
    print(f"  Vision: {vision_model.name} (detection)")
    print(f"  STT:    {stt_model.name}")
    print(f"  TTS:    System voices (Apple AVFoundation)")
    print(f"  LLM:    Ready for MLX integration")
    print("\nEndpoints:")
    print("  POST /v1/images/detect         - Object detection")
    print("  POST /v1/audio/transcriptions  - Speech-to-text")
    print("  POST /v1/audio/speech          - Text-to-speech")
    print("  POST /v1/chat/completions      - LLM chat (future)")
    print("  GET  /v1/models                - List models")
    print("  GET  /health                   - Health check")
    print("\n" + "="*70 + "\n")

    # Check if models exist
    if not vision_model.exists():
        print(f"⚠️  Vision model not found: {vision_model}")
        vision_model = None
        vision_type = None
    else:
        vision_type = "detection"

    if not stt_model.exists():
        print(f"⚠️  STT model not found: {stt_model}")
        stt_model = None

    # Start server with all available modalities
    start_server(
        host="0.0.0.0",
        port=8000,
        vision_model=str(vision_model) if vision_model else None,
        vision_type=vision_type,
        stt_model=str(stt_model) if stt_model else None,
        model_name="tiny",  # Whisper model name
        enable_tts=True,  # System TTS
        tts_voice="Albert",
    )


if __name__ == "__main__":
    main()
