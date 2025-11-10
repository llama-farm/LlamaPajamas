#!/usr/bin/env python3
"""
Multi-modal server demonstration.

Starts a server with Vision (detection) + STT (Whisper) backends.

Usage:
    # Start server
    uv run python examples/multimodal_server_demo.py

    # In another terminal, test with curl:
    # Object detection
    curl -X POST http://localhost:8000/v1/images/detect \
      -H "Content-Type: application/json" \
      -d '{"image": "data:image/jpeg;base64,...", "confidence_threshold": 0.5}'

    # STT transcription
    curl -X POST http://localhost:8000/v1/audio/transcriptions \
      -F file=@audio.wav \
      -F model=whisper-tiny

    # Health check
    curl http://localhost:8000/health

    # List models
    curl http://localhost:8000/v1/models
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_run_coreml.server import start_server

logging.basicConfig(level=logging.INFO)


def main():
    """Start multi-modal server with Vision + STT."""

    # Paths to models (adjust as needed)
    base_dir = Path(__file__).parent.parent.parent

    # Vision: YOLO-v8n detection model (INT8)
    vision_model = base_dir / "quant/models/yolo-v8n/coreml/fp16/model.mlpackage"

    # STT: Whisper-tiny encoder (INT8)
    stt_model = base_dir / "quant/models/whisper-tiny/coreml/int8/encoder.mlpackage"

    print("\n" + "="*70)
    print("🚀 Multi-Modal Server Demo")
    print("="*70)
    print("\nStarting server with:")
    print(f"  Vision: {vision_model.name} (detection)")
    print(f"  STT:    {stt_model.name}")
    print("\nEndpoints:")
    print("  POST /v1/images/detect         - Object detection")
    print("  POST /v1/audio/transcriptions  - Speech-to-text")
    print("  POST /v1/audio/speech          - Text-to-speech")
    print("  GET  /v1/models                - List models")
    print("  GET  /health                   - Health check")
    print("\n" + "="*70 + "\n")

    # Check if models exist
    if not vision_model.exists():
        print(f"⚠️  Vision model not found: {vision_model}")
        print("   Export with: cd quant && uv run python scripts/export_coreml_quantized.py")
        vision_model = None
        vision_type = None
    else:
        vision_type = "detection"

    if not stt_model.exists():
        print(f"⚠️  STT model not found: {stt_model}")
        print("   Export with: cd quant && uv run python scripts/export_whisper_coreml.py")
        stt_model = None

    if not vision_model and not stt_model:
        print("\n❌ No models found! Exiting.")
        sys.exit(1)

    # Start server
    start_server(
        host="0.0.0.0",
        port=8000,
        vision_model=str(vision_model) if vision_model else None,
        vision_type=vision_type,
        stt_model=str(stt_model) if stt_model else None,
        model_name="tiny",  # Whisper model name for decoder
        enable_tts=True,    # Enable system TTS
        tts_voice="Samantha",  # Default voice
    )


if __name__ == "__main__":
    main()
