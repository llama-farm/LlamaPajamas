"""CoreML multi-modal API server.

Starts a FastAPI server with vision and speech endpoints.
"""

import logging
from pathlib import Path
from typing import Optional

import uvicorn
from llama_pajamas_run_core.server_multimodal import create_multimodal_app

from .backends import CoreMLVisionBackend, CoreMLSTTBackend, SystemTTSBackend

logger = logging.getLogger(__name__)


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    vision_model: Optional[str] = None,
    vision_type: Optional[str] = None,
    stt_model: Optional[str] = None,
    tts_model: Optional[str] = None,
    **kwargs,
):
    """Start CoreML multi-modal API server.

    Args:
        host: Server host (default: 0.0.0.0)
        port: Server port (default: 8000)
        vision_model: Path to CoreML vision model (.mlpackage)
        vision_type: Type of vision model (detection, classification, embedding)
        stt_model: Path to CoreML STT model (.mlpackage)
        tts_model: Path to CoreML TTS model (.mlpackage)
        **kwargs: Additional server options
    """
    # Initialize backends
    vision_backend = None
    stt_backend = None
    tts_backend = None

    if vision_model and vision_type:
        logger.info(f"Loading vision model: {vision_model} ({vision_type})")
        vision_backend = CoreMLVisionBackend()
        vision_backend.load_model(vision_model, vision_type, **kwargs)
        logger.info("✅ Vision backend loaded")

    if stt_model:
        logger.info(f"Loading STT model: {stt_model}")
        stt_backend = CoreMLSTTBackend()
        stt_backend.load_model(stt_model, **kwargs)
        logger.info("✅ STT backend loaded")

    if tts_model or kwargs.get("enable_tts"):
        logger.info("Loading TTS backend (System TTS)")
        tts_backend = SystemTTSBackend()
        tts_backend.load_model(voice=kwargs.get("tts_voice", "Samantha"))
        logger.info("✅ TTS backend loaded")

    if not any([vision_backend, stt_backend, tts_backend]):
        logger.warning("No models loaded! Starting server with no backends.")

    # Create FastAPI app
    app = create_multimodal_app(
        llm_backend=None,  # No LLM backend for CoreML (use MLX for that)
        vision_backend=vision_backend,
        stt_backend=stt_backend,
        tts_backend=tts_backend,
    )

    # Start server
    logger.info(f"Starting CoreML multi-modal server on {host}:{port}")
    logger.info("Available endpoints:")
    if vision_backend:
        logger.info("  POST /v1/images/detect      - Object detection")
        logger.info("  POST /v1/images/classify    - Image classification")
        logger.info("  POST /v1/images/embed       - Image embeddings")
    if stt_backend:
        logger.info("  POST /v1/audio/transcriptions - Speech-to-text (OpenAI-compatible)")
    if tts_backend:
        logger.info("  POST /v1/audio/speech        - Text-to-speech (OpenAI-compatible)")
    logger.info("  GET  /v1/models              - List loaded models")
    logger.info("  GET  /health                 - Health check")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m llama_pajamas_run_coreml.server --vision-model <path> --vision-type <type>")
        print("       python -m llama_pajamas_run_coreml.server --stt-model <path>")
        print("       python -m llama_pajamas_run_coreml.server --tts-model <path>")
        sys.exit(1)

    # Simple arg parsing
    kwargs = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--vision-model":
            kwargs["vision_model"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--vision-type":
            kwargs["vision_type"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--stt-model":
            kwargs["stt_model"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--tts-model":
            kwargs["tts_model"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--port":
            kwargs["port"] = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    start_server(**kwargs)
