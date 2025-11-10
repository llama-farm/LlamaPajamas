"""Extended OpenAI-compatible API server with multi-modal support (Vision + Speech).

Adds to the standard LLM API:
- Vision: /v1/images/* endpoints (detection, classification, embeddings)
- Speech: /v1/audio/* endpoints (transcriptions, speech)
- Multi-modal: /v1/chat/completions with image support (like GPT-4V)
"""

import base64
import io
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from .backends import VisionBackend, STTBackend, TTSBackend


# ============================================================================
# Vision API Models (Custom - inspired by OpenAI but adapted for our needs)
# ============================================================================

class ImageDetectionRequest(BaseModel):
    """Request for object detection."""

    image: str = Field(..., description="Base64-encoded image or URL")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class ImageClassificationRequest(BaseModel):
    """Request for image classification."""

    image: str = Field(..., description="Base64-encoded image or URL")
    top_k: int = Field(default=5, ge=1, le=100)


class ImageEmbeddingRequest(BaseModel):
    """Request for image embeddings."""

    image: str = Field(..., description="Base64-encoded image or URL")


# ============================================================================
# Speech API Models (OpenAI-compatible)
# ============================================================================

class AudioTranscriptionRequest(BaseModel):
    """OpenAI-compatible transcription request.

    See: https://platform.openai.com/docs/api-reference/audio/createTranscription
    """

    model: str = Field(default="whisper-1", description="Model to use (e.g., 'whisper-1')")
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en', 'zh')")
    prompt: Optional[str] = Field(default=None, description="Optional text to guide transcription")
    response_format: str = Field(default="json", description="Format: json, text, srt, verbose_json, vtt")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class AudioSpeechRequest(BaseModel):
    """OpenAI-compatible TTS request.

    See: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    model: str = Field(default="tts-1", description="Model to use (e.g., 'tts-1', 'tts-1-hd')")
    input: str = Field(..., description="Text to synthesize (max 4096 chars)")
    voice: str = Field(default="alloy", description="Voice: alloy, echo, fable, onyx, nova, shimmer")
    response_format: str = Field(default="mp3", description="Format: mp3, opus, aac, flac, wav, pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed (0.25 to 4.0)")


# ============================================================================
# Helper Functions
# ============================================================================

def decode_image(image_data: str):
    """Decode base64 image or load from URL.

    Args:
        image_data: Base64 string or URL

    Returns:
        PIL.Image.Image or np.ndarray
    """
    from PIL import Image
    import numpy as np

    # Check if it's a base64 string
    if image_data.startswith("data:image"):
        # Format: data:image/jpeg;base64,<base64_data>
        header, base64_data = image_data.split(",", 1)
        image_bytes = base64.b64decode(base64_data)
    elif image_data.startswith("http://") or image_data.startswith("https://"):
        # URL - fetch image
        import requests
        response = requests.get(image_data)
        image_bytes = response.content
    else:
        # Assume raw base64
        image_bytes = base64.b64decode(image_data)

    # Load as PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    return image


def decode_audio(audio_bytes: bytes) -> tuple:
    """Decode audio file to numpy array + sample rate.

    Args:
        audio_bytes: Audio file bytes (wav, mp3, etc.)

    Returns:
        (audio_array, sample_rate)
    """
    import soundfile as sf
    import numpy as np

    # Load audio
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert to float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, sr


def encode_audio(audio_array, sample_rate: int, format: str = "wav") -> bytes:
    """Encode audio array to bytes.

    Args:
        audio_array: numpy array (float32, mono)
        sample_rate: Sample rate in Hz
        format: Output format (wav, mp3, etc.)

    Returns:
        Audio file bytes
    """
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format=format.upper())
    buffer.seek(0)
    return buffer.read()


# ============================================================================
# Extended Server Factory
# ============================================================================

def create_multimodal_app(
    llm_backend=None,
    vision_backend: Optional[VisionBackend] = None,
    stt_backend: Optional[STTBackend] = None,
    tts_backend: Optional[TTSBackend] = None,
) -> FastAPI:
    """Create FastAPI application with multi-modal support.

    Args:
        llm_backend: LLM backend (MLX, GGUF) - optional
        vision_backend: Vision backend (CoreML, TensorRT) - optional
        stt_backend: Speech-to-text backend - optional
        tts_backend: Text-to-speech backend - optional

    Returns:
        FastAPI application with multi-modal endpoints
    """
    app = FastAPI(
        title="Llama-Pajamas Multi-Modal Runtime",
        description="OpenAI-compatible API with vision and speech support",
        version="0.2.0",
    )

    # ========================================================================
    # Vision Endpoints
    # ========================================================================

    @app.post("/v1/images/detect")
    async def detect_objects(request: ImageDetectionRequest):
        """Detect objects in an image.

        Custom endpoint (not in OpenAI spec).
        """
        if vision_backend is None:
            raise HTTPException(status_code=501, detail="Vision backend not loaded")

        if not vision_backend.is_loaded:
            raise HTTPException(status_code=503, detail="Vision model not loaded")

        if vision_backend.model_type != "detection":
            raise HTTPException(
                status_code=400,
                detail=f"Model type is '{vision_backend.model_type}', not 'detection'"
            )

        # Decode image
        image = decode_image(request.image)

        # Run detection
        detections = vision_backend.detect(
            image,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
        )

        return {
            "object": "image.detection",
            "created": int(time.time()),
            "detections": [det.to_dict() for det in detections],
        }

    @app.post("/v1/images/classify")
    async def classify_image(request: ImageClassificationRequest):
        """Classify an image.

        Custom endpoint (not in OpenAI spec).
        """
        if vision_backend is None:
            raise HTTPException(status_code=501, detail="Vision backend not loaded")

        if not vision_backend.is_loaded:
            raise HTTPException(status_code=503, detail="Vision model not loaded")

        if vision_backend.model_type != "classification":
            raise HTTPException(
                status_code=400,
                detail=f"Model type is '{vision_backend.model_type}', not 'classification'"
            )

        # Decode image
        image = decode_image(request.image)

        # Run classification
        results = vision_backend.classify(image, top_k=request.top_k)

        return {
            "object": "image.classification",
            "created": int(time.time()),
            "predictions": [res.to_dict() for res in results],
        }

    @app.post("/v1/images/embed")
    async def embed_image(request: ImageEmbeddingRequest):
        """Generate image embeddings (CLIP-style).

        Custom endpoint (not in OpenAI spec).
        """
        if vision_backend is None:
            raise HTTPException(status_code=501, detail="Vision backend not loaded")

        if not vision_backend.is_loaded:
            raise HTTPException(status_code=503, detail="Vision model not loaded")

        if vision_backend.model_type != "embedding":
            raise HTTPException(
                status_code=400,
                detail=f"Model type is '{vision_backend.model_type}', not 'embedding'"
            )

        # Decode image
        image = decode_image(request.image)

        # Generate embedding
        embedding = vision_backend.embed(image)

        return {
            "object": "image.embedding",
            "created": int(time.time()),
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
        }

    # ========================================================================
    # Speech Endpoints (OpenAI-compatible)
    # ========================================================================

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(default="whisper-1"),
        language: Optional[str] = Form(default=None),
        prompt: Optional[str] = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
    ):
        """Transcribe audio to text (OpenAI-compatible).

        See: https://platform.openai.com/docs/api-reference/audio/createTranscription
        """
        if stt_backend is None:
            raise HTTPException(status_code=501, detail="STT backend not loaded")

        if not stt_backend.is_loaded:
            raise HTTPException(status_code=503, detail="STT model not loaded")

        # Read audio file
        audio_bytes = await file.read()
        audio, sr = decode_audio(audio_bytes)

        # Run transcription
        result = stt_backend.transcribe(audio, sample_rate=sr, language=language)

        # Format response based on response_format
        if response_format == "json":
            return {
                "text": result.text,
            }
        elif response_format == "verbose_json":
            return result.to_dict()
        elif response_format == "text":
            return Response(content=result.text, media_type="text/plain")
        elif response_format == "srt":
            # TODO: Convert segments to SRT format
            raise HTTPException(status_code=501, detail="SRT format not yet implemented")
        elif response_format == "vtt":
            # TODO: Convert segments to VTT format
            raise HTTPException(status_code=501, detail="VTT format not yet implemented")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid response_format: {response_format}")

    @app.post("/v1/audio/speech")
    async def create_speech(request: AudioSpeechRequest):
        """Generate speech from text (OpenAI-compatible).

        See: https://platform.openai.com/docs/api-reference/audio/createSpeech
        """
        if tts_backend is None:
            raise HTTPException(status_code=501, detail="TTS backend not loaded")

        if not tts_backend.is_loaded:
            raise HTTPException(status_code=503, detail="TTS model not loaded")

        if len(request.input) > 4096:
            raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")

        # Map OpenAI voice names to speaker IDs
        # TODO: Make this configurable based on model
        voice_map = {
            "alloy": 0,
            "echo": 1,
            "fable": 2,
            "onyx": 3,
            "nova": 4,
            "shimmer": 5,
        }
        speaker_id = voice_map.get(request.voice, 0)

        # Synthesize speech
        audio = tts_backend.synthesize(
            text=request.input,
            speaker_id=speaker_id,
            # TODO: Handle speed parameter
        )

        # Encode to requested format
        audio_bytes = encode_audio(audio, 22050, format=request.response_format)

        # Determine media type
        media_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        media_type = media_types.get(request.response_format, "audio/wav")

        return Response(content=audio_bytes, media_type=media_type)

    # ========================================================================
    # Health & Info Endpoints
    # ========================================================================

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        models = []

        if llm_backend is not None:
            models.append({
                "id": "llama-pajamas-llm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama-pajamas",
                "capabilities": ["chat", "completions"],
            })

        if vision_backend is not None and vision_backend.is_loaded:
            models.append({
                "id": f"llama-pajamas-vision-{vision_backend.model_type}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama-pajamas",
                "capabilities": ["vision", vision_backend.model_type],
            })

        if stt_backend is not None and stt_backend.is_loaded:
            models.append({
                "id": "llama-pajamas-stt",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama-pajamas",
                "capabilities": ["speech-to-text", "transcription"],
            })

        if tts_backend is not None and tts_backend.is_loaded:
            models.append({
                "id": "llama-pajamas-tts",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama-pajamas",
                "capabilities": ["text-to-speech", "synthesis"],
            })

        return {
            "object": "list",
            "data": models,
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        status = {
            "status": "ok",
            "backends": {
                "llm": llm_backend is not None,
                "vision": vision_backend is not None and vision_backend.is_loaded if vision_backend else False,
                "stt": stt_backend is not None and stt_backend.is_loaded if stt_backend else False,
                "tts": tts_backend is not None and tts_backend.is_loaded if tts_backend else False,
            },
        }
        return status

    return app
