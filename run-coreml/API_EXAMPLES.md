# CoreML Multi-Modal API Examples

OpenAI-compatible endpoints for vision and speech on Apple Silicon.

## Starting the Server

```bash
# Vision only (detection)
python -m llama_pajamas_run_coreml.server \
  --vision-model models/yolo-v8n.mlpackage \
  --vision-type detection \
  --port 8000

# STT only
python -m llama_pajamas_run_coreml.server \
  --stt-model models/whisper-base.mlpackage \
  --port 8000

# TTS only
python -m llama_pajamas_run_coreml.server \
  --tts-model models/fastspeech2.mlpackage \
  --port 8000

# All backends
python -m llama_pajamas_run_coreml.server \
  --vision-model models/yolo-v8n.mlpackage \
  --vision-type detection \
  --stt-model models/whisper-base.mlpackage \
  --tts-model models/fastspeech2.mlpackage \
  --port 8000
```

## Vision API

### Object Detection

```bash
# Detect objects in an image
curl -X POST http://localhost:8000/v1/images/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }'

# Response:
{
  "object": "image.detection",
  "created": 1699564800,
  "detections": [
    {
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95
    }
  ]
}
```

### Image Classification

```bash
curl -X POST http://localhost:8000/v1/images/classify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://example.com/cat.jpg",
    "top_k": 5
  }'

# Response:
{
  "object": "image.classification",
  "created": 1699564800,
  "predictions": [
    {"class_id": 281, "class_name": "tabby cat", "confidence": 0.89},
    {"class_id": 282, "class_name": "tiger cat", "confidence": 0.08}
  ]
}
```

### Image Embeddings (CLIP)

```bash
curl -X POST http://localhost:8000/v1/images/embed \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }'

# Response:
{
  "object": "image.embedding",
  "created": 1699564800,
  "embedding": [0.123, -0.456, ...],  // 512-D vector
  "dimension": 512
}
```

## Speech API (OpenAI-Compatible)

### Speech-to-Text (Whisper)

```bash
# Transcribe audio file
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@recording.wav" \
  -F "model=whisper-1" \
  -F "language=en" \
  -F "response_format=json"

# Response (simple JSON):
{
  "text": "Hello, this is a test recording."
}

# Response (verbose JSON):
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "response_format=verbose_json"

{
  "text": "Hello, this is a test recording.",
  "language": "en",
  "segments": [
    {
      "text": "Hello,",
      "start_time": 0.0,
      "end_time": 0.5,
      "confidence": 0.98
    },
    {
      "text": "this is a test recording.",
      "start_time": 0.5,
      "end_time": 2.1,
      "confidence": 0.95
    }
  ]
}
```

### Text-to-Speech

```bash
# Generate speech from text
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of text to speech synthesis.",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3

# Available voices: alloy, echo, fable, onyx, nova, shimmer
# Available formats: mp3, opus, aac, flac, wav, pcm
```

## Python Client Examples

### Vision

```python
import requests
import base64

# Load image
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Object detection
response = requests.post(
    "http://localhost:8000/v1/images/detect",
    json={
        "image": f"data:image/jpeg;base64,{image_b64}",
        "confidence_threshold": 0.5,
    }
)

detections = response.json()["detections"]
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

### Speech-to-Text

```python
import requests

# Transcribe audio
with open("recording.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": f},
        data={
            "model": "whisper-1",
            "language": "en",
            "response_format": "verbose_json",
        }
    )

result = response.json()
print(f"Transcription: {result['text']}")
for segment in result["segments"]:
    print(f"  [{segment['start_time']:.1f}s - {segment['end_time']:.1f}s] {segment['text']}")
```

### Text-to-Speech

```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello world, this is a test.",
        "voice": "alloy",
        "response_format": "mp3",
    }
)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

## Health & Models

### List Models

```bash
curl http://localhost:8000/v1/models

# Response:
{
  "object": "list",
  "data": [
    {
      "id": "llama-pajamas-vision-detection",
      "object": "model",
      "created": 1699564800,
      "owned_by": "llama-pajamas",
      "capabilities": ["vision", "detection"]
    },
    {
      "id": "llama-pajamas-stt",
      "object": "model",
      "created": 1699564800,
      "owned_by": "llama-pajamas",
      "capabilities": ["speech-to-text", "transcription"]
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:8000/health

# Response:
{
  "status": "ok",
  "backends": {
    "llm": false,
    "vision": true,
    "stt": true,
    "tts": false
  }
}
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid input (e.g., wrong model type, empty text)
- `501 Not Implemented` - Backend not loaded (e.g., vision endpoint but no vision model)
- `503 Service Unavailable` - Model not loaded

Example error response:
```json
{
  "detail": "Vision backend not loaded"
}
```
