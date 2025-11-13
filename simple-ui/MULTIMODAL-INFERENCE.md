# Multimodal Inference - Vision & Speech

## ✅ Implementation Complete

The Simple UI now has full support for vision and speech inference using the existing multimodal server infrastructure from `run-coreml/`.

### 🚀 Server Types

**Multimodal Server** (Port 8000)
- Vision: Object detection (YOLO), classification (ViT), embeddings (CLIP)
- Speech-to-Text: Whisper models with ANE acceleration
- Text-to-Speech: System TTS

**Endpoints:**
- `POST /v1/images/detect` - Object detection
- `POST /v1/images/classify` - Image classification
- `POST /v1/audio/transcriptions` - Speech-to-text (OpenAI-compatible)
- `POST /v1/audio/speech` - Text-to-speech
- `GET /v1/models` - List loaded models
- `GET /health` - Health check

### 📋 How to Use

#### 1. Start Multimodal Server

**Option A: Via UI**
1. Go to **Server** tab
2. Select "Multimodal" server type
3. Enter vision model path (optional): `./quant/models/yolo-v8n/coreml/fp16/model.mlpackage`
4. Enter STT model path (optional): `./quant/models/whisper-tiny/coreml/int8/encoder.mlpackage`
5. Click "Start Optimized Server"
6. Server runs on port 8000

**Option B: Via CLI**
```bash
cd run-coreml
uv run python examples/multimodal_server_demo.py
```

#### 2. Use Inference Modes

**Chat Mode (LLM)**
- Uses `llama_pajamas_run` Python API
- Streaming text generation
- GGUF or MLX backend

**Image Mode (Vision)**
- Connects to multimodal server `/v1/images/detect`
- Upload image for object detection
- Returns bounding boxes with labels and confidence scores

**Voice Mode (Speech)**
- Connects to multimodal server `/v1/audio/transcriptions`
- Upload audio file for transcription
- OpenAI-compatible Whisper API

### 🎯 Usage Examples

**Vision Detection:**
```bash
# Upload image via UI Inference tab (Image mode)
# Or via curl:
curl -X POST http://localhost:8000/v1/images/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "confidence_threshold": 0.5}'
```

**Speech-to-Text:**
```bash
# Upload audio via UI Inference tab (Voice mode)
# Or via curl:
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-tiny \
  -F response_format=json
```

### 📊 Features

**Vision Inference:**
- Object detection with YOLO
- Image classification with ViT
- Bounding box visualization
- Confidence scores
- Multiple object detection

**Speech Inference:**
- Whisper model transcription
- OpenAI-compatible API
- Multiple audio formats (WAV, FLAC, MP3)
- Language detection
- Timestamp support

### 🧪 Testing

**Test Multimodal Server:**
```bash
# Start server
cd run-coreml
uv run python examples/multimodal_server_demo.py

# Test in another terminal
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

**Test from UI:**
1. Go to **Server** tab
2. Start Multimodal server
3. Go to **Inference** tab
4. Switch to **Image** mode
5. Upload an image
6. See detection results with bounding boxes!

### 📁 Implementation Files

**Server:**
- `/app/api/server/start-multimodal/route.ts` - NEW multimodal server API
- `/components/ServerPanel.tsx` - Added multimodal option

**Inference:**
- `/components/InferencePanel.tsx` - Image & voice modes enabled
- Image upload with preview
- Audio file upload
- Results visualization

### ✅ Status

- ✅ Multimodal server integration
- ✅ Vision inference (object detection)
- ✅ Speech inference (transcription)
- ✅ Image upload UI
- ✅ Audio upload UI
- ✅ Results visualization
- ✅ Uses existing `run-coreml/` infrastructure

**All modes now fully functional!** 🚀

###  Backend Support

| Feature | Backend | Status |
|---------|---------|--------|
| LLM Chat | GGUF, MLX | ✅ Working |
| Vision Detection | CoreML (YOLO) | ✅ Working |
| Vision Classification | CoreML (ViT) | ✅ Working |
| Speech-to-Text | CoreML (Whisper) | ✅ Working |
| Text-to-Speech | System TTS | ✅ Working |
| ONNX Vision | ONNX Runtime | 🔜 Coming Soon |
| TensorRT Vision | TensorRT | 🔜 Coming Soon |
