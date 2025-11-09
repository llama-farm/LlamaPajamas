# Llama-Pajamas CoreML Runtime

**Apple Silicon Multi-Modal Runtime** for vision, speech, and LLMs.

## Features

- **Vision**: Object detection (YOLO), classification (ViT), embeddings (CLIP)
- **Speech**: STT (Whisper), TTS (FastSpeech2)
- **Optimized**: Apple Neural Engine (ANE) acceleration on M1/M2/M3 and iOS

## Status

🚧 **Week 1-2: Infrastructure Setup (In Progress)**

- ✅ Base classes defined (VisionBackend, STTBackend, TTSBackend)
- ✅ Package structure created
- ✅ Backend stubs implemented
- ⏳ Converters (Week 3-6)
- ⏳ Full implementation (Week 3-8)

## Installation

```bash
# Development install
cd run-coreml
uv sync

# Or with pip
pip install -e .
```

## Dependencies

- `coremltools>=7.0` - CoreML conversion and runtime
- `Pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Array operations
- `librosa>=0.10.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O

## Planned Usage

### Vision

```bash
# Object detection
llama-pajamas-coreml detect \
  --model models/yolo-v8n.mlpackage \
  --image photo.jpg \
  --confidence 0.5

# Image classification
llama-pajamas-coreml classify \
  --model models/vit-base.mlpackage \
  --image photo.jpg \
  --top-k 5

# Image embeddings
llama-pajamas-coreml embed \
  --model models/clip-vit-b32.mlpackage \
  --image photo.jpg
```

### Speech

```bash
# Speech-to-text
llama-pajamas-coreml transcribe \
  --model models/whisper-base.mlpackage \
  --audio recording.wav \
  --language en

# Text-to-speech
llama-pajamas-coreml synthesize \
  --model models/fastspeech2.mlpackage \
  --text "Hello world" \
  --output speech.wav
```

## Development Roadmap

### Week 1-2: Infrastructure ✅ (Current)
- [x] Create package structure
- [x] Define base classes
- [x] Implement backend stubs
- [x] Set up dependencies

### Week 3-4: Vision Implementation
- [ ] Implement YOLO detection pipeline
- [ ] Implement CLIP embeddings
- [ ] Implement ViT classification
- [ ] PyTorch → CoreML converters
- [ ] ANE optimization (FP16, NHWC)
- [ ] Target: 30+ FPS @ 640x640 on M3 Max

### Week 5-6: Speech Implementation
- [ ] Implement Whisper STT
- [ ] Implement FastSpeech2 TTS
- [ ] Streaming audio support
- [ ] Mel spectrogram on device
- [ ] Target: <100ms STT latency

### Week 7: Multi-Modal Integration
- [ ] CLIP + MLX LLM pipeline
- [ ] Zero-copy CoreML → MLX
- [ ] Multi-modal examples

### Week 8: Testing + Release
- [ ] End-to-end tests
- [ ] Performance benchmarks
- [ ] Documentation
- [ ] v0.2.0 release

## Architecture

```
run-coreml/
├── llama_pajamas_run_coreml/
│   ├── backends/
│   │   ├── vision.py           # CoreMLVisionBackend
│   │   ├── speech_stt.py       # CoreMLSTTBackend
│   │   └── speech_tts.py       # CoreMLTTSBackend
│   ├── converters/             # PyTorch → CoreML (Week 3-6)
│   │   ├── vision.py           # YOLO, CLIP, ViT
│   │   └── speech.py           # Whisper, FastSpeech2
│   ├── __main__.py             # CLI
│   └── __init__.py
└── pyproject.toml
```

## License

Same as llama-pajamas main project.
