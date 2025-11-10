# Runtime Architecture Plan: CoreML & TensorRT

Based on existing MLX/GGUF runtime pattern, extended for multi-modal support.

---

## Current Pattern (LLM-Only)

### Structure
```
run-core/                         # Shared infrastructure
├── llama_pajamas_run_core/
│   ├── backends/
│   │   ├── base.py              # Abstract Backend class
│   │   └── __init__.py
│   ├── server.py                # FastAPI server (OpenAI-compatible)
│   ├── manifest_loader.py       # Load manifest.json
│   ├── model_loader.py          # Auto-detect backend
│   └── config.py                # Runtime config

run-mlx/                          # MLX-specific (Apple Silicon LLMs)
├── llama_pajamas_run_mlx/
│   ├── backend.py               # MLXBackend(Backend)
│   ├── __main__.py              # CLI: llama-pajamas-mlx
│   └── __init__.py
└── pyproject.toml               # Deps: mlx, mlx-lm

run-gguf/                         # GGUF-specific (Universal LLMs)
├── llama_pajamas_run_gguf/
│   ├── backend.py               # GGUFBackend(Backend)
│   ├── __main__.py              # CLI: llama-pajamas-gguf
│   └── __init__.py
└── pyproject.toml               # Deps: llama-cpp-python
```

### Backend Interface (base.py)
```python
class Backend(ABC):
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load LLM model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, ...) -> str | Iterator[str]:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def chat_completion(self, messages: List[Dict], ...) -> Dict | Iterator[Dict]:
        """OpenAI-compatible chat completions."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Free resources."""
        pass
```

**Works well for**: LLM-only runtimes (MLX, GGUF)

**Problem for multi-modal**: Vision/Speech have different interfaces than LLMs

---

## Proposed Pattern (Multi-Modal)

### Key Insight
CoreML and TensorRT handle **multiple modalities**:
- **Vision**: Detection (YOLO), Classification (ViT), Embeddings (CLIP)
- **Speech**: STT (Whisper), TTS (FastSpeech2)
- **LLMs**: (Maybe) - CoreML for iOS apps, TensorRT-LLM for servers

**Solution**: Separate backends per modality, shared base classes

### Structure

```
run-core/                         # Shared infrastructure (EXTENDED)
├── llama_pajamas_run_core/
│   ├── backends/
│   │   ├── base.py              # LLMBackend(ABC) - existing
│   │   ├── vision_base.py       # VisionBackend(ABC) - NEW
│   │   ├── speech_base.py       # SpeechBackend(ABC) - NEW
│   │   └── __init__.py
│   ├── server.py                # FastAPI server (EXTENDED for multi-modal)
│   ├── manifest_loader.py
│   ├── model_loader.py          # EXTENDED: auto-detect modality
│   └── config.py

run-coreml/                       # CoreML-specific (Apple Silicon multi-modal)
├── llama_pajamas_run_coreml/
│   ├── backends/
│   │   ├── vision.py            # CoreMLVisionBackend(VisionBackend)
│   │   ├── speech_stt.py        # CoreMLSTTBackend(SpeechBackend)
│   │   ├── speech_tts.py        # CoreMLTTSBackend(SpeechBackend)
│   │   └── __init__.py
│   ├── converters/              # PyTorch/HF → CoreML conversion
│   │   ├── vision.py            # Convert YOLO, CLIP, ViT
│   │   ├── speech.py            # Convert Whisper, FastSpeech2
│   │   └── __init__.py
│   ├── __main__.py              # CLI: llama-pajamas-coreml
│   └── __init__.py
└── pyproject.toml               # Deps: coremltools, Pillow, librosa

run-tensorrt/                     # TensorRT-specific (NVIDIA multi-modal)
├── llama_pajamas_run_tensorrt/
│   ├── backends/
│   │   ├── vision.py            # TensorRTVisionBackend(VisionBackend)
│   │   ├── speech_stt.py        # TensorRTSTTBackend(SpeechBackend)
│   │   ├── speech_tts.py        # TensorRTTTSBackend(SpeechBackend)
│   │   └── __init__.py
│   ├── converters/              # PyTorch/ONNX → TensorRT engines
│   │   ├── vision.py            # Build YOLO, CLIP engines
│   │   ├── speech.py            # Build Whisper, FastSpeech2 engines
│   │   └── __init__.py
│   ├── __main__.py              # CLI: llama-pajamas-tensorrt
│   └── __init__.py
└── pyproject.toml               # Deps: tensorrt, pycuda, Pillow, librosa
```

### New Base Classes

#### VisionBackend (vision_base.py)
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from PIL import Image


class DetectionResult:
    """Object detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float


class VisionBackend(ABC):
    """Abstract base class for vision inference."""

    @abstractmethod
    def load_model(self, model_path: str, model_type: str, **kwargs) -> None:
        """Load vision model.

        Args:
            model_path: Path to model file (.mlpackage, .engine, etc)
            model_type: One of: 'detection', 'classification', 'embedding'
            **kwargs: Backend-specific parameters
        """
        pass

    @abstractmethod
    def detect(self, image: Image.Image | np.ndarray, **kwargs) -> List[DetectionResult]:
        """Object detection (YOLO, Detr).

        Args:
            image: PIL Image or numpy array (HWC, RGB)
            **kwargs: confidence_threshold, iou_threshold, etc.

        Returns:
            List of detection results
        """
        pass

    @abstractmethod
    def classify(self, image: Image.Image | np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Image classification (ViT, ResNet).

        Returns:
            List of {'class_id': int, 'class_name': str, 'confidence': float}
        """
        pass

    @abstractmethod
    def embed(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Generate image embeddings (CLIP).

        Returns:
            Embedding vector (e.g., 512-D for CLIP)
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass
```

#### SpeechBackend (speech_base.py)
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator
import numpy as np


class TranscriptionSegment:
    """Speech-to-text segment."""
    text: str
    start_time: float  # seconds
    end_time: float
    confidence: float


class SpeechBackend(ABC):
    """Abstract base class for speech inference."""

    @abstractmethod
    def load_model(self, model_path: str, model_type: str, **kwargs) -> None:
        """Load speech model.

        Args:
            model_path: Path to model file
            model_type: One of: 'stt', 'tts'
            **kwargs: Backend-specific parameters
        """
        pass


class STTBackend(SpeechBackend):
    """Speech-to-Text backend."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate in Hz
            language: Language code (e.g., 'en', 'zh')

        Returns:
            {'text': str, 'segments': List[TranscriptionSegment]}
        """
        pass

    @abstractmethod
    def transcribe_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int = 16000,
        **kwargs
    ) -> Iterator[str]:
        """Streaming transcription.

        Args:
            audio_stream: Iterator yielding audio chunks

        Returns:
            Iterator yielding partial transcriptions
        """
        pass


class TTSBackend(SpeechBackend):
    """Text-to-Speech backend."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text
            sample_rate: Output sample rate
            speaker_id: Speaker/voice ID

        Returns:
            Audio samples (float32, mono)
        """
        pass

    @abstractmethod
    def synthesize_streaming(
        self,
        text: str,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """Streaming TTS.

        Returns:
            Iterator yielding audio chunks
        """
        pass
```

---

## Implementation Plan

### Phase 1: CoreML Multi-Modal (8 weeks)

#### Week 1-2: Core Infrastructure
**Tasks**:
1. Extend `run-core` with new base classes:
   - Add `vision_base.py` with `VisionBackend` ABC
   - Add `speech_base.py` with `STTBackend` and `TTSBackend` ABCs
   - Extend `model_loader.py` to detect modality from manifest

2. Create `run-coreml` package structure:
   - Initialize package with proper dependencies
   - Add `backends/` directory
   - Add `converters/` directory

**Deliverables**:
- `run-core/llama_pajamas_run_core/backends/vision_base.py`
- `run-core/llama_pajamas_run_core/backends/speech_base.py`
- `run-coreml/` package skeleton

#### Week 3-4: CoreML Vision
**Tasks**:
1. Implement `CoreMLVisionBackend`:
   - Load `.mlpackage` models
   - Detection interface (YOLO-v8)
   - Classification interface (ViT)
   - Embedding interface (CLIP)
   - ANE optimization (FP16, NHWC layout)

2. Implement conversion pipeline:
   - PyTorch → CoreML converter for YOLO
   - PyTorch → CoreML converter for CLIP
   - PyTorch → CoreML converter for ViT
   - Automatic ANE optimization

3. Testing:
   - Convert YOLO-v8n from HuggingFace
   - Convert CLIP-ViT-B/32
   - Test on Mac M1 64GB
   - Benchmark FPS @ 640x640

**Deliverables**:
- `run-coreml/llama_pajamas_run_coreml/backends/vision.py`
- `run-coreml/llama_pajamas_run_coreml/converters/vision.py`
- Working YOLO-v8n detection @ 30+ FPS

#### Week 5-6: CoreML Speech
**Tasks**:
1. Implement `CoreMLSTTBackend`:
   - Load Whisper CoreML models
   - Transcribe audio chunks
   - Streaming transcription (10s chunks, 10ms hop)
   - Mel spectrogram on device

2. Implement `CoreMLTTSBackend`:
   - Load FastSpeech2 + HiFi-GAN
   - Synthesize speech from text
   - Streaming synthesis (50-100ms frames)
   - Audio playback integration

3. Implement conversion pipeline:
   - PyTorch Whisper → CoreML
   - PyTorch FastSpeech2 → CoreML
   - Optimize mel preprocessing

4. Testing:
   - Convert Whisper-base-en
   - Convert FastSpeech2
   - Test on Mac M1 64GB
   - Benchmark latency

**Deliverables**:
- `run-coreml/llama_pajamas_run_coreml/backends/speech_stt.py`
- `run-coreml/llama_pajamas_run_coreml/backends/speech_tts.py`
- `run-coreml/llama_pajamas_run_coreml/converters/speech.py`
- Working Whisper STT @ <100ms latency
- Working FastSpeech2 TTS @ RT factor 0.3

#### Week 7: Multi-Modal Integration
**Tasks**:
1. Implement multi-modal pipelines:
   - CLIP (CoreML) → embeddings → LLM (MLX)
   - Camera (Vision) → detections → LLM (MLX) → description
   - Audio (STT) → text → LLM (MLX) → response → TTS

2. Zero-copy buffer sharing:
   - CoreML → MLX via unified memory
   - Verify no CPU copies
   - Benchmark end-to-end latency

3. CLI integration:
   ```bash
   llama-pajamas-coreml detect --image photo.jpg --model yolo-v8n
   llama-pajamas-coreml transcribe --audio recording.wav --model whisper-base
   llama-pajamas-coreml synthesize --text "Hello world" --model fastspeech2
   ```

**Deliverables**:
- Multi-modal pipeline examples
- Zero-copy verification
- CLI commands working

#### Week 8: Testing + Documentation
**Tasks**:
1. End-to-end testing:
   - Vision: YOLO detection pipeline
   - Speech: Whisper transcription pipeline
   - Multi-modal: CLIP + LLM pipeline
   - Performance benchmarks

2. Documentation:
   - Conversion guide (PyTorch → CoreML)
   - API documentation
   - Multi-modal examples
   - iOS deployment guide

3. Release v0.2.0:
   - Package on PyPI
   - GitHub release
   - Documentation site

**Deliverables**:
- Comprehensive documentation
- v0.2.0 release
- iOS deployment guide

---

### Phase 2: TensorRT Multi-Modal (6 weeks)

#### Week 1-2: TensorRT Vision
**Tasks**:
1. Implement `TensorRTVisionBackend`:
   - Build TensorRT engines from ONNX
   - Detection (YOLO-v8)
   - Classification (ViT)
   - Embedding (CLIP)
   - FP16 + INT8 calibration
   - Engine caching

2. Implement conversion pipeline:
   - PyTorch/ONNX → TensorRT engine builder
   - INT8 calibration datasets
   - Engine optimization flags

3. Testing:
   - Build YOLO-v8n engine
   - Build CLIP engine
   - Test on RTX 4070
   - Benchmark FPS

**Deliverables**:
- `run-tensorrt/llama_pajamas_run_tensorrt/backends/vision.py`
- `run-tensorrt/llama_pajamas_run_tensorrt/converters/vision.py`
- Working YOLO-v8 @ 60+ FPS on RTX 4070

#### Week 3-4: TensorRT Speech
**Tasks**:
1. Implement `TensorRTSTTBackend` and `TensorRTTTSBackend`
2. Build engines for Whisper, FastSpeech2
3. CUDA graphs for low latency
4. Streaming audio pipelines

**Deliverables**:
- Speech backends working
- <50ms STT latency on RTX 4070

#### Week 5: TensorRT-LLM Evaluation
**Tasks**:
1. Set up benchmarking harness:
   - RTX 4070 (desktop)
   - Jetson Orin (edge)
   - A100 (server)

2. Benchmark GGUF+CUDA vs TensorRT-LLM:
   - Throughput (tokens/sec)
   - Latency (ms to first token)
   - Memory usage (VRAM)
   - Quality (perplexity)

3. **Decision**: Add TensorRT-LLM or stick with GGUF+CUDA
   - If >50% faster → implement TensorRT-LLM backend
   - If <30% faster → stick with GGUF+CUDA

**Deliverables**:
- Benchmark results
- Go/no-go decision on TensorRT-LLM

#### Week 6: Integration + Release
**Tasks**:
1. Multi-modal pipelines (TRT vision → GGUF/TRT-LLM)
2. DLPack zero-copy (CUDA pointers)
3. Jetson Orin testing
4. Release v0.3.0

**Deliverables**:
- v0.3.0 release
- TensorRT documentation
- Jetson deployment guide

---

## Package Dependencies

### run-core (EXTENDED)
```toml
[project]
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    # No vision/speech deps here - kept in specific runtimes
]
```

### run-coreml
```toml
[project]
dependencies = [
    "llama-pajamas-run-core",
    "coremltools>=7.0.0",      # CoreML conversion + runtime
    "Pillow>=10.0.0",          # Image processing
    "numpy>=1.24.0",           # Array operations
    "librosa>=0.10.0",         # Audio processing
    "soundfile>=0.12.0",       # Audio I/O
]
```

### run-tensorrt
```toml
[project]
dependencies = [
    "llama-pajamas-run-core",
    "tensorrt>=8.6.0",         # TensorRT (NVIDIA SDK)
    "pycuda>=2022.1",          # CUDA bindings
    "Pillow>=10.0.0",          # Image processing
    "numpy>=1.24.0",           # Array operations
    "librosa>=0.10.0",         # Audio processing
    "soundfile>=0.12.0",       # Audio I/O
    # Optional: TensorRT-LLM (if evaluation succeeds)
    # "tensorrt-llm>=0.7.0",
]
```

---

## CLI Usage Examples

### CoreML
```bash
# Vision detection
llama-pajamas-coreml detect \
  --model models/yolo-v8n.mlpackage \
  --image photo.jpg \
  --confidence 0.5

# Speech transcription
llama-pajamas-coreml transcribe \
  --model models/whisper-base.mlpackage \
  --audio recording.wav \
  --language en

# Text-to-speech
llama-pajamas-coreml synthesize \
  --model models/fastspeech2.mlpackage \
  --text "Hello world" \
  --output speech.wav

# Multi-modal: Vision → LLM
llama-pajamas-multi \
  --vision-model coreml:yolo-v8n \
  --llm-model mlx:qwen-8b-q4 \
  --image photo.jpg \
  --prompt "Describe what you see"
```

### TensorRT
```bash
# Vision detection (with engine caching)
llama-pajamas-tensorrt detect \
  --model models/yolo-v8n.onnx \
  --engine-cache ./engines/ \
  --image photo.jpg \
  --precision fp16

# Speech transcription
llama-pajamas-tensorrt transcribe \
  --model models/whisper-base.onnx \
  --engine-cache ./engines/ \
  --audio recording.wav

# Multi-modal: Vision → LLM (TensorRT + GGUF)
llama-pajamas-multi \
  --vision-model tensorrt:yolo-v8n \
  --llm-model gguf:qwen-8b-q4 \
  --image photo.jpg \
  --prompt "What do you see?"
```

---

## Migration from run-onnx

We already have ONNX backends code in `run-onnx/`. Should we:

### Option A: Abandon ONNX completely
- Delete `run-onnx/` code
- Start fresh with CoreML and TensorRT
- **Pro**: Clean slate, no ONNX baggage
- **Con**: Waste existing work

### Option B: Repurpose ONNX code
- Keep ONNX as fallback for unsupported platforms
- CoreML and TensorRT take priority
- **Pro**: Maintains portability option
- **Con**: More maintenance burden

### Option C: Extract useful patterns
- Use ONNX backend structure as template
- Rewrite for CoreML/TensorRT
- Delete ONNX code after extraction
- **Pro**: Learn from ONNX mistakes, apply to CoreML/TensorRT
- **Con**: Still need to rewrite everything

**Recommendation**: **Option C** - Extract patterns, then delete ONNX

The ONNX work taught us:
- How to structure multi-backend runtimes
- How to handle >2GB models
- How to optimize graph operations
- How to do zero-copy buffer sharing

Apply these lessons to CoreML and TensorRT, then remove ONNX.

---

## Success Criteria

### CoreML (v0.2.0)
- ✅ YOLO-v8n detection @ 30+ FPS (640x640) on M3 Max
- ✅ CLIP embeddings @ <50ms per image
- ✅ Whisper-base STT @ <100ms latency (10s chunk)
- ✅ FastSpeech2 TTS @ RT factor <0.3
- ✅ Zero-copy CoreML → MLX verified
- ✅ Multi-modal pipelines working end-to-end
- ✅ iOS deployment guide complete

### TensorRT (v0.3.0)
- ✅ YOLO-v8n detection @ 60+ FPS (640x640) on RTX 4070
- ✅ CLIP embeddings @ <20ms per image
- ✅ Whisper-base STT @ <50ms latency
- ✅ FastSpeech2 TTS @ RT factor <0.2
- ✅ DLPack zero-copy verified
- ✅ TensorRT-LLM decision made (add or skip)
- ✅ Jetson Orin deployment working

---

## Next Actions

1. **Immediate**: Extend `run-core` with vision/speech base classes
2. **Week 1-2**: Implement CoreML vision backend + converters
3. **Week 3-4**: Implement CoreML speech backends
4. **Week 5-6**: Multi-modal integration + testing
5. **Week 7-8**: Documentation + v0.2.0 release

Then repeat for TensorRT in Phase 2.
