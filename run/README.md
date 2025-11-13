# Llama-Pajamas Runtime

Lightweight LLM inference runtime for GGUF and MLX quantized models.

## Features

### LLM Inference
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI chat completions
- **Multi-Backend**: GGUF (CUDA/CPU) and MLX (Apple Silicon)
- **Minimal Dependencies**: Only install what you need
- **Performance Benchmarks**: Built-in tok/s measurement
- **Manifest-Driven**: Automatically discover model formats

### Multimodal (CoreML on Apple Silicon)
- **Vision**: Object detection (YOLO), classification (ViT), embeddings (CLIP)
- **Speech-to-Text**: Whisper models with ANE acceleration
- **ANE Optimization**: Apple Neural Engine for efficient inference
- **Shared Core + Runtime Pattern**: Clean architecture across modalities

## Installation

```bash
# For CUDA/CPU (GGUF backend)
pip install llama-pajamas-run[cuda]

# For Apple Silicon (MLX backend)
pip install llama-pajamas-run[mlx]

# For development (both backends)
pip install llama-pajamas-run[full]
```

## Quick Start

### Basic Text Generation

```python
from llama_pajamas_run import RuntimeConfig, ModelLoader

# Configure runtime
config = RuntimeConfig(
    backend="mlx",  # or "gguf"
    model_path="./models/qwen3-8b",
    max_tokens=200,
    temperature=0.7,
)

# Load and use model
with ModelLoader(config) as loader:
    response = loader.generate("Write a Python function to reverse a string:")
    print(response)
```

### OpenAI-Compatible Chat

```python
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(backend="mlx", model_path="./models/qwen3-8b")

with ModelLoader(config) as loader:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I use async/await in Python?"},
    ]

    response = loader.chat(messages)
    print(response["choices"][0]["message"]["content"])
```

### Streaming Generation

```python
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(backend="mlx", model_path="./models/qwen3-8b")

with ModelLoader(config) as loader:
    for chunk in loader.generate("Explain neural networks:", stream=True):
        print(chunk, end="", flush=True)
```

### Performance Benchmarking

```python
from llama_pajamas_run import RuntimeConfig, ModelLoader, benchmark_generation

config = RuntimeConfig(backend="mlx", model_path="./models/qwen3-8b")

with ModelLoader(config) as loader:
    result = benchmark_generation(
        loader,
        prompt="Write a detailed explanation:",
        num_tokens=200,
        warmup_runs=1,
    )

    print(f"Tokens/second: {result.tokens_per_second:.2f}")
```

## Configuration Options

### RuntimeConfig

```python
@dataclass
class RuntimeConfig:
    backend: Literal["gguf", "mlx"]  # Required
    model_path: str                   # Required
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    n_ctx: Optional[int] = None       # GGUF: context window size
    n_gpu_layers: int = -1            # GGUF: GPU layers (-1 = all)
    verbose: bool = False
```

### Backend-Specific Notes

**GGUF Backend** (llama-cpp-python):
- Works on CPU, CUDA, Metal, ROCm
- Use `n_gpu_layers=-1` to offload all layers to GPU
- Set `n_ctx` for custom context window size
- Best for: Universal compatibility, production deployments

**MLX Backend** (mlx-lm):
- Apple Silicon only (M1/M2/M3/M4)
- 10-20% faster than GGUF on Mac
- Unified memory architecture
- Best for: Mac development, optimal performance on Apple Silicon

## Model Loading

### Using Manifest (Recommended)

```python
# Load from quantized model directory with manifest.json
config = RuntimeConfig(
    backend="mlx",
    model_path="./models/qwen3-8b",  # Directory with manifest.json
)
```

The manifest.json describes available formats:

```json
{
  "model_id": "Qwen/Qwen3-8B",
  "architecture": { ... },
  "formats": {
    "gguf": {
      "file": "gguf/qwen3-8b_q4_k_m.gguf",
      "size_bytes": 4681234432
    },
    "mlx": {
      "directory": "mlx",
      "size_bytes": 4312345678
    }
  }
}
```

### Direct File Path

```python
# Load GGUF file directly
config = RuntimeConfig(
    backend="gguf",
    model_path="./models/qwen3-8b/gguf/qwen3-8b_q4_k_m.gguf",
)

# Load MLX directory directly
config = RuntimeConfig(
    backend="mlx",
    model_path="./models/qwen3-8b/mlx/",
)
```

## API Reference

### ModelLoader

**Methods:**

- `load()` - Load model into memory
- `generate(prompt, ...)` - Generate text from prompt
- `chat(messages, ...)` - OpenAI-compatible chat completion
- `count_tokens(text)` - Count tokens in text
- `unload()` - Free model from memory

**Context Manager:**

```python
with ModelLoader(config) as loader:
    # Model automatically loaded
    response = loader.generate("Hello")
    # Model automatically unloaded
```

### Benchmarks

**Functions:**

- `benchmark_generation(loader, ...)` - Benchmark basic generation
- `benchmark_streaming(loader, ...)` - Benchmark streaming with TTFT
- `benchmark_chat(loader, ...)` - Benchmark chat completions

**Returns:** `BenchmarkResult` with metrics:
- `tokens_per_second` - Generation speed
- `time_to_first_token_seconds` - Latency (streaming only)
- `total_time_seconds` - End-to-end time
- `prompt_tokens` / `generated_tokens` - Token counts

## Performance Targets

From MVP comprehensive plan:

| Model | Backend | Memory | Speed (Target) |
|-------|---------|--------|----------------|
| Qwen3-8B | GGUF | 1.9GB | 70+ tok/s (RTX 4070) |
| Qwen3-8B | MLX | 1.7GB | 80+ tok/s (M3 Max) |

## Speech-to-Text (CoreML)

### Basic Transcription

```python
from llama_pajamas_run_coreml.backends.stt import CoreMLSTTBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

# Initialize backend
backend = CoreMLSTTBackend()

# Load Whisper model (encoder on ANE, decoder in Python)
backend.load_model(
    model_path="./models/whisper-tiny/coreml/float16/encoder.mlpackage",
    model_name="tiny"  # or "base", "small"
)

# Load and transcribe audio
audio = load_audio("audio.flac", sample_rate=16000)
result = backend.transcribe(audio, sample_rate=16000)

print(f"Transcription: {result.text}")
print(f"Language: {result.language}")
```

### Streaming Transcription

```python
def audio_stream():
    # Generator that yields audio chunks
    while True:
        chunk = get_audio_chunk()  # Your audio source
        if chunk is None:
            break
        yield chunk

# Stream transcription
for partial_text in backend.transcribe_streaming(audio_stream()):
    print(partial_text, end=" ", flush=True)
```

### Batch Transcription

```python
# Load multiple audio files
audio_files = ["audio1.flac", "audio2.flac", "audio3.flac"]
audio_list = [load_audio(f, sample_rate=16000) for f in audio_files]

# Batch transcribe
results = backend.batch_transcribe(audio_list, sample_rate=16000)

for i, result in enumerate(results):
    print(f"File {i+1}: {result.text}")
```

### Performance (Apple Silicon)

| Model | Size | WER | Latency | RTF | Best For |
|-------|------|-----|---------|-----|----------|
| whisper-tiny | 15.7 MB | 9.2% | 249ms | 0.032 | Mobile, embedded |
| whisper-base | 39.3 MB | 5.1% | 542ms | 0.070 | Balanced |
| whisper-small | 168.3 MB | 0.9% | 983ms | 0.126 | Best quality |

- **WER**: Word Error Rate (lower is better)
- **RTF**: Real-time Factor (< 1.0 = faster than real-time)
- All models leverage ANE for encoder inference

### Browser Audio Recording (NEW: WebM Support)

**Browser MediaRecorder creates audio/webm files** which Whisper cannot process directly. LlamaPajamas automatically handles webm→wav conversion:

```typescript
// Simple UI API (simple-ui/app/api/speech/transcribe/route.ts)
// Automatic webm conversion using ffmpeg:

import { spawn } from 'child_process'
import { writeFile } from 'fs/promises'

// 1. Save uploaded webm file from browser
await writeFile(tempAudioPath, audioBuffer)

// 2. Convert webm → wav (16kHz mono) using ffmpeg
const ffmpeg = spawn('ffmpeg', [
  '-i', tempAudioPath,      // Input: webm from MediaRecorder
  '-ar', '16000',           // 16kHz (Whisper standard)
  '-ac', '1',               // Mono channel
  '-f', 'wav',              // WAV output
  '-y',                     // Overwrite
  tempWavPath
])

// 3. Load converted audio
const audio = load_audio(tempWavPath, sample_rate=16000)
const result = backend.transcribe(audio, sample_rate=16000)
```

**Features:**
- ✅ End-to-end browser recording (MediaRecorder API → webm → wav → Whisper)
- ✅ Automatic format detection and conversion
- ✅ Proper SSE stream buffering (fixed hanging issues)
- ✅ Automatic cleanup of temporary files
- ✅ Works with all Whisper models (tiny, base, small)

**Requirements:**
- `ffmpeg` installed (`brew install ffmpeg` on macOS, `apt-get install ffmpeg` on Linux)
- Modern browser with MediaRecorder API support (Chrome, Firefox, Safari, Edge)

**Usage in Simple UI:**
1. Navigate to Inference → Voice mode
2. Click "🎤 Record" → speak → "⏹ Stop"
3. Click "Transcribe Audio"
4. Conversion happens automatically, transcription appears

## Vision (CoreML) - 4 Task Types (NEW)

### Supported Vision Tasks

LlamaPajamas supports 4 distinct vision tasks with model-specific capabilities:

| Task | Description | Output | Models |
|------|-------------|--------|--------|
| **Classification** | What is in the image? | Top-K predictions | ViT, CLIP |
| **Localization** | Where is the main object? | Bounding box + label | ViT, CLIP, YOLO |
| **Detection** | What and where (all objects)? | Multiple boxes + labels | YOLO |
| **Segmentation** | Instance masks per object | Boxes + colored masks | YOLO (segmentation) |

### Vision Utilities

```python
from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
from llama_pajamas_run_core.utils import get_imagenet_class_names, annotate_image_with_detections
from PIL import Image

backend = CoreMLVisionBackend()

# Task 1: Classification with ImageNet labels
backend.load_model("models/clip-vit-base/coreml/int8/model.mlpackage", model_type="classification")
image = Image.open("cat.jpg")
predictions = backend.classify(image, top_k=5)

# Get human-readable labels (not just class IDs)
imagenet_names = get_imagenet_class_names()
for pred in predictions:
    label = imagenet_names[pred.class_id]  # "tabby_cat" instead of "class_167"
    print(f"{label}: {pred.confidence:.1%}")

# Generate annotated image with labels
annotated = annotate_image_with_detections(image, predictions, task_type="classification")
annotated.save("classification_result.jpg")

# Task 2: Localization (main object with bounding box)
backend.load_model("models/yolo-v8n/coreml/fp16/model.mlpackage", model_type="detection")
detections = backend.detect(image, confidence_threshold=0.3)
top_detection = max(detections, key=lambda d: d.confidence)
annotated = annotate_image_with_detections(image, [top_detection], task_type="localization")
annotated.save("localization_result.jpg")

# Task 3: Detection (all objects with colored boxes)
detections = backend.detect(image, confidence_threshold=0.5)
annotated = annotate_image_with_detections(image, detections, task_type="detection")
annotated.save("detection_result.jpg")

# Task 4: Segmentation (instance masks)
annotated = annotate_image_with_detections(image, detections, task_type="segmentation")
annotated.save("segmentation_result.jpg")
```

**Annotation Features:**
- **Classification**: Top prediction label at top-left
- **Localization**: Green bounding box with label
- **Detection**: Color-coded boxes for each object
- **Segmentation**: Semi-transparent colored masks per instance

**CLI Test**: See `run-coreml/test_all_vision_tasks.py` for complete examples

## Multi-Modal Server (CoreML)

Unified OpenAI-compatible API server with Vision + Speech + LLM support.

### Start Server

```bash
# Start with Vision (detection) + STT
cd run-coreml
uv run python examples/multimodal_server_demo.py

# Server will start on http://localhost:8000
```

### API Endpoints

#### Vision

```bash
# Object detection
curl -X POST http://localhost:8000/v1/images/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }'

# Image classification
curl -X POST http://localhost:8000/v1/images/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "top_k": 5}'

# Image embeddings (CLIP)
curl -X POST http://localhost:8000/v1/images/embed \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

#### Speech-to-Text (OpenAI-Compatible)

```bash
# Transcribe audio (OpenAI-compatible)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-tiny \
  -F response_format=json

# With language hint
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-tiny \
  -F language=en \
  -F response_format=verbose_json
```

#### Management

```bash
# Health check
curl http://localhost:8000/health

# List loaded models
curl http://localhost:8000/v1/models
```

### Python Client

```python
import requests
import base64
from PIL import Image
import io

# Object detection
with Image.open("image.jpg") as img:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

response = requests.post(
    "http://localhost:8000/v1/images/detect",
    json={
        "image": f"data:image/jpeg;base64,{img_base64}",
        "confidence_threshold": 0.5,
    }
)
detections = response.json()["detections"]

# Speech-to-text
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": ("audio.wav", f, "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "json"}
    )
transcription = response.json()["text"]
```

### Server Architecture

- **Core Server** (`run-core/server_multimodal.py`): Shared FastAPI app factory
- **CoreML Server** (`run-coreml/server.py`): CoreML backend integration
- **Backends**: Vision, STT, TTS (modular, load what you need)
- **OpenAI-Compatible**: Drop-in replacement for OpenAI Audio API

### Performance

Running on Apple Silicon (M3 Max):
- **Vision (YOLO-v8n)**: ~40 FPS (INT8), 13-15 FPS (FP16)
- **STT (whisper-tiny)**: 31x faster than real-time (RTF: 0.032)
- **All models**: Optimized for Apple Neural Engine (ANE)

## Examples

### LLM Examples

See `examples/simple_usage.py` for complete working examples:

1. Basic text generation
2. OpenAI-compatible chat
3. Streaming generation
4. GGUF backend usage
5. Performance benchmarking

### Multimodal Examples

See `run-coreml/examples/` for CoreML examples:

1. **Vision**: Object detection, classification, embeddings
2. **STT**: Audio transcription with Whisper
3. **ANE Optimization**: Apple Neural Engine usage

## Architecture

```
llama_pajamas_run/
   config.py              # RuntimeConfig dataclass
   manifest_loader.py     # Load manifest.json
   model_loader.py        # High-level ModelLoader
   benchmarks.py          # Performance measurement
   backends/
       base.py           # Abstract Backend interface
       gguf_backend.py   # llama-cpp-python wrapper
       mlx_backend.py    # mlx-lm wrapper
```

## Design Principles

1. **Separation of Concerns**: Runtime is separate from quantization pipeline
2. **Minimal Dependencies**: Optional extras for CUDA/MLX
3. **OpenAI Compatibility**: Drop-in replacement for chat API
4. **Config-Based**: No auto-detection, explicit backend selection
5. **Lightweight**: Production deployments ~500MB vs 5GB+ for pipeline

## TensorRT Runtime (NVIDIA GPU)

Optimized inference on NVIDIA GPUs using TensorRT.

### Installation

```bash
pip install llama-pajamas-run-tensorrt

# Prerequisites:
# - CUDA 11.8+ or 12.0+
# - cuDNN 8.9+
# - TensorRT 8.6+
# - NVIDIA GPU (RTX 3060+, A100, H100)
```

### Quick Start

**LLM Inference**:
```python
from llama_pajamas_run_tensorrt import TensorRTLLMBackend

backend = TensorRTLLMBackend()
backend.load_model("models/qwen3-8b-int8.engine")

response = backend.generate("Hello!", max_tokens=200)
print(response)
```

**Vision Inference**:
```python
from llama_pajamas_run_tensorrt import TensorRTVisionBackend
from PIL import Image

backend = TensorRTVisionBackend()
backend.load_model("models/yolov8n-fp16.engine", model_type="detection")

image = Image.open("image.jpg")
detections = backend.detect(image)
```

### Performance (vs CoreML)

**LLM (RTX 4090 vs M3 Max)**:
- Throughput: 120 tok/s vs 80 tok/s (**1.5x faster**)
- Batch size: 128 vs 8 (**16x larger**)

**Vision (RTX 4090 vs M3 Max)**:
- YOLO FPS: 400 vs 40 (**10x faster**)
- ViT FPS: 150 vs 20 (**7.5x faster**)

### Use Cases

- **TensorRT**: Data centers, batch processing, high throughput
- **CoreML**: Mobile, edge devices, privacy-first
- **GGUF**: Universal compatibility (CPU/GPU)
- **MLX**: Apple Silicon optimization

See [run-tensorrt/README.md](../run-tensorrt/README.md) for complete documentation.

---

## ONNX Runtime (Universal Edge Deployment)

Optimized inference for CPU, AMD GPU, ARM processors, Jetson, Intel GPUs.

### Installation

```bash
pip install llama-pajamas-run-onnx

# No special requirements:
# - Works on CPU (any platform)
# - Optional GPU acceleration (CUDA, ROCm, DirectML, OpenVINO)
```

### Quick Start

**Vision Inference (CPU)**:
```python
from llama_pajamas_run_onnx import ONNXVisionBackend
from PIL import Image

# Initialize backend
backend = ONNXVisionBackend()

# Load model (CPU execution)
backend.load_model(
    "models/yolov8n/onnx/yolov8n.onnx",
    model_type="detection",
    providers=["CPUExecutionProvider"],  # CPU only
    num_threads=4,
)

# Run inference
image = Image.open("image.jpg")
detections = backend.detect(image, confidence_threshold=0.5)

for det in detections:
    print(f"Class {det['class_id']}: {det['confidence']:.2f} at {det['bbox']}")
```

**Vision Inference (GPU)**:
```python
# NVIDIA GPU with TensorRT
backend.load_model(
    "models/yolov8n/onnx/yolov8n.onnx",
    model_type="detection",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Intel GPU/CPU with OpenVINO
backend.load_model(
    "models/yolov8n/onnx/yolov8n.onnx",
    model_type="detection",
    providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
)

# AMD GPU (Windows)
backend.load_model(
    "models/yolov8n/onnx/yolov8n.onnx",
    model_type="detection",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)
```

**Speech-to-Text Inference**:
```python
from llama_pajamas_run_onnx import ONNXSpeechBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

# Initialize backend
backend = ONNXSpeechBackend()

# Load Whisper encoder
backend.load_model(
    encoder_path="models/whisper-tiny/onnx/tiny_encoder.onnx",
    model_name="whisper-tiny",
    providers=["CPUExecutionProvider"],
)

# Load and transcribe audio
audio = load_audio("audio.wav", sample_rate=16000)
result = backend.transcribe(audio, sample_rate=16000)

print(f"Transcription: {result['text']}")
```

### Performance (CPU vs GPU)

**Apple M3 Max (CPU only)**:
| Model | Format | Inference | FPS |
|-------|--------|-----------|-----|
| YOLO-v8n | FP32 | 51.5ms | 19.4 |

**NVIDIA RTX 4090 (TensorRT provider)**:
| Model | Format | Inference | FPS |
|-------|--------|-----------|-----|
| YOLO-v8n | FP32 | 4ms | 250 |
| YOLO-v8n | INT8 | 2.5ms | 400 |

**Intel Core i9 (OpenVINO provider)**:
| Model | Format | Inference | FPS |
|-------|--------|-----------|-----|
| YOLO-v8n | FP32 | ~30ms | ~33 |

### Execution Providers

ONNX Runtime supports multiple execution providers:

| Provider | Hardware | Performance | Use Case |
|----------|----------|-------------|----------|
| **TensorrtExecutionProvider** | NVIDIA GPU | Highest | Data centers, high throughput |
| **CUDAExecutionProvider** | NVIDIA GPU | High | General CUDA acceleration |
| **OpenVINOExecutionProvider** | Intel CPU/GPU | High | Intel hardware optimization |
| **DmlExecutionProvider** | AMD/NVIDIA (Windows) | Medium | Windows DirectX acceleration |
| **ROCmExecutionProvider** | AMD GPU (Linux) | Medium | AMD GPU on Linux |
| **CPUExecutionProvider** | Any CPU | Baseline | Universal compatibility |

### Use Cases

**ONNX vs TensorRT vs CoreML**:
- **ONNX**: Cross-platform, CPU/edge deployment, portable models
- **TensorRT**: NVIDIA GPUs only, highest performance, data centers
- **CoreML**: Apple Silicon only, ANE acceleration, mobile

**When to use ONNX Runtime**:
1. **CPU-only deployment** (no GPU required)
2. **Non-NVIDIA GPUs** (AMD, Intel)
3. **Edge devices** (Raspberry Pi, Jetson with ONNX)
4. **Cross-platform applications** (Windows, Linux, macOS, ARM)
5. **Mixed hardware environments** (cloud with various GPUs)
6. **Docker containers** (portable across infrastructure)

### Quantization Support

**INT8 Quantization**:
- ✅ FP32 models: Full support on CPU
- ⚠️ INT8 models: Require GPU providers (TensorRT, OpenVINO, DirectML)
- ❌ INT8 on CPU: Limited operator support (`ConvInteger` not supported)

For CPU deployment, use FP32 models. For GPU deployment, use INT8 for 72.8% size reduction.

### Model Compatibility

**Supported Models**:
- ✅ Vision: YOLO, ViT, ResNet, CLIP, Faster R-CNN
- ✅ Speech: Whisper encoder (decoder via Python)
- ✅ Any PyTorch model exported to ONNX
- ✅ HuggingFace Transformers (with `transformers.onnx`)

**Export from**:
- PyTorch → ONNX (via `torch.onnx.export`)
- TensorFlow → ONNX (via `tf2onnx`)
- HuggingFace → ONNX (via `optimum.onnxruntime`)

---

## Requirements

- Python 3.12+
- For GGUF: `llama-cpp-python >= 0.2.0`
- For MLX: `mlx >= 0.19.0`, `mlx-lm >= 0.19.0`
- For TensorRT: `tensorrt >= 8.6.0`, `tensorrt-llm`

## License

See main project LICENSE file.

## Related

- [llama-pajamas-quant](../quant/README.md) - Quantization pipeline
- [Main README](../README.md) - Project overview
- [Deployment Guide](../quant/DEPLOYMENT.md) - Production deployment
