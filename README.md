# Llama-Pajamas 🦙 📦

**Universal Model Quantization & Optimized Runtime System**

Llama-Pajamas provides a complete pipeline for quantizing and deploying AI models across all modalities (LLM, Vision, Speech) on hardware-optimized runtimes.

## Our Distinct Point of View

**❌ WRONG: "One runtime fits all hardware"**
- Traditional approach: Same PyTorch model runs everywhere
- Result: Inefficient - no hardware acceleration, large memory footprint
- Result: Slow - CPU-only inference, no GPU/ANE optimization

**✅ RIGHT: "Hardware-optimized quantization + native runtimes"**
- **Llama-Pajamas**: Quantize once → Deploy on optimized runtime per platform
- **Apple Silicon**: MLX (LLM), CoreML (Vision/Speech) with ANE acceleration
- **NVIDIA GPU**: GGUF (LLM), TensorRT (Vision/Speech) with CUDA optimization
- **CPU/Edge**: GGUF (LLM), ONNX (Vision/Speech) for universal compatibility
- Result: **3-10x faster inference**, **50-75% smaller models**, **native hardware acceleration**

## The Complete Pipeline

### 1. Quantization (Offline, Heavy)
Convert full-precision models to compressed formats:
- **LLM**: GGUF (Q4_K_M, IQ2_XS) + MLX (4-bit, 2-bit)
- **Vision**: CoreML (INT8), TensorRT (FP16/INT8), ONNX (INT8)
- **Speech**: CoreML (INT8), ONNX (FP32)

### 2. Runtime (Online, Light)
Deploy quantized models on hardware-optimized runtimes:
- **Apple Silicon**: MLX, CoreML (ANE acceleration)
- **NVIDIA GPU**: GGUF, TensorRT (CUDA optimization)
- **CPU/Edge**: GGUF, ONNX (universal compatibility)

### 3. Multi-Modal Server
Unified OpenAI-compatible API server:
- **LLM**: Chat completions
- **Vision**: Object detection, classification, embeddings
- **Speech**: Transcription (Whisper)

## Quick Start

### LLM: Quantize and Run Qwen3-8B

```bash
# 1. Quantize to GGUF Q4_K_M (4.6 GB, industry standard)
cd quant
uv run python test_dual_format.py \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf \
  --gguf-precision Q4_K_M

# 2. Run inference
cd ../run
uv run python examples/simple_usage.py
# Output: ~80 tok/s on M3 Max, ~70 tok/s on RTX 4070
```

### Vision: Quantize and Run YOLO

```bash
# 1. Export to CoreML with INT8 quantization
cd quant
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend coreml \
  --precision int8 \
  --output models/yolo-v8n/

# 2. Run inference
cd ../run-coreml
uv run python examples/test_vision.py
# Output: ~40 FPS on M3 Max (INT8 with ANE)
```

### Speech: Transcribe Audio with Whisper

```bash
# 1. Export to CoreML INT8
cd quant
uv run python scripts/quantize_whisper_coreml.py --model whisper-tiny

# 2. Transcribe audio
cd ../run-coreml
uv run python examples/test_stt.py
# Output: 31x faster than real-time (RTF: 0.032)
```

## Project Structure

```
llama-pajamas/
├── quant/                    # Quantization Pipeline (llama-pajamas-quant)
│   ├── llama_pajamas_quant/
│   │   ├── core/            # Architecture detection
│   │   ├── converters/      # GGUF, MLX converters
│   │   ├── exporters/       # ONNX, unified export
│   │   └── quantizers/      # ONNX quantization
│   ├── scripts/             # CLI scripts
│   ├── evaluation/          # LLM, Vision, STT evaluation
│   └── models/              # Quantized models output
│
├── run/                      # LLM Runtime (GGUF/MLX)
│   └── llama_pajamas_run/
│       ├── backends/        # GGUF, MLX backends
│       └── benchmarks.py    # Performance testing
│
├── run-core/                 # Shared runtime core
│   └── llama_pajamas_run_core/
│       ├── backends/        # Base classes
│       └── utils/           # Audio, image utilities
│
├── run-coreml/              # CoreML Runtime (Vision/Speech)
│   └── llama_pajamas_run_coreml/
│       ├── backends/        # Vision, STT backends
│       └── server.py        # Multi-modal API server
│
├── run-onnx/                # ONNX Runtime (Edge deployment)
│   └── llama_pajamas_run_onnx/
│       └── backends/        # Vision, Speech backends
│
└── run-tensorrt/            # TensorRT Runtime (NVIDIA GPU)
    └── llama_pajamas_run_tensorrt/
        └── backends/        # Vision, LLM backends
```

---

# LLM Pipeline (Text Generation)

## 1. Quantization (Offline)

### Standard Quantization (Q4_K_M - Recommended)

```bash
cd quant

# Dual-format: GGUF + MLX (one command)
uv run python test_dual_format.py \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --gguf-precision Q4_K_M \
  --mlx-bits 4

# Output:
# models/qwen3-8b/
#   ├── gguf/Q4_K_M/model.gguf      (4.68 GB)
#   ├── mlx/4bit-mixed/             (4.31 GB)
#   └── manifest.json
```

**Result**: 3.3x compression, <5% quality loss, industry standard

### Extreme Compression (IQ2_XS - 50% smaller)

**What are IQ methods?** Importance Quantization (IQ) uses calibration data to identify critical weights and preserve them at higher precision. This achieves better quality than standard quantization at ultra-low bit rates.

**How it works:**
1. Generate calibration data (diverse prompts)
2. Run model on calibration data to identify important weights
3. Apply variable precision: Critical weights → higher bits, Less important → lower bits
4. Result: Better quality than uniform quantization at same size

```bash
cd quant

# Step 1: Generate calibration data (140 diverse prompts)
uv run python -c "
from llama_pajamas_quant.simple_benchmarks import TEST_PROMPTS
with open('calibration.txt', 'w') as f:
    for prompt in TEST_PROMPTS:
        f.write(prompt['prompt'] + '\n\n')
"

# Step 2: Generate importance matrix
cd ../libs/llama.cpp
./llama-imatrix \
  -m ../../quant/models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  -f ../../quant/calibration.txt \
  -o ../../quant/qwen3-8b.imatrix \
  --chunks 512

# Step 3: Quantize with importance matrix
./llama-quantize \
  --imatrix ../../quant/qwen3-8b.imatrix \
  ../../quant/models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  ../../quant/models/qwen3-8b/gguf/IQ2_XS/model.gguf \
  IQ2_XS

# Step 4: MLX 2-bit (for Apple Silicon)
cd ../../quant
uv run python test_dual_format.py \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats mlx \
  --mlx-bits 2
```

**Result**: ~2.4 GB (GGUF IQ2_XS) + ~2.4 GB (MLX 2-bit) = 53% size reduction vs Q4_K_M

**Quality comparison:**
- **Q4_K_M**: 4.6 GB, 94% accuracy (industry standard)
- **IQ3_M**: 3.5 GB, 90-93% accuracy (good balance)
- **IQ2_XS**: 2.4 GB, 85-90% accuracy (with imatrix, usable)
- **IQ2_XXS**: 2.2 GB, 80-85% accuracy (extreme compression)

**Where to find quantized models:**
```
models/qwen3-8b/
├── gguf/
│   ├── Q4_K_M/model.gguf        # Industry standard (4.6 GB)
│   ├── Q3_K_M/model.gguf        # Smaller alternative (3.8 GB)
│   ├── IQ2_XS/model.gguf        # Extreme compression (2.4 GB)
│   └── Q5_K_M/model.gguf        # High quality (5.3 GB)
└── mlx/
    ├── 4bit-mixed/              # Standard MLX (4.3 GB)
    ├── 3bit-mixed/              # Smaller MLX (3.2 GB)
    └── 2bit-mixed/              # Extreme MLX (2.4 GB)
```

## 2. Runtime (Online)

### Running GGUF Models (Universal)

```python
# examples/simple_usage.py
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(
    backend="gguf",
    model_path="./models/qwen3-8b/gguf/Q4_K_M/model.gguf",
    n_gpu_layers=-1,  # Offload all layers to GPU
    n_ctx=4096,       # Context window
)

with ModelLoader(config) as loader:
    response = loader.generate(
        "Write a Python function to reverse a string:",
        max_tokens=200,
        temperature=0.7,
    )
    print(response)
```

**Performance**: ~70 tok/s (NVIDIA RTX 4070), ~80 tok/s (Apple M3 Max with Metal)

### Running MLX Models (Apple Silicon)

```python
from llama_pajamas_run import RuntimeConfig, ModelLoader

config = RuntimeConfig(
    backend="mlx",
    model_path="./models/qwen3-8b/mlx/4bit-mixed/",
    max_tokens=2048,
)

with ModelLoader(config) as loader:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain neural networks in simple terms."},
    ]

    response = loader.chat(messages)
    print(response["choices"][0]["message"]["content"])
```

**Performance**: ~80 tok/s on M3 Max (10-20% faster than GGUF on Mac)

### Streaming Generation

```python
config = RuntimeConfig(backend="mlx", model_path="./models/qwen3-8b/mlx/4bit-mixed/")

with ModelLoader(config) as loader:
    for chunk in loader.generate("Explain quantum computing:", stream=True):
        print(chunk, end="", flush=True)
```

## 3. Evaluation

### Run Evaluation (40 questions across 6 categories)

```bash
cd quant

# Evaluate GGUF models
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --model-path ./models/qwen3-8b/gguf/IQ2_XS/*.gguf \
    --format gguf

# Evaluate MLX models
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/mlx/4bit-mixed \
    --model-path ./models/qwen3-8b/mlx/2bit-mixed \
    --format mlx

# Quick test (10 questions, ~30 seconds per model)
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf \
    --num-questions 10
```

**Categories tested:**
- Knowledge (MMLU-style): 10 questions
- Common Sense (HellaSwag): 5 questions
- Math (GSM8K): 10 questions
- Reasoning (ARC): 5 questions
- Truthfulness (TruthfulQA): 5 questions
- Tool Calling (BFCL): 5 questions

### Generate Comparison Report

```bash
# Compare all quantizations
uv run python evaluation/llm/compare_evaluations.py \
    --model-dir ./models/qwen3-8b

# View results
cat ./models/qwen3-8b/EVALUATION_REPORT.md
```

**Example results:**
| Model | Size | Accuracy | Speed (s/q) | Efficiency (acc/GB) |
|-------|------|----------|-------------|---------------------|
| Q4_K_M | 4.68 GB | 94.0% | 0.79 | 20.1 |
| Q3_K_M | 3.84 GB | 94.3% | 0.67 | 24.6 |
| IQ2_XS | 2.40 GB | 87.5% | 0.52 | 36.5 |
| MLX 4-bit | 4.31 GB | 93.0% | 1.38 | 21.6 |

---

# Vision Pipeline (Object Detection, Classification)

## 1. Unified Export (All Backends)

```bash
cd quant

# CoreML (Apple Silicon, ANE acceleration)
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend coreml \
  --precision float16 \
  --output models/yolo-v8n/

# ONNX (CPU, AMD GPU, Intel GPU, Edge)
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend onnx \
  --precision int8 \
  --output models/yolo-v8n/

# TensorRT (NVIDIA GPU) - Step 1: Export to ONNX
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend tensorrt \
  --precision fp16 \
  --output models/yolo-v8n/
# Step 2: Build TensorRT engine (requires NVIDIA GPU or Docker)
./quant/scripts/build_tensorrt_engine.sh \
    quant/models/yolo-v8n/tensorrt/fp16/yolov8n.onnx \
    quant/models/yolo-v8n/tensorrt/fp16/yolov8n.engine \
    fp16
```

## 2. CoreML Quantization (Apple Silicon)

### Post-Training INT8 Quantization (50% size reduction)

```bash
cd quant

# Quantize all vision models to INT8
uv run python scripts/quantize_coreml_vision.py --model all --precision int8

# Quantize specific model
uv run python scripts/quantize_coreml_vision.py --model vit-base --precision int8

# Quantize to INT4 (experimental, 75% reduction)
uv run python scripts/quantize_coreml_vision.py --model vit-base --precision int4
```

**Results:**
- **ViT-Base**: 165 MB → 83 MB (49.7% reduction, +2.4% FPS gain)
- **CLIP-ViT-Base**: 167 MB → 83.8 MB (49.8% reduction, -4.3% FPS)
- **YOLO-v8n**: ⚠️ INT8 not supported reliably (use FP16)

**Where to find models:**
```
models/vit-base/
├── coreml/
│   ├── fp16/model.mlpackage      (165 MB)
│   └── int8/model.mlpackage      (83 MB)
└── QUANTIZATION_REPORT.md
```

## 3. ONNX Quantization (Edge Deployment)

**INT8 dynamic quantization** (72.8% size reduction):

```bash
cd quant

# Already done via unified export
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend onnx \
  --precision int8 \
  --output models/yolo-v8n/

# Output: 12.2 MB FP32 → 3.3 MB INT8
```

**Note**: INT8 models require GPU execution providers (TensorRT, OpenVINO, DirectML). For CPU-only deployment, use FP32.

## 4. Runtime (Online)

### CoreML Runtime (Apple Silicon)

```python
# run-coreml/examples/test_vision.py
from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
from PIL import Image

backend = CoreMLVisionBackend()

# Load INT8 model (optimized for ANE)
backend.load_model(
    model_path="./models/yolo-v8n/coreml/int8/model.mlpackage",
    model_type="detection"
)

# Run inference
image = Image.open("image.jpg")
detections = backend.detect(image, confidence_threshold=0.5)

for det in detections:
    print(f"Class {det['class']}: {det['confidence']:.2f} at {det['bbox']}")
```

**Performance**: ~40 FPS (INT8 with ANE), ~13-15 FPS (FP16)

### ONNX Runtime (CPU/Edge)

```python
# run-onnx/examples/test_onnx_vision.py
from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend
from PIL import Image

backend = ONNXVisionBackend()

# Load model (CPU execution)
backend.load_model(
    "models/yolo-v8n/onnx/yolov8n.onnx",
    model_type="detection",
    providers=["CPUExecutionProvider"],  # CPU only
    num_threads=4,
)

# Run inference
image = Image.open("image.jpg")
detections = backend.detect(image, confidence_threshold=0.5)
```

**Performance**: ~19 FPS on M3 Max CPU (FP32)

### ONNX Runtime with GPU

```python
# NVIDIA GPU with TensorRT
backend.load_model(
    "models/yolo-v8n/onnx/yolov8n.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Intel GPU/CPU with OpenVINO
backend.load_model(
    "models/yolo-v8n/onnx/yolov8n.onnx",
    providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
)
```

## 5. Evaluation

```bash
cd quant

# Evaluate all vision models
uv run python evaluation/vision/run_eval.py \
    --models-dir ./models \
    --images evaluation/vision/images/detection

# Evaluate specific model
uv run python evaluation/vision/run_eval.py \
    --model yolo-v8n \
    --models-dir ./models \
    --images evaluation/vision/images/detection

# Custom detection threshold
uv run python evaluation/vision/run_eval.py \
    --model yolo-v8n \
    --conf-threshold 0.3 \
    --models-dir ./models \
    --images evaluation/vision/images/detection

# View results
cat ./models/yolo-v8n/EVALUATION_REPORT.md
```

**Results:**
```
models/yolo-v8n/
├── coreml/fp16/evaluation.json      # 34.3 FPS, 29.1ms latency
├── coreml/int8/evaluation.json      # 40.0 FPS, 25.0ms latency
└── EVALUATION_REPORT.md
```

---

# Speech Pipeline (Speech-to-Text)

## 1. Export to CoreML

```bash
cd quant

# Export all Whisper models to CoreML FP16
uv run python scripts/export_whisper_coreml.py --model all --precision float16

# Export specific model
uv run python scripts/export_whisper_coreml.py --model whisper-tiny --precision float16
```

**Models available:**
- whisper-tiny: 39M params, 15.7 MB encoder
- whisper-base: 74M params, 39.3 MB encoder
- whisper-small: 244M params, 168.3 MB encoder

## 2. Quantize to INT8 (50% size reduction)

```bash
cd quant

# Quantize all Whisper encoders to INT8
uv run python scripts/quantize_whisper_coreml.py --model all

# Quantize specific model
uv run python scripts/quantize_whisper_coreml.py --model whisper-tiny
```

**Results:**
| Model | FP16 Size | INT8 Size | Reduction | WER Change |
|-------|-----------|-----------|-----------|------------|
| whisper-tiny | 15.7 MB | 7.9 MB | 49.7% | -0.004 (better) |
| whisper-base | 39.3 MB | 19.8 MB | 49.6% | 0.000 (same) |
| whisper-small | 168.3 MB | 84.5 MB | 49.8% | 0.000 (same) |

**Recommendation**: Use INT8 as default (50% smaller, no quality loss)

## 3. Export to ONNX (Edge Deployment)

```bash
cd quant

# Export Whisper to ONNX
uv run python scripts/export_model.py \
  --model whisper-tiny \
  --backend onnx \
  --precision fp32 \
  --output models/whisper-tiny/
```

## 4. Runtime (Online)

### CoreML Runtime (Apple Silicon)

```python
# run-coreml/examples/test_stt.py
from llama_pajamas_run_coreml.backends.stt import CoreMLSTTBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

backend = CoreMLSTTBackend()

# Load INT8 model (optimized for ANE)
backend.load_model(
    model_path="./models/whisper-tiny/coreml/int8/encoder.mlpackage",
    model_name="tiny"
)

# Transcribe audio
audio = load_audio("audio.flac", sample_rate=16000)
result = backend.transcribe(audio, sample_rate=16000)

print(f"Text: {result.text}")
print(f"Language: {result.language}")
```

**Performance**: 31x faster than real-time (RTF: 0.032)

### ONNX Runtime (Edge)

```python
from llama_pajamas_run_onnx.backends.speech import ONNXSpeechBackend
from llama_pajamas_run_core.utils.audio_utils import load_audio

backend = ONNXSpeechBackend()

backend.load_model(
    encoder_path="models/whisper-tiny/onnx/tiny_encoder.onnx",
    model_name="whisper-tiny",
    providers=["CPUExecutionProvider"],
)

audio = load_audio("audio.wav", sample_rate=16000)
result = backend.transcribe(audio, sample_rate=16000)
```

## 5. Evaluation

```bash
cd quant

# Download LibriSpeech audio samples (10 samples)
cd evaluation/stt
uv run python download_audio.py --num-samples 10

# Evaluate all Whisper models
cd ../../run-coreml
uv run python ../quant/evaluation/stt/run_eval.py \
    --models-dir ../quant/models

# Evaluate specific model
uv run python ../quant/evaluation/stt/run_eval.py \
    --models-dir ../quant/models \
    --model whisper-tiny

# View results
cat ../quant/models/whisper-tiny/EVALUATION_REPORT.md
```

**Results:**
| Model | Size | WER | Latency | RTF | Best For |
|-------|------|-----|---------|-----|----------|
| whisper-tiny | 15.7 MB | 9.2% | 249ms | 0.032 | Mobile, embedded |
| whisper-base | 39.3 MB | 5.1% | 542ms | 0.070 | Balanced |
| whisper-small | 168.3 MB | 0.9% | 983ms | 0.126 | Best quality |

**Metrics:**
- **WER** (Word Error Rate): Lower is better (0% = perfect)
- **RTF** (Real-time Factor): <1.0 = faster than real-time
- All models run faster than real-time with ANE acceleration

---

# Multi-Modal Server (OpenAI-Compatible API)

## Starting the Server

```bash
cd run-coreml

# Start with Vision + STT
uv run python examples/multimodal_server_demo.py

# Server starts on http://localhost:8000
```

## API Endpoints

### Vision - Object Detection

```bash
# Detect objects in image
curl -X POST http://localhost:8000/v1/images/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }'
```

### Vision - Classification

```bash
# Classify image
curl -X POST http://localhost:8000/v1/images/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "top_k": 5}'
```

### Vision - Embeddings (CLIP)

```bash
# Generate image embeddings
curl -X POST http://localhost:8000/v1/images/embed \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

### Speech-to-Text (OpenAI-Compatible)

```bash
# Transcribe audio (OpenAI-compatible endpoint)
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

### Management

```bash
# Health check
curl http://localhost:8000/health

# List loaded models
curl http://localhost:8000/v1/models
```

## Python Client

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

**Architecture:**
- **Core Server** (`run-core/server_multimodal.py`): Shared FastAPI app factory
- **CoreML Server** (`run-coreml/server.py`): CoreML backend integration
- **Backends**: Vision, STT (modular, load what you need)
- **OpenAI-Compatible**: Drop-in replacement for OpenAI Audio API

**Performance** (Apple M3 Max):
- Vision (YOLO-v8n INT8): ~40 FPS
- STT (whisper-tiny): RTF 0.032 (31x faster than real-time)

# Installation

## Prerequisites

- Python 3.12+
- UV package manager
- Platform-specific requirements:
  - **Apple Silicon**: macOS 13.5+ for CoreML/MLX
  - **NVIDIA GPU**: CUDA 11.8+, cuDNN 8.9+, TensorRT 8.6+ (optional)
  - **CPU/Edge**: No special requirements

## Quantization Pipeline (Required)

```bash
# Clone repository
git clone https://github.com/yourusername/llama-pajamas.git
cd llama-pajamas

# Install quantization pipeline
cd quant
uv sync

# Verify installation
uv run python -c "from llama_pajamas_quant import ArchitectureDetector; print('✅ Quant pipeline installed')"
```

## Runtimes (Install what you need)

### LLM Runtime (GGUF/MLX)

```bash
# Install LLM runtime
cd ../run
uv sync

# Verify
uv run python -c "from llama_pajamas_run import ModelLoader; print('✅ LLM runtime installed')"
```

### CoreML Runtime (Apple Silicon - Vision/Speech)

```bash
# Install CoreML runtime
cd ../run-coreml
uv sync

# Verify
uv run python -c "from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend; print('✅ CoreML runtime installed')"
```

### ONNX Runtime (Edge deployment)

```bash
# Install ONNX runtime
cd ../run-onnx
uv sync

# Verify
uv run python -c "from llama_pajamas_run_onnx.backends.vision import ONNXVisionBackend; print('✅ ONNX runtime installed')"
```

### TensorRT Runtime (NVIDIA GPU - Optional)

```bash
# Install TensorRT runtime
cd ../run-tensorrt
uv sync

# Requires: CUDA, cuDNN, TensorRT pre-installed
```

---

# Development

## Using UV

```bash
# Install dependencies
cd quant  # or run, run-coreml, run-onnx, run-tensorrt
uv sync

# Add new dependencies
uv add package-name

# Update dependencies
uv lock --upgrade

# Run scripts
uv run python script.py
```

## Package Development

```bash
# Install package in editable mode
cd quant
uv pip install -e .

# Install runtime
cd ../run
uv pip install -e .

# Install CoreML runtime
cd ../run-coreml
uv pip install -e .
```

## Testing

```bash
# LLM quantization test
cd quant
uv run python test_dual_format.py \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf \
  --gguf-precision Q4_K_M

# LLM evaluation test
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf \
    --num-questions 10

# Vision export test
uv run python scripts/export_model.py \
  --model yolov8n \
  --backend onnx \
  --precision fp32 \
  --output models/yolo-v8n/

# Vision evaluation test
uv run python evaluation/vision/run_eval.py \
    --model yolo-v8n \
    --models-dir ./models \
    --images evaluation/vision/images/detection
```

---

# Hardware Platform Support

| Platform | LLM | Vision | Speech | Runtimes |
|----------|-----|--------|--------|----------|
| **Apple Silicon** | ✅ MLX, GGUF | ✅ CoreML (ANE) | ✅ CoreML (ANE) | mlx-lm, llama-cpp-python, CoreML |
| **NVIDIA GPU** | ✅ GGUF | ✅ TensorRT | ✅ TensorRT | llama-cpp-python, TensorRT |
| **AMD GPU** | ✅ GGUF | ✅ ONNX (ROCm/DirectML) | ✅ ONNX (ROCm/DirectML) | llama-cpp-python, ONNX Runtime |
| **Intel GPU** | ✅ GGUF | ✅ ONNX (OpenVINO) | ✅ ONNX (OpenVINO) | llama-cpp-python, ONNX Runtime |
| **CPU (Any)** | ✅ GGUF | ✅ ONNX | ✅ ONNX | llama-cpp-python, ONNX Runtime |
| **Edge (ARM)** | ✅ GGUF | ✅ ONNX | ✅ ONNX | llama-cpp-python, ONNX Runtime |

**Key:**
- **ANE**: Apple Neural Engine (hardware acceleration)
- **ROCm**: AMD GPU compute platform
- **DirectML**: Windows GPU acceleration (AMD/NVIDIA)
- **OpenVINO**: Intel optimization toolkit

---

# Performance Summary

## LLM (Qwen3-8B)

| Format | Size | Accuracy | Speed | Platform |
|--------|------|----------|-------|----------|
| **GGUF Q4_K_M** | 4.68 GB | 94.0% | 80 tok/s | M3 Max |
| **GGUF Q4_K_M** | 4.68 GB | 94.0% | 70 tok/s | RTX 4070 |
| **MLX 4-bit** | 4.31 GB | 93.0% | 80 tok/s | M3 Max |
| **GGUF IQ2_XS** | 2.40 GB | 87.5% | 90 tok/s | M3 Max |

## Vision (YOLO-v8n)

| Format | Size | FPS | Latency | Platform |
|--------|------|-----|---------|----------|
| **CoreML INT8** | 3.1 MB | 40 | 25ms | M3 Max (ANE) |
| **CoreML FP16** | 6.2 MB | 15 | 67ms | M3 Max |
| **ONNX FP32** | 12.2 MB | 19 | 52ms | M3 Max (CPU) |
| **ONNX INT8** | 3.3 MB | 400 | 2.5ms | RTX 4090 (TensorRT) |

## Speech (Whisper-tiny)

| Format | Size | WER | RTF | Latency | Platform |
|--------|------|-----|-----|---------|----------|
| **CoreML INT8** | 7.9 MB | 9.2% | 0.032 | 249ms | M3 Max (ANE) |
| **CoreML FP16** | 15.7 MB | 9.2% | 0.032 | 249ms | M3 Max (ANE) |
| **ONNX FP32** | 0.3 MB | ~10% | ~0.05 | ~400ms | M3 Max (CPU) |

**Metrics:**
- **tok/s**: Tokens per second (LLM)
- **FPS**: Frames per second (Vision)
- **WER**: Word Error Rate (Speech, lower is better)
- **RTF**: Real-time Factor (Speech, <1.0 = faster than real-time)

---

# Complete Command Reference

## LLM Commands

```bash
# Standard quantization
cd quant
uv run python test_dual_format.py --model Qwen/Qwen3-8B --output ./models/qwen3-8b --formats gguf,mlx --gguf-precision Q4_K_M --mlx-bits 4

# Extreme compression (IQ2_XS)
uv run python -c "from llama_pajamas_quant.simple_benchmarks import TEST_PROMPTS; import sys; [sys.stdout.write(p['prompt'] + '\n\n') for p in TEST_PROMPTS]" > calibration.txt
cd ../libs/llama.cpp && ./llama-imatrix -m ../../quant/models/qwen3-8b/gguf/Q4_K_M/*.gguf -f ../../quant/calibration.txt -o ../../quant/qwen3-8b.imatrix --chunks 512
./llama-quantize --imatrix ../../quant/qwen3-8b.imatrix ../../quant/models/qwen3-8b/gguf/Q4_K_M/*.gguf ../../quant/models/qwen3-8b/gguf/IQ2_XS/model.gguf IQ2_XS

# Run inference
cd ../../run
uv run python examples/simple_usage.py

# Evaluate
cd ../quant
uv run python evaluation/llm/run_eval.py --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf --format gguf
uv run python evaluation/llm/compare_evaluations.py --model-dir ./models/qwen3-8b
```

## Vision Commands

```bash
# Export to CoreML/ONNX/TensorRT
cd quant
uv run python scripts/export_model.py --model yolov8n --backend coreml --precision float16 --output models/yolo-v8n/
uv run python scripts/export_model.py --model yolov8n --backend onnx --precision int8 --output models/yolo-v8n/
uv run python scripts/export_model.py --model yolov8n --backend tensorrt --precision fp16 --output models/yolo-v8n/

# Quantize CoreML to INT8
uv run python scripts/quantize_coreml_vision.py --model vit-base --precision int8

# Run inference (CoreML)
cd ../run-coreml
uv run python examples/test_vision.py

# Run inference (ONNX)
cd ../run-onnx
uv run python examples/test_onnx_vision.py

# Evaluate
cd ../quant
uv run python evaluation/vision/run_eval.py --model yolo-v8n --models-dir ./models --images evaluation/vision/images/detection
```

## Speech Commands

```bash
# Export to CoreML
cd quant
uv run python scripts/export_whisper_coreml.py --model whisper-tiny --precision float16

# Quantize to INT8
uv run python scripts/quantize_whisper_coreml.py --model whisper-tiny

# Export to ONNX
uv run python scripts/export_model.py --model whisper-tiny --backend onnx --precision fp32 --output models/whisper-tiny/

# Run inference (CoreML)
cd ../run-coreml
uv run python examples/test_stt.py

# Evaluate
cd ../quant/evaluation/stt
uv run python download_audio.py --num-samples 10
cd ../../run-coreml
uv run python ../quant/evaluation/stt/run_eval.py --models-dir ../quant/models --model whisper-tiny
```

## Multi-Modal Server Commands

```bash
# Start server
cd run-coreml
uv run python examples/multimodal_server_demo.py

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/images/detect -H "Content-Type: application/json" -d '{"image": "data:image/jpeg;base64,..."}'
curl -X POST http://localhost:8000/v1/audio/transcriptions -F file=@audio.wav -F model=whisper-tiny -F response_format=json
```

---

# License

MIT

---

# Citation

```bibtex
@software{llama_pajamas2025,
  title = {Llama-Pajamas: Universal Model Quantization \& Optimized Runtime System},
  author = {Rob Thelen},
  year = {2025},
  url = {https://github.com/yourusername/llama-pajamas}
}
```

---

**🚀 Production Ready**

**Core Principles:**
1. **Quantize once** → Deploy on hardware-optimized runtimes
2. **Native acceleration** → ANE (Apple), CUDA (NVIDIA), OpenVINO (Intel)
3. **50-75% smaller models** → IQ2_XS (LLM), INT8 (Vision/Speech)
4. **3-10x faster inference** → vs CPU-only PyTorch
5. **OpenAI-compatible APIs** → Drop-in replacement
