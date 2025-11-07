# Llama-Pajamas: Pipeline & Runtime Separation

## Executive Summary

**Llama-Pajamas consists of TWO separate components**:

1. **Quantization Pipeline** (`llama-pajamas-quant`): Converts models
   - Input: HuggingFace models (FP16/BF16)
   - Output: Quantized models (GGUF, MLX)
   - Heavy dependencies (llama.cpp, calibration datasets, validation)
   - Used once per model (CI/CD, developer workstation)

2. **Inference Runtime** (`llama-pajamas-run`): Runs models
   - Input: Quantized models (GGUF, MLX)
   - Output: Text generation, embeddings
   - Lightweight (only inference libraries)
   - Deployed everywhere (production, edge, mobile)

**Why Separate?**
- ✅ **Modularity**: Use runtime without pipeline (consume pre-quantized models)
- ✅ **Deployment**: Runtime is 10x smaller (no conversion tools)
- ✅ **CI/CD**: Pipeline runs once, runtime deployed many times
- ✅ **Licensing**: Runtime pure MIT, pipeline may include GPL tools
- ✅ **Performance**: Runtime optimized for inference only

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION PIPELINE                         │
│                   (llama-pajamas-quant)                          │
│                                                                   │
│  Input: HuggingFace Models (FP16/BF16)                          │
│                                                                   │
│  ┌────────────┐   ┌──────────────┐   ┌────────────────┐        │
│  │ HF Cache   │──▶│ Quantizer    │──▶│ Model Registry │        │
│  │ Reader     │   │ Engine       │   │ & Metadata     │        │
│  └────────────┘   └──────────────┘   └────────────────┘        │
│                           │                                       │
│                           ├──────────┬──────────┐               │
│                           ▼          ▼          ▼               │
│                    ┌──────────┐ ┌───────┐ ┌─────────┐          │
│                    │   GGUF   │ │  MLX  │ │ Future  │          │
│                    │ Generator│ │ Gen.  │ │ Formats │          │
│                    └──────────┘ └───────┘ └─────────┘          │
│                           │          │          │               │
│  Output: Quantized Models (GGUF, MLX, ...)                      │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ Artifact Transfer
                             │ (filesystem, registry, S3, etc.)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE RUNTIME                            │
│                    (llama-pajamas-run)                           │
│                                                                   │
│  Input: Quantized Models (GGUF, MLX)                            │
│                                                                   │
│  ┌────────────┐   ┌──────────────┐   ┌────────────────┐        │
│  │ Model      │──▶│ Backend      │──▶│ Inference      │        │
│  │ Loader     │   │ Manager      │   │ Engine         │        │
│  └────────────┘   └──────────────┘   └────────────────┘        │
│                           │                                       │
│                           ├──────────┬──────────┐               │
│                           ▼          ▼          ▼               │
│                    ┌──────────┐ ┌───────┐ ┌─────────┐          │
│                    │   GGUF   │ │  MLX  │ │ Future  │          │
│                    │ Backend  │ │Backend│ │ Backends│          │
│                    └──────────┘ └───────┘ └─────────┘          │
│                                                                   │
│  Output: Generated Text, Embeddings, etc.                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Quantization Pipeline

### Purpose

Convert models from HuggingFace (FP16/BF16) to quantized formats (GGUF, MLX).

### Responsibilities

1. **Model Discovery**: Read from HuggingFace cache or download
2. **Architecture Detection**: Identify model type (dense, MoE, etc.)
3. **Quantization**: Convert to target precision (INT4, INT8, etc.)
4. **Calibration**: Use calibration data for optimal quantization
5. **Validation**: Measure quality (perplexity, benchmarks)
6. **Export**: Write quantized models with metadata

### Dependencies (Heavy)

```yaml
Required:
  - transformers (HuggingFace)
  - torch (model loading)
  - safetensors (checkpoint loading)
  - llama.cpp conversion scripts (HF → GGUF)
  - llama-quantize binary (GGUF quantization)
  - mlx + mlx-lm (MLX quantization)
  - datasets (calibration data: C4, WikiText)
  - numpy, scipy (numerical operations)

Optional:
  - optimum (ONNX export)
  - bitsandbytes (advanced quantization)
  - accelerate (multi-GPU)

Size: ~5GB installed
```

### Project Structure

```
llama-pajamas-quant/
├── pyproject.toml
├── src/
│   └── llama_pajamas_quant/
│       ├── __init__.py
│       ├── __main__.py        # CLI entry point
│       │
│       ├── core/
│       │   ├── quantizer.py      # Main quantization engine
│       │   ├── calibrator.py     # Calibration data handling
│       │   ├── validator.py      # Quality validation
│       │   ├── detector.py       # Architecture detection
│       │   └── registry.py       # Model metadata registry
│       │
│       ├── converters/
│       │   ├── gguf.py           # GGUF conversion
│       │   ├── mlx.py            # MLX conversion
│       │   └── base.py           # Converter interface
│       │
│       ├── cache/
│       │   ├── hf_cache.py       # HuggingFace cache integration
│       │   └── download.py       # Model download
│       │
│       └── cli/
│           ├── main.py           # Main CLI
│           ├── quantize.py       # Quantize command
│           ├── validate.py       # Validate command
│           └── info.py           # Model info command
│
├── libs/
│   ├── llama.cpp/                # Git submodule
│   └── mlx-lm/                   # MLX language models
│
└── tests/
    ├── test_quantizer.py
    ├── test_converters.py
    └── test_validation.py
```

### CLI Interface

```bash
# Quantize a model (primary use case)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --precision q4_k_m:4bit \
  --validate

# Output structure:
# ./models/qwen3-8b/
#   ├── gguf/
#   │   ├── qwen3-8b-q4_k_m.gguf
#   │   └── metadata.json
#   ├── mlx/
#   │   ├── weights.safetensors
#   │   ├── config.json
#   │   └── metadata.json
#   └── manifest.json            # Overall model card

# Validate quality
llama-pajamas-quant validate \
  --original Qwen/Qwen3-8B \
  --quantized ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --metrics perplexity,mmlu

# Show model info
llama-pajamas-quant info Qwen/Qwen3-8B

# List available quantization methods
llama-pajamas-quant list-methods
```

### Python API

```python
from llama_pajamas_quant import Quantizer, QuantConfig

# Configure quantization
config = QuantConfig(
    formats=["gguf", "mlx"],
    gguf_method="Q4_K_M",
    mlx_bits=4,
    mlx_group_size=64,
    calibration_samples=256,
    validate=True
)

# Quantize
quantizer = Quantizer(config)
result = quantizer.convert(
    model="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b"
)

# Access results
print(f"GGUF: {result.gguf_path}")
print(f"MLX: {result.mlx_path}")
print(f"Compression: {result.compression_ratio:.1f}x")
print(f"Quality loss: {result.perplexity_increase:.1f}%")
```

### Output Format (Model Artifact)

```yaml
# manifest.json (root model descriptor)
{
  "model_id": "Qwen/Qwen3-8B",
  "quantized_at": "2025-01-15T10:30:00Z",
  "pipeline_version": "0.1.0",
  "architecture": {
    "family": "dense_decoder",
    "type": "qwen3",
    "params_total": "8.2B",
    "params_non_embedding": "6.95B"
  },
  "formats": [
    {
      "type": "gguf",
      "path": "gguf/qwen3-8b-q4_k_m.gguf",
      "method": "Q4_K_M",
      "size_mb": 1847,
      "compression": "8.9x"
    },
    {
      "type": "mlx",
      "path": "mlx/",
      "method": "4bit_mixed",
      "size_mb": 1623,
      "compression": "10.2x"
    }
  ],
  "validation": {
    "perplexity_fp16": 8.23,
    "perplexity_quantized": 8.56,
    "increase_pct": 4.0,
    "mmlu_fp16": 74.2,
    "mmlu_quantized": 71.8
  }
}

# gguf/metadata.json (GGUF-specific)
{
  "format": "gguf",
  "quantization_method": "Q4_K_M",
  "group_size": 128,
  "llama_cpp_version": "b1234",
  "compatible_backends": ["cpu", "cuda", "metal", "rocm"],
  "runtime_requirements": "llama-pajamas-run >= 0.1.0"
}

# mlx/metadata.json (MLX-specific)
{
  "format": "mlx",
  "quantization": {
    "body_bits": 4,
    "embedding_bits": 6,
    "group_size": 64
  },
  "mlx_version": "0.29.4",
  "compatible_backends": ["mlx"],
  "runtime_requirements": "llama-pajamas-run[mlx] >= 0.1.0"
}
```

---

## Component 2: Inference Runtime

### Purpose

Load and run quantized models for inference.

### Responsibilities

1. **Model Loading**: Load GGUF or MLX models from disk
2. **Backend Selection**: Auto-detect or manually choose backend
3. **Inference**: Generate text, embeddings, etc.
4. **API**: Provide Python API and optional REST server

### Dependencies (Lightweight)

```yaml
Required:
  - llama-cpp-python (GGUF inference)
  - mlx + mlx-lm (MLX inference, Mac only)
  - numpy (basic operations)

Optional:
  - fastapi (REST API server)
  - uvicorn (ASGI server)
  - openai (OpenAI-compatible API)

Size: ~500MB installed (vs 5GB for pipeline)
```

### Project Structure

```
llama-pajamas-run/
├── pyproject.toml
├── src/
│   └── llama_pajamas_run/
│       ├── __init__.py
│       ├── __main__.py         # CLI entry point
│       │
│       ├── runtime/
│       │   ├── backend.py         # IBackend interface
│       │   ├── manager.py         # Backend manager
│       │   ├── loader.py          # Model loader
│       │   └── inference.py       # Inference engine
│       │
│       ├── backends/
│       │   ├── gguf_backend.py    # GGUF via llama.cpp
│       │   ├── mlx_backend.py     # MLX backend
│       │   └── base.py            # Base backend
│       │
│       ├── api/
│       │   ├── server.py          # REST API server
│       │   ├── openai.py          # OpenAI-compatible
│       │   └── client.py          # Python client
│       │
│       └── cli/
│           ├── main.py            # Main CLI
│           ├── run.py             # Run command
│           ├── serve.py           # Serve command
│           └── benchmark.py       # Benchmark command
│
└── tests/
    ├── test_backends.py
    ├── test_inference.py
    └── test_api.py
```

### CLI Interface

```bash
# Run inference (primary use case)
llama-pajamas-run generate \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --prompt "Write a Python function to calculate fibonacci" \
  --max-tokens 500

# Interactive mode
llama-pajamas-run chat \
  --model ./models/qwen3-8b/mlx/ \
  --backend mlx

# Start REST API server
llama-pajamas-run serve \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --port 8080 \
  --api-style openai

# Benchmark
llama-pajamas-run benchmark \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend cuda \
  --num-tokens 500

# Auto-detect backend
llama-pajamas-run generate \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend auto \
  --prompt "Hello world"
```

### Python API

```python
from llama_pajamas_run import Runtime, InferenceConfig

# Load model (auto-detects backend)
runtime = Runtime(
    model_path="./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf",
    backend="auto"  # or "cuda", "mlx", "cpu"
)

# Configure generation
config = InferenceConfig(
    max_tokens=500,
    temperature=0.7,
    top_p=0.9,
    stream=True
)

# Generate (streaming)
for token in runtime.generate("Write a story about AI", config):
    print(token, end="", flush=True)

# Generate (non-streaming)
result = runtime.generate("What is Python?", config)
print(result.text)
print(f"Speed: {result.tokens_per_second:.1f} tok/s")

# Unload model
runtime.unload()
```

### REST API (OpenAI-compatible)

```python
# Start server
llama-pajamas-run serve \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --port 8080 \
  --api-style openai

# Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[
        {"role": "user", "content": "What is quantum computing?"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

---

## Workflow: Pipeline → Runtime

### Development Workflow

```bash
# STEP 1: Quantize model (run once, heavy)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --precision q4_k_m:4bit \
  --validate

# Output: ./models/qwen3-8b/{gguf,mlx}/

# STEP 2: Run model (run many times, lightweight)
llama-pajamas-run chat \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend auto
```

### CI/CD Workflow

```yaml
# .github/workflows/quantize-models.yml
name: Quantize Models

on:
  workflow_dispatch:
    inputs:
      model_id:
        description: 'HuggingFace model ID'
        required: true
        default: 'Qwen/Qwen3-8B'

jobs:
  quantize:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v4

      - name: Install Pipeline
        run: pip install llama-pajamas-quant

      - name: Quantize Model
        run: |
          llama-pajamas-quant convert \
            --model ${{ inputs.model_id }} \
            --output ./artifacts \
            --formats gguf,mlx \
            --precision q4_k_m:4bit \
            --validate

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: quantized-models
          path: ./artifacts/

      - name: Push to S3
        run: aws s3 sync ./artifacts/ s3://my-models/
```

### Production Deployment

```dockerfile
# Dockerfile (production - runtime only)
FROM python:3.11-slim

# Install ONLY runtime (lightweight)
RUN pip install llama-pajamas-run[cuda]

# Copy pre-quantized model from S3/artifact store
COPY --from=artifacts /models/qwen3-8b /models/qwen3-8b

# Start server
CMD ["llama-pajamas-run", "serve", \
     "--model", "/models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf", \
     "--port", "8080"]
```

---

## Interface Between Pipeline and Runtime

### Model Artifact Contract

The pipeline produces a **standard artifact format** that the runtime consumes:

```
model-name/
├── manifest.json          # ← Runtime reads this first
├── gguf/
│   ├── model-*.gguf
│   └── metadata.json
└── mlx/
    ├── weights.safetensors
    ├── config.json
    └── metadata.json
```

**Runtime reads `manifest.json` to discover available formats:**

```python
# Runtime model loader
import json
from pathlib import Path

class ModelLoader:
    def load(self, model_path: Path, backend: str = "auto"):
        # Read manifest
        manifest = json.loads((model_path / "manifest.json").read_text())

        # Find compatible format
        if backend == "auto":
            backend = self._detect_best_backend()

        # Load appropriate format
        for fmt in manifest["formats"]:
            if fmt["type"] == "gguf" and backend in ["cpu", "cuda", "metal"]:
                return self._load_gguf(model_path / fmt["path"])
            elif fmt["type"] == "mlx" and backend == "mlx":
                return self._load_mlx(model_path / fmt["path"])

        raise RuntimeError(f"No compatible format for backend: {backend}")
```

### Version Compatibility

```yaml
# manifest.json includes compatibility requirements
{
  "runtime_requirements": {
    "llama-pajamas-run": ">=0.1.0,<1.0.0",
    "llama-cpp-python": ">=0.2.0",
    "mlx": ">=0.29.0"
  },
  "format_versions": {
    "gguf": "3",
    "mlx": "1"
  }
}
```

**Runtime checks compatibility on load:**

```python
import importlib.metadata

def check_compatibility(manifest):
    runtime_version = importlib.metadata.version("llama-pajamas-run")
    required = manifest["runtime_requirements"]["llama-pajamas-run"]

    if not version_matches(runtime_version, required):
        raise RuntimeError(
            f"Runtime {runtime_version} incompatible with model "
            f"(requires {required})"
        )
```

---

## Packaging Strategy

### Pipeline Package

```toml
# llama-pajamas-quant/pyproject.toml
[project]
name = "llama-pajamas-quant"
version = "0.1.0"
description = "Quantization pipeline for LLM models"
dependencies = [
    "transformers>=4.35.0",
    "torch>=2.1.0",
    "safetensors>=0.4.0",
    "datasets>=2.14.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
mlx = ["mlx>=0.29.0", "mlx-lm>=0.2.0"]
onnx = ["optimum>=1.14.0", "onnxruntime>=1.16.0"]
full = ["llama-pajamas-quant[mlx,onnx]"]

[project.scripts]
llama-pajamas-quant = "llama_pajamas_quant.cli.main:main"
```

### Runtime Package

```toml
# llama-pajamas-run/pyproject.toml
[project]
name = "llama-pajamas-run"
version = "0.1.0"
description = "Lightweight inference runtime for quantized LLMs"
dependencies = [
    "llama-cpp-python>=0.2.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
mlx = ["mlx>=0.29.0", "mlx-lm>=0.2.0"]
cuda = ["llama-cpp-python[cuda]"]
rocm = ["llama-cpp-python[rocm]"]
server = ["fastapi>=0.104.0", "uvicorn>=0.24.0"]
full = ["llama-pajamas-run[mlx,cuda,server]"]

[project.scripts]
llama-pajamas-run = "llama_pajamas_run.cli.main:main"
```

### Installation Examples

```bash
# Developer (needs both)
pip install llama-pajamas-quant[full]
pip install llama-pajamas-run[full]

# Production (runtime only)
pip install llama-pajamas-run[cuda,server]

# Mac developer
pip install llama-pajamas-quant[mlx]
pip install llama-pajamas-run[mlx]

# CI/CD (pipeline only)
pip install llama-pajamas-quant
```

---

## Advantages of Separation

### 1. Deployment Efficiency

| Scenario | Pipeline | Runtime | Total |
|----------|----------|---------|-------|
| **Development** | ✅ Installed | ✅ Installed | 5.5GB |
| **Production** | ❌ Not needed | ✅ Installed | 500MB |
| **CI/CD** | ✅ Installed | ❌ Not needed | 5GB |
| **Edge Device** | ❌ Not needed | ✅ Installed | 500MB |

**Savings**: 10x smaller deployment size

### 2. Security & Licensing

```yaml
Pipeline:
  - May include GPL tools (llama.cpp)
  - Heavy dependencies (PyTorch)
  - Not security-critical (offline use)

Runtime:
  - Pure MIT/Apache (clean licensing)
  - Minimal dependencies
  - Security-critical (handles user input)
```

### 3. Development Velocity

```yaml
Use Cases:
  - Model experimentation: Pipeline only
  - Production deployment: Runtime only
  - Testing different backends: Runtime only
  - Creating new quantized models: Pipeline only
  - API development: Runtime only
```

### 4. User Flexibility

```bash
# User A: Just wants to run pre-quantized models
pip install llama-pajamas-run
llama-pajamas-run chat --model ./model.gguf

# User B: Wants to quantize custom models
pip install llama-pajamas-quant
llama-pajamas-quant convert --model my-org/my-model

# User C: Full development
pip install llama-pajamas-quant llama-pajamas-run
# Use both as needed
```

---

## Implementation Phases

### Phase 1: Pipeline MVP (Week 1-2)

**Focus**: Core quantization functionality

```bash
# Deliverable
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --validate
```

**Success**: Produces valid GGUF and MLX artifacts

### Phase 2: Runtime MVP (Week 2-3)

**Focus**: Load and run quantized models

```bash
# Deliverable
llama-pajamas-run chat \
  --model ./models/qwen3-8b/gguf/model.gguf \
  --backend auto
```

**Success**: Runs on CUDA (Linux) and MLX (Mac)

### Phase 3: Polish (Week 3-4)

**Focus**: Integration, docs, testing

- [x] End-to-end workflow tested
- [x] Documentation complete
- [x] CI/CD examples
- [x] Docker images
- [x] PyPI packages

---

## Summary

**Clear Separation**:
- **Pipeline** = Heavy, offline, converts models
- **Runtime** = Light, online, runs models

**Benefits**:
- ✅ 10x smaller production deployments
- ✅ Faster iteration (runtime-only changes)
- ✅ Better security posture (clean runtime)
- ✅ User flexibility (use one or both)

**Interface**: Standard artifact format (manifest.json + formats)

**Ready to implement**: Clear architecture, separate repos/packages, defined interface.
