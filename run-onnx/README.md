# llama-pajamas-run-onnx

ONNX Runtime for llama-pajamas with hardware-optimized execution providers.

## Overview

This package provides ONNX Runtime inference for llama-pajamas models with support for:

- **CoreML** (Apple Neural Engine on M1/M2/M3)
- **TensorRT** (NVIDIA GPUs) - Coming soon
- **CPU** (Optimized for Intel/AMD/ARM) - Coming soon

## Installation

```bash
# Base package (includes CoreML support on macOS)
pip install llama-pajamas-run-onnx

# Optional: CUDA/TensorRT support (NVIDIA GPUs)
# pip install llama-pajamas-run-onnx[tensorrt]  # Coming soon
```

## Quick Start

### 1. Convert Model to ONNX

First, convert your HuggingFace model to ONNX using the quantization pipeline:

```python
from llama_pajamas_quant.converters.onnx import ONNXConverter, TargetSpec

# Create converter
converter = ONNXConverter()

# Convert for CoreML (Apple Silicon)
target_specs = TargetSpec(
    target_eps=["CoreML"],
    target_precisions=["int8"],  # FP16, INT8 supported
    optimization_hints={
        "attention_type": "gqa",  # If GQA model
        "gqa_ratio": 4,           # 4:1 Q:KV ratio
    }
)

# Convert
results = converter.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b/onnx/",
    target_specs=target_specs,
)

# Output:
# ./models/qwen3-8b/onnx/
#   ├── manifest.json
#   ├── CoreML/
#   │   ├── fp16/model.onnx
#   │   └── int8/model.onnx
#   └── base/model.onnx
```

### 2. Run Inference

```python
from llama_pajamas_run_onnx import ONNXSession
from llama_pajamas_run_onnx.backends.base import GenerationConfig
from pathlib import Path

# Load from manifest (recommended)
session = ONNXSession.from_manifest(
    manifest_path=Path("./models/qwen3-8b/onnx/manifest.json"),
    ep="CoreML",       # Execution provider
    precision="int8",  # Target precision
)

# Or load directly from model path
session = ONNXSession.from_model_path(
    model_path=Path("./models/qwen3-8b/onnx/CoreML/int8/model.onnx"),
    ep="CoreML",
)

# Generate text
result = session.generate(
    prompt="Write a Python function to reverse a string",
    config=GenerationConfig(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
    ),
)

print(result.text)
print(f"Speed: {result.tokens_per_second:.1f} tok/s")
```

### 3. Streaming Generation

```python
# Stream tokens as they're generated
for token in session.generate_stream(
    prompt="Explain quantum computing",
    config=GenerationConfig(max_tokens=500),
):
    print(token, end="", flush=True)
```

### 4. Benchmarking

```python
# Benchmark performance
benchmark = session.benchmark(
    prompt="The quick brown fox jumps over the lazy dog",
    num_tokens=100,
)

print(f"Throughput: {benchmark['tokens_per_second']:.1f} tok/s")
print(f"Total time: {benchmark['total_time_ms']:.0f} ms")
```

## CoreML Backend (Apple Silicon)

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- ONNX Runtime >= 1.17.0
- Model quantized with INT8 symmetric (ANE requirement)

### Performance Targets

| Device      | Model Size | Precision | Expected Speed  |
|-------------|------------|-----------|-----------------|
| M3 Max      | Qwen3-8B   | INT8      | 50-70 tok/s     |
| M2 Max      | Qwen3-8B   | INT8      | 40-60 tok/s     |
| M1 Max      | Qwen3-8B   | INT8      | 30-50 tok/s     |
| M3 Max      | Qwen3-8B   | FP16      | 30-40 tok/s     |

### CoreML-Specific Options

```python
from llama_pajamas_run_onnx.backends import CoreMLBackend

backend = CoreMLBackend(
    model_path=Path("model.onnx"),
    provider_options={
        "use_ane": True,           # Use Apple Neural Engine
        "compute_units": "ALL",    # "CPU_AND_NE" | "CPU_ONLY" | "ALL"
    },
)
```

### CoreML Quantization Requirements

The Apple Neural Engine (ANE) requires:
- **INT8 symmetric quantization** (zero-point = 0)
- **Per-channel** quantization for weights
- **Per-tensor** quantization for activations
- **No INT4 support** (ANE limitation)

This is automatically handled by the ONNXQuantizer when targeting CoreML.

## TensorRT Backend (Coming Soon)

### Features

- INT8/INT4 QDQ quantization
- FP16/FP8 mixed precision
- 100+ tok/s on RTX 4070 (Qwen3-8B INT8)
- 130+ tok/s on RTX 4090 (Qwen3-8B INT4)

### Example (Preview)

```python
# Convert for TensorRT
target_specs = TargetSpec(
    target_eps=["TensorRT"],
    target_precisions=["int8", "int4"],
)

converter.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b/onnx/",
    target_specs=target_specs,
)

# Run on TensorRT
session = ONNXSession.from_manifest(
    manifest_path=Path("./models/qwen3-8b/onnx/manifest.json"),
    ep="TensorRT",
    precision="int4",
)
```

## CPU Backend (Coming Soon)

### Features

- INT4 MatMulNBits operators
- INT8 dynamic quantization
- Optimized for AVX2/AVX-512/NEON

## Architecture

### Pipeline-First Design

llama-pajamas uses a **pipeline-first** architecture where users specify target hardware during conversion, not at runtime:

```python
# ✅ CORRECT: User specifies targets
target_specs = TargetSpec(
    target_eps=["CoreML", "TensorRT"],  # Generate for both
    target_precisions=["int8"],
    optimization_hints={
        "attention_type": "gqa",   # Model architecture hints
        "gqa_ratio": 4,
    }
)

# ❌ WRONG: No auto-detection at conversion time
# The pipeline doesn't detect architecture - users specify what they want
```

### Backend Architecture

```
ONNXSession
    ↓
Backend (Abstract)
    ├── CoreMLBackend (Apple ANE)
    ├── TensorRTBackend (NVIDIA) - Coming soon
    └── CPUBackend (Universal) - Coming soon
```

Each backend implements:
- `_load_model()`: Load ONNX model with EP
- `generate()`: Text generation
- `generate_stream()`: Streaming generation
- `benchmark()`: Performance testing

## API Reference

### ONNXSession

High-level session manager:

```python
session = ONNXSession.from_manifest(
    manifest_path: Path,
    ep: str,                        # CoreML, TensorRT, CPU
    precision: str = "fp16",        # fp16, int8, int4
    provider_options: Dict = None,  # EP-specific options
)

session = ONNXSession.from_model_path(
    model_path: Path,
    ep: str,
    provider_options: Dict = None,
)

result: GenerationResult = session.generate(
    prompt: str,
    config: GenerationConfig = None,
)

tokens: Iterator[str] = session.generate_stream(
    prompt: str,
    config: GenerationConfig = None,
)

benchmark: Dict = session.benchmark(
    prompt: str = "...",
    num_tokens: int = 100,
)
```

### GenerationConfig

```python
from llama_pajamas_run_onnx.backends.base import GenerationConfig

config = GenerationConfig(
    max_tokens=512,           # Max tokens to generate
    temperature=0.7,          # Sampling temperature (0.0 = greedy)
    top_p=0.9,                # Nucleus sampling
    top_k=50,                 # Top-k sampling
    repetition_penalty=1.0,   # Repetition penalty
    stream=False,             # Enable streaming
)
```

### GenerationResult

```python
result = session.generate(prompt, config)

result.text                  # Generated text (str)
result.tokens                # Token IDs (List[int])
result.num_tokens            # Number of tokens (int)
result.tokens_per_second     # Generation speed (float)
result.metadata              # Additional info (Dict)
```

## Development

### Running Tests

```bash
# Test ONNX converter
cd quant/
python test_onnx_converter.py --model Qwen/Qwen3-0.6B

# Test CoreML backend
cd run-onnx/
python -m llama_pajamas_run_onnx.backends.coreml_backend \
    models/qwen3-0.6b/onnx/CoreML/fp16/model.onnx \
    "Hello, how are you?"
```

### Project Structure

```
run-onnx/
├── llama_pajamas_run_onnx/
│   ├── __init__.py
│   ├── session.py              # ONNXSession (high-level API)
│   └── backends/
│       ├── __init__.py
│       ├── base.py             # Backend abstract base class
│       ├── coreml_backend.py   # CoreML backend (✅ Complete)
│       ├── tensorrt_backend.py # TensorRT backend (⏳ Coming soon)
│       └── cpu_backend.py      # CPU backend (⏳ Coming soon)
├── pyproject.toml
└── README.md
```

## Roadmap

### Week 1 (Current)
- ✅ CoreML FP16 export
- ✅ CoreML INT8 symmetric quantization
- ✅ CoreML backend runtime
- ⏳ Benchmark on Mac M3 Max (target: 50-70 tok/s)

### Week 2
- ⏳ TensorRT INT8/INT4 export
- ⏳ TensorRT backend runtime
- ⏳ Multi-EP generation (single convert → multiple EPs)

### Week 3
- ⏳ CPU INT4 MatMulNBits
- ⏳ Edge device optimization (Jetson)

### Week 4
- ⏳ Unified CLI (GGUF + MLX + ONNX)
- ⏳ Quality validation suite
- ⏳ v0.2.0 release

## License

Same as llama-pajamas project.

## Contributing

See main llama-pajamas CONTRIBUTING.md.
