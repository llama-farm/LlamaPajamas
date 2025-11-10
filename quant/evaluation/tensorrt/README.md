# TensorRT Model Evaluation

Evaluation pipeline for TensorRT-optimized models on NVIDIA GPUs.

## Overview

This directory contains evaluation tools for TensorRT models, parallel to the CoreML evaluation pipeline.

**Supported Tasks**:
- LLM inference benchmarking (tokens/sec, latency)
- Vision model evaluation (FPS, mAP, accuracy)
- Quantization quality assessment (INT8 vs FP16)
- Multi-GPU scalability testing

## Quick Start

### Benchmark LLM

```bash
# Benchmark single model
uv run python evaluation/tensorrt/benchmark_llm.py \
    --model models/qwen3-8b/tensorrt/int8/model.engine

# Benchmark all TensorRT LLMs
uv run python evaluation/tensorrt/benchmark_llm.py --model all
```

### Benchmark Vision

```bash
# Benchmark object detection
uv run python evaluation/tensorrt/benchmark_vision.py \
    --model models/yolo-v8n/tensorrt/int8/model.engine \
    --task detection

# Benchmark classification
uv run python evaluation/tensorrt/benchmark_vision.py \
    --model models/vit-base/tensorrt/fp16/model.engine \
    --task classification
```

## Metrics

### LLM Metrics
- **Tokens/second**: Throughput (higher is better)
- **Latency**: Time to first token (ms)
- **Memory**: GPU VRAM usage (GB)
- **Batch efficiency**: Throughput scaling with batch size

### Vision Metrics
- **FPS**: Frames per second
- **mAP**: Mean average precision (detection)
- **Top-1/Top-5**: Classification accuracy
- **Latency**: Inference time per image (ms)

## Evaluation Pipeline

### 1. Export Models
```bash
# Export LLM to TensorRT
cd quant
uv run python scripts/export_tensorrt_llm.py --model qwen3-8b --dtype int8

# Export vision model
uv run python scripts/export_tensorrt_vision.py --model yolo-v8n --precision int8
```

### 2. Benchmark Performance
```bash
# LLM benchmark
uv run python evaluation/tensorrt/benchmark_llm.py --model all

# Vision benchmark
uv run python evaluation/tensorrt/benchmark_vision.py --model all
```

### 3. Compare Results
```bash
# Compare TensorRT vs CoreML
uv run python evaluation/compare_backends.py \
    --backend1 tensorrt \
    --backend2 coreml \
    --model yolo-v8n
```

## Directory Structure

```
evaluation/tensorrt/
├── README.md                    # This file
├── benchmark_llm.py             # LLM benchmarking
├── benchmark_vision.py          # Vision benchmarking
├── calibration/                 # INT8 calibration datasets
├── results/                     # Benchmark results (JSON)
└── scripts/                     # Helper scripts
    ├── generate_report.py       # Generate comparison reports
    └── plot_results.py          # Visualize benchmarks
```

## Expected Performance

### LLM (NVIDIA RTX 4090)
| Model | Precision | Batch | Throughput | Latency | VRAM |
|-------|-----------|-------|------------|---------|------|
| Qwen3-8B | INT8 | 1 | 120 tok/s | 8ms | 6 GB |
| Qwen3-8B | INT8 | 8 | 800 tok/s | 10ms | 8 GB |
| Qwen3-8B | FP16 | 1 | 80 tok/s | 12ms | 16 GB |
| Qwen3-8B | INT4 | 1 | 150 tok/s | 7ms | 4 GB |

### Vision (NVIDIA RTX 4090)
| Model | Precision | Batch | FPS | Latency | VRAM |
|-------|-----------|-------|-----|---------|------|
| YOLO-v8n | INT8 | 1 | 400 | 2.5ms | 500 MB |
| YOLO-v8n | INT8 | 8 | 2000 | 4ms | 1 GB |
| ViT-Base | FP16 | 1 | 150 | 7ms | 2 GB |

## Comparison: TensorRT vs CoreML

### LLM
| Metric | TensorRT (RTX 4090) | CoreML (M3 Max) | Speedup |
|--------|---------------------|-----------------|---------|
| Throughput | 120 tok/s | 80 tok/s | 1.5x |
| Latency | 8ms | 12ms | 1.5x |
| Batch Size | Up to 128 | Up to 8 | 16x |

### Vision
| Metric | TensorRT (RTX 4090) | CoreML (M3 Max) | Speedup |
|--------|---------------------|-----------------|---------|
| YOLO FPS | 400 | 40 | 10x |
| ViT FPS | 150 | 20 | 7.5x |

## Implementation Status

### ✅ Completed
- Benchmark script templates
- Evaluation pipeline structure
- Metadata schemas
- Documentation

### 🚧 In Progress
- TensorRT-LLM integration
- CUDA preprocessing
- Calibration dataset generation

### 📋 TODO
- Implement actual benchmarking (requires TensorRT-LLM)
- Add visualization scripts
- Multi-GPU testing
- Continuous integration

## Requirements

**Hardware**:
- NVIDIA GPU (RTX 3060+, A100, H100)
- 8+ GB VRAM (for 8B models)

**Software**:
- CUDA 11.8+ or 12.0+
- cuDNN 8.9+
- TensorRT 8.6+
- TensorRT-LLM (for LLM inference)

**Install**:
```bash
pip install tensorrt>=8.6.0
pip install tensorrt-llm
pip install pycuda
```

## Next Steps

1. **Install TensorRT-LLM**
   - Follow: https://github.com/NVIDIA/TensorRT-LLM

2. **Export Models**
   - Run export scripts in `quant/scripts/`

3. **Run Benchmarks**
   - Execute benchmark scripts
   - Compare with CoreML results

4. **Generate Reports**
   - Use comparison scripts
   - Create visualization plots

## Resources

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA Performance Optimization](https://docs.nvidia.com/deeplearning/performance/)
- [CoreML Evaluation](../coreml/README.md)

## Related

- [TensorRT Runtime](../../run-tensorrt/README.md)
- [CoreML Evaluation](../coreml/README.md)
- [Quantization Guide](../../quant/README.md)
