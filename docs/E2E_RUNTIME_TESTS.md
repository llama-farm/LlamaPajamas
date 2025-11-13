# End-to-End Runtime Tests

Comprehensive end-to-end tests for LlamaPajamas that validate model export, quantization, and runtime execution across all supported backends.

## Overview

The E2E test suite validates the complete pipeline for three model types:

1. **LLM (Language Models)** - Tests: Qwen2.5-1.5B-Instruct
   - Formats: GGUF, MLX
   - Quantizations: 8-bit (Q8_0), 4-bit (Q4_K_M), IQ2 (IQ2_XS)
   - Runtimes: llama-cpp-python (GGUF), mlx-lm (MLX)

2. **Vision (Object Detection)** - Tests: YOLOv8n
   - Formats: CoreML, ONNX, TensorRT
   - Quantizations: FP16, INT8, INT4 (CoreML only)
   - Runtimes: CoreML (macOS), ONNX Runtime, TensorRT (NVIDIA GPU)

3. **Speech (Speech-to-Text)** - Tests: Whisper-tiny
   - Formats: CoreML, ONNX
   - Quantizations: FP16, INT8
   - Runtimes: CoreML (macOS), ONNX Runtime

## Quick Start

### Run All Tests

```bash
cd quant/tests
python run_e2e_tests.py
```

### Run Specific Test Type

```bash
# LLM tests only
python run_e2e_tests.py --tests llm

# Vision tests only
python run_e2e_tests.py --tests vision

# Speech tests only
python run_e2e_tests.py --tests speech

# Multiple specific tests
python run_e2e_tests.py --tests llm vision
```

### Platform-Specific Options

```bash
# Skip TensorRT (no NVIDIA GPU)
python run_e2e_tests.py --skip-tensorrt

# Skip MLX (not on macOS)
python run_e2e_tests.py --skip-mlx

# Skip CoreML (not on macOS)
python run_e2e_tests.py --skip-coreml
```

### Custom Output Directory

```bash
python run_e2e_tests.py --output-dir ./my_test_results
```

### Keep Test Artifacts

```bash
python run_e2e_tests.py --no-cleanup
```

## Individual Test Scripts

Each test type can be run independently:

### LLM Tests

```bash
python test_e2e_llm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir ./llm_results \
  --no-cleanup
```

**What it tests:**
1. Downloads the model from HuggingFace
2. Exports to GGUF (FP16)
3. Exports to MLX (FP16)
4. Quantizes GGUF: Q8_0, Q4_K_M, IQ2_XS
5. Quantizes MLX: 8-bit, 4-bit
6. Tests each runtime with generation
7. Measures tokens/second performance

**Expected output:**
- Quantized models in various formats
- Runtime performance metrics
- Generation test results
- `e2e_test_results.json` with detailed results

### Vision Tests

```bash
python test_e2e_vision.py \
  --model yolov8n \
  --output-dir ./vision_results \
  --no-cleanup \
  --skip-tensorrt  # if no NVIDIA GPU
```

**What it tests:**
1. Downloads YOLOv8n from Ultralytics
2. Exports to CoreML (FP16) - macOS only
3. Quantizes CoreML: INT8, INT4
4. Exports to ONNX (FP32)
5. Quantizes ONNX: INT8
6. Exports to TensorRT: FP16, INT8 - NVIDIA GPU only
7. Tests each runtime with object detection
8. Measures FPS performance

**Expected output:**
- Quantized vision models
- FPS benchmarks
- Detection test results
- `e2e_test_results.json`

### Speech Tests

```bash
python test_e2e_speech.py \
  --model openai/whisper-tiny \
  --output-dir ./speech_results \
  --no-cleanup
```

**What it tests:**
1. Downloads Whisper-tiny from HuggingFace
2. Exports to CoreML (FP16) - macOS only
3. Quantizes CoreML: INT8
4. Exports to ONNX (FP32)
5. Quantizes ONNX: INT8
6. Tests each runtime with transcription
7. Measures Real-Time Factor (RTF)

**Expected output:**
- Quantized speech models
- RTF benchmarks
- Transcription test results
- `e2e_test_results.json`

## TensorRT Docker Testing

For TensorRT tests, use Docker to ensure proper environment setup.

### Prerequisites

1. **NVIDIA GPU** with compute capability 7.5+ (Turing, Ampere, Hopper, Ada, Blackwell)
2. **NVIDIA Driver** installed
3. **NVIDIA Container Toolkit** installed
4. **Docker** and **Docker Compose**

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build and Run TensorRT Tests

```bash
cd docker

# Build the TensorRT container
docker compose -f docker-compose.tensorrt.yml build

# Run vision tests only
docker compose -f docker-compose.tensorrt.yml run --rm tensorrt-e2e-test

# Run all tests (LLM, Vision, Speech)
docker compose -f docker-compose.tensorrt.yml run --rm tensorrt-all-tests

# Interactive shell for debugging
docker compose -f docker-compose.tensorrt.yml run --rm tensorrt-shell
```

### TensorRT Container Details

- **Base Image**: `nvcr.io/nvidia/tensorrt:25.10-py3`
- **CUDA Version**: 13.0.2
- **TensorRT Version**: Latest (25.10)
- **Python Version**: 3.10+
- **GPU Support**: Automatic via NVIDIA Container Runtime

## CI/CD Pipeline

The E2E tests are integrated into GitHub Actions and run automatically on:

- **Push to main/develop** - Validates runtime integrity
- **Pull Requests** - Prevents regressions
- **Daily Schedule** - Nightly comprehensive tests
- **Manual Trigger** - On-demand testing

### Pipeline Jobs

1. **e2e-cpu-linux** - CPU-only tests on Ubuntu
   - GGUF (CPU)
   - ONNX (CPU)

2. **e2e-macos-apple-silicon** - macOS M1 runner
   - MLX
   - CoreML
   - ONNX

3. **e2e-gpu-nvidia** - Self-hosted NVIDIA GPU runner
   - GGUF (GPU)
   - TensorRT
   - ONNX (GPU)

4. **e2e-tensorrt-docker** - TensorRT in Docker
   - Full TensorRT validation

5. **generate-report** - Aggregates results
   - Creates summary report
   - Comments on PRs

### Manually Trigger Pipeline

```bash
# Via GitHub CLI
gh workflow run e2e-runtime-tests.yml \
  -f test_type=all

# Or via web UI
# Actions → E2E Runtime Tests → Run workflow
```

## Understanding Test Results

### Test Output Structure

```
test_results/
├── e2e_test_summary.json      # Overall summary
├── llm/                        # LLM test results
│   ├── e2e_test_results.json
│   ├── gguf/
│   │   ├── fp16/
│   │   ├── Q8_0/
│   │   ├── Q4_K_M/
│   │   └── IQ2_XS/
│   └── mlx/
│       ├── fp16/
│       ├── 8bit/
│       └── 4bit/
├── vision/                     # Vision test results
│   ├── e2e_test_results.json
│   ├── coreml/
│   ├── onnx/
│   └── tensorrt/
└── speech/                     # Speech test results
    ├── e2e_test_results.json
    ├── coreml/
    └── onnx/
```

### Result JSON Format

```json
{
  "timestamp": "2025-01-15 10:30:00",
  "platform": "Darwin",
  "tests": [
    {
      "format": "GGUF",
      "precision": "Q4_K_M",
      "name": "4-bit",
      "model_path": "/path/to/model.gguf",
      "runtime": {
        "success": true,
        "elapsed": 2.5,
        "tokens_per_sec": 45.2,
        "response": "Generated text..."
      }
    }
  ]
}
```

### Performance Metrics

**LLM:**
- `tokens_per_sec` - Higher is better
- Typical range: 20-100 tokens/sec (CPU), 100-500 (GPU)

**Vision:**
- `fps` - Frames Per Second, higher is better
- Typical range: 10-50 FPS (CPU), 100-500 (GPU)

**Speech:**
- `rtf` - Real-Time Factor, lower is better
- RTF < 1.0 means faster than real-time
- Typical range: 0.1-0.5 RTF

## Troubleshooting

### Common Issues

#### 1. Model Download Failures

```bash
# Set HuggingFace token if needed
export HF_TOKEN=your_token_here

# Or use local cache
export HF_HOME=/path/to/cache
```

#### 2. CUDA Out of Memory (TensorRT)

```bash
# Reduce batch size or use smaller model
python test_e2e_vision.py --model yolov8n  # instead of yolov8s
```

#### 3. MLX Not Available

MLX only works on Apple Silicon (M1/M2/M3/M4):
```bash
# Auto-skip on non-macOS
python run_e2e_tests.py --skip-mlx
```

#### 4. TensorRT Container Issues

```bash
# Check GPU visibility
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check runtime
docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### 5. Slow Downloads

```bash
# Use mirrors or local cache
export HF_ENDPOINT=https://hf-mirror.com

# Or download separately
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

### Debug Mode

Run tests with verbose output:

```bash
# Python logging
export PYTHONVERBOSE=1

# Keep all files
python run_e2e_tests.py --no-cleanup

# Run single test
python test_e2e_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --no-cleanup
```

## Platform Requirements

### Linux (CPU)
- Python 3.11+
- GCC/G++ 11+
- 16GB RAM minimum
- 50GB disk space

### Linux (NVIDIA GPU)
- Above + NVIDIA GPU (Turing or newer)
- NVIDIA Driver 525+
- CUDA 12.1+
- TensorRT 8.6+

### macOS (Apple Silicon)
- macOS 13+
- M1/M2/M3/M4 chip
- Xcode Command Line Tools
- 16GB RAM minimum

### Windows (Experimental)
- Windows 10/11
- Python 3.11+
- Visual Studio 2019+
- (TensorRT requires WSL2 + NVIDIA GPU)

## Performance Baselines

Expected performance on reference hardware:

### LLM (Qwen2.5-1.5B)

| Platform | Format | Precision | Tokens/sec |
|----------|--------|-----------|------------|
| M1 Max | MLX | 4-bit | 85 |
| M1 Max | GGUF | Q4_K_M | 42 |
| RTX 4090 | GGUF | Q4_K_M | 350 |
| AMD Ryzen 9 (CPU) | GGUF | Q4_K_M | 28 |

### Vision (YOLOv8n)

| Platform | Format | Precision | FPS |
|----------|--------|-----------|-----|
| M1 Max | CoreML | INT8 | 180 |
| RTX 4090 | TensorRT | FP16 | 485 |
| RTX 4090 | TensorRT | INT8 | 720 |
| AMD Ryzen 9 (CPU) | ONNX | FP32 | 35 |

### Speech (Whisper-tiny)

| Platform | Format | Precision | RTF |
|----------|--------|-----------|-----|
| M1 Max | CoreML | INT8 | 0.08 |
| RTX 4090 | ONNX | FP32 | 0.05 |
| AMD Ryzen 9 (CPU) | ONNX | FP32 | 0.25 |

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Add platform detection
3. Include performance metrics
4. Update documentation
5. Add to CI/CD pipeline

Example test template:

```python
def test_new_runtime(self, model_path: Path, name: str) -> Dict:
    """Test new runtime with a model."""
    print(f"\n   Testing new runtime: {name}...")

    try:
        # Import backend
        from new_runtime import Backend

        # Load model
        backend = Backend(model_path=str(model_path))

        # Run inference
        start_time = time.time()
        result = backend.infer(test_input)
        elapsed = time.time() - start_time

        # Calculate metrics
        metric = calculate_metric(result, elapsed)

        print(f"✅ Runtime test passed")
        print(f"   Metric: {metric:.2f}")

        return {
            "success": True,
            "elapsed": elapsed,
            "metric": metric
        }
    except Exception as e:
        print(f"❌ Runtime test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
```

## License

These tests are part of LlamaPajamas and follow the same license.

## Support

- **Issues**: https://github.com/llama-farm/LlamaPajamas/issues
- **Discussions**: https://github.com/llama-farm/LlamaPajamas/discussions
- **Documentation**: https://llamapajamas.readthedocs.io
