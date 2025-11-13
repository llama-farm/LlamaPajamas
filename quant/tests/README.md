# LlamaPajamas E2E Runtime Tests

End-to-end tests for validating model export, quantization, and runtime execution.

## Quick Start

### Practical Runtime Tests (Recommended)

For quick validation of runtimes without full quantization pipeline:

```bash
# Test runtimes with pre-quantized models (Python 3.11+)
python test_runtime_e2e.py

# Or test just GGUF
python test_simple_gguf.py
```

These tests work with Python 3.11+ and minimal dependencies.

### Full Pipeline Tests

For complete export, quantization, and evaluation testing (requires Python 3.12+):

```bash
# Run all tests
python run_e2e_tests.py

# Run specific test
python run_e2e_tests.py --tests llm
python run_e2e_tests.py --tests vision
python run_e2e_tests.py --tests speech

# Skip platform-specific runtimes
python run_e2e_tests.py --skip-tensorrt --skip-mlx --skip-coreml
```

## Test Files

### Practical Runtime Tests
- `test_runtime_e2e.py` - Runtime tests with pre-quantized models (Python 3.11+)
- `test_simple_gguf.py` - Simple GGUF runtime test (Python 3.11+)
- `check_environment.py` - Environment checker

### Full Pipeline Tests (Python 3.12+)
- `run_e2e_tests.py` - Main test orchestrator
- `test_e2e_llm.py` - LLM tests (Qwen2.5-1.5B, GGUF/MLX)
- `test_e2e_vision.py` - Vision tests (YOLOv8n, CoreML/ONNX/TensorRT)
- `test_e2e_speech.py` - Speech tests (Whisper, CoreML/ONNX)

## Models Tested

### LLM
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Formats**: GGUF, MLX
- **Quantizations**: Q8_0, Q4_K_M, IQ2_XS, 8-bit, 4-bit

### Vision
- **Model**: YOLOv8n
- **Formats**: CoreML, ONNX, TensorRT
- **Quantizations**: FP16, INT8, INT4

### Speech
- **Model**: openai/whisper-tiny
- **Formats**: CoreML, ONNX
- **Quantizations**: FP16, INT8

## Requirements

### Python Version
- **Practical Runtime Tests**: Python 3.11+
- **Full Pipeline Tests**: Python 3.12+ (due to package dependencies)

### All Platforms
- 16GB RAM (recommended)
- 50GB disk space (for full tests, 10GB for runtime tests)

### macOS (Apple Silicon)
- MLX and CoreML tests
- M1/M2/M3/M4 chip

### Linux/Windows (NVIDIA GPU)
- TensorRT tests
- NVIDIA GPU (Turing or newer)
- CUDA 12.1+

## Documentation

See [E2E_RUNTIME_TESTS.md](../../docs/E2E_RUNTIME_TESTS.md) for detailed documentation.

## TensorRT Docker

```bash
cd ../docker
docker compose -f docker-compose.tensorrt.yml run --rm tensorrt-e2e-test
```

## CI/CD

Tests run automatically on:
- Push to main/develop
- Pull requests
- Daily schedule (2 AM UTC)
- Manual trigger

## Output

Results are saved to timestamped directories:
```
e2e_test_results/YYYYMMDD_HHMMSS/
├── e2e_test_summary.json
├── llm/
├── vision/
└── speech/
```

## Support

- Issues: https://github.com/llama-farm/LlamaPajamas/issues
- Docs: https://llamapajamas.readthedocs.io
