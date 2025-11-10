# STT (Speech-to-Text) Evaluation

## Models Exported

| Model | Parameters | Encoder Size (FP16) | Status |
|-------|-----------|---------------------|--------|
| **whisper-tiny** | 39M | 15.7 MB | ✅ Exported |
| **whisper-base** | 74M | 39.3 MB | ✅ Exported |
| **whisper-small** | 244M | 168.3 MB | ✅ Exported |

## Directory Structure

```
evaluation/stt/
├── README.md                # This file
├── test_samples.json        # Reference transcriptions
└── run_eval.py             # Evaluation script (TODO)
```

## Evaluation Metrics

- **WER (Word Error Rate)**: Primary accuracy metric
- **Latency**: Time to transcribe (ms)
- **Real-time Factor (RTF)**: Processing time / audio duration
- **Model Size**: CoreML package size

## Test Samples

10 reference samples across 3 categories:
- **Short** (3-4 sec): Quick phrases
- **Medium** (4-6 sec): Common sentences
- **Long** (6-7 sec): Complex statements

## Next Steps

1. **Generate Audio Files**: Use macOS `say` command or download samples
2. **Implement Evaluation Script**: `run_eval.py` with CoreML integration
3. **Quantize Models**: INT8/INT4 quantization for each encoder
4. **Benchmark**: Run evaluation on all models and precisions
5. **Compare**: Generate comparison reports (FP16 vs INT8 vs INT4)

## Quick Start (TODO)

```bash
# Generate synthetic audio samples
uv run python evaluation/stt/generate_audio.py

# Evaluate all models
uv run python evaluation/stt/run_eval.py \
    --models-dir ./models \
    --audio evaluation/stt/audio

# Quantize models
uv run python scripts/quantize_whisper.py --model all --precision int8

# Re-evaluate quantized models
uv run python evaluation/stt/run_eval.py \
    --models-dir ./models \
    --audio evaluation/stt/audio \
    --precision int8
```

## Hardware Requirements

- **Apple Silicon**: M1, M2, M3, M4 (for ANE acceleration)
- **iOS**: 15+ (for mobile deployment)
- **Compute Units**: ALL (CPU + GPU + ANE)

## Model Details

### Whisper Architecture

**Encoder** (exported to CoreML):
- Input: Mel-spectrogram features (80 filterbanks)
- Output: Audio embeddings
- Hardware: ANE-optimized on Apple Silicon

**Decoder** (Python):
- Input: Audio embeddings from encoder
- Output: Text tokens (autoregressive)
- Note: Decoder runs in Python due to autoregressive nature

### Export Details

Models exported using:
```bash
uv run python scripts/export_whisper_coreml.py --model all --precision float16
```

- **Format**: CoreML MLProgram
- **Precision**: FP16 (default), INT8 (quantized), INT4 (experimental)
- **Deployment Target**: iOS 15+
- **Compute Units**: ALL (enables ANE acceleration)

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
