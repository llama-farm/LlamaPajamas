# Llama-Pajamas Quantization Examples

This directory contains example configurations and workflows for the llama-pajamas quantization pipeline.

## Batch Processing

### Basic Batch Quantization

Process multiple models in parallel:

```bash
cd quant
llama-pajamas-quant batch --config examples/batch-config.yaml --parallel 2
```

**What it does:**
- Processes 3 models in parallel (2 at a time)
- Generates GGUF and MLX formats
- Saves to `./models/` directory

### IQ Batch Quantization

For IQ quantization, use a workflow approach:

```bash
# 1. Generate calibration data
llama-pajamas-quant iq generate-calibration \
    --output calibration.txt \
    --num-samples 512

# 2. Quantize base model to Q4_K_M first
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf \
    --gguf-precision Q4_K_M \
    --output ./models/qwen3-8b

# 3. Generate importance matrix (reuse for all IQ variants)
llama-pajamas-quant iq generate-matrix \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --calibration calibration.txt \
    --output qwen3-8b.imatrix

# 4. Quantize to multiple IQ precisions
for precision in IQ2_XS IQ2_XXS IQ3_XS IQ3_M IQ4_XS; do
  llama-pajamas-quant iq quantize \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --imatrix qwen3-8b.imatrix \
    --precision $precision \
    --output ./models/qwen3-8b/gguf/$precision/
done
```

## Configuration Files

### batch-config.yaml

Standard batch processing for multiple models.

**Features:**
- Parallel processing
- Multiple formats (GGUF, MLX)
- Configurable precision

**Settings:**
```yaml
parallel: 2              # Process 2 models at a time
skip_existing: true      # Skip if output exists
evaluate: false          # Don't run evaluation
```

### batch-iq-config.yaml

IQ quantization workflow example.

**Workflow:**
1. Quantize to Q4_K_M (base model)
2. Generate calibration data
3. Generate importance matrix
4. Quantize to IQ variants

## End-to-End Examples

### Example 1: Quick Test (Small Model)

```bash
# Use small model for testing (faster)
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-0.6B \
    --formats gguf \
    --gguf-precision Q4_K_M \
    --output ./test-models/qwen3-0.6b \
    --no-benchmark

# IQ quantization test
llama-pajamas-quant iq generate-calibration \
    --output test-calibration.txt \
    --num-samples 100

llama-pajamas-quant iq quantize \
    --model ./test-models/qwen3-0.6b/gguf/Q4_K_M/*.gguf \
    --calibration test-calibration.txt \
    --precision IQ2_XS \
    --output ./test-models/qwen3-0.6b/gguf/IQ2_XS/
```

### Example 2: Full Production Workflow

```bash
# 1. Quantize base model (dual-format)
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf,mlx \
    --gguf-precision Q4_K_M \
    --mlx-bits 4 \
    --output ./models/qwen3-8b

# 2. Generate calibration data (512 samples)
llama-pajamas-quant iq generate-calibration \
    --output calibration.txt \
    --num-samples 512

# 3. IQ extreme compression
llama-pajamas-quant iq quantize \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --calibration calibration.txt \
    --precision IQ2_XS \
    --output ./models/qwen3-8b/gguf/IQ2_XS/

# 4. Evaluate all variants
cd ../run
uv run python ../quant/evaluation/llm/run_eval.py \
    --model-path ../quant/models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf \
    --num-questions 140

uv run python ../quant/evaluation/llm/run_eval.py \
    --model-path ../quant/models/qwen3-8b/gguf/IQ2_XS/*.gguf \
    --format gguf \
    --num-questions 140
```

### Example 3: Using Pre-existing Calibration Data

The repository includes pre-generated calibration data:

```bash
# Use existing calibration data
llama-pajamas-quant iq quantize \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --calibration ./calibration_data/calibration.txt \
    --precision IQ2_XS \
    --output ./models/qwen3-8b/gguf/IQ2_XS/
```

## Direct Binary Access

For advanced users who prefer direct llama.cpp tools:

```bash
# Setup symlinks (one-time)
bash bin/setup-symlinks.sh

# Use directly
./bin/llama-imatrix \
    -m ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    -f calibration.txt \
    -o qwen3-8b.imatrix \
    --chunks 512

./bin/llama-quantize \
    --imatrix qwen3-8b.imatrix \
    ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    ./models/qwen3-8b/gguf/IQ2_XS/model.gguf \
    IQ2_XS
```

## Hardware Detection

Detect your hardware and get recommended settings:

```bash
# Detect hardware
llama-pajamas-quant hardware detect

# Generate runtime config
llama-pajamas-quant hardware config \
    --model-size 7-8B \
    --use-case speed \
    --output runtime-config.json
```

## Tips

### 1. Start Small
- Use Qwen3-0.6B for testing (2-3 minutes)
- Move to larger models once workflow is validated

### 2. Reuse Importance Matrix
- Generate once, use for multiple IQ precisions
- Saves significant time (~10-15 minutes per precision)

### 3. Calibration Data
- Use existing `calibration_data/calibration.txt` (512 samples)
- Or generate custom: `llama-pajamas-quant iq generate-calibration`

### 4. Parallel Processing
- Set `parallel: N` in batch config
- Recommended: N = CPU cores / 2

### 5. Quality vs Size
- **Q4_K_M**: Best quality/size ratio (standard)
- **IQ3_XS**: Good balance (30% smaller than Q4_K_M)
- **IQ2_XS**: Extreme compression (50% smaller, usable quality)
- **IQ2_XXS**: Maximum compression (55% smaller, lower quality)

## Troubleshooting

### "Binary not found"
```bash
cd quant
uv run python scripts/build_llama_cpp.py
bash ../bin/setup-symlinks.sh
```

### "Calibration file not found"
```bash
llama-pajamas-quant iq generate-calibration --output calibration.txt
# Or use existing:
ls calibration_data/calibration.txt
```

### "Out of memory"
- Reduce `--n-chunks` for imatrix generation
- Use smaller model for testing
- Process models sequentially (parallel: 1)

## See Also

- [Main README](../README.md) - Full documentation
- [CLI Reorganization Plan](../.plans/CLI-REORGANIZATION-PLAN.md)
- [IQ Tools Accessibility](../.plans/IQ-TOOLS-ACCESSIBILITY.md)
- [Evaluation README](../evaluation/README.md)
