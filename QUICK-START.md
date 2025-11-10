# Llama-Pajamas Quick Start

Get started with llama-pajamas quantization in 5 minutes!

## Installation

```bash
# Clone repository
git clone https://github.com/llama-farm/LlamaPajamas.git
cd LlamaPajamas

# Install quantization pipeline
cd quant
uv sync
uv pip install -e .

# Setup IQ tools (optional)
cd ..
bash bin/setup-symlinks.sh
```

## Quick Examples

### 1. Standard Quantization (GGUF Q4_K_M)

```bash
cd quant

# Quantize a model
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-0.6B \
    --formats gguf \
    --gguf-precision Q4_K_M \
    --output ./models/qwen3-0.6b

# Takes ~2-3 minutes, outputs 4.68GB Q4_K_M file
```

### 2. IQ Quantization (Extreme Compression)

```bash
cd quant

# Run automated test workflow
bash examples/test-iq-workflow.sh

# Or manual steps:
# 1. Generate calibration data
llama-pajamas-quant iq generate-calibration \
    --output calibration.txt \
    --num-samples 512

# 2. Quantize to IQ2_XS (50% smaller than Q4_K_M)
llama-pajamas-quant iq quantize \
    --model ./models/qwen3-0.6b/gguf/Q4_K_M/*.gguf \
    --calibration calibration.txt \
    --precision IQ2_XS \
    --output ./models/qwen3-0.6b/gguf/IQ2_XS/
```

### 3. Dual-Format (GGUF + MLX)

```bash
# Best for Apple Silicon
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf,mlx \
    --gguf-precision Q4_K_M \
    --mlx-bits 4 \
    --output ./models/qwen3-8b
```

### 4. Batch Processing

```bash
# Process multiple models
llama-pajamas-quant batch \
    --config examples/batch-config.yaml \
    --parallel 2
```

### 5. Hardware Detection

```bash
# Detect your hardware
llama-pajamas-quant hardware detect

# Output example:
# Platform: Apple M1 Max (64GB)
# Recommended backend: mlx
# Capabilities: metal, neon, fp16
```

## Common Workflows

### Workflow 1: Quick Test (2-3 minutes)

```bash
cd quant

# Small model for testing
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-0.6B \
    --formats gguf \
    --output ./test-models/qwen3-0.6b \
    --no-benchmark
```

### Workflow 2: Production Deployment

```bash
cd quant

# 1. Standard quantization
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf,mlx \
    --output ./models/qwen3-8b

# 2. Extreme compression (IQ)
llama-pajamas-quant iq generate-calibration --output calibration.txt
llama-pajamas-quant iq quantize \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --calibration calibration.txt \
    --precision IQ2_XS \
    --output ./models/qwen3-8b/gguf/IQ2_XS/

# 3. Evaluate
cd ../run
uv run python ../quant/evaluation/llm/run_eval.py \
    --model-path ../quant/models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf
```

### Workflow 3: Direct Binary Access

```bash
# Setup (one-time)
bash bin/setup-symlinks.sh

# Use llama.cpp tools directly
./bin/llama-imatrix -m model.gguf -f calibration.txt -o output.imatrix
./bin/llama-quantize --imatrix output.imatrix model.gguf out.gguf IQ2_XS
```

## CLI Commands Reference

```bash
# Main commands
llama-pajamas-quant quantize llm     # Quantize LLMs
llama-pajamas-quant iq               # IQ quantization
llama-pajamas-quant hardware detect  # Detect hardware
llama-pajamas-quant export           # Export models
llama-pajamas-quant batch            # Batch processing

# IQ subcommands
llama-pajamas-quant iq generate-calibration  # Create calibration data
llama-pajamas-quant iq generate-matrix       # Create importance matrix
llama-pajamas-quant iq quantize             # Full IQ workflow
llama-pajamas-quant iq run-binary           # Direct binary access
```

## Model Sizes & Quality

| Precision | Size (8B model) | Quality | Use Case |
|-----------|----------------|---------|----------|
| Q4_K_M | 4.68 GB | 94% | Standard (recommended) |
| IQ3_XS | 3.3 GB | 90-93% | Good balance |
| IQ2_XS | 2.4 GB | 85-90% | Extreme compression |
| IQ2_XXS | 2.2 GB | 80-85% | Maximum compression |

## Tips

1. **Start Small**: Test with Qwen3-0.6B (2-3 min) before larger models
2. **Reuse Calibration**: Generate once, use for all IQ precisions
3. **Use Examples**: `quant/examples/` has ready-to-use configs
4. **Check Hardware**: Run `hardware detect` for optimal settings
5. **Parallel Processing**: Set `parallel: N` in batch configs

## Troubleshooting

### "Binary not found"
```bash
cd quant
uv run python scripts/build_llama_cpp.py
bash ../bin/setup-symlinks.sh
```

### "Out of memory"
- Use smaller model (Qwen3-0.6B)
- Reduce `--n-chunks` for imatrix
- Process sequentially (parallel: 1)

### "Module not found"
```bash
cd quant
uv sync
uv pip install -e .
```

## Next Steps

- 📖 Read [Main README](README.md) for full documentation
- 📊 See [Evaluation README](quant/evaluation/README.md) for benchmarking
- 🎯 Check [Examples](quant/examples/) for workflows
- 📋 Review [Plans](.plans/) for architecture details

## Getting Help

```bash
# Command help
llama-pajamas-quant --help
llama-pajamas-quant iq --help
llama-pajamas-quant iq quantize --help

# Example workflows
ls quant/examples/
cat quant/examples/README.md
```
