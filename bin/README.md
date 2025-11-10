# Llama-Pajamas Binary Tools

This directory contains symlinks to commonly used llama.cpp tools for easy access.

## Quick Start

### Setup (One-time)
```bash
# Create symlinks
bash bin/setup-symlinks.sh
```

### Usage

**Method 1: Direct access from root**
```bash
./bin/llama-imatrix --help
./bin/llama-quantize --help
./bin/llama-cli --help
```

**Method 2: Add to PATH**
```bash
# Temporary (current shell only)
source bin/setup-env.sh

# Now you can run directly
llama-imatrix --help
llama-quantize --help

# Make permanent (add to ~/.bashrc or ~/.zshrc)
export PATH="/path/to/llama-pajamas/bin:$PATH"
```

## Available Tools

### Core Quantization
- `llama-imatrix` - Generate importance matrix for IQ quantization
- `llama-quantize` - Quantize models to various precisions

### Inference & Testing
- `llama-cli` - Command-line inference
- `llama-server` - HTTP server for model inference
- `llama-perplexity` - Evaluate model perplexity
- `llama-bench` - Benchmark model performance

### Utilities
- `llama-gguf` - GGUF file utilities
- `llama-gguf-split` - Split large GGUF files

## Examples

### IQ Quantization Workflow

**Step 1: Generate importance matrix**
```bash
./bin/llama-imatrix \
    -m ./quant/models/qwen3-8b/gguf/Q4_K_M/model.gguf \
    -f ./quant/calibration.txt \
    -o ./quant/qwen3-8b.imatrix \
    --chunks 512
```

**Step 2: Quantize with imatrix**
```bash
./bin/llama-quantize \
    --imatrix ./quant/qwen3-8b.imatrix \
    ./quant/models/qwen3-8b/gguf/Q4_K_M/model.gguf \
    ./quant/models/qwen3-8b/gguf/IQ2_XS/model.gguf \
    IQ2_XS
```

### Quick Inference Test
```bash
./bin/llama-cli \
    -m ./quant/models/qwen3-8b/gguf/Q4_K_M/model.gguf \
    -p "Explain quantum computing:" \
    -n 200
```

### Benchmark Model
```bash
./bin/llama-bench \
    -m ./quant/models/qwen3-8b/gguf/Q4_K_M/model.gguf
```

## Troubleshooting

### Symlinks not working?
```bash
# Re-run setup
bash bin/setup-symlinks.sh
```

### Binaries not found?
Build llama.cpp first:
```bash
cd quant
uv run python scripts/build_llama_cpp.py
```

### Permission denied?
Make scripts executable:
```bash
chmod +x bin/*.sh
```

## High-Level CLI

For a simpler interface, use the llama-pajamas-quant CLI:
```bash
# Instead of manual imatrix workflow
llama-pajamas-quant iq quantize \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --calibration ./calibration.txt \
    --precision IQ2_XS \
    --output ./models/qwen3-8b/gguf/IQ2_XS/
```

See main README for full CLI documentation.
