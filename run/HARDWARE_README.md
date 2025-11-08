# Hardware-Optimized Runtime Guide

llama-pajamas includes **automatic hardware detection and optimization** for maximum performance across different platforms.

## Quick Start (Any Hardware)

The fastest way to get started - let llama-pajamas detect your hardware and configure optimally:

```bash
# Auto-configure and run (one command!)
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf \
  --auto-configure \
  --verbose
```

This automatically:
1. Detects your hardware (CPU, GPU, RAM/VRAM, cores)
2. Generates optimal settings (gpu layers, threads, batch sizes, context)
3. Loads and runs the model with maximum performance

## Hardware-Specific Guides

### Apple Silicon (M1/M2/M3/M4)

**Recommended Format**: MLX (10-20% faster than GGUF on Mac)

**Auto-configure (easiest)**:
```bash
# MLX format
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/mlx/4bit-mixed/ \
  --auto-configure

# GGUF format (also works)
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  --auto-configure
```

**Manual configuration**:
```bash
# Generate config once
uv run python quant/scripts/generate_runtime_config.py \
  --model-size 7-8B \
  --output runtime_config.json

# Run with config
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/mlx/4bit-mixed/ \
  --config runtime_config.json
```

**Expected Performance**:
- **M1 (16GB)**: ~60 tokens/sec (8B Q4_K_M)
- **M2 (16GB)**: ~65 tokens/sec (8B Q4_K_M)
- **M3 (16GB)**: ~70 tokens/sec (8B Q4_K_M)
- **M1 Max (64GB)**: ~75 tokens/sec (8B Q5_K_M), can run 30B+ models

**Optimal Settings** (auto-configured):
- **Threads**: Performance cores only (4-8)
- **GPU Layers**: All layers on Metal GPU (`-1`)
- **Prompt Batch**: 512-1024 (M2/M3 benefit from larger)
- **Decode Batch**: 16-32
- **Context**: 4096-8192 (depending on RAM)

---

### NVIDIA GPUs (RTX/Tesla)

**Recommended Format**: GGUF (universal, excellent CUDA support)

**Auto-configure**:
```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  --auto-configure
```

**Expected Performance**:
- **RTX 3060 (12GB)**: ~55 tokens/sec (8B Q4_K_M)
- **RTX 3090 (24GB)**: ~100 tokens/sec (8B Q4_K_M), 60 tokens/sec (13B Q4_K_M)
- **RTX 4090 (24GB)**: ~120 tokens/sec (8B Q4_K_M), 65 tokens/sec (13B Q4_K_M)
- **Tesla A100 (40GB)**: ~130 tokens/sec (8B Q4_K_M), 75 tokens/sec (13B Q4_K_M)

**Optimal Settings** (auto-configured):
- **Threads**: 8-16
- **GPU Layers**: All layers on CUDA (`-1`)
- **Prompt Batch**: 1024-2048 (desktop VRAM allows large batches)
- **Decode Batch**: 32-64
- **Context**: 4096-8192

**Tips**:
- Keep GPU clocks high (disable power throttling)
- On Linux: `sudo nvidia-smi -pm 1` for persistence mode
- Watch VRAM usage: `nvidia-smi dmon`

---

### AMD GPUs (RX 7000 series, MI series)

**Recommended Format**: GGUF with ROCm backend

**Auto-configure**:
```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  --auto-configure
```

**Expected Performance**:
- **RX 7900 XT/XTX (20-24GB)**: ~80 tokens/sec (8B Q4_K_M)

**Optimal Settings** (auto-configured):
- **Threads**: 8
- **GPU Layers**: All layers on ROCm (`-1`)
- **Prompt Batch**: 512-768 (ROCm prefers moderate batches)
- **Decode Batch**: 16-24
- **Context**: 4096

**Requirements**:
- ROCm 5.6+ installed
- Linux kernel 5.15+ recommended
- Set `HSA_OVERRIDE_GFX_VERSION` if needed for your GPU

---

### CPU-Only (Intel/AMD x86, ARM)

**Recommended Format**: GGUF (optimized CPU kernels)

**Auto-configure**:
```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  --auto-configure
```

**Expected Performance**:
- **16GB RAM**: ~15 tokens/sec (8B Q4_K_M)
- **64GB RAM**: ~20 tokens/sec (8B Q4_K_M), 12 tokens/sec (13B Q4_K_M)

**Optimal Settings** (auto-configured):
- **Threads**: Physical cores (not hyperthreads)
- **GPU Layers**: 0 (CPU-only)
- **Prompt Batch**: 256-512
- **Decode Batch**: 8-16
- **Context**: 2048-4096

**Tips**:
- Build llama.cpp with AVX2/AVX-512 support (`-march=native`)
- Use `numactl` on multi-socket systems
- Prefer Q3_K_M or Q4_K_M for smaller RAM footprint
- Consider smaller models (7B vs 13B) for better responsiveness

---

## Use Case Optimization

### Long Context (64K+ tokens)

```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
  --auto-configure \
  --use-case long_context
```

**Adjustments**:
- Context: 2x larger (8192 → 16384)
- Prompt batch: 2x smaller (1024 → 512)
- Decode batch: 2x smaller (32 → 16)

### Speed (Maximum tokens/sec)

```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q3_K_M/*.gguf \
  --auto-configure \
  --use-case speed
```

**Adjustments**:
- Precision: Lower (Q4_K_M → Q3_K_M)
- Prompt batch: 2x larger (1024 → 2048)
- Decode batch: 2x larger (32 → 64)

### Quality (Best accuracy)

```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q5_K_M/*.gguf \
  --auto-configure \
  --use-case quality
```

**Adjustments**:
- Precision: Higher (Q4_K_M → Q5_K_M)
- Conservative batch sizes
- Full context window

---

## Advanced: Manual Configuration

### 1. Detect Hardware

```bash
uv run python quant/scripts/detect_hardware.py --format summary
```

Output:
```
Hardware Profile: Apple M1 (64GB)
Platform ID: apple_silicon_m1_64gb

System:
  OS: darwin
  Architecture: arm64

CPU:
  Model: Apple M1 Max
  Performance cores: 8
  Efficiency cores: 2

Memory:
  RAM: 64.0 GB
  VRAM: 64.0 GB

GPU:
  Model: Apple M1 Max
  Type: metal

Recommended Backend: mlx
Capabilities: metal, neon, fp16
```

### 2. Generate Configuration

```bash
uv run python quant/scripts/generate_runtime_config.py \
  --model-size 7-8B \
  --use-case general \
  --output runtime_config.json
```

Output (`runtime_config.json`):
```json
{
  "model": {
    "size": "7-8B",
    "precision": "Q5_K_M"
  },
  "backend": "metal",
  "settings": {
    "n_gpu_layers": -1,
    "n_threads": 8,
    "n_batch": 1024,
    "n_ubatch": 32,
    "n_ctx": 8192
  },
  "metadata": {
    "hardware_profile": "apple_silicon_m1_64gb",
    "expected_tokens_per_sec": 75
  }
}
```

### 3. Run with Config

```bash
uv run python -m llama_pajamas_run.cli \
  --model ./models/qwen3-8b/gguf/Q5_K_M/*.gguf \
  --config runtime_config.json \
  --verbose
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: Crash, slow swapping, or "failed to allocate" error

**Solutions**:
1. Use lower precision: Q5_K_M → Q4_K_M → Q3_K_M
2. Reduce context: `--use-case general` (instead of `long_context`)
3. Use smaller model: 13B → 8B → 7B

**Auto-fix**: The config generator automatically downgrades settings if VRAM insufficient

### Slow Performance

**Check**:
1. GPU layers offloaded? (`n_gpu_layers: -1` in config)
2. Threads match physical cores? (not hyperthreads)
3. Running on battery? (Mac thermal throttling)

**Tips**:
- Mac: Plug in power adapter, keep screen brightness low
- NVIDIA: Check `nvidia-smi` shows GPU utilization >80%
- CPU: Monitor with `htop`, ensure cores not oversubscribed

### "Backend not available"

**CUDA**: Install CUDA toolkit + `llama-cpp-python[cuda]`
**ROCm**: Install ROCm + compatible kernel
**Metal**: Mac only, comes with macOS
**MLX**: Mac only, install `mlx-lm`

---

## Hardware Profiles Reference

Full hardware profiles database: `quant/config/hardware_profiles.json`

Includes empirically-tuned settings for:
- Apple Silicon: M1/M2/M3/M4 (8GB to 64GB)
- NVIDIA: RTX 3060/3090/4090, Tesla A100/V100
- AMD: RX 7900 XT/XTX, MI250/MI300
- CPU: Intel/AMD x86, ARM (Graviton, etc.)

Based on: `.plans/runtime-optimizations.md` empirical guidance

---

## Performance Expectations

| Hardware | Model | Precision | Tokens/sec | Notes |
|----------|-------|-----------|------------|-------|
| M1 (16GB) | 8B | Q4_K_M | ~60 | Unified memory |
| M3 (16GB) | 8B | Q4_K_M | ~70 | Larger batches |
| M1 Max (64GB) | 8B | Q5_K_M | ~75 | Higher precision |
| M1 Max (64GB) | 13B | Q4_K_M | ~50 | Larger model |
| RTX 3060 (12GB) | 8B | Q4_K_M | ~55 | Entry GPU |
| RTX 3090 (24GB) | 8B | Q4_K_M | ~100 | High-end GPU |
| RTX 4090 (24GB) | 8B | Q4_K_M | ~120 | Top-tier GPU |
| RX 7900 XTX (24GB) | 8B | Q4_K_M | ~80 | AMD flagship |
| CPU (16GB) | 8B | Q4_K_M | ~15 | Slower but works |

*Performance varies by actual hardware, model architecture, and workload*

---

## Additional Resources

- **Quantization Guide**: `quant/README.md`
- **Runtime Optimizations**: `.plans/runtime-optimizations.md`
- **Hardware Profiles**: `quant/config/hardware_profiles.json`
- **Evaluation System**: `quant/scripts/EVALUATION_README.md`

---

Built with llama-pajamas - architecture-aware quantization for maximum performance 🚀
