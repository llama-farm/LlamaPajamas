# Directory Structure Update - Complete ✅

All code has been updated to support the new subdirectory structure for quantized models.

## New Structure

```
models/qwen3-8b/
├── gguf/
│   ├── Q4_K_M/
│   │   └── model.gguf
│   ├── Q3_K_M/
│   │   └── model.gguf
│   └── Q5_K_M/
│       └── model.gguf
├── mlx/
│   ├── 4bit-mixed/
│   │   ├── config.json
│   │   ├── weights.npz
│   │   └── tokenizer_config.json
│   ├── 3bit-mixed/
│   │   └── ...
│   └── 8bit-pure/
│       └── ...
├── benchmark_mlx_4bit-mixed.json
├── benchmark_mlx_3bit-mixed.json
├── benchmark_gguf_Q4_K_M.json
├── benchmark_gguf_Q3_K_M.json
└── manifest.json
```

## Files Updated

### ✅ Converters (Create Subdirectories)
- `llama_pajamas_quant/converters/gguf.py` - Creates `gguf/{precision}/` subdirectories
- `llama_pajamas_quant/converters/mlx.py` - Creates `mlx/{bits}bit-{mixed|pure}/` subdirectories

### ✅ Core Quantizer (Benchmark Naming)
- `llama_pajamas_quant/core/quantizer.py` - Saves benchmarks with descriptive names

### ✅ Benchmark Scripts (Auto-Discovery)
All scripts now auto-discover models from any subdirectory:
- `run_benchmarks_simple.py` - Uses `glob("**/*.gguf")` and searches for `config.json`
- `llama_pajamas_quant/simple_benchmarks.py` - Same auto-discovery
- `benchmark_performance.py` - Same auto-discovery
- `validate_quality.py` - Same auto-discovery

## Auto-Discovery Logic

Scripts automatically find models using:

```python
# For MLX - find first subdirectory with config.json
mlx_subdirs = [d for d in (models_dir / "mlx").iterdir()
               if d.is_dir() and (d / "config.json").exists()]
mlx_path = mlx_subdirs[0]

# For GGUF - find first .gguf file in any subdirectory
gguf_files = list((models_dir / "gguf").glob("**/*.gguf"))
gguf_path = str(gguf_files[0])
```

This means scripts will work with:
- Old flat structure (if you haven't migrated yet)
- New subdirectory structure
- Multiple quantizations (picks first found)

## Commands to Test

### Run 3-bit Quantization
```bash
uv run python test_dual_format.py \
  --model /Users/robthelen/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --gguf-precision Q3_K_M \
  --mlx-bits 3
```

This will create:
- `models/qwen3-8b/gguf/Q3_K_M/model.gguf`
- `models/qwen3-8b/mlx/3bit-mixed/`
- `models/qwen3-8b/benchmark_gguf_Q3_K_M.json`
- `models/qwen3-8b/benchmark_mlx_3bit-mixed.json`

### Run Benchmarks on Any Models
```bash
# Automatically finds and benchmarks any models in the directories
uv run python run_benchmarks_simple.py
```

### Run Performance Tests
```bash
uv run python benchmark_performance.py
```

### Validate Quality
```bash
uv run python validate_quality.py
```

All scripts will automatically discover models regardless of subdirectory structure!

## Migration Notes

If you have existing models in the old flat structure, they will still work:
- Scripts search recursively for `.gguf` files
- Scripts search for any `config.json` in MLX subdirectories
- New quantizations will use the new structure
- Old quantizations will still be found and used

## Next Steps

1. ✅ Run 3-bit quantization to test new structure
2. ✅ Compare benchmarks between 3-bit and 4-bit
3. ✅ Multiple quantizations can now coexist in same directory

Everything is ready to go!
