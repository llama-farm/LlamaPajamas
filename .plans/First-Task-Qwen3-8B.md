# First Task: Qwen3-8B Complete Pipeline

## Objective

**Build end-to-end pipeline for Qwen/Qwen3-8B**:
1. ✅ Download from HuggingFace
2. ✅ Quantize to GGUF (Q4_K_M) and MLX (4-bit)
3. ✅ Validate quality (perplexity, benchmarks)
4. ✅ Run on CUDA (Linux) and MLX (Mac)

**Success Criteria**:
```bash
# Pipeline: Convert model
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --precision q4_k_m:4bit

# Runtime: Run model
llama-pajamas-run chat \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend cuda

# Result: <2GB quantized model, <5% quality loss, 60+ tok/s
```

---

## Model: Qwen3-8B

### Specifications

```yaml
Model ID: Qwen/Qwen3-8B
Release: May 2025 (Qwen3 series)

Parameters:
  Total: 8.2B
  Non-embedding: 6.95B

Architecture:
  Type: Dense Decoder-Only Transformer
  Layers: 36
  Attention: Grouped Query Attention (GQA)
    Query heads: 32
    KV heads: 8 (4:1 ratio)
  Context: 32,768 tokens (native)
  Extended: 131,072 tokens (YaRN scaling, factor 4.0)

Training:
  Format: BF16
  Stages: Pretraining + Post-training
  Languages: 100+ languages and dialects

Capabilities:
  - Thinking mode (complex reasoning, temp=0.6, top_p=0.95)
  - Non-thinking mode (efficient dialogue, temp=0.7, top_p=0.8)
  - Multilingual instruction-following
  - Translation
  - Code generation

Baseline Memory (BF16):
  Model weights: ~16.4GB (8.2B × 2 bytes)
  KV cache (8K context): ~1.2GB
  Activations: ~2GB
  Total: ~19.6GB

Community:
  Quantized versions: 173+
  Fine-tuned models: 503+
  Adapters: 323+
```

### Why Qwen3-8B?

**Perfect First Target**:
- ✅ Modern architecture (Qwen3, May 2025)
- ✅ Good size (8B = accessible but non-trivial)
- ✅ GQA attention (4:1 query:KV ratio = efficient)
- ✅ Thinking mode (demonstrates reasoning)
- ✅ 100+ languages (multilingual testing)
- ✅ Strong community (173 quantized versions = proven)
- ✅ Both Instruct and Base available

---

## Target Configurations

### Configuration 1: Balanced (MVP Target)

```yaml
Name: Qwen3-8B-Q4-Balanced
Goal: Best quality/size for consumer hardware

GGUF Q4_K_M:
  Method: Q4_K_M (4-bit K-quant medium)
  Group size: 128
  Layers: All quantized uniformly
  Expected size: ~1.9GB
  Expected quality: -3 to -5% vs BF16

MLX 4-bit Mixed:
  Body: 4-bit, group size 64
  Embeddings: 6-bit (sensitive)
  Output layer: 6-bit (critical for distribution)
  Expected size: ~1.7GB
  Expected quality: -3 to -5% vs BF16

Expected Performance:
  CUDA (RTX 4070): 65-80 tokens/sec
  MLX (M3 Max): 75-90 tokens/sec
  Memory: <2GB
  Quality: >95% of baseline

Use Cases:
  - Development
  - Consumer GPUs (RTX 3060+, 12GB)
  - Apple Silicon (M2+, 16GB)
  - Production applications
```

### Configuration 2: Ultra-Compressed (Stretch)

```yaml
Name: Qwen3-8B-Q3-Ultra
Goal: Maximum compression for edge

GGUF Q3_K_M:
  Method: Q3_K_M (3-bit K-quant medium)
  Group size: 64
  Expected size: ~1.2GB
  Expected quality: -8 to -12% vs BF16

MLX 4-bit Aggressive:
  All layers: 4-bit uniform
  Group size: 64
  Expected size: ~1.1GB
  Expected quality: -8 to -12% vs BF16

Expected Performance:
  CUDA (RTX 3060): 40-50 tokens/sec
  MLX (M2 Pro): 30-40 tokens/sec
  Memory: <1.5GB
  Quality: ~90% of baseline

Use Cases:
  - Edge devices
  - Limited VRAM
  - Cost-sensitive deployments
```

### Configuration 3: High Quality (Reference)

```yaml
Name: Qwen3-8B-Q6-HQ
Goal: Minimal degradation

GGUF Q6_K:
  Method: Q6_K (6-bit K-quant)
  Group size: 128
  Expected size: ~3.2GB
  Expected quality: -1 to -2% vs BF16

MLX 8-bit:
  All layers: INT8 uniform
  Group size: 128
  Expected size: ~3.0GB
  Expected quality: -1 to -2% vs BF16

Expected Performance:
  CUDA (RTX 4090): 110-130 tokens/sec
  MLX (M3 Ultra): 130-150 tokens/sec
  Memory: ~3.5GB
  Quality: >98% of baseline

Use Cases:
  - Quality-critical
  - Professional development
  - Reference baseline
```

---

## Implementation Plan

### Phase 1: Pipeline Setup (Days 1-2)

**Goal**: Basic infrastructure

**Tasks**:
1. Project structure (llama-pajamas-quant)
2. HuggingFace cache integration
3. Model download functionality
4. Architecture detector (Qwen3 detection)

**Deliverable**:
```bash
llama-pajamas-quant info Qwen/Qwen3-8B

# Output:
# Model: Qwen3-8B
# Architecture: Dense Decoder (GQA 4:1)
# Parameters: 8.2B (6.95B non-embedding)
# Context: 32K (native), 128K (extended)
# Downloaded: ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B
```

### Phase 2: GGUF Quantization (Days 3-4)

**Goal**: GGUF Q4_K_M working

**Tasks**:
1. llama.cpp integration (HF → GGUF conversion)
2. llama-quantize wrapper (FP16 GGUF → Q4_K_M)
3. Metadata generation
4. Basic validation (load test)

**Deliverable**:
```bash
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf \
  --precision q4_k_m

# Output: ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf (~1.9GB)
```

**Testing**:
```python
# Verify GGUF loads
from llama_cpp import Llama

model = Llama(
    model_path="./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1
)

output = model("Hello, world!", max_tokens=50)
print(output["choices"][0]["text"])
# Should generate coherent text
```

### Phase 3: MLX Quantization (Days 5-6)

**Goal**: MLX 4-bit mixed working

**Tasks**:
1. mlx-lm integration (HF → MLX conversion)
2. Mixed precision quantization (4-bit + 6-bit)
3. Metadata generation
4. Basic validation (load test)

**Deliverable**:
```bash
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats mlx \
  --precision 4bit \
  --mixed-precision \
  --embedding-bits 6

# Output: ./models/qwen3-8b/mlx/ (~1.7GB)
```

**Testing**:
```python
# Verify MLX loads
from mlx_lm import load, generate

model, tokenizer = load("./models/qwen3-8b/mlx/")
response = generate(model, tokenizer, "Hello, world!", max_tokens=50)
print(response)
# Should generate coherent text
```

### Phase 4: Quality Validation (Days 7-8)

**Goal**: Automated quality checks

**Tasks**:
1. Perplexity calculation (WikiText2)
2. Sample generation comparison
3. Validation report generation
4. manifest.json creation

**Deliverable**:
```bash
llama-pajamas-quant validate \
  --original Qwen/Qwen3-8B \
  --quantized ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --metrics perplexity

# Output:
# Perplexity (BF16):     8.23
# Perplexity (Q4_K_M):   8.56
# Increase:              4.0%
# Status:                ✅ PASS (<5% threshold)
```

**Validation Report**:
```json
{
  "model": "Qwen/Qwen3-8B",
  "format": "gguf",
  "method": "Q4_K_M",
  "validation": {
    "perplexity": {
      "baseline": 8.23,
      "quantized": 8.56,
      "increase_pct": 4.0,
      "threshold": 5.0,
      "pass": true
    },
    "samples": [
      {
        "prompt": "Write a Python function to sort",
        "baseline": "def sort_list(lst):\n    return sorted(lst)",
        "quantized": "def sort_list(lst):\n    return sorted(lst)",
        "match": true
      }
    ]
  }
}
```

### Phase 5: Runtime (Days 9-10)

**Goal**: Load and run quantized models

**Tasks**:
1. Model loader (reads manifest.json)
2. Backend manager (CUDA, MLX, CPU)
3. Inference engine
4. CLI interface

**Deliverable**:
```bash
# GGUF on CUDA
llama-pajamas-run chat \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend cuda

# MLX on Mac
llama-pajamas-run chat \
  --model ./models/qwen3-8b/mlx/ \
  --backend mlx

# Both work, generate coherent text at target speed
```

**Success Criteria**:
- CUDA: 65+ tokens/sec (RTX 4070)
- MLX: 75+ tokens/sec (M3 Max)
- Quality: Coherent, on-topic responses
- Memory: <2GB

### Phase 6: Polish & Documentation (Days 11-12)

**Goal**: Complete, documented system

**Tasks**:
1. Error handling (graceful failures)
2. Progress bars (user feedback)
3. Documentation (README, examples)
4. Testing (unit + integration)

**Deliverable**: Production-ready v0.1.0

---

## Testing Strategy

### Unit Tests

```python
# test_quantizer.py
def test_gguf_conversion():
    """Test HF → GGUF conversion"""
    result = quantizer.convert(
        model="Qwen/Qwen3-8B",
        format="gguf",
        method="Q4_K_M"
    )
    assert result.gguf_path.exists()
    assert result.size_mb < 2000  # <2GB

def test_mlx_conversion():
    """Test HF → MLX conversion"""
    result = quantizer.convert(
        model="Qwen/Qwen3-8B",
        format="mlx",
        bits=4
    )
    assert result.mlx_path.exists()
    assert (result.mlx_path / "weights.safetensors").exists()

def test_architecture_detection():
    """Test Qwen3 architecture detection"""
    detector = ArchitectureDetector()
    arch = detector.detect("Qwen/Qwen3-8B")
    assert arch.model_type == "qwen3"
    assert arch.num_kv_heads == 8
    assert arch.attention_type == "gqa"
```

### Integration Tests

```python
# test_end_to_end.py
def test_full_pipeline():
    """Test complete pipeline: download → quantize → validate"""
    # Quantize
    result = pipeline.run(
        model="Qwen/Qwen3-8B",
        formats=["gguf", "mlx"],
        validate=True
    )

    # Check outputs
    assert result.gguf_path.exists()
    assert result.mlx_path.exists()
    assert result.validation.perplexity_increase < 5.0

    # Load and generate
    runtime = Runtime(result.gguf_path)
    output = runtime.generate("Hello", max_tokens=10)
    assert len(output.text) > 0

def test_backend_consistency():
    """Test GGUF and MLX produce similar outputs"""
    prompt = "Write a Python function to calculate fibonacci"

    gguf_runtime = Runtime(gguf_path, backend="cpu")
    mlx_runtime = Runtime(mlx_path, backend="mlx")

    gguf_output = gguf_runtime.generate(prompt, max_tokens=100, temperature=0.0)
    mlx_output = mlx_runtime.generate(prompt, max_tokens=100, temperature=0.0)

    # Outputs should be very similar (temperature=0)
    similarity = compute_similarity(gguf_output.text, mlx_output.text)
    assert similarity > 0.9  # >90% similar
```

### Performance Benchmarks

```bash
# Benchmark GGUF on CUDA
llama-pajamas-run benchmark \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend cuda \
  --num-tokens 500 \
  --context-lengths 2048,8192,32768

# Expected:
# Context 2K:   75 tok/s
# Context 8K:   68 tok/s
# Context 32K:  45 tok/s

# Benchmark MLX on Mac
llama-pajamas-run benchmark \
  --model ./models/qwen3-8b/mlx/ \
  --backend mlx \
  --num-tokens 500

# Expected (M3 Max):
# Context 2K:   85 tok/s
# Context 8K:   78 tok/s
# Context 32K:  52 tok/s
```

---

## CLI Examples

### Pipeline (Conversion)

```bash
# Basic conversion (balanced config)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --precision q4_k_m:4bit

# Ultra compression (aggressive)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b-ultra \
  --formats gguf,mlx \
  --precision q3_k_m:4bit \
  --aggressive

# High quality (reference)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b-hq \
  --formats gguf,mlx \
  --precision q6_k:8bit

# With validation
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx \
  --validate \
  --calibration-samples 512
```

### Runtime (Inference)

```bash
# Interactive chat (GGUF)
llama-pajamas-run chat \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --backend cuda

# Interactive chat (MLX)
llama-pajamas-run chat \
  --model ./models/qwen3-8b/mlx/ \
  --backend mlx

# Single generation
llama-pajamas-run generate \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --prompt "Write a Python function to reverse a string" \
  --max-tokens 200 \
  --temperature 0.7

# With thinking mode (Qwen3 feature)
llama-pajamas-run generate \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --prompt "Solve: If x + 2 = 5, what is x?" \
  --mode thinking \
  --temperature 0.6 \
  --top-p 0.95

# Start API server
llama-pajamas-run serve \
  --model ./models/qwen3-8b/gguf/qwen3-8b-q4_k_m.gguf \
  --port 8080 \
  --api-style openai
```

---

## Success Metrics

### Functional Requirements

- [x] ✅ Download Qwen3-8B from HuggingFace
- [x] ✅ Convert to GGUF Q4_K_M successfully
- [x] ✅ Convert to MLX 4-bit successfully
- [x] ✅ Generate manifest.json with metadata
- [x] ✅ Validate quality (perplexity <5% increase)
- [x] ✅ Load and run on CUDA (Linux)
- [x] ✅ Load and run on MLX (Mac)

### Performance Targets

| Config | Format | Memory | Quality | Speed (RTX 4070) | Speed (M3 Max) |
|--------|--------|--------|---------|------------------|----------------|
| Balanced | GGUF Q4 | 1.9GB | -4% | 70 t/s | N/A |
| Balanced | MLX 4-bit | 1.7GB | -4% | N/A | 80 t/s |
| Ultra | GGUF Q3 | 1.2GB | -10% | 45 t/s | N/A |
| Ultra | MLX 4-bit | 1.1GB | -10% | N/A | 35 t/s |
| HQ | GGUF Q6 | 3.2GB | -1.5% | 120 t/s | N/A |
| HQ | MLX 8-bit | 3.0GB | -1% | N/A | 140 t/s |

### Compression Achievements

- **Balanced**: **10.3x** (19.6GB → 1.9GB GGUF), **11.5x** (19.6GB → 1.7GB MLX)
- **Ultra**: **16.3x** (19.6GB → 1.2GB GGUF), **17.8x** (19.6GB → 1.1GB MLX)
- **HQ**: **6.1x** (19.6GB → 3.2GB GGUF), **6.5x** (19.6GB → 3.0GB MLX)

### Quality Thresholds

**Must Pass**:
- Perplexity increase: <5% (balanced), <12% (ultra), <2% (HQ)
- Coherent generation: 100% of test prompts
- Thinking mode works: Correctly solves reasoning tasks
- Multilingual: Generate correct responses in 5+ languages

---

## Timeline

| Days | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Pipeline Setup | `llama-pajamas-quant info` working |
| 3-4 | GGUF Quantization | GGUF Q4_K_M produced |
| 5-6 | MLX Quantization | MLX 4-bit produced |
| 7-8 | Quality Validation | Validation passing |
| 9-10 | Runtime | Both backends running |
| 11-12 | Polish | Documentation, tests |

**Total**: 12 days (2-3 weeks calendar time)

---

## Next Steps After Qwen3-8B

Once Qwen3-8B is complete, the pipeline is **proven** and can handle:

1. **Qwen3-30B-A3B** (MoE): Tests expert-aware quantization
2. **GPT-OSS-20B** (MoE + sparse attention): Tests attention patterns
3. **Any Qwen/LLaMA/Mistral model**: Architecture detector generalizes

**The hard part is Qwen3-8B**. Once working, the rest is configuration! 🚀
