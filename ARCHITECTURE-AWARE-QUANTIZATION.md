# Architecture-Aware Quantization in Llama-Pajamas

## Overview

Llama-Pajamas implements **architecture-aware quantization** that automatically detects and applies optimal quantization strategies for different model architectures, from standard transformers to cutting-edge MoE and hybrid Mamba-2/Transformer models.

## How It Works

### 1. Automatic Architecture Detection

Every quantization workflow starts with architecture detection:

```python
from llama_pajamas_quant.core import Quantizer

quantizer = Quantizer()  # Initializes with ArchitectureDetector

result = quantizer.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b",
    formats=["gguf", "mlx"]
)
```

**What happens behind the scenes:**

```
[1/4] Detecting model architecture...
├─ Loads config.json
├─ Analyzes architecture components
├─ Detects: model_type, family, attention_type
├─ Identifies: GQA ratio, MoE configuration, special layers
└─ Recommends quantization strategy

Detected: llm (qwen3)
Parameters: 8.0B
Attention: grouped_query_attention (GQA)
GQA Ratio: 4:1

[2/4] Converting to GGUF with Q4_K_M...
[3/4] Converting to MLX with 4-bit...
[4/4] Generating manifest...
```

### 2. Supported Architectures

#### Standard Dense Transformers
**Models**: GPT-3, LLaMA 2, Mistral 7B, older Qwen

**Strategy**:
- Multi-Head Attention (MHA) or Grouped Query Attention (GQA)
- Standard FFN layers
- **Precision**: W4A16 or W8A8
- **Group Size**: 64-128
- **Expected**: <1% perplexity degradation (INT8)

```bash
# Automatically detected and optimized
llama-pajamas-quant quantize llm \
    --model mistralai/Mistral-7B-v0.1 \
    --formats gguf,mlx \
    --output ./models/mistral-7b
```

#### Grouped Query Attention (GQA)
**Models**: LLaMA 3, Qwen3, Gemma 3

**Strategy**:
- GQA with 4:1 or 8:1 ratio
- **Attention Heads**: Higher precision for Q, K, V projections
- **KV Cache**: Optimized for reduced memory
- **Group Size**: 128 for 7B-13B models

```bash
# GQA detected automatically
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf,mlx \
    --output ./models/qwen3-8b

# Output shows GQA detection:
# GQA Ratio: 4:1
```

#### Mixture-of-Experts (MoE)
**Models**:
- Qwen3-235B-A22B (235B total, 22B active, 128 experts)
- DeepSeek V3 (671B total, 37B active)
- Mixtral 8x7B (47B total, 13B active, 8 experts)

**Strategy**:
- **Expert-Aware Mixed Precision**:
  - Router Network: FP16 or INT8 (critical)
  - Shared Experts: INT8 (always active)
  - Frequently Used Experts: INT8
  - Rarely Used Experts: INT4 or INT3
- **Expert-Balanced Calibration**:
  - Profiles expert activation frequencies
  - Oversamples underutilized experts
  - Per-expert quantization parameters

```bash
# MoE detected automatically
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-235B-A22B \
    --formats gguf \
    --gguf-precision Q4_K_M \
    --output ./models/qwen3-235b

# Output shows MoE detection:
# MoE: 128 experts, 8 active
# Strategy: Expert-aware mixed precision
```

#### Hybrid Mamba-2/Transformer (Granite 4.0)
**Models**: IBM Granite 4.0 (9:1 Mamba:Transformer ratio)

**Strategy**:
- **Mamba-2 Blocks** (90% of model):
  - State-space models with linear complexity
  - INT8 for most weights
  - Higher precision for state transition matrices
- **Transformer Blocks** (10% of model):
  - Standard transformer quantization
  - KV cache only for transformer layers
- **70% memory reduction** vs standard transformers

```bash
# Hybrid architecture detected
llama-pajamas-quant quantize llm \
    --model ibm/granite-4.0-h-small \
    --formats gguf \
    --output ./models/granite-4.0

# Output shows hybrid detection:
# Architecture: Hybrid Mamba-2/Transformer (9:1)
# Strategy: Per-block-type quantization
```

### 3. Architecture Detection Code

The detection logic in `core/detector.py`:

```python
class ArchitectureDetector:
    def detect(self, model_path: str) -> ArchitectureInfo:
        """Detect model architecture from config."""

        # Load config
        config = self._load_config(model_path)

        # Detect basic architecture
        family = self._detect_family(config)
        attention_type = self._detect_attention(config)

        # Detect advanced features
        is_gqa = self._detect_gqa(config)
        is_moe = self._detect_moe(config)
        is_hybrid = self._detect_hybrid(config)

        # Build architecture info
        arch = ArchitectureInfo(
            model_type="llm",
            family=family,
            attention_type=attention_type,
            is_gqa=is_gqa,
            is_moe=is_moe,
            # ... more fields
        )

        return arch
```

### 4. Quantization Strategy Recommendations

Each detected architecture returns optimal settings:

```python
arch = detector.detect("Qwen/Qwen3-235B-A22B")
strategy = arch.recommend_quantization()

# Returns:
{
    "precision_allocation": {
        "router": "int8",
        "shared_experts": "int8",
        "frequent_experts": "int8",
        "rare_experts": "int4",
        "attention": "int8",
    },
    "group_size": 128,
    "calibration": {
        "method": "expert_balanced",
        "samples": 512,
    },
    "expected_compression": "8-12x",
    "quality_target": "<2% perplexity degradation",
}
```

## Quantization Methods by Architecture

### Standard Transformers

| Precision | Size (8B) | Perplexity | Use Case |
|-----------|-----------|------------|----------|
| Q4_K_M | 4.68 GB | +0.4685 | Standard (recommended) |
| Q5_K_M | 5.21 GB | +0.1316 | High quality |
| Q3_K_M | 3.78 GB | +1.2 | Smaller, good quality |
| IQ2_XS* | 2.40 GB | +2-3% | Extreme compression |

*Requires importance matrix (IQ quantization)

### MoE Models

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Router | INT8/FP16 | Critical for expert selection |
| Shared Experts | INT8 | Always active, high utilization |
| Top 20% Experts | INT8 | Frequently activated |
| Bottom 80% Experts | INT4 | Sparse activation tolerates aggression |
| Attention | INT8 | Standard |

### Hybrid (Granite 4.0)

| Block Type | Precision | Memory | Speed |
|------------|-----------|--------|-------|
| Mamba-2 (90%) | INT8 | Fixed state size | 2x faster |
| Transformer (10%) | INT8 | KV cache only | Standard |
| **Total** | **Mixed** | **70% reduction** | **2x overall** |

## IQ Quantization (Importance-Based)

For extreme compression with architecture awareness:

```bash
# Step 1: Generate calibration data
llama-pajamas-quant iq generate-calibration \
    --output calibration.txt \
    --num-samples 512

# Step 2: Standard quantization first
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf \
    --gguf-precision Q4_K_M \
    --output ./models/qwen3-8b

# Step 3: IQ extreme compression
llama-pajamas-quant iq quantize \
    --model ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --calibration calibration.txt \
    --precision IQ2_XS \
    --output ./models/qwen3-8b/gguf/IQ2_XS/

# Automatically applies architecture-aware:
# - Per-layer importance analysis
# - MoE expert-specific importance
# - Hybrid block-type importance
```

## Vision & Speech Models

### Vision (CoreML)
```bash
# Architecture detection for vision models
llama-pajamas-quant export \
    --model yolov8n \
    --backend coreml \
    --precision fp16 \
    --output ./models/yolo-v8n/

# Then quantize
llama-pajamas-quant quantize vision \
    --model yolov8n \
    --precision int8 \
    --output ./models/yolo-v8n/coreml/int8/
```

**Detected**: Detection head, classification layers, backbone
**Strategy**: Higher precision for detection head, INT8 for backbone

### Speech (Whisper CoreML)
```bash
# Export Whisper encoder
uv run python scripts/export_whisper_coreml.py \
    --model whisper-tiny

# Quantize
llama-pajamas-quant quantize speech \
    --model whisper-tiny \
    --precision int8 \
    --output ./models/whisper-tiny/coreml/int8/
```

**Detected**: Encoder-only architecture
**Strategy**: INT8 with attention preservation

## Multi-Modal Architecture Support

| Architecture Type | Detection | Strategy | Support |
|-------------------|-----------|----------|---------|
| Dense Transformer | ✅ Auto | W4A16/W8A8 | Full |
| GQA | ✅ Auto | KV cache optimized | Full |
| MoE | ✅ Auto | Expert-aware mixed | Full |
| Hybrid Mamba-2 | ✅ Auto | Per-block-type | Full |
| Vision (YOLO, ViT, CLIP) | ✅ Auto | Layer-specific | Full |
| Speech (Whisper) | ✅ Auto | Encoder-optimized | Full |

## Verification

You can verify architecture detection:

```bash
# Check what was detected
llama-pajamas-quant quantize llm \
    --model Qwen/Qwen3-8B \
    --formats gguf \
    --output ./models/test \
    --verbose

# Output includes full detection:
# [1/4] Detecting model architecture...
# Detected: llm (qwen3)
# Parameters: 8.0B
# Attention: grouped_query_attention (GQA)
# GQA Ratio: 4:1
# Architecture-aware strategy:
#   - Precision: Q4_K_M
#   - Group size: 128
#   - KV cache: Optimized for GQA
#   - Expected: <1% perplexity degradation
```

## Reference Documentation

For detailed architecture-specific strategies, see:
- `.plans/Model-Architecture-Strategy.md` - Full quantization strategies per architecture
- `.plans/Novel-Architectures-Granite-GPTOSS.md` - Hybrid and MoE architectures
- `quant/llama_pajamas_quant/core/detector.py` - Detection implementation
- `quant/llama_pajamas_quant/core/architecture.py` - Architecture definitions

## Summary

**Llama-Pajamas automatically:**
1. ✅ Detects your model's architecture
2. ✅ Applies optimal quantization strategy
3. ✅ Handles GQA, MoE, Hybrid architectures
4. ✅ Works across LLM, Vision, Speech modalities
5. ✅ Provides architecture-specific calibration
6. ✅ Optimizes for quality vs size tradeoffs

**No manual configuration needed** - just run the quantization command and let the architecture detector handle the rest!
