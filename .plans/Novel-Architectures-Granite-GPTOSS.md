# Novel Architectures: IBM Granite 4.0 & OpenAI GPT-OSS

## Executive Summary

**IBM Granite 4.0** and **OpenAI GPT-OSS** represent **groundbreaking architectural innovations** (2025) that challenge traditional transformer assumptions:

- **Granite 4.0**: Hybrid **Mamba-2/Transformer** (9:1 ratio) + fine-grained MoE with shared experts
- **GPT-OSS**: Dense MoE with **alternating sparse/dense attention** + native MXFP4 post-training

Both architectures achieve **dramatic efficiency gains** (70% memory reduction, 2x faster) and require **specialized quantization strategies** beyond standard approaches.

This document provides **architecture-specific quantization playbooks** for production deployment.

---

## IBM Granite 4.0 Architecture Deep-Dive

### Architecture Overview

**Revolutionary Hybrid Design**: 9 Mamba-2 blocks : 1 Transformer block

```
Input
  ↓
[Mamba-2 Block 1] → [MoE Block] → Residual
[Mamba-2 Block 2] → [MoE Block] → Residual
[Mamba-2 Block 3] → [MoE Block] → Residual
[Mamba-2 Block 4] → [MoE Block] → Residual
[Mamba-2 Block 5] → [MoE Block] → Residual
[Mamba-2 Block 6] → [MoE Block] → Residual
[Mamba-2 Block 7] → [MoE Block] → Residual
[Mamba-2 Block 8] → [MoE Block] → Residual
[Mamba-2 Block 9] → [MoE Block] → Residual
[Transformer Block] → [MoE Block] → Residual  ← Every 10th layer
  ↓
Output
```

### Model Variants

| Model | Total Params | Active Params | Architecture | MoE Type |
|-------|-------------|---------------|--------------|----------|
| **Granite-4.0-H-Small** | 32B | 9B | Hybrid Mamba-2/Transformer | Fine-grained MoE with shared experts |
| **Granite-4.0-H-Tiny** | 7B | 1B | Hybrid Mamba-2/Transformer | Fine-grained MoE with shared experts |
| **Granite-4.0-H-Micro** | 3B | 3B | Hybrid Mamba-2/Transformer | Dense FFN (no MoE) |
| **Granite-4.0-Micro** | 3B | 3B | Pure Transformer | Dense FFN |

### Technical Specifications

**Attention Mechanism**:
- Grouped Query Attention (GQA)
- Transformer blocks use standard self-attention
- Rotary Positional Embeddings (RoPE)

**State Space Model (Mamba-2)**:
- Captures global context with linear complexity O(n)
- Maintains hidden state across sequence
- 9x more layers than transformers (90% of model)

**MoE Configuration**:
- **Shared experts**: Always activated (critical knowledge)
- **Specialized experts**: Conditionally activated (task-specific)
- Fine-grained routing per token
- Enables better parameter efficiency

**Other Components**:
- SwiGLU activation functions
- RMSNorm for normalization
- Shared input/output embeddings

**Context & Training**:
- Trained on samples up to 512K tokens
- Evaluated up to 128K tokens
- 15T tokens total training data

### Performance Characteristics

**Memory Efficiency**:
- **70% RAM reduction** vs standard transformers (long context)
- KV cache only needed for 1/10 layers (transformer blocks)
- Mamba-2 state is fixed size regardless of sequence length

**Speed**:
- **2x faster inference** vs comparable transformers
- Linear complexity for Mamba blocks vs quadratic attention
- Reduced memory bandwidth requirements

**Quality**:
- Competitive with standard transformers 2x their active size
- Better long-context handling (512K training)

---

## Granite 4.0 Quantization Strategy

### Challenge: Dual Architecture Quantization

**Problem**: Mamba-2 and Transformer blocks have different sensitivities

1. **Mamba-2 blocks**: State-space models with recurrent dynamics
   - Hidden state updates are sensitive to quantization
   - Linear projections for state transitions
   - Convolutional components

2. **Transformer blocks**: Standard attention mechanisms
   - Q, K, V projections and attention scores
   - Only 10% of layers but critical for local context

3. **MoE blocks**: Mixed expert activation
   - Shared experts (always on) vs specialized experts
   - Router sensitivity
   - Expert imbalance

### Component-Specific Precision Allocation

```yaml
Mamba-2 Blocks (90% of model):
  State Transition Matrices (A, B, C, D):
    Precision: INT8 minimum, FP16 for stability-critical models
    Rationale: Recurrent dynamics accumulate errors
    Group size: 64-128

  Input/Output Projections:
    Precision: INT8
    Rationale: Standard linear layers, robust

  Convolution Weights:
    Precision: INT8
    Rationale: Local context, less sensitive

  Hidden State (Runtime):
    Precision: FP16 or BF16 (do not quantize)
    Rationale: Accumulated across sequence, error compounds

Transformer Blocks (10% of model):
  Q, K, V Projections:
    Precision: INT8 or FP16
    Rationale: Only 1/10 layers, can afford higher precision

  Attention Scores:
    Precision: FP16 or BF16
    Rationale: Numerical stability for softmax

  Output Projection:
    Precision: INT8

MoE Blocks (if present):
  Router:
    Precision: INT8
    Rationale: Expert selection accuracy critical

  Shared Experts:
    Precision: INT8
    Rationale: Always active, high utilization

  Specialized Experts:
    Precision: INT4 for low-frequency, INT8 for high-frequency
    Rationale: Sparse activation tolerates compression

Embeddings & Output:
  Embedding Table: INT8 or shared with output
  Output Projection: INT8
```

### Recommended Quantization Configurations

#### Configuration 1: High Quality (Recommended)

```python
granite_high_quality = {
    "mamba_blocks": {
        "ssm_matrices": "int8",  # A, B, C, D matrices
        "projections": "int8",
        "convolutions": "int8",
        "hidden_state": "fp16",  # Do not quantize!
        "group_size": 128
    },
    "transformer_blocks": {
        "qkv_proj": "int8",
        "attention_scores": "fp16",
        "output_proj": "int8",
        "group_size": 128
    },
    "moe_blocks": {
        "router": "int8",
        "shared_experts": "int8",
        "specialized_experts": "int8",  # Conservative
        "group_size": 128
    },
    "expected_quality": "1-3% degradation",
    "expected_memory": "~8GB for 32B model (vs 64GB FP16)",
    "compression_ratio": "8x"
}
```

#### Configuration 2: Balanced (Default)

```python
granite_balanced = {
    "mamba_blocks": {
        "ssm_matrices": "int8",
        "projections": "int8",
        "convolutions": "int4",  # More aggressive
        "hidden_state": "fp16",
        "group_size": 128
    },
    "transformer_blocks": {
        "qkv_proj": "int8",
        "attention_scores": "bf16",
        "output_proj": "int8",
        "group_size": 128
    },
    "moe_blocks": {
        "router": "int8",
        "shared_experts": "int8",
        "specialized_experts": "int4",  # Mixed precision
        "group_size": 128
    },
    "expected_quality": "3-5% degradation",
    "expected_memory": "~6GB for 32B model",
    "compression_ratio": "10x"
}
```

#### Configuration 3: Extreme Compression (Edge)

```python
granite_extreme = {
    "mamba_blocks": {
        "ssm_matrices": "int8",  # Keep stable
        "projections": "int4",
        "convolutions": "int4",
        "hidden_state": "bf16",  # Reduce precision slightly
        "group_size": 64
    },
    "transformer_blocks": {
        "qkv_proj": "int8",  # Keep higher for quality
        "attention_scores": "bf16",
        "output_proj": "int4",
        "group_size": 64
    },
    "moe_blocks": {
        "router": "int8",
        "shared_experts": "int8",
        "specialized_experts": "int3",  # Very aggressive
        "group_size": 64
    },
    "expected_quality": "5-10% degradation",
    "expected_memory": "~4GB for 32B model",
    "compression_ratio": "16x"
}
```

### Native Format Support

**IBM Granite Native Format**: BF16 checkpoints

**Quantization Options**:
1. **Standard GGUF**: llama.cpp supports Mamba-2 + Transformer hybrid
2. **MXFP4/MXFP6**: AMD Quark for AMD Instinct GPUs (high accuracy retention)
3. **MLX**: Apple Silicon optimization (if Mamba-2 kernels available)

### Calibration Strategy

**Hybrid Calibration Required**:

```yaml
Calibration Dataset:
  Size: 512 samples
  Sequence length: 8K-32K tokens (leverage long context)
  Distribution:
    - 50% general text (C4, WikiText)
    - 30% code (Granite strong on coding)
    - 20% STEM (technical domain)

Calibration Process:
  1. Profile Mamba-2 hidden state statistics
     - Track state magnitude and variance across layers
     - Identify outlier neurons in state transitions
     - Use percentile clipping (99.9%) for activations

  2. Profile Transformer attention patterns
     - Standard attention calibration for 10% of layers
     - Focus on local context patterns

  3. Profile MoE expert activations
     - Balance expert sampling as per MoE strategy
     - Ensure shared experts well-calibrated (high frequency)

  4. End-to-end validation
     - Test on long sequences (64K+)
     - Verify Mamba-2 state stability over time
     - Check transformer quality on local reasoning

Validation Metrics:
  - Perplexity: <5% increase
  - Long-context tasks: Needle-in-haystack at 128K
  - Code generation: HumanEval, MBPP
  - STEM reasoning: MATH, GSM8K
```

### Implementation Notes

**Critical Considerations**:

1. **Mamba-2 State Precision**: Never quantize below BF16
   - Errors accumulate across sequence
   - Long contexts (128K+) amplify drift
   - FP16/BF16 maintains stability

2. **Transformer Block Importance**: 10% of layers, 50% of quality
   - These layers parse local context critically
   - Consider higher precision (INT8) vs Mamba-2 (can use INT4)

3. **Memory Pattern**: KV cache only for transformer layers
   - 90% memory savings vs full attention model
   - Focus quantization on weights, not cache

4. **llama.cpp Support**: Check Mamba-2 kernel availability
   - Hybrid models require both Mamba-2 and Transformer kernels
   - As of 2025, llama.cpp adding Mamba support

---

## OpenAI GPT-OSS Architecture Deep-Dive

### Architecture Overview

**Sparse MoE with Alternating Attention Patterns**

```
Layer Pattern (Repeating):
[Dense Attention Layer]
  ↓
[Sparse Attention Layer] (locally banded)
  ↓
[MoE Block] (128 experts, 4 active)
  ↓
Next Layer...
```

### Model Variants

| Model | Total Params | Active Params | Experts/Layer | Active Experts | Context |
|-------|-------------|---------------|---------------|----------------|---------|
| **gpt-oss-120b** | 117B | 5.1B | 128 | 4 | 128K |
| **gpt-oss-20b** | 21B | 3.6B | (MoE config) | (varies) | 128K |

### Technical Specifications

**Attention Mechanism**:
- **Alternating Dense/Sparse Pattern** (similar to GPT-3)
  - Dense layers: Full attention O(n²)
  - Sparse layers: Locally banded attention O(n·w) where w = window size
- **Grouped Multi-Query Attention** (group size = 8)
  - 8 query heads share 1 key-value pair
  - Reduces KV cache by 8x

**Positional Encoding**:
- Rotary Positional Embeddings (RoPE)
- Native 128K context length support

**MoE Configuration** (gpt-oss-120b):
- 36 layers total
- 128 experts per MoE layer
- 4 experts active per token
- Top-k routing (k=4)

**Tokenizer**:
- o200k_harmony tokenizer
- Optimized for code and technical text

**Post-Training Quantization**:
- **Native MXFP4** quantization for MoE weights
- Enables 120B model on single 80GB GPU
- 20B model runs in 16GB memory

### Performance Characteristics

**Quality**:
- gpt-oss-120b matches o4-mini on reasoning benchmarks
- gpt-oss-20b matches o3-mini
- Strong on: Codeforces, MMLU, tool use, mathematics

**Efficiency**:
- 120B runs on single H100 (80GB) with MXFP4
- 20B runs on consumer hardware (16GB)
- ~5.1B active params (vs 120B total) = 96% sparsity

---

## GPT-OSS Quantization Strategy

### Challenge: Triple Complexity

1. **Sparse MoE**: 128 experts, only 4 active per token
2. **Alternating Attention**: Dense vs banded sparse patterns
3. **Native MXFP4**: Already post-trained with quantization

### Component-Specific Precision Allocation

```yaml
Attention Layers (Alternating):
  Dense Attention Layers:
    Q projections: INT8
    K, V projections: INT8
    KV cache: INT8 (full attention needs quality)
    Attention scores: FP16
    Output projection: INT8

  Sparse Attention Layers (Locally Banded):
    Q projections: INT8
    K, V projections: INT8
    KV cache: INT4 (local window, more tolerant)
    Attention scores: FP16
    Output projection: INT8

  Grouped Multi-Query Attention:
    - KV cache already 8x smaller due to GQA
    - INT8 → additional 2x savings
    - Total: 16x KV cache reduction vs standard MHA FP16

MoE Expert Blocks:
  Router Network:
    Precision: INT8 or FP16
    Rationale: Top-k selection (k=4) must be accurate
    Note: 128-way classification is sensitive

  Expert Weights (128 experts total):
    Strategy: Per-expert precision allocation

    Top 10% experts (by activation frequency):
      Precision: INT8
      Rationale: High utilization, quality critical

    Middle 60% experts:
      Precision: INT4
      Rationale: Moderate use, balance quality/memory

    Bottom 30% experts:
      Precision: INT3 or MXFP4
      Rationale: Rarely activated, aggressive OK

  Shared Components (if any):
    Precision: INT8

Native MXFP4 Considerations:
  - Model already post-trained with MXFP4 for MoE
  - Converting to INT4 may degrade from baseline
  - Consider keeping MXFP4 for experts, INT8 for attention
```

### Recommended Quantization Configurations

#### Configuration 1: Native MXFP4 (Optimal for AMD)

```python
gpt_oss_native_mxfp4 = {
    "attention_dense": {
        "qkv": "int8",
        "kv_cache": "int8",
        "scores": "fp16",
        "output": "int8"
    },
    "attention_sparse": {
        "qkv": "int8",
        "kv_cache": "int4",  # Local window tolerates it
        "scores": "fp16",
        "output": "int8"
    },
    "moe_experts": "mxfp4",  # Keep native format
    "router": "fp16",  # Critical for 128-way routing
    "target_hardware": "AMD MI300X with MXFP4 support",
    "expected_quality": "<1% degradation (maintains post-training)",
    "expected_memory": "~35GB for 120B model (single 80GB GPU)",
    "note": "Optimal - leverages native post-training"
}
```

#### Configuration 2: Mixed INT8/INT4 (Universal)

```python
gpt_oss_mixed_precision = {
    "attention_dense": {
        "qkv": "int8",
        "kv_cache": "int8",
        "scores": "fp16",
        "output": "int8"
    },
    "attention_sparse": {
        "qkv": "int8",
        "kv_cache": "int4",
        "scores": "fp16",
        "output": "int8"
    },
    "moe_experts": {
        "top_10pct": "int8",  # Frequent experts
        "middle_60pct": "int4",  # Medium frequency
        "bottom_30pct": "int3"  # Rare experts
    },
    "router": "int8",
    "target_hardware": "NVIDIA/Universal",
    "expected_quality": "2-4% degradation",
    "expected_memory": "~30GB for 120B model",
    "compression_ratio": "4x vs FP16"
}
```

#### Configuration 3: Aggressive GGUF (Consumer Hardware)

```python
gpt_oss_gguf_aggressive = {
    "format": "gguf",
    "method": "Q4_K_M",  # llama.cpp standard
    "attention_all": "int4",  # Unified for simplicity
    "moe_experts": "int4_uniform",  # All experts INT4
    "router": "int8",
    "kv_cache": "int4",
    "target_hardware": "Consumer GPU (24GB), CPU offload",
    "expected_quality": "5-8% degradation",
    "expected_memory": "~24GB for 120B model (partial GPU offload)",
    "note": "Max accessibility, quality trade-off acceptable"
}
```

### Attention Pattern-Aware Calibration

**Critical: Preserve Sparse Attention Patterns**

```yaml
Calibration Strategy:
  Dense Attention Layers:
    - Standard calibration (full context attention)
    - Sequence length: 4K-8K tokens
    - Samples: 256

  Sparse Attention Layers (Banded):
    - Focus on local window statistics
    - Test different window sizes (512, 1024, 2048)
    - Ensure band pattern preserved post-quantization
    - Samples: 256

  MoE Expert Calibration:
    - Expert-balanced sampling (64 samples per expert minimum)
    - Track activation frequency distribution
    - Oversample prompts that activate rare experts
    - Total samples: 8192 (64 × 128 experts)

  Router Calibration:
    - Validate top-k selection accuracy
    - Compare expert selection pre/post quantization
    - Target: >95% top-4 overlap
    - Use diverse prompt set (code, STEM, general)

Validation:
  - Perplexity on long sequences (32K+)
  - Code generation (HumanEval, MBPP)
  - Tool use accuracy (function calling)
  - Reasoning tasks (MATH, AIME)
  - Expert activation distribution (should match FP16)
```

### Native MXFP4 vs Standard Quantization

**Decision Matrix**:

| Hardware | Recommended Format | Rationale |
|----------|-------------------|-----------|
| AMD MI300X/350X | MXFP4 (native) | Hardware support, preserves post-training |
| NVIDIA H100 | Mixed INT8/INT4 | TensorRT-optimized INT kernels |
| Consumer NVIDIA | GGUF Q4_K_M | llama.cpp mature, good compression |
| Apple Silicon | MLX INT4 or GGUF | Unified memory benefits, Metal acceleration |
| CPU | GGUF Q4_K_M | Portable, AVX optimization |

**Key Insight**: GPT-OSS was **post-trained with MXFP4** on experts
- Converting to INT4 degrades from this baseline
- Keeping MXFP4 (if hardware supports) optimal
- INT4 acceptable if MXFP4 unavailable (2-4% further degradation)

---

## Architecture Detection & Auto-Configuration

### Enhanced Architecture Detector

```python
# src/llama_pajamas/core/architecture_detector.py

class ArchitectureDetector:
    """Enhanced to detect Granite 4.0 and GPT-OSS"""

    def detect(self, model_path: Path) -> ArchitectureInfo:
        """Detect architecture with support for novel designs"""

        config = self._load_config(model_path)

        # Detect IBM Granite 4.0
        if self._is_granite_4(config):
            return self._detect_granite_4(config)

        # Detect OpenAI GPT-OSS
        if self._is_gpt_oss(config):
            return self._detect_gpt_oss(config)

        # ... existing detection logic

    def _is_granite_4(self, config: Dict) -> bool:
        """Check if model is Granite 4.0"""
        model_type = config.get("model_type", "").lower()
        architectures = str(config.get("architectures", [])).lower()

        # Granite 4 has "mamba" in architecture
        return (
            "granite" in model_type or
            "mamba" in architectures or
            "hybrid_mamba" in model_type
        )

    def _detect_granite_4(self, config: Dict) -> ArchitectureInfo:
        """Parse Granite 4.0 specific config"""

        # Detect model size
        num_layers = config.get("num_hidden_layers", 0)
        hidden_size = config.get("hidden_size", 0)

        # Detect Mamba vs Transformer ratio
        num_mamba_layers = config.get("num_mamba_layers", int(num_layers * 0.9))
        num_transformer_layers = num_layers - num_mamba_layers

        # Detect MoE configuration
        num_experts = config.get("num_experts", None)
        num_experts_active = config.get("num_experts_per_tok", None)
        has_shared_experts = config.get("num_shared_experts", 0) > 0

        return ArchitectureInfo(
            family="hybrid_mamba_transformer",
            model_type="granite_4.0",
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=config.get("num_attention_heads", 0),
            num_kv_heads=config.get("num_key_value_heads"),
            is_moe=num_experts is not None,
            num_experts=num_experts,
            num_experts_active=num_experts_active,
            attention_type="gqa",
            # Granite-specific
            num_mamba_layers=num_mamba_layers,
            num_transformer_layers=num_transformer_layers,
            mamba_transformer_ratio=f"{num_mamba_layers}:{num_transformer_layers}",
            has_shared_experts=has_shared_experts,
            max_position_embeddings=config.get("max_position_embeddings", 128000)
        )

    def _is_gpt_oss(self, config: Dict) -> bool:
        """Check if model is GPT-OSS"""
        model_type = config.get("model_type", "").lower()
        architectures = str(config.get("architectures", [])).lower()

        return (
            "gpt-oss" in model_type or
            "gptoss" in model_type or
            ("openai" in str(config.get("_name_or_path", "")).lower() and
             "oss" in str(config.get("_name_or_path", "")).lower())
        )

    def _detect_gpt_oss(self, config: Dict) -> ArchitectureInfo:
        """Parse GPT-OSS specific config"""

        num_layers = config.get("num_hidden_layers", 36)

        # GPT-OSS specific: alternating dense/sparse attention
        attention_pattern = config.get("attention_pattern", "alternating")

        # MoE configuration
        num_experts = config.get("num_experts", 128)
        num_experts_active = config.get("num_experts_per_tok", 4)

        # Grouped multi-query attention
        num_attention_heads = config.get("num_attention_heads", 0)
        gqa_group_size = config.get("gqa_group_size", 8)
        num_kv_heads = num_attention_heads // gqa_group_size

        # Native quantization
        native_quantization = config.get("quantization", None)
        is_mxfp4 = native_quantization == "mxfp4"

        return ArchitectureInfo(
            family="sparse_moe_alternating_attention",
            model_type="gpt_oss",
            num_layers=num_layers,
            hidden_size=config.get("hidden_size", 0),
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            is_moe=True,
            num_experts=num_experts,
            num_experts_active=num_experts_active,
            attention_type="grouped_multiquery_alternating",
            # GPT-OSS specific
            attention_pattern=attention_pattern,
            gqa_group_size=gqa_group_size,
            native_quantization=native_quantization,
            is_mxfp4_native=is_mxfp4,
            max_position_embeddings=config.get("max_position_embeddings", 128000)
        )
```

### Quantization Recommendations

```python
# Add to ArchitectureInfo class

def recommend_quantization(self) -> Dict:
    """Enhanced with Granite and GPT-OSS support"""

    if self.family == "hybrid_mamba_transformer":
        return self._recommend_granite_4()

    elif self.family == "sparse_moe_alternating_attention":
        return self._recommend_gpt_oss()

    # ... existing recommendations

def _recommend_granite_4(self) -> Dict:
    """Granite 4.0 specific recommendations"""

    # Size-based precision
    if self.num_experts and self.num_experts_active:
        # MoE variant (H-Small, H-Tiny)
        strategy = "hybrid_mamba_moe"
        precision = "mixed"
    else:
        # Dense variant (H-Micro, Micro)
        strategy = "hybrid_mamba_dense"
        precision = "int8"

    return {
        "strategy": strategy,
        "mamba_blocks": {
            "ssm_matrices": "int8",
            "projections": "int8",
            "convolutions": "int8",
            "hidden_state": "fp16",  # Never quantize
            "note": "Mamba-2 state must stay FP16 for stability"
        },
        "transformer_blocks": {
            "qkv": "int8",
            "attention_scores": "fp16",
            "output": "int8",
            "note": "Only 10% of layers, can afford quality"
        },
        "moe_blocks": {
            "router": "int8",
            "shared_experts": "int8",
            "specialized_experts": "int4"
        } if self.is_moe else None,
        "calibration": {
            "method": "hybrid_mamba_aware",
            "samples": 512,
            "sequence_length": "16K-32K",
            "focus": "Mamba-2 state stability, long context"
        },
        "expected_compression": "8-10x",
        "expected_quality_loss": "3-5%",
        "notes": [
            f"Hybrid Mamba-2/Transformer ratio: {self.mamba_transformer_ratio}",
            "70% memory reduction vs standard transformers",
            "2x faster inference due to linear Mamba complexity",
            "Critical: Never quantize Mamba hidden state below FP16"
        ]
    }

def _recommend_gpt_oss(self) -> Dict:
    """GPT-OSS specific recommendations"""

    # Check for native MXFP4
    if self.is_mxfp4_native:
        recommended_format = "mxfp4"
        note = "Keep native MXFP4 for experts (post-trained)"
    else:
        recommended_format = "mixed_int8_int4"
        note = "Convert to INT4, accept 2-4% quality loss"

    return {
        "strategy": "sparse_moe_alternating_attention",
        "precision": "mixed",
        "attention_dense": {
            "qkv": "int8",
            "kv_cache": "int8",
            "scores": "fp16"
        },
        "attention_sparse": {
            "qkv": "int8",
            "kv_cache": "int4",  # Local window tolerates it
            "scores": "fp16",
            "note": "Banded sparse attention less sensitive"
        },
        "moe_experts": {
            "format": recommended_format,
            "router": "int8",
            "top_10pct": "int8",
            "middle_60pct": "int4",
            "bottom_30pct": "int3"
        },
        "grouped_multiquery": {
            "group_size": self.gqa_group_size,
            "kv_cache_reduction": f"{self.gqa_group_size}x from GQA",
            "additional_quantization": "2x from INT8",
            "total_reduction": f"{self.gqa_group_size * 2}x"
        },
        "calibration": {
            "method": "expert_balanced_alternating_attention",
            "expert_samples": 64,
            "total_samples": self.num_experts * 64,
            "focus": "Expert activation distribution, attention patterns"
        },
        "hardware_specific": {
            "amd_mi300x": "Use native MXFP4",
            "nvidia_h100": "Mixed INT8/INT4",
            "consumer": "GGUF Q4_K_M"
        },
        "expected_compression": "4x (MXFP4) to 8x (INT4)",
        "expected_quality_loss": "<1% (MXFP4) to 2-4% (INT4)",
        "notes": [
            f"MoE: {self.num_experts} experts, {self.num_experts_active} active",
            f"Alternating attention: Dense + Sparse pattern",
            f"GQA group size: {self.gqa_group_size}",
            "Model post-trained with MXFP4 on experts",
            f"Runs on single 80GB GPU with MXFP4 (120B model)"
        ]
    }
```

---

## Testing & Validation

### Granite 4.0 Specific Tests

```yaml
Test Suite: Granite 4.0
  Architecture Validation:
    - Verify Mamba-2 state stability across 128K context
    - Check transformer block quality (only 10% of layers)
    - Validate MoE expert activation distribution
    - Test shared expert utilization

  Quality Metrics:
    - Perplexity: <5% degradation on long sequences (64K+)
    - Code generation: HumanEval >90% of FP16
    - Long context: RULER benchmark at 128K
    - STEM reasoning: MATH, GSM8K >95% of FP16

  Memory & Speed:
    - Verify 70% memory reduction vs standard transformer
    - Confirm 2x speedup vs comparable transformer
    - KV cache size: 10x smaller (only transformer layers)

  Edge Cases:
    - Very long sequences (256K, 512K with RoPE scaling)
    - Mamba state overflow detection
    - Expert load balancing
```

### GPT-OSS Specific Tests

```yaml
Test Suite: GPT-OSS
  Architecture Validation:
    - Verify sparse attention pattern preservation
    - Check dense vs sparse layer quality balance
    - Validate top-k expert selection (k=4)
    - Test 128-way router accuracy

  Quality Metrics:
    - Codeforces rating: Match o4-mini
    - MMLU: >95% of FP16
    - Tool use: TauBench function calling accuracy
    - Mathematics: AIME 2024/2025 >90% of FP16

  Native MXFP4 Validation:
    - Compare MXFP4 vs INT4 quality
    - Verify expert weight quantization stability
    - Test on AMD MI300X hardware

  Memory & Efficiency:
    - 120B model on single 80GB GPU (MXFP4)
    - 20B model in 16GB memory
    - GQA KV cache: 16x reduction vs standard MHA
```

---

## Deployment Recommendations

### Granite 4.0 Deployment

**Recommended Platforms**:
1. **NVIDIA GPUs**: Use GGUF Q4_K_M or mixed INT8/INT4
   - RTX 4090: 32B model (H-Small) with INT4
   - H100: Multiple models, higher precision

2. **AMD GPUs**: Use MXFP4/MXFP6 via AMD Quark
   - MI300X: Optimal for Granite 4.0
   - Native MXFP support, excellent quality retention

3. **Apple Silicon**: MLX (if Mamba-2 kernels available)
   - M3 Max/Ultra: 32B model feasible
   - Unified memory benefits hybrid architecture

4. **CPU**: llama.cpp with Mamba-2 support
   - Check for Mamba kernel availability
   - Use Q4_K_M for memory efficiency

**Production Configuration** (32B H-Small):
```bash
# NVIDIA H100
llama-pajamas quantize ibm-granite/granite-4.0-h-small \
  --method Q4_K_M \
  --architecture granite-4 \
  --preserve-mamba-state fp16 \
  --output ./models/granite-4-h-small-q4

# AMD MI300X with MXFP4
llama-pajamas quantize ibm-granite/granite-4.0-h-small \
  --method MXFP4 \
  --architecture granite-4 \
  --backend rocm \
  --output ./models/granite-4-h-small-mxfp4
```

### GPT-OSS Deployment

**Recommended Platforms**:
1. **AMD MI300X/350X**: Native MXFP4 (optimal)
   - 120B model on single GPU
   - Preserves post-training quantization
   - Best quality

2. **NVIDIA H100**: Mixed INT8/INT4
   - 120B model on single GPU
   - TensorRT optimizations
   - Slight quality trade-off vs MXFP4

3. **Consumer NVIDIA (24GB)**: GGUF Q4_K_M with CPU offload
   - 20B model full GPU
   - 120B model with partial offload
   - llama.cpp mature

4. **Apple Silicon**: MLX INT4
   - 20B model on M3 Max (36GB)
   - Unified memory benefits
   - Metal acceleration

**Production Configuration** (120B):
```bash
# AMD MI300X (optimal - native MXFP4)
llama-pajamas quantize openai/gpt-oss-120b \
  --method native \
  --architecture gpt-oss \
  --keep-mxfp4 \
  --backend rocm \
  --output ./models/gpt-oss-120b-mxfp4

# NVIDIA H100 (mixed precision)
llama-pajamas quantize openai/gpt-oss-120b \
  --method mixed \
  --precision int8:int4 \
  --architecture gpt-oss \
  --backend cuda \
  --output ./models/gpt-oss-120b-mixed

# Consumer NVIDIA (GGUF)
llama-pajamas quantize openai/gpt-oss-120b \
  --method Q4_K_M \
  --backend cuda \
  --cpu-offload auto \
  --output ./models/gpt-oss-120b-q4
```

---

## Summary: Architecture Comparison

| Feature | Granite 4.0 | GPT-OSS | Qwen3 | Gemma 3 |
|---------|-------------|---------|-------|---------|
| **Core Innovation** | Hybrid Mamba/Transformer | Sparse Attention + MoE | Dense MoE + GQA | Hybrid Attention |
| **Memory Efficiency** | 70% reduction | 4-8x compression | 6-8x compression | 5x KV cache |
| **Speed** | 2x faster | Standard | Standard | Faster (long context) |
| **Context** | 128K-512K | 128K | 128K | 128K |
| **Quantization Challenge** | Mamba state stability | Attention pattern preservation | Expert balancing | Local/global balance |
| **Best Precision** | INT8/INT4 hybrid | MXFP4 native | FP8 native | INT8/INT4 |
| **Compression Ratio** | 8-10x | 4-8x | 4-6x | 5-8x |
| **Quality Loss** | 3-5% | 1-4% | 2-5% | 2-4% |

**Key Takeaway**: Each architecture requires **specific quantization strategies**:
- **Granite 4.0**: Never quantize Mamba state, focus on transformer layers
- **GPT-OSS**: Preserve native MXFP4 if possible, balance dense/sparse attention
- **Qwen3**: Expert-balanced calibration, FP8 for native support
- **Gemma 3**: Aggressive local KV cache quantization, conservative global

**Llama-Pajamas is powerful** because it understands these nuances and quantizes accordingly! 🚀
