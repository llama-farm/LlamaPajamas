# Qwen3-8B Quantization Comparison

## Overview

This document compares the quantized versions of Qwen3-8B against each other, evaluated using GPT-5 nano as an LLM judge.

## Evaluation Results

### Quality Scores (0-10 scale)

| Metric | MLX 4-bit | GGUF Q4_K_M | Difference |
|--------|-----------|-------------|------------|
| **Overall Quality** | 3.9/10 | 1.6/10 | **+2.3** (MLX wins) |
| **Accuracy** | 4.4/10 | 1.3/10 | **+3.1** (MLX wins) |
| **Coherence** | 4.0/10 | 2.0/10 | **+2.0** (MLX wins) |
| **Relevance** | 4.1/10 | 1.6/10 | **+2.6** (MLX wins) |

### Performance

| Metric | MLX 4-bit | GGUF Q4_K_M | Difference |
|--------|-----------|-------------|------------|
| **Speed (s/prompt)** | 3.36s | 2.25s | **-1.11s** (GGUF faster) |
| **Model Size** | 4.3GB | 4.7GB | **-0.4GB** (MLX smaller) |

## Winner: MLX 4-bit

**MLX is the clear winner for quality**, scoring 2.3 points higher on overall quality (3.9 vs 1.6).

**GGUF is faster** (2.25s vs 3.36s per prompt) but at the cost of significantly lower quality.

## Detailed Analysis by Category

### Code Generation (3 prompts)
- **MLX**: Struggled with incomplete responses (2/10 average)
- **GGUF**: Failed most code generation tasks (0-1/10)

Example failure: Both models generated descriptions instead of actual Python functions.

### Technical Explanations (2 prompts)
- **MLX**: Better at explaining concepts (4-6/10)
- **GGUF**: Minimal explanations, often incomplete (3-6/10)

### Factual Questions (1 prompt)
- **MLX**: Excellent - 9/10 on list vs tuple question
- **GGUF**: Poor - 1/10 on same question

### Conceptual (1 prompt)
- **MLX**: Fair - 4/10 on sync vs async
- **GGUF**: Poor - 1/10 on sync vs async

## Key Findings

###  1. Both models show quality degradation

The low scores (1.6-3.9/10) suggest both quantized models are producing incomplete or repetitive responses. This is likely due to:
- **Token limit too low** (150 tokens) - not enough for complete answers
- **Quantization artifacts** - 4-bit compression affecting generation quality
- **Temperature settings** - may need tuning

### 2. MLX significantly outperforms GGUF

Despite both having low absolute scores, MLX is **144% better** than GGUF (3.9 vs 1.6).

### 3. GGUF is faster but sacrifices quality

GGUF generates **33% faster** (2.25s vs 3.36s) but with **60% lower quality** (1.6 vs 3.9).

## Recommendations

### For Quality-Critical Applications
✅ **Use MLX 4-bit**
- Better accuracy, coherence, and relevance
- More complete responses
- Worth the 1.1s speed penalty

### For Speed-Critical Applications
⚠️ **Use GGUF Q4_K_M with caution**
- Faster generation
- Significantly lower quality
- May need higher precision (Q5_K_M or Q6_K)

### For Production Use
⚠️ **Neither model is production-ready as-is**

Both models need improvements:
1. **Increase max_tokens** to 300-500 for complete answers
2. **Test higher precision**: Q5_K_M, Q6_K, or 8-bit quantization
3. **Tune generation parameters** (temperature, top_p, etc.)
4. **Evaluate against baseline FP16** to measure actual degradation

## Next Steps

### 1. Evaluate Baseline (Recommended)
Run the original Qwen3-8B in FP16 to establish a quality baseline:
```bash
uv run python evaluate_baseline.py
```

### 2. Test Higher Precision
Try less aggressive quantization:
```python
# Q5_K_M - Better quality, slightly larger
quantizer.convert(
    "Qwen/Qwen3-8B",
    "./models/qwen3-8b-q5",
    formats=["gguf"],
    gguf_precision="Q5_K_M",
    evaluate=True
)

# 8-bit MLX - Closer to FP16 quality
quantizer.convert(
    "Qwen/Qwen3-8B",
    "./models/qwen3-8b-mlx-8bit",
    formats=["mlx"],
    mlx_bits=8,
    evaluate=True
)
```

### 3. Increase Token Limits
Update evaluation to allow longer responses:
- Current: 150 tokens
- Recommended: 300-500 tokens

### 4. Compare Multiple Configurations
Create a comparison matrix:
- FP16 (baseline)
- MLX 8-bit
- MLX 4-bit (current)
- GGUF Q6_K
- GGUF Q5_K_M
- GGUF Q4_K_M (current)
- GGUF Q4_0

## Cost Analysis

**Evaluation cost**: $0.0006 for both models (14 prompts total)

This makes it **very affordable** to experiment with different quantization configurations!

## Files

- `models/qwen3-8b/evaluation_mlx.json` - MLX detailed results
- `models/qwen3-8b/evaluation_gguf.json` - GGUF detailed results

## Conclusion

**MLX 4-bit is the winner** for Qwen3-8B quantization based on quality metrics, though both models show room for improvement. The evaluation system successfully identifies quality differences between quantization methods, enabling data-driven optimization.

For production use, consider:
1. Testing higher precision (Q5_K_M, Q6_K, 8-bit)
2. Evaluating against FP16 baseline
3. Increasing token limits for complete responses
4. Running with domain-specific prompts for your use case
