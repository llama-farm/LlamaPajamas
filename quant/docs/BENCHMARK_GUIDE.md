# Open-Source Benchmark Guide

## Overview

Llama-Pajamas uses **industry-standard open-source benchmarks** to evaluate quantized model quality. This provides objective, reproducible metrics without API costs or coding bias.

## Benchmark Suite

We use a **balanced suite** optimized for speed vs coverage:

### 1. MMLU (Multi-task Language Understanding)
**Weight**: 30%
**Samples**: 100
**What it tests**: General knowledge across 57 subjects (math, science, history, etc.)
**Why it matters**: Measures broad reasoning capability
**Example**: "What is the atomic number of carbon?" → Tests factual knowledge

### 2. HellaSwag
**Weight**: 20%
**Samples**: 100
**What it tests**: Common sense reasoning and sentence completion
**Why it matters**: Tests real-world understanding
**Example**: "A person is riding a bike. They..." → Tests natural continuation

### 3. ARC-Easy (AI2 Reasoning Challenge)
**Weight**: 15%
**Samples**: 100
**What it tests**: Grade-school science questions
**Why it matters**: Tests basic reasoning and QA
**Example**: "Why do leaves change color in fall?" → Tests reasoning

### 4. TruthfulQA
**Weight**: 10%
**Samples**: 50
**What it tests**: Truthfulness and resistance to false beliefs
**Why it matters**: Prevents hallucinations and misinformation
**Example**: "What happens if you crack your knuckles?" → Tests factuality

### 5. BFCL (Berkeley Function Calling Leaderboard)
**Weight**: 25%
**Samples**: 100
**What it tests**: Tool/function calling accuracy
**Why it matters**: Critical for agentic AI applications
**Example**: Given functions, correctly call `search(query="weather")` → Tests tool use

## Why This Suite?

### ✅ No Coding Bias
- Only BFCL involves structured output
- Most benchmarks test reasoning, knowledge, common sense
- Won't unfairly penalize models that aren't code-focused

### ✅ Balanced Coverage
- **30%** General knowledge (MMLU)
- **25%** Tool calling (BFCL)
- **20%** Common sense (HellaSwag)
- **15%** Reasoning (ARC)
- **10%** Truthfulness (TruthfulQA)

### ✅ Fast Yet Comprehensive
- **Total**: ~450 samples
- **Time**: 15-20 minutes per model
- **vs Full benchmarks**: 3+ hours per model
- **Statistical validity**: 50-100 samples per task provides good signal

### ✅ Industry Standard
- Used by major leaderboards (HuggingFace, LMSYS)
- Reproducible and well-documented
- Community-vetted datasets

## Usage

### Quick Start
```bash
# Run all benchmarks on both models
uv run python run_benchmarks.py
```

### Output Format
```
BENCHMARK COMPARISON
================================================================================
Benchmark                 MLX 4-bit       GGUF Q4_K_M     Difference
--------------------------------------------------------------------------------
mmlu_abstract_algebra     65.23%          58.42%          +6.81%
hellaswag                 72.15%          68.91%          +3.24%
arc_easy                  78.32%          74.18%          +4.14%
truthfulqa_mc2            45.67%          42.33%          +3.34%
bfcl                      54.12%          48.76%          +5.36%
--------------------------------------------------------------------------------
Weighted Average          64.85%          59.23%          +5.62%
Total Time                18.5min         17.2min         +1.3min
================================================================================
```

### Detailed Results
All benchmarks save detailed JSON results:
- `models/qwen3-8b/lm_eval_mlx/results.json` - MLX detailed results
- `models/qwen3-8b/lm_eval_gguf/results.json` - GGUF detailed results

## Integration with Quantization Pipeline

### Auto-benchmark during quantization
```python
from llama_pajamas_quant import Quantizer

quantizer = Quantizer()
result = quantizer.convert(
    "Qwen/Qwen3-8B",
    "./models/qwen3-8b",
    formats=["gguf", "mlx"],
    benchmark=True,  # Run benchmarks after quantization
)

# Access results
print(f"MLX Score: {result['benchmarks']['mlx'].avg_score:.2%}")
print(f"GGUF Score: {result['benchmarks']['gguf'].avg_score:.2%}")
```

### Standalone benchmarking
```python
from llama_pajamas_quant.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

# Benchmark MLX
mlx_results = runner.run_mlx_benchmarks(
    "./models/qwen3-8b-mlx",
    quant_config={"bits": 4}
)

# Benchmark GGUF
gguf_results = runner.run_gguf_benchmarks(
    "./models/qwen3-8b.gguf",
    quant_config={"precision": "Q4_K_M"}
)

# Compare
runner.compare_results([mlx_results, gguf_results])
```

## Custom Benchmark Configuration

### Adjust sample sizes
```python
custom_config = {
    "mmlu": {
        "name": "MMLU",
        "num_samples": 200,  # Double the samples
        "tasks": ["mmlu_abstract_algebra"],
        "metric": "acc",
        "weight": 0.5,
    },
    "hellaswag": {
        "name": "HellaSwag",
        "num_samples": 200,
        "tasks": ["hellaswag"],
        "metric": "acc_norm",
        "weight": 0.5,
    },
}

runner = BenchmarkRunner(benchmark_config=custom_config)
```

### Run full benchmarks (slow but comprehensive)
```bash
# Use lm-eval directly for full benchmarks
lm_eval --model hf \
  --model_args pretrained=./models/qwen3-8b-mlx,dtype=float16 \
  --tasks mmlu,hellaswag,arc_easy,truthfulqa_mc,bfcl \
  --output_path ./full_results
```

## Interpreting Scores

### Score Ranges (typical for 8B models)

| Score | Interpretation |
|-------|----------------|
| 70%+ | Excellent - Near SOTA for size |
| 60-70% | Good - Production quality |
| 50-60% | Fair - Usable with limitations |
| 40-50% | Poor - Significant degradation |
| <40% | Failed - Not usable |

### Quantization Quality Loss

Typical degradation from FP16 baseline:

| Quantization | Expected Loss |
|--------------|---------------|
| 8-bit | 0-2% |
| Q6_K | 1-3% |
| Q5_K_M | 2-4% |
| Q4_K_M | 3-6% |
| Q4_0 | 5-10% |

**Example**: If FP16 scores 70%, Q4_K_M should score 64-67% (3-6% loss).

## Troubleshooting

### "Task not found" error
Some tasks may require additional setup:
```bash
# Download required datasets
lm_eval --tasks list  # See available tasks
```

### GGUF backend issues
Ensure llama-cpp-python is installed:
```bash
uv add llama-cpp-python
```

### Memory errors
Reduce batch size or sample count:
```bash
lm_eval --batch_size 1 --limit 50
```

## Best Practices

### 1. Baseline Comparison
Always benchmark the original FP16 model first:
```bash
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen3-8B \
  --tasks mmlu,hellaswag,arc_easy,truthfulqa_mc,bfcl
```

### 2. Consistent Hardware
Run all benchmarks on the same hardware for fair comparison.

### 3. Multiple Runs
For critical decisions, average 2-3 runs to account for variance.

### 4. Domain-Specific Tests
Add benchmarks relevant to your use case:
- **Math**: GSM8K, MATH
- **Code**: HumanEval, MBPP
- **Conversation**: MT-Bench, AlpacaEval

## Cost Comparison

### vs GPT-5 nano LLM-as-judge:
- **Open-source benchmarks**: $0 (free, runs locally)
- **GPT-5 nano**: ~$0.0006 per model
- **Time**: Similar (15-20 min vs 10-15 min)
- **Quality**: Industry-standard metrics vs subjective scoring

### vs Full manual evaluation:
- **Open-source benchmarks**: Automated, objective
- **Manual**: Expensive, subjective, time-consuming
- **Reproducibility**: High vs Low

## Benchmark Details

### MMLU Subjects Tested
By default, we test 3 subjects for speed:
- Abstract Algebra (math reasoning)
- Anatomy (scientific knowledge)
- Business Ethics (applied reasoning)

Full MMLU has 57 subjects across:
- STEM, Humanities, Social Sciences, Other

### BFCL Categories
- Simple function calling
- Multiple function calling
- Parallel function calling
- Function calling with context

## Roadmap

Future enhancements:
- [ ] Add GSM8K for math reasoning
- [ ] Add HumanEval for code (optional)
- [ ] Add MT-Bench for conversation
- [ ] GPU acceleration for faster benchmarking
- [ ] Automatic regression detection
- [ ] HTML dashboard for results

## References

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [HellaSwag Paper](https://arxiv.org/abs/1905.07830)
- [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [TruthfulQA Paper](https://arxiv.org/abs/2109.07958)
