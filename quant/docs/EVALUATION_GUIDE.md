# LLM-as-Judge Evaluation Guide

## Overview

Llama-Pajamas now includes an integrated **LLM-as-judge evaluation system** that automatically assesses the quality of quantized models using OpenAI's GPT-5 nano. This allows you to:

- **Compare quantization configurations** objectively
- **Track quality across experiments** automatically
- **Make data-driven decisions** about compression vs. quality tradeoffs
- **Iterate quickly** on quantization strategies

## Quick Start

### 1. Setup

Ensure you have an OpenAI API key configured:

```bash
# Add to .env file
OPEN_AI_KEY=sk-proj-...your-key-here...
```

### 2. Basic Usage

Enable evaluation when quantizing:

```python
from llama_pajamas_quant import Quantizer

quantizer = Quantizer()
result = quantizer.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b",
    formats=["gguf", "mlx"],
    evaluate=True,  # Enable evaluation
)

# Access results
for fmt, evaluation in result["evaluations"].items():
    print(f"{fmt}: Quality {evaluation.avg_quality:.1f}/10")
```

## How It Works

### Evaluation Process

1. **Quantization** - Model is quantized to requested format(s)
2. **Test Generation** - 7 standard prompts are run through the model
3. **LLM Judging** - GPT-5 nano evaluates each response on:
   - **Accuracy** (0-10): Is the information correct?
   - **Coherence** (0-10): Is it well-structured?
   - **Relevance** (0-10): Does it address the prompt?
   - **Quality** (0-10): Overall assessment
4. **Aggregation** - Scores are averaged across all prompts
5. **Storage** - Results saved as `evaluation_{format}.json`

### Standard Test Prompts

The system uses 7 carefully selected prompts across different categories:

1. **Code**: "Write a Python function to calculate the factorial of a number"
2. **Technical Explanation**: "Explain the concept of recursion in programming"
3. **Factual**: "What is the difference between a list and a tuple in Python?"
4. **Code**: "Write a function to reverse a string"
5. **Technical Explanation**: "Explain what a decorator is in Python"
6. **Code**: "Write a sorting algorithm (your choice) in Python"
7. **Conceptual**: "What are the key differences between synchronous and asynchronous programming?"

## Usage Examples

### Example 1: Compare Quantization Methods

```python
from llama_pajamas_quant import Quantizer

quantizer = Quantizer()

# Test Q4_0 (aggressive)
result_q4_0 = quantizer.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-q4-0",
    formats=["gguf"],
    gguf_precision="Q4_0",
    evaluate=True,
)

# Test Q4_K_M (balanced)
result_q4_k_m = quantizer.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-q4-k-m",
    formats=["gguf"],
    gguf_precision="Q4_K_M",
    evaluate=True,
)

# Compare
print("Q4_0 Quality:", result_q4_0["evaluations"]["gguf"].avg_quality)
print("Q4_K_M Quality:", result_q4_k_m["evaluations"]["gguf"].avg_quality)
```

### Example 2: MLX vs GGUF

```python
from llama_pajamas_quant import Quantizer

quantizer = Quantizer()

result = quantizer.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b",
    formats=["gguf", "mlx"],
    evaluate=True,
)

# Compare formats
mlx = result["evaluations"]["mlx"]
gguf = result["evaluations"]["gguf"]

print(f"MLX:  Quality {mlx.avg_quality:.1f}/10, Speed {mlx.avg_generation_time:.2f}s")
print(f"GGUF: Quality {gguf.avg_quality:.1f}/10, Speed {gguf.avg_generation_time:.2f}s")
```

### Example 3: Standalone Evaluation

Evaluate an already-quantized model:

```python
from llama_pajamas_quant import ModelEvaluator

evaluator = ModelEvaluator(judge_model="gpt-5-nano")

# Evaluate MLX model
evaluation = evaluator.evaluate_mlx(
    model_path="./models/qwen3-8b-mlx",
    quant_config={"quantization": "mlx", "bits": 4}
)

# Save results
evaluator.save_evaluation(evaluation, "./my_evaluation.json")

print(f"Quality: {evaluation.avg_quality:.1f}/10")
```

## Evaluation Results

### JSON Output Format

Evaluation results are saved as JSON files with this structure:

```json
{
  "model_name": "qwen3-8b-mlx",
  "model_path": "./models/qwen3-8b-mlx",
  "quantization_format": "mlx",
  "quantization_config": {
    "quantization": "mlx",
    "bits": 4,
    "mixed_precision": true
  },
  "prompt_results": [
    {
      "prompt": "Write a Python function...",
      "category": "code",
      "model_response": "def factorial(n)...",
      "generation_time_seconds": 2.5,
      "accuracy_score": 8.0,
      "coherence_score": 9.0,
      "relevance_score": 10.0,
      "quality_score": 8.5,
      "judge_reasoning": "The function is correct and well-structured...",
      "timestamp": "2025-11-07T..."
    }
  ],
  "aggregate_scores": {
    "accuracy": 8.2,
    "coherence": 8.8,
    "relevance": 9.1,
    "quality": 8.4,
    "generation_time": 2.6
  },
  "evaluation_timestamp": "2025-11-07T...",
  "judge_model": "gpt-5-nano",
  "hardware_info": {
    "platform": "Darwin",
    "processor": "arm",
    "python_version": "3.12.0",
    "ram_gb": "64.0"
  }
}
```

### Interpreting Scores

| Score | Interpretation |
|-------|----------------|
| 9-10  | Excellent - Production ready |
| 7-8   | Good - Acceptable for most use cases |
| 5-6   | Fair - Consider higher precision |
| 3-4   | Poor - Significant quality degradation |
| 0-2   | Failed - Not usable |

## Configuration

### Custom Evaluation Prompts

Create custom evaluation prompts for your domain:

```python
from llama_pajamas_quant import ModelEvaluator, EvaluationPrompt

# Define custom prompts
custom_prompts = [
    EvaluationPrompt(
        prompt="Explain quantum entanglement:",
        category="physics",
        expected_qualities=["accurate", "accessible", "comprehensive"]
    ),
    EvaluationPrompt(
        prompt="Write a Rust function to parse JSON:",
        category="code",
        expected_qualities=["correct", "idiomatic", "safe"]
    ),
]

# Use custom prompts
evaluator = ModelEvaluator(
    judge_model="gpt-5-nano",
    prompts=custom_prompts
)

evaluation = evaluator.evaluate_mlx(
    model_path="./models/my-model",
    quant_config={"quantization": "mlx", "bits": 4}
)
```

### Different Judge Models

While gpt-5-nano is recommended for cost-efficiency, you can use other models:

```python
# Use GPT-4o for more rigorous evaluation
quantizer.convert(
    model_path="Qwen/Qwen3-8B",
    output_dir="./models/qwen3-8b",
    evaluate=True,
    judge_model="gpt-4o",  # More expensive but more detailed
)
```

## CLI Usage

### Test API Connection

```bash
cd quant
uv run python test_judge_api.py
```

### Run Evaluation Examples

```bash
# Example 1: Basic evaluation
uv run python examples/quantize_with_evaluation.py --example 1

# Example 2: Compare configurations
uv run python examples/quantize_with_evaluation.py --example 2

# Example 3: MLX-only
uv run python examples/quantize_with_evaluation.py --example 3
```

## Cost Considerations

### GPT-5 Nano Pricing

- **Input**: $0.05 per 1M tokens
- **Output**: $0.40 per 1M tokens

### Typical Evaluation Costs

For standard 7-prompt evaluation:

- **Input tokens**: ~2,000 tokens (prompts + responses)
- **Output tokens**: ~500 tokens (scores + reasoning)
- **Cost per evaluation**: ~$0.0002 (0.02 cents)
- **Cost for dual-format**: ~$0.0004 (0.04 cents)

**Example**: Comparing 5 different quantization configs:
- 5 configs × $0.0004 = **$0.002 (0.2 cents)**

Very affordable for experimentation! 🎉

## Best Practices

### 1. Baseline Comparison

Always evaluate a baseline (FP16 or BF16) for reference:

```python
# First, evaluate unquantized model with a runtime
# Then compare against quantized versions
```

### 2. Consistent Hardware

Run evaluations on the same hardware for fair comparisons:
- Generation time varies by hardware
- Quality scores should be consistent

### 3. Statistical Significance

For critical decisions, run multiple evaluations:
- Prompts use temperature=1 (some randomness)
- Average across 2-3 runs for stability

### 4. Domain-Specific Prompts

Use prompts relevant to your use case:
- Code models → coding prompts
- Medical models → medical prompts
- General chat → diverse prompts

## Troubleshooting

### API Key Not Found

```
❌ OPEN_AI_KEY not found in environment
```

**Solution**: Create `.env` file in quant directory:

```bash
OPEN_AI_KEY=sk-proj-your-key-here
```

### Model Not Supported

```
Error code: 400 - model 'gpt-5-nano' not found
```

**Solution**: Verify model name or use fallback:

```python
judge_model="gpt-4o-mini"  # Alternative
```

### Evaluation Too Slow

**Solution**: Reduce number of prompts:

```python
from llama_pajamas_quant.evaluator import STANDARD_PROMPTS

# Use fewer prompts
evaluator = ModelEvaluator(
    prompts=STANDARD_PROMPTS[:3]  # Only 3 prompts
)
```

## Roadmap

Future enhancements planned:

- [ ] **Multi-model comparison reports** - HTML dashboards
- [ ] **Perplexity calculations** - Additional quality metrics
- [ ] **Regression detection** - Alert on quality drops
- [ ] **Custom metric weights** - Weight scores by importance
- [ ] **Batch evaluation** - Evaluate multiple models in parallel

## Examples

See `examples/quantize_with_evaluation.py` for complete working examples!

## Questions?

Check the main README or open an issue on GitHub.
