# Evaluation System

This directory contains scripts for evaluating quantized models and generating comparison reports.

## Overview

The evaluation system has two main scripts:

1. **`evaluate_model.py`** - Evaluates one or more models and saves results in model-specific folders
2. **`compare_evaluations.py`** - Generates a markdown comparison report from all evaluations

## Workflow

### Step 1: Evaluate Models

Evaluate individual models or multiple models at once. Results are saved in model-specific folders:

```bash
# Evaluate single GGUF model
uv run python scripts/evaluate_model.py \
    --model-path ./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf \
    --format gguf

# Evaluate multiple GGUF models at once
uv run python scripts/evaluate_model.py \
    --model-path ./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf \
    --model-path ./models/qwen3-8b/gguf/Q3_K_M/qwen3-8b-q3_k_m.gguf \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf \
    --format gguf

# Evaluate MLX model
uv run python scripts/evaluate_model.py \
    --model-path ./models/qwen3-8b/mlx/4bit-mixed \
    --format mlx

# Limit to first N questions (for quick testing)
uv run python scripts/evaluate_model.py \
    --model-path ./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf \
    --format gguf \
    --num-questions 50
```

**Output:**
- GGUF: `./models/qwen3-8b/gguf/IQ2_XS/evaluation.json`
- MLX: `./models/qwen3-8b/mlx/4bit-mixed/evaluation.json`

### Step 2: Generate Comparison Report

After evaluating all models, generate a comprehensive markdown comparison:

```bash
uv run python scripts/compare_evaluations.py --model-dir ./models/qwen3-8b
```

**Output:**
- `./models/qwen3-8b/EVALUATION_REPORT.md`

## Evaluation Format

Each `evaluation.json` contains:

```json
{
  "format": "gguf",
  "model_path": "./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf",
  "accuracy": 0.85,
  "correct": 102,
  "total": 120,
  "avg_time": 0.45,
  "category_stats": {
    "knowledge": {
      "correct": 23,
      "total": 25,
      "accuracy": 0.92
    },
    "math": {
      "correct": 20,
      "total": 25,
      "accuracy": 0.80
    }
  },
  "results": [...],
  "timestamp": "2025-01-08 14:30:00"
}
```

## Benchmark Questions

The evaluation uses 120 questions across 6 categories:

- **Knowledge** (25 questions) - MMLU-style general knowledge
- **Common Sense** (20 questions) - HellaSwag-style reasoning
- **Math** (25 questions) - GSM8K-style arithmetic
- **Reasoning** (20 questions) - ARC-style scientific reasoning
- **Truthfulness** (20 questions) - TruthfulQA-style myth-busting
- **Tool Calling** (30 questions) - Function calling ability

## Example Workflow

Complete evaluation workflow for all Qwen3-8B quantizations:

```bash
# 1. Evaluate all GGUF models
uv run python scripts/evaluate_model.py \
    --model-path ./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf \
    --model-path ./models/qwen3-8b/gguf/Q3_K_M/qwen3-8b-q3_k_m.gguf \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf \
    --format gguf

# 2. Evaluate all MLX models
uv run python scripts/evaluate_model.py \
    --model-path ./models/qwen3-8b/mlx/2bit-mixed \
    --model-path ./models/qwen3-8b/mlx/3bit-mixed \
    --model-path ./models/qwen3-8b/mlx/4bit-mixed \
    --format mlx

# 3. Generate comparison report
uv run python scripts/compare_evaluations.py --model-dir ./models/qwen3-8b

# 4. View report
cat ./models/qwen3-8b/EVALUATION_REPORT.md
```

## Directory Structure

After running evaluations:

```
models/qwen3-8b/
├── gguf/
│   ├── IQ2_XS/
│   │   ├── qwen3-8b-f16-iq2_xs.gguf
│   │   └── evaluation.json          ← Evaluation results
│   ├── Q3_K_M/
│   │   ├── qwen3-8b-q3_k_m.gguf
│   │   └── evaluation.json          ← Evaluation results
│   └── Q4_K_M/
│       ├── qwen3-8b-q4_k_m.gguf
│       └── evaluation.json          ← Evaluation results
├── mlx/
│   ├── 2bit-mixed/
│   │   ├── config.json
│   │   ├── weights.safetensors
│   │   └── evaluation.json          ← Evaluation results
│   ├── 3bit-mixed/
│   │   └── evaluation.json          ← Evaluation results
│   └── 4bit-mixed/
│       └── evaluation.json          ← Evaluation results
└── EVALUATION_REPORT.md             ← Comparison report
```

## Tips

1. **Quick Testing**: Use `--num-questions 50` to test with just 50 questions (~2-3 min)
2. **Full Evaluation**: Omit `--num-questions` for all 120 questions (~5-8 min per model)
3. **Batch Evaluation**: Evaluate multiple models in one command using multiple `--model-path` arguments
4. **MLX vs GGUF**: Run separate commands for MLX and GGUF formats (they use different backends)

## Troubleshooting

### Model not found
```bash
# Check model paths
ls -la ./models/qwen3-8b/gguf/*/
ls -la ./models/qwen3-8b/mlx/*/
```

### No evaluations found
```bash
# Make sure you ran evaluate_model.py first
find ./models/qwen3-8b -name "evaluation.json"
```

### MLX import error
```bash
# Install MLX dependencies
uv pip install mlx mlx-lm
```

### GGUF import error
```bash
# Install llama-cpp-python
uv pip install llama-cpp-python
```
