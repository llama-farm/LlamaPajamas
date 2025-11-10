# Llama-Pajamas Evaluation Suite

Comprehensive evaluation system for LLM and vision models across multiple runtimes (GGUF, MLX, CoreML, ONNX).

## Directory Structure

```
evaluation/
├── README.md                    # This file
├── llm/                         # LLM evaluation
│   ├── questions.json          # 140 standardized questions
│   ├── run_eval.py             # Evaluation script
│   └── compare_evaluations.py  # Comparison/reporting tool
└── vision/                      # Vision evaluation
    ├── dataset.json            # Vision dataset metadata
    ├── run_eval.py             # Vision evaluation script
    └── images/                 # Test images
        ├── detection/          # Object detection (5 images)
        ├── classification/     # Image classification (1 image)
        └── embedding/          # Vision embeddings
```

## LLM Evaluation

### Quick Start

```bash
cd /Users/robthelen/llama-pajamas/quant

# Evaluate single GGUF model (140 questions, ~2-3 minutes)
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf

# Evaluate multiple GGUF models
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q3_K_M/*.gguf \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf

# Evaluate MLX model
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/mlx/4bit-mixed \
    --format mlx

# Quick test with fewer questions (10 questions, ~30 seconds)
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf \
    --num-questions 10

# Use LLM-as-judge for summarization evaluation (optional)
export OPENAI_API_KEY=sk-...
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf \
    --use-llm-judge

# Generate comparison report
uv run python evaluation/llm/compare_evaluations.py \
    --model-dir ./models/qwen3-8b
```

### Questions

The evaluation uses **140 standard questions** across 6 categories:

- **Knowledge** (25): MMLU-style general knowledge
- **Common Sense** (20): HellaSwag-style reasoning
- **Math** (25): GSM8K-style arithmetic
- **Reasoning** (20): ARC-style science/logic
- **Truthfulness** (20): TruthfulQA-style fact-checking
- **Tool Calling** (30): BFCL-style function selection

**Extended questions** (20 additional):
- **Tool Calling Extended** (5): Complex multi-function scenarios with dependencies
- **Summarization** (5): Requires LLM-as-judge evaluation (optional)
- **Code Generation** (5): Multi-language code generation tasks
- **Long Context** (5): Multi-step reasoning with extended context

Questions are defined in:
- `llm/questions.json` - 140 standard questions
- `llm/extended_questions.json` - 20 extended questions
- `llm/llm_judge.py` - Optional LLM-as-judge for summarization

### Results

Evaluation results are saved in each model's directory:

```
models/qwen3-8b/
   gguf/Q4_K_M/evaluation.json     # Individual results
   mlx/4bit-mixed/evaluation.json
   EVALUATION_REPORT.md              # Comparison report
```

## Vision Evaluation

### Quick Start

```bash
cd /Users/robthelen/llama-pajamas/quant

# Evaluate all vision models
uv run python evaluation/vision/run_eval.py \
    --models-dir ../models \
    --images evaluation/vision/images/detection

# Evaluate specific model
uv run python evaluation/vision/run_eval.py \
    --model yolo-v8n \
    --models-dir ../models \
    --images evaluation/vision/images/detection

# Custom parameters for detection
uv run python evaluation/vision/run_eval.py \
    --model yolo-v8n \
    --conf-threshold 0.3 \
    --models-dir ../models \
    --images evaluation/vision/images/detection
```

### Tasks

- **Object Detection** (YOLO-v8n): FPS, latency, detections/image, confidence
- **Image Classification** (ViT-base): FPS, latency, top-1 confidence
- **Vision Embeddings** (CLIP-ViT-base): FPS, latency, similarity scores

### Results

Evaluation results are saved in each model's directory:

```
models/yolo-v8n/
   coreml/fp16/evaluation.json      # Performance & quality metrics
   EVALUATION_REPORT.md               # Human-readable report
```

## Dataset Expansion

### Current Dataset

- **LLM**: 140 standard questions + 20 extended questions
  - Standard: 140 questions (knowledge, math, reasoning, tool calling, etc.)
  - Extended: 20 questions (long context, complex tool calling, summarization, code generation)
- **Vision**: 5-6 test images (expandable to 200-300)

### Download Vision Dataset (200-300 images)

```bash
# Download 200 diverse images from Open Images V7
cd evaluation/vision
uv run python download_dataset.py --num-images 200

# Download more images (300-500)
uv run python download_dataset.py --num-images 300

# Custom output directory
uv run python download_dataset.py --num-images 200 --output ./custom_images

# This creates:
# - images/detection/*.jpg (~50% of images, diverse objects)
# - images/classification/*.jpg (~35% of images, various categories)
# - images/embedding/*.jpg (~15% for similarity testing)
# - images/dataset_metadata.json (dataset info with source URLs)
```

**Download Results** (tested with 200 requested images):
- ✅ Successfully downloaded: **176 images** (88% success rate)
- ❌ Failed: 24 images (404 errors from expired Flickr URLs)
- 📊 Source: Open Images validation set (41,620 images available)
- 📜 License: CC BY 4.0
- 🎯 Includes: Diverse real-world objects, scenes, and contexts

**Note**: The downloader uses Open Images V7 validation set. Some URLs may be unavailable (404 errors) as original Flickr photos are removed over time. 80-90% success rate is typical.

## Adding New Evaluations

### Add LLM Questions

Edit `llm/questions.json`:

```json
{
  "questions": [
    {
      "prompt": "Your question here\nA) Option 1\nB) Option 2\nAnswer:",
      "expected": "A",
      "category": "knowledge"
    }
  ]
}
```

### Add Vision Images

```bash
# Copy images to appropriate directory
cp /path/to/images/*.jpg evaluation/vision/images/detection/

# Update dataset.json with new count
```

## Comparison & Reporting

### LLM Models

```bash
uv run python evaluation/llm/compare_evaluations.py \
    --model-dir ./models/qwen3-8b
```

Generates `models/qwen3-8b/EVALUATION_REPORT.md` with:
- Overall accuracy by format/variant
- Category breakdown
- Speed comparison
- Size efficiency (accuracy per GB)

### Vision Models

Comparison reports are auto-generated per model:

```bash
cat models/yolo-v8n/EVALUATION_REPORT.md
```

## Integration with CI/CD

```bash
# Run full evaluation suite
./evaluation/run_all.sh

# Check if accuracy meets threshold
uv run python evaluation/check_thresholds.py \
    --min-accuracy 0.85 \
    --max-latency-ms 100
```

## LLM-as-Judge (Optional)

For summarization and code generation tasks, simple string matching isn't sufficient. Enable optional LLM-as-judge evaluation:

```bash
# Set API key (OpenAI or Anthropic)
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

# Run evaluation with LLM judge
uv run python evaluation/llm/run_eval.py \
    --model-path ./models/qwen3-8b/gguf/Q4_K_M/*.gguf \
    --format gguf \
    --use-llm-judge

# Use specific judge model
export LLM_JUDGE_MODEL=gpt-5-nano

# Set custom score threshold (default: 7.0/10)
uv run python evaluation/llm/run_eval.py \
    --use-llm-judge \
    --judge-threshold 8.0
```

**LLM Judge evaluates**:
- **Summarization**: Accuracy, completeness, conciseness, coherence
- **Code Generation**: Correctness, code quality, completeness, efficiency
- **Long Context**: Complex reasoning quality

**Cost Note**: LLM judge adds API costs. Disable for routine testing, enable for thorough evaluation.

## Extended Evaluation Questions

The `extended_questions.json` file contains 20 additional questions for advanced testing:

1. **Complex Tool Calling** (5 questions): Multi-step function orchestration
   - Example: "Email John about meeting, schedule it, create todo for slides"
   - Tests: Function dependency understanding, parameter extraction

2. **Summarization** (5 questions): Requires LLM judge
   - Example: Summarize research papers, news articles, meeting transcripts
   - Tests: Information compression, key point extraction

3. **Code Generation** (5 questions): Multi-language programming
   - Example: "Write Python function for LCS using dynamic programming"
   - Tests: Algorithm correctness, code quality, documentation

4. **Long Context** (5 questions): Extended inputs with multi-step reasoning
   - Example: Complex business scenarios, research study analysis
   - Tests: Information retention, logical reasoning across paragraphs

Use extended questions for comprehensive model assessment:
```bash
# Merge standard + extended questions (160 total)
python -c "
import json
std = json.load(open('evaluation/llm/questions.json'))
ext = json.load(open('evaluation/llm/extended_questions.json'))
std['questions'].extend(ext['questions'])
std['total_questions'] = len(std['questions'])
json.dump(std, open('evaluation/llm/questions_full.json', 'w'), indent=2)
"

# Run evaluation with all 160 questions
uv run python evaluation/llm/run_eval.py --questions evaluation/llm/questions_full.json
```

## Notes

- **LLM evaluations** take ~2-3 minutes per model (140 questions)
- **Extended evaluations** take ~5-10 minutes per model (160 questions + LLM judge)
- **Vision evaluations** take ~30 seconds per model (5-6 images)
- **Vision with 300 images** takes ~5-10 minutes per model
- Results are deterministic (temperature=0.1 for LLM)
- All evaluations use production runtime configurations
- LLM judge is optional and adds API costs
