# Multi-Domain Calibration Data for GGUF/IQ Quantization

This directory contains domain-specific calibration datasets optimized for importance quantization (IQ) of language models. Each domain focuses on specific use cases to create hyper-quantized models that excel in targeted applications.

## Overview

Importance Quantization (IQ) uses calibration data to determine which model weights are most critical for specific tasks. By using domain-specific calibration data, you can create ultra-compressed models that maintain high quality for your specific use case.

## Available Domains

### Built-in Domains (Seed Data)

Located in `domains/` directory:

1. **Tool Calling** (`calibration_tool_calling.txt`)
   - **Samples**: 80 prompts
   - **Use Case**: Function calling, API interactions, structured outputs
   - **Focus**: JSON generation, tool selection, parameter extraction
   - **Example**: "Search the web for recent news about quantum computing and summarize the top 3 articles"

2. **Summarization** (`calibration_summarization.txt`)
   - **Samples**: 25 prompts
   - **Use Case**: Text compression, abstraction, key point extraction
   - **Focus**: Technical articles, business reports, scientific papers
   - **Example**: "Summarize this 2000-word technical article on Kubernetes architecture"

3. **RAG - Retrieval-Augmented Generation** (`calibration_rag.txt`)
   - **Samples**: 13 context-question pairs
   - **Use Case**: Question answering over documents
   - **Focus**: Context comprehension, factual extraction, reasoning
   - **Example**: "Given this PostgreSQL documentation, what should pool_size be set to?"

4. **Military** (`calibration_military.txt`)
   - **Samples**: 20 prompts
   - **Use Case**: Military, defense, tactical planning
   - **Focus**: Operations planning, threat assessment, logistics
   - **Example**: "Analyze tactical advantages of forward operating bases in mountainous terrain"

5. **Medical** (`calibration_medical.txt`)
   - **Samples**: 25 prompts
   - **Use Case**: Healthcare, clinical diagnosis, treatment planning
   - **Focus**: Pathophysiology, pharmacology, differential diagnosis
   - **Example**: "A 45-year-old presents with acute chest pain. Describe differential diagnosis and interventions"

6. **Tone Analysis** (`calibration_tone_analysis.txt`)
   - **Samples**: 25 prompts
   - **Use Case**: Sentiment detection, emotional intelligence
   - **Focus**: Emotion recognition, sarcasm detection, communication style
   - **Example**: "Identify the tone and passive-aggressive elements in this customer email"

7. **General** (`calibration_general.txt`)
   - **Samples**: Combined tool_calling + summarization + rag
   - **Use Case**: Balanced general-purpose quantization
   - **Focus**: Broad coverage across multiple task types

## Usage

### 1. Using CLI with Domain Selection

The simplest way to use domain-specific calibration:

```bash
# Quantize with medical domain calibration
llama-pajamas-quant iq quantize \
  --model model.gguf \
  --domain medical \
  --precision IQ2_XS \
  --output ./output/

# Quantize with military domain calibration
llama-pajamas-quant iq quantize \
  --model model.gguf \
  --domain military \
  --precision IQ3_XS \
  --output ./output/
```

### 2. Using Calibration Files Directly

```bash
# Use a specific calibration file
llama-pajamas-quant iq quantize \
  --model model.gguf \
  --calibration ./calibration_data/domains/calibration_medical.txt \
  --precision IQ2_XS \
  --output ./output/
```

### 3. Exporting Built-in Calibration Data

```bash
# Export a specific domain
llama-pajamas-quant calibration export \
  --domain medical \
  --output ./my_calibration.txt

# List all available domains
llama-pajamas-quant calibration list
```

## Generating Synthetic Domain-Specific Data

For maximum optimization, generate synthetic calibration data tailored to your specific use case:

### Basic Synthetic Generation

```bash
llama-pajamas-quant calibration generate \
  --domain medical \
  --output ./calibration_data \
  --num-samples 200 \
  --provider anthropic
```

### Custom Synthetic Generation

Provide your own purpose description and examples:

```bash
llama-pajamas-quant calibration generate \
  --domain medical \
  --output ./calibration_data \
  --num-samples 250 \
  --custom-purpose "Emergency medicine triage and acute care decision support" \
  --custom-examples \
    "A trauma patient arrives with multiple injuries..." \
    "Prioritize treatment for a mass casualty event..." \
    "Assess ABCDE in a critical patient..." \
  --provider anthropic
```

### Requirements for Synthetic Generation

You need an API key for either:
- **Anthropic Claude** (recommended): Set `ANTHROPIC_API_KEY` environment variable
- **OpenAI GPT-4**: Set `OPENAI_API_KEY` environment variable

```bash
# Using Anthropic (default)
export ANTHROPIC_API_KEY="your-key-here"
llama-pajamas-quant calibration generate --domain medical --output ./data --num-samples 200

# Using OpenAI
export OPENAI_API_KEY="your-key-here"
llama-pajamas-quant calibration generate --domain medical --output ./data --num-samples 200 --provider openai
```

## Choosing the Right Domain

| Use Case | Recommended Domain | Sample Count | Notes |
|----------|-------------------|--------------|-------|
| General chatbot | `general` | 118+ | Balanced coverage |
| API/Function calling agent | `tool_calling` | 80 | Structured outputs |
| Document Q&A system | `rag` | 13 | Context comprehension |
| Medical assistant | `medical` | 25+ | Clinical knowledge |
| Defense/tactical analysis | `military` | 20+ | Strategic reasoning |
| Sentiment analysis | `tone_analysis` | 25+ | Emotional intelligence |
| Content summarization | `summarization` | 25 | Compression ability |

## Best Practices

### 1. Sample Count
- **Minimum**: 150 samples for basic calibration
- **Recommended**: 200-300 samples for optimal results
- **Maximum**: 500+ samples for very large models

### 2. Domain Selection
- **Single-purpose models**: Use domain-specific calibration (medical, military, tone_analysis)
- **Multi-purpose models**: Use `general` or combine multiple domains
- **Specialized workflows**: Generate synthetic data with custom examples

### 3. Combining Domains

Create a custom combined calibration file:

```bash
cat domains/calibration_medical.txt \
    domains/calibration_tone_analysis.txt \
    > custom_medical_empathy.txt

llama-pajamas-quant iq quantize \
  --calibration custom_medical_empathy.txt \
  --model model.gguf \
  --precision IQ2_XS \
  --output ./output/
```

### 4. Quality vs Size Trade-off

| Precision | Size Reduction | Quality | Best For |
|-----------|---------------|---------|----------|
| IQ2_XS | ~2 bits | Good with calibration | Maximum compression |
| IQ3_XS | ~3 bits | Very good | Balanced |
| IQ4_XS | ~4 bits | Excellent | Quality-focused |

## Workflow Examples

### Example 1: Medical Diagnosis Assistant

```bash
# Step 1: Generate synthetic medical calibration data (300 samples)
llama-pajamas-quant calibration generate \
  --domain medical \
  --output ./medical_cal \
  --num-samples 300

# Step 2: Quantize model with medical calibration
llama-pajamas-quant iq quantize \
  --model Qwen2.5-7B-Instruct-F16.gguf \
  --calibration ./medical_cal/calibration_medical.txt \
  --precision IQ3_XS \
  --output ./medical_model/

# Result: 7B model compressed to ~2.6GB optimized for medical queries
```

### Example 2: Military Intelligence Analysis

```bash
# Use built-in military domain
llama-pajamas-quant iq quantize \
  --model Llama-3-8B-F16.gguf \
  --domain military \
  --precision IQ2_XS \
  --output ./military_model/

# Result: 8B model compressed to ~2.2GB optimized for military analysis
```

### Example 3: Customer Support (Tone Analysis + Tool Calling)

```bash
# Combine tone analysis and tool calling
cat domains/calibration_tone_analysis.txt \
    domains/calibration_tool_calling.txt \
    > customer_support_cal.txt

llama-pajamas-quant iq quantize \
  --model model.gguf \
  --calibration customer_support_cal.txt \
  --precision IQ3_XS \
  --output ./support_model/

# Result: Model optimized for empathetic responses and tool use
```

## File Structure

```
calibration_data/
├── README.md                           # This file
├── domains/                            # Pre-built domain calibration files
│   ├── calibration_tool_calling.txt    # 80 samples (6.3 KB)
│   ├── calibration_summarization.txt   # 25 samples (80.9 KB)
│   ├── calibration_rag.txt             # 13 samples (28.8 KB)
│   ├── calibration_military.txt        # 20 samples (3.4 KB)
│   ├── calibration_medical.txt         # 25 samples (5.3 KB)
│   ├── calibration_tone_analysis.txt   # 25 samples (5.5 KB)
│   └── calibration_general.txt         # Combined (116.0 KB)
├── calibration.txt                     # Original combined calibration (legacy)
└── evaluation.txt                      # Evaluation dataset (legacy)
```

## Technical Details

### Calibration Data Format

Plain text file with one prompt per line (or structured format for RAG):

```
Prompt 1 text here
Prompt 2 text here
Prompt 3 text here
```

For RAG domain:
```
Context 1:
[Long context document]

Question: [Question about context]

Context 2:
[Another context document]

Question: [Another question]
```

### How IQ Quantization Works

1. **IMatrix Generation**: Feed calibration data through model to measure weight importance
2. **Importance Scoring**: Calculate which weights activate most for your domain
3. **Smart Quantization**: Preserve high-importance weights, aggressively quantize low-importance ones
4. **Result**: Dramatically smaller model with maintained quality for your specific use case

## Extending with New Domains

To create a new domain:

1. **Create seed examples** (5-10 representative prompts)
2. **Generate synthetic data**:
   ```bash
   llama-pajamas-quant calibration generate \
     --domain custom_domain \
     --custom-purpose "Your detailed domain description" \
     --custom-examples "Example 1" "Example 2" "Example 3" \
     --output ./calibration_data \
     --num-samples 200
   ```
3. **Test and iterate**: Evaluate quantized model quality
4. **Refine**: Adjust examples and regenerate if needed

## Performance Tips

1. **Use adequate samples**: 200-300 is the sweet spot for most models
2. **Match your use case**: Domain-specific calibration >> general calibration
3. **Test different precisions**: IQ3_XS often best quality/size balance
4. **Validate output**: Always test quantized model on your actual workload

## Questions?

- List available domains: `llama-pajamas-quant calibration list`
- Export seed data: `llama-pajamas-quant calibration export --domain <domain>`
- Generate synthetic: `llama-pajamas-quant calibration generate --help`
- Quantize with domain: `llama-pajamas-quant iq quantize --domain <domain>`

## References

- [IQ Quantization Guide](../docs/iq-quantization-guide.md)
- [Architecture-Aware Quantization](../docs/architecture-aware-quantization.md)
- [CLI Documentation](../README.md)
