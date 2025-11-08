# lm-eval Integration Guide

## What lm-eval Requires

To work with lm-eval, a model backend must implement **3 core methods**:

### 1. `loglikelihood(requests)` → `list[tuple[float, bool]]`
**Purpose**: Compute log-probabilities for multiple-choice tasks (MMLU, HellaSwag, ARC, TruthfulQA)

**Input**: List of (context, continuation) pairs
- Example: ("Question: What is 2+2?\nA) 3\nB) 4\nC) 5\nAnswer:", "B")

**Output**: List of (log_prob, is_greedy) tuples
- `log_prob`: Sum of log probabilities for continuation tokens
- `is_greedy`: Whether continuation has highest probability

**How it works**:
1. Tokenize context + continuation
2. Run model forward pass to get logits
3. Calculate log-softmax over vocabulary
4. Sum log-probs at continuation token positions
5. Compare to greedy decoding to set `is_greedy`

### 2. `generate_until(requests)` → `list[str]`
**Purpose**: Generate free-form text (GSM8K, open-ended tasks)

**Input**: List of prompts with generation parameters
- `prompt`: Input text
- `until`: Stop sequences (e.g., ["\n", "Question:"])
- `max_tokens`: Maximum generation length

**Output**: List of generated strings

### 3. `loglikelihood_rolling(requests)` → `list[tuple[float, bool]]`
**Purpose**: Calculate rolling perplexity (less common, used for language modeling eval)

**Input**: Single long text
**Output**: Log-likelihood across entire sequence

---

## Current Runtime Capabilities

### ✅ What Our Runtimes Support

**MLX Backend** (`llama_pajamas_run/backends/mlx_backend.py`):
- ✅ `generate()` - Text generation
- ✅ `chat_completion()` - Chat format
- ✅ Streaming
- ❌ **No loglikelihood support**

**GGUF Backend** (`llama_pajamas_run/backends/gguf_backend.py`):
- ✅ `generate()` - Text generation (via `llama_cpp`)
- ✅ `chat_completion()` - Chat format
- ✅ Streaming
- ✅ **Has `logits_all=True` parameter** - can expose logits!
- ⚠️ **Partial loglikelihood support** - llama-cpp-python exposes logits, but we don't use them

---

## What We'd Need to Add

### Option 1: Full lm-eval Integration (Most Work, Best Compatibility)

Create `lm_eval` model adapters for our backends:

```python
# llama_pajamas_quant/lm_eval_models/mlx.py
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from llama_pajamas_run.backends.mlx_backend import MLXBackend

@register_model("llama-pajamas-mlx")
class LlamaPajamasMLX(LM):
    def __init__(self, model_path: str):
        self.backend = MLXBackend()
        self.backend.load_model(model_path)

    def loglikelihood(self, requests):
        """PROBLEM: MLX doesn't expose per-token logits easily!"""
        # Would need to:
        # 1. Get model.forward() to return logits
        # 2. Tokenize each request
        # 3. Calculate log-softmax
        # 4. Sum log-probs at continuation positions
        raise NotImplementedError("MLX doesn't expose logits in mlx-lm API")

    def generate_until(self, requests):
        """This works!"""
        results = []
        for req in requests:
            text = self.backend.generate(
                prompt=req.arguments[0],
                max_tokens=req.arguments[1].get('max_gen_toks', 100),
                stop=req.arguments[1].get('until', [])
            )
            results.append(text)
        return results

    def loglikelihood_rolling(self, requests):
        """Also needs logits"""
        raise NotImplementedError("MLX doesn't expose logits")
```

**GGUF version** would be similar but could use llama-cpp-python's logits:

```python
@register_model("llama-pajamas-gguf")
class LlamaPajamasGGUF(LM):
    def __init__(self, model_path: str):
        self.backend = GGUFBackend()
        self.backend.load_model(model_path)

    def loglikelihood(self, requests):
        """This CAN work with llama-cpp-python!"""
        results = []
        for context, continuation in requests:
            # Tokenize
            ctx_tokens = self.backend.model.tokenize(context.encode())
            cont_tokens = self.backend.model.tokenize(continuation.encode())

            # Run model with logits_all=True
            output = self.backend.model(
                ctx_tokens + cont_tokens,
                logits_all=True  # This returns all token logits!
            )

            # Calculate log-prob from logits
            # ... (complex logit processing here) ...

        return results
```

**Challenges**:
- ❌ **MLX**: `mlx-lm` doesn't expose per-token logits in the public API
  - Would need to modify `mlx-lm` or use lower-level MLX model.forward()
  - Significant engineering effort
- ✅ **GGUF**: llama-cpp-python has `logits_all=True` - can implement!
- 📦 Need to package as lm-eval plugin or register models

### Option 2: Minimal Integration (Generate-Only Tasks)

Only support tasks that use `generate_until`:
- ✅ GSM8K (math)
- ✅ HumanEval (code)
- ✅ Some instruction-following benchmarks
- ❌ MMLU, HellaSwag, ARC, TruthfulQA (need loglikelihood)

**Simpler to implement but loses 80% of benchmarks.**

### Option 3: Server-Based Approach (Current Attempt)

Use OpenAI-compatible servers + lm-eval's API backend:
- ✅ Works with `generate_until` tasks
- ❌ **Doesn't work** - chat-completions API doesn't expose logits
- ❌ Even completions API doesn't expose per-token logprobs in our servers

**This is what failed earlier** - the fundamental limitation.

### Option 4: Keep Simple Benchmarks (Current Working Solution)

Our custom benchmark runner:
- ✅ Tests same capabilities (MMLU, HellaSwag, GSM8K, ARC, TruthfulQA)
- ✅ Uses actual runtimes directly
- ✅ Fast (2-3 min vs 15-20 min)
- ✅ Already working!
- ⚠️ Not "industry standard" (but results are still valid)
- ⚠️ Smaller sample size (7 questions vs 100s)

---

## Recommended Path Forward

### Short-term (Now): Use Simple Benchmarks
- Already implemented and working
- Tests correct capabilities
- Fast iteration
- Scale to 50-100 questions for better statistical validity

### Medium-term (If Needed): GGUF lm-eval Integration
**GGUF is feasible** because llama-cpp-python exposes logits:

1. Create `llama_pajamas_quant/lm_eval_models/gguf.py`
2. Implement `loglikelihood()` using `logits_all=True`
3. Register with lm-eval: `@register_model("llama-pajamas-gguf")`
4. Run benchmarks: `lm_eval --model llama-pajamas-gguf --model_args model_path=...`

**Estimated effort**: 1-2 days

### Long-term (If Critical): MLX lm-eval Integration
**MLX is harder** - need to modify how we use mlx-lm:

1. Fork mlx-lm or use lower-level API
2. Access `model.forward()` directly for logits
3. Implement tokenization + logit processing
4. Same registration process

**Estimated effort**: 3-5 days (requires deeper MLX knowledge)

---

## Code Changes Summary

### For GGUF lm-eval Support (Feasible)

**New file**: `llama_pajamas_quant/lm_eval_models/gguf.py`
```python
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from llama_pajamas_run.backends.gguf_backend import GGUFBackend
import numpy as np

@register_model("llama-pajamas-gguf")
class LlamaPajamasGGUF(LM):
    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self.backend = GGUFBackend()
        self.backend.load_model(model_path, **kwargs)

    def loglikelihood(self, requests):
        results = []
        for ctx, cont in requests:
            # Get tokens
            ctx_toks = self.backend.model.tokenize(ctx.encode())
            cont_toks = self.backend.model.tokenize(cont.encode())
            full_toks = ctx_toks + cont_toks

            # Get logits for all tokens
            self.backend.model.reset()
            self.backend.model.eval(full_toks)
            logits = self.backend.model.eval_logits  # Access raw logits

            # Calculate log-softmax and sum for continuation
            log_probs = np.log(softmax(logits, axis=-1))
            cont_log_prob = sum(log_probs[len(ctx_toks):, cont_toks])

            # Check if greedy
            greedy_toks = np.argmax(logits, axis=-1)
            is_greedy = all(greedy_toks[len(ctx_toks):] == cont_toks)

            results.append((cont_log_prob, is_greedy))

        return results

    def generate_until(self, requests):
        # Already works!
        return [
            self.backend.generate(
                req.args[0],
                max_tokens=req.args[1].get('max_gen_toks', 100),
                stop=req.args[1].get('until', [])
            )
            for req in requests
        ]
```

**Changes to pyproject.toml**:
```toml
[project.entry-points."lm_eval.models"]
llama_pajamas_gguf = "llama_pajamas_quant.lm_eval_models.gguf:LlamaPajamasGGUF"
```

### For MLX lm-eval Support (Harder)

**Would need**:
1. New file: `llama_pajamas_quant/lm_eval_models/mlx.py`
2. Modify `llama_pajamas_run/backends/mlx_backend.py` to expose:
   - `get_logits(tokens)` method
   - Direct model access for forward passes
3. Handle tokenization manually (mlx-lm wraps this)

---

## Bottom Line

**Question**: "What would we have to do to fix the runtimes to work with lm-eval?"

**Answer**:

1. **GGUF**: ~1-2 days of work
   - Implement `loglikelihood()` using llama-cpp-python's logits API
   - Register as lm-eval model
   - **Definitely feasible**

2. **MLX**: ~3-5 days of work
   - Requires low-level MLX API usage
   - mlx-lm doesn't expose what we need
   - **Harder, may not be worth it**

3. **Alternative**: Keep simple benchmarks
   - Already working
   - Just expand question set from 7 → 50-100
   - Gets 95% of the value with 5% of the effort

**My recommendation**: Stick with simple benchmarks for now, implement GGUF lm-eval integration if/when you need "industry standard" validation for papers, documentation, or marketing.
