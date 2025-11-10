# ONNX Integration: Implementation Roadmap

**Current State**: Plan complete, ready for implementation
**Goal**: Add ONNX as 3rd format alongside GGUF and MLX
**Timeline**: 4 weeks (Week 1-2 running in parallel with current MVP work)

---

## Week 1: Foundation + TensorRT (Days 1-7)

### Day 1-2: Package Structure + Basic Export

**Tasks**:
1. Create `quant/llama_pajamas_quant/converters/onnx.py`
2. Create `quant/llama_pajamas_quant/optimizers/onnx_graph.py`
3. Create `quant/llama_pajamas_quant/optimizers/onnx_quant.py`
4. Add ONNX dependencies to `quant/pyproject.toml`
5. Basic export test (FP16, no quantization yet)

**Files to Create**:
```
quant/
├── llama_pajamas_quant/
│   ├── converters/
│   │   ├── gguf.py          # ✅ Exists
│   │   ├── mlx.py           # ✅ Exists
│   │   └── onnx.py          # ← NEW
│   └── optimizers/          # ← NEW directory
│       ├── __init__.py
│       ├── onnx_graph.py    # Graph optimizations
│       └── onnx_quant.py    # Quantization strategies
```

**Dependencies** (`quant/pyproject.toml`):
```toml
[project]
dependencies = [
    # ... existing ...
    "onnx>=1.15.0",
    "onnxruntime>=1.17.0",
    "optimum[onnxruntime]>=1.16.0",
]

[project.optional-dependencies]
onnx = [
    "onnx>=1.15.0",
    "onnxruntime>=1.17.0",
    "optimum[onnxruntime]>=1.16.0",
]
```

**Deliverable**:
```bash
# Test basic ONNX export
cd quant
uv run python test_onnx_export.py

# Should create:
# ./models/qwen3-8b/onnx/base/fp16/model.onnx (~16GB)
```

**Success Criteria**:
- ✅ ONNX graph exported successfully
- ✅ No errors from onnx.checker.check_model()
- ✅ Model loads in onnxruntime
- ✅ Single forward pass works

---

### Day 3-4: Graph Optimizations + TensorRT INT8

**Tasks**:
1. Implement `ONNXGraphOptimizer` with hint-driven optimizations
2. Implement `ONNXQuantizer.quantize_for_tensorrt()` (INT8 QDQ)
3. Test on Qwen3-8B with GQA hints
4. Validate graph structure (QDQ nodes inserted correctly)

**Deliverable**:
```bash
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-eps TensorRT \
  --target-precisions int8 \
  --optimization-hints '{"attention_type": "gqa", "gqa_ratio": 4}'

# Should create:
# ./models/qwen3-8b/onnx/tensorrt/int8/
#   ├── model.onnx (~8GB INT8)
#   ├── metadata.json
```

**Success Criteria**:
- ✅ INT8 quantization applied (model size ~50% of FP16)
- ✅ QDQ nodes present in graph
- ✅ GQA optimizations applied (if hints provided)
- ✅ metadata.json contains target_specs

---

### Day 5-7: TensorRT Runtime + Benchmarking

**Tasks**:
1. Create `run-onnx/` package structure
2. Implement `llama_pajamas_run_onnx/backends/tensorrt_backend.py`
3. Implement `llama_pajamas_run_onnx/session.py` (SessionOptions wrapper)
4. Load TensorRT INT8 model and benchmark on RTX GPU

**Package Structure**:
```
run-onnx/
├── llama_pajamas_run_onnx/
│   ├── __init__.py
│   ├── session.py              # SessionOptions manager
│   ├── loader.py               # Model loader (reads metadata)
│   └── backends/
│       ├── __init__.py
│       ├── base.py             # Base backend interface
│       ├── tensorrt_backend.py # ← Start here
│       ├── coreml_backend.py   # Week 2
│       ├── cuda_backend.py     # Week 2
│       └── cpu_backend.py      # Week 2
├── pyproject.toml
└── README.md
```

**Dependencies** (`run-onnx/pyproject.toml`):
```toml
[project]
name = "llama-pajamas-run-onnx"
version = "0.1.0"
dependencies = [
    "onnxruntime>=1.17.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
tensorrt = [
    "onnxruntime-gpu>=1.17.0",  # Includes TensorRT EP
]
cuda = [
    "onnxruntime-gpu>=1.17.0",
]
coreml = [
    "onnxruntime>=1.17.0",  # CoreML on Mac
]
```

**Deliverable**:
```bash
# Install runtime
cd run-onnx
uv pip install -e .

# Run interactive chat
llama-pajamas-run-onnx chat \
  --model ./models/qwen3-8b/onnx/tensorrt/int8/model.onnx \
  --backend tensorrt

# Expected: 100+ tok/s on RTX 4070
```

**Success Criteria**:
- ✅ TensorRT EP loads successfully
- ✅ Engine cache working (fast subsequent loads)
- ✅ Performance: 100+ tok/s on RTX 4070 (8B INT8)
- ✅ Quality: Coherent text generation
- ✅ IOBinding working (tensors stay on GPU)

**MILESTONE 1**: TensorRT INT8 working, faster than GGUF CUDA baseline

---

## Week 2: CoreML + Multi-EP + INT4 (Days 8-14)

### Day 8-9: CoreML Export + Quantization

**Tasks**:
1. Implement `ONNXQuantizer.quantize_for_coreml()` (symmetric INT8)
2. Test on M3 Max (or GitHub Actions Mac runner)
3. Validate ANE utilization

**Deliverable**:
```bash
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-eps CoreML \
  --target-precisions int8 \
  --optimization-hints '{"attention_type": "gqa", "gqa_ratio": 4}'

# Should create:
# ./models/qwen3-8b/onnx/coreml/int8/
#   ├── model.onnx (~8GB INT8, symmetric)
#   ├── metadata.json
```

**Success Criteria**:
- ✅ Symmetric INT8 (ANE requirement)
- ✅ No unsupported ops (all ops ANE-compatible or fallback to CPU/GPU)
- ✅ Model size ~8GB

---

### Day 10-11: CoreML Runtime + Testing

**Tasks**:
1. Implement `coreml_backend.py`
2. Test on M3 Max
3. Monitor ANE utilization (Activity Monitor)

**Deliverable**:
```bash
llama-pajamas-run-onnx chat \
  --model ./models/qwen3-8b/onnx/coreml/int8/model.onnx \
  --backend coreml

# Expected: 50-70 tok/s on M3 Max
```

**Success Criteria**:
- ✅ CoreML EP loads successfully
- ✅ Performance: 50-70 tok/s on M3 Max
- ✅ ANE utilization visible in Activity Monitor
- ✅ Compare vs MLX (MLX should be 10-20% faster, which is expected)

---

### Day 12-13: INT4 Support (TensorRT + CPU)

**Tasks**:
1. Implement `ONNXQuantizer.quantize_int4_tensorrt()`
2. Implement `ONNXQuantizer.quantize_int4_cpu()` (MatMulNBits)
3. Test both INT4 variants

**Deliverable**:
```bash
# TensorRT INT4
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-eps TensorRT \
  --target-precisions int4 \
  --optimization-hints '{"attention_type": "gqa", "gqa_ratio": 4}'

# CPU INT4
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-eps CPU \
  --target-precisions int4

# Should create:
# ./models/qwen3-8b/onnx/tensorrt/int4/ (~3.2GB)
# ./models/qwen3-8b/onnx/cpu/int4/ (~3.2GB)
```

**Success Criteria**:
- ✅ TensorRT INT4: ~3.2GB model size
- ✅ CPU INT4: MatMulNBits operator present
- ✅ Performance: TensorRT INT4 1.5x faster than INT8
- ✅ Quality: <8% loss vs FP16

---

### Day 14: Multi-EP Generation + Testing

**Tasks**:
1. Test generating multiple EPs in single convert command
2. Validate all combinations work
3. Benchmark comparison

**Deliverable**:
```bash
# Generate for multiple targets at once
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-eps TensorRT,CoreML,CPU \
  --target-precisions int8,int4 \
  --optimization-hints '{"attention_type": "gqa", "gqa_ratio": 4}'

# Should create 6 models:
# - onnx/tensorrt/int8/
# - onnx/tensorrt/int4/
# - onnx/coreml/int8/
# - onnx/cpu/int8/
# - onnx/cpu/int4/
```

**Success Criteria**:
- ✅ All 6 models generated successfully
- ✅ Each model loads in appropriate backend
- ✅ Performance comparison table generated

**MILESTONE 2**: Multi-EP support working, INT4 validated

---

## Week 3: Edge Devices + Optimization (Days 15-21)

### Day 15-16: Jetson Support (TensorRT INT8)

**Tasks**:
1. Tune TensorRT settings for Jetson (small VRAM)
2. Test on Jetson Orin (or document without hardware)
3. Create Jetson deployment guide

**Target Settings**:
```python
# Jetson-specific TensorRT options
trt_opts = {
    "trt_fp16_enable": False,  # Force INT8 only
    "trt_int8_enable": True,
    "trt_max_workspace_size": 1<<30,  # 1GB (conserve VRAM)
    "trt_engine_cache_enable": True,
}
```

**Success Criteria** (if hardware available):
- ✅ 8B INT8 model fits in 8GB VRAM
- ✅ Performance: 25-35 tok/s on Jetson Orin

---

### Day 17-18: Small GPU Support (RTX 3060)

**Tasks**:
1. Test INT4 on RTX 3060 (12GB VRAM)
2. Compare INT4 vs INT8 performance/quality
3. Document optimal settings

**Success Criteria**:
- ✅ INT4: 75+ tok/s on RTX 3060
- ✅ INT8: 65+ tok/s on RTX 3060
- ✅ Quality comparison documented

---

### Day 19-20: CPU Backend + Optimization

**Tasks**:
1. Implement `cpu_backend.py`
2. Test INT4 MatMulNBits on CPU
3. Test INT8 oneDNN on CPU

**Deliverable**:
```bash
llama-pajamas-run-onnx chat \
  --model ./models/qwen3-8b/onnx/cpu/int4/model.onnx \
  --backend cpu

# Expected: 12-18 tok/s on modern CPU (AVX-512)
```

**Success Criteria**:
- ✅ CPU INT4: 30-40% faster than INT8
- ✅ Reasonable performance on CPU-only systems
- ✅ Auto-detect AVX2/AVX-512 and use appropriate kernels

---

### Day 21: Edge Deployment Documentation

**Tasks**:
1. Create `EDGE_DEPLOYMENT.md`
2. Document Jetson, small GPU, CPU deployment
3. Hardware-specific optimization guides

**MILESTONE 3**: Edge deployment validated

---

## Week 4: Integration + Polish (Days 22-28)

### Day 22-23: Unified CLI Integration

**Tasks**:
1. Update main `llama-pajamas-quant` CLI to support ONNX
2. Update main `llama-pajamas-run` CLI to auto-select backend
3. Test unified workflow

**Deliverable**:
```bash
# Unified convert (GGUF + MLX + ONNX)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats gguf,mlx,onnx \
  --gguf-precision Q4_K_M \
  --mlx-bits 4 \
  --onnx-target-eps TensorRT,CoreML \
  --onnx-precisions int8

# Unified run (auto-selects best format)
llama-pajamas-run chat \
  --model ./models/qwen3-8b \
  --auto-select-format  # Picks ONNX TensorRT on NVIDIA, MLX on Mac, GGUF fallback
```

**Success Criteria**:
- ✅ Single command generates all 3 formats
- ✅ Runtime auto-detects optimal format per hardware
- ✅ Graceful fallback if EP not available

---

### Day 24-25: Quality Validation

**Tasks**:
1. Run benchmark suite on all ONNX models
2. Compare vs GGUF/MLX baselines
3. Document quality/speed tradeoffs

**Benchmark Plan**:
```bash
# Run comprehensive benchmarks
uv run python quant/scripts/benchmark_all_formats.py \
  --model ./models/qwen3-8b \
  --num-questions 120 \
  --formats gguf,mlx,onnx \
  --output ./models/qwen3-8b/BENCHMARK_COMPARISON.md
```

**Success Criteria**:
- ✅ ONNX TensorRT INT8: <5% quality loss vs FP16
- ✅ ONNX TensorRT INT4: <8% quality loss vs FP16
- ✅ Performance meets targets (100+ tok/s TensorRT, 50+ CoreML)

---

### Day 26-27: Documentation

**Tasks**:
1. Create `run-onnx/README.md` (comprehensive runtime guide)
2. Update main `README.md` with ONNX section
3. Create ONNX-specific examples
4. Document hardware profiles for ONNX

**Documentation Outline**:
```
run-onnx/README.md
├── Quick Start
├── Hardware-Specific Guides
│   ├── NVIDIA (TensorRT)
│   ├── Apple Silicon (CoreML)
│   ├── Jetson (Edge)
│   └── CPU-only
├── Performance Optimization
├── INT4 vs INT8 Guide
├── Troubleshooting
└── API Reference
```

**Success Criteria**:
- ✅ All hardware platforms documented
- ✅ Performance expectations clear
- ✅ Examples tested and working

---

### Day 28: Final Polish + Release Prep

**Tasks**:
1. Final testing on all platforms
2. CI/CD setup for ONNX tests
3. Version bump and release notes
4. Performance comparison table

**Deliverable**: Ready for v0.2.0 release with ONNX support

**MILESTONE 4**: ONNX integration complete and production-ready

---

## Post-Week 4: Future Phases

### Phase 5: Vision Models (2 weeks)
- CLIP, YOLO, Qwen3-VL support
- Vision-specific optimizations
- Mobile deployment (CoreML on iOS)

### Phase 6: STT Models (2 weeks)
- Whisper support (encoder/decoder split)
- KV-cache optimization
- Streaming inference

### Phase 7: TTS Models (2 weeks)
- VITS, Tacotron support
- Vocoder optimization
- Real-time synthesis

### Phase 8: Multimodal (3 weeks)
- Vision-Language models
- Component reuse strategy
- End-to-end pipelines

---

## Success Metrics (End of Week 4)

**Functional**:
- ✅ Qwen3-8B converts to ONNX (TensorRT, CoreML, CPU)
- ✅ INT8 and INT4 both working
- ✅ Multi-EP generation working
- ✅ All 3 runtimes functional (TensorRT, CoreML, CPU)

**Performance**:
- ✅ TensorRT INT8: 100+ tok/s (RTX 4070)
- ✅ TensorRT INT4: 130+ tok/s (RTX 4090)
- ✅ CoreML INT8: 50-70 tok/s (M3 Max)
- ✅ CPU INT4: 12-18 tok/s (modern CPU)

**Quality**:
- ✅ TensorRT INT8: <5% loss vs FP16
- ✅ TensorRT INT4: <8% loss vs FP16
- ✅ CoreML INT8: <6% loss vs FP16

**Integration**:
- ✅ Unified CLI (quant + run)
- ✅ Auto-format selection
- ✅ Comprehensive documentation
- ✅ All tests passing

---

## Risk Mitigation

**Risk 1: TensorRT EP not available**
- Mitigation: Fallback to CUDA EP
- Test: Validate CUDA fallback works

**Risk 2: CoreML quality poor**
- Mitigation: Use INT8 instead of lower precision
- Test: Early quality validation on M3 Max

**Risk 3: INT4 quality degradation**
- Mitigation: Document clearly, provide INT8 alternative
- Test: Compare INT4 vs INT8 quality on benchmarks

**Risk 4: Performance targets not met**
- Mitigation: Profile and optimize hot paths
- Test: Continuous benchmarking during development

---

## Next Steps

**Immediate (Day 1)**:
1. Create package structure (`quant/optimizers/`, `run-onnx/`)
2. Add ONNX dependencies to pyproject.toml
3. Implement basic `ONNXConverter.convert()` (FP16 export only)
4. Test basic export on Qwen3-8B

**Ready to start?** Let me know and I'll begin with Day 1 implementation!
