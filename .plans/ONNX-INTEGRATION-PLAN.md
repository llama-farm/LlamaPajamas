# Llama-Pajamas ONNX Integration Plan

**Version**: 0.2.0 (Post-MVP Extension)
**Focus**: Apple Silicon + Desktop/Small NVIDIA GPUs
**Modality**: LLMs (with roadmap for Vision, TTS)
**Timeline**: 4 weeks post-MVP
**Philosophy**: Thin and deep - optimize the hell out of target platforms

---

## Executive Summary

### Our ONNX Point of View

**❌ WRONG: "ONNX as universal export format"**
- Traditional approach: Export → Deploy same ONNX everywhere
- Result: Suboptimal performance (20-40% slower than specialized runtimes)
- Result: Large binaries (full ORT + all EPs = 500MB+)

**✅ RIGHT: "ONNX as hardware-specific optimization target"**
- **Llama-Pajamas ONNX**: Generate EP-specific optimized models per hardware
- Result: Apple Silicon gets CoreML-optimized QDQ models (ANE acceleration)
- Result: NVIDIA gets TensorRT-compiled engines (FP16/INT8 kernels)
- Result: 10-30% faster than generic ONNX, 300MB+ smaller runtimes
- Result: Architecture-aware quantization BEFORE ONNX export (MoE-balanced calibration)

### Why ONNX After GGUF/MLX?

**Complementary strengths**:

```yaml
GGUF (llama.cpp):
  Best for: Inference flexibility, rapid deployment, CPU excellence
  Weaknesses: Mobile deployment, embedded systems, cloud-edge optimization

MLX:
  Best for: Apple Silicon maximum performance
  Weaknesses: Apple-only, no mobile/iOS support yet

ONNX:
  Best for: Production deployment, mobile (iOS/Android), edge devices, cloud serving
  Strengths: Execution Provider ecosystem, compiler optimizations, mobile runtimes
  Fills gaps: iOS (CoreML), Android (QNN/NNAPI), Jetson (TensorRT), Web (WASM/WebGPU)
```

**Our strategy**: Generate all three formats, let deployment context choose optimal runtime.

---

## Part 1: The ONNX Runtime Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  LLAMA-PAJAMAS ONNX SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         QUANTIZATION PIPELINE (llama-pajamas-quant)        │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                             │ │
│  │  HuggingFace Model (FP16/BF16)                             │ │
│  │         +                                                   │ │
│  │  Target Architectures (USER SPECIFIES)                    │ │
│  │    - target_eps: ["TensorRT", "CoreML", "CPU"]            │ │
│  │    - target_precisions: ["int8", "fp16"]                  │ │
│  │    - optimization_hints: {moe_experts: 128, gqa_ratio: 4} │ │
│  │         ↓                                                   │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │      ONNX Converter (NEW - receives targets)        │   │ │
│  │  ├─────────────────────────────────────────────────────┤   │ │
│  │  │  Input: model, target_specs, calibration_data      │   │ │
│  │  │                                                      │   │ │
│  │  │  1. Export to ONNX (torch.onnx / Optimum)          │   │ │
│  │  │     - Use user-provided optimization hints         │   │ │
│  │  │     - Dynamic axes for flexible deployment         │   │ │
│  │  │  2. Graph Optimization (target-aware)              │   │ │
│  │  │     - Apply hints: MoE routing, GQA KV-cache, etc │   │ │
│  │  │     - EP-specific fusions (TensorRT vs CoreML)    │   │ │
│  │  │  3. Generate EP-Specific Models:                   │   │ │
│  │  │     FOR EACH target_ep:                            │   │ │
│  │  │       - TensorRT: QDQ INT8 (per-channel)          │   │ │
│  │  │       - CoreML: QDQ INT8 (symmetric, ANE)         │   │ │
│  │  │       - CPU: QOperator INT8 (oneDNN)              │   │ │
│  │  │  4. EP-Specific Compilation (optional):            │   │ │
│  │  │     - TensorRT: Build + cache engines             │   │ │
│  │  │     - OpenVINO: IR export                          │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │         ↓                                                   │ │
│  │  Output: models/{name}/onnx/{ep}/{precision}/              │ │
│  │    - model.onnx (base graph)                               │ │
│  │    - model_optimized_{ep}.onnx (EP-specific)               │ │
│  │    - model_{ep}.trt (TensorRT engine, if applicable)       │ │
│  │    - metadata.json (target_specs, settings)                │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         │                                        │
│                         │ Deploy artifacts                       │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │      ONNX RUNTIME (llama-pajamas-run-onnx) (NEW)           │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                             │ │
│  │  Quantized ONNX Artifacts                                  │ │
│  │         ↓                                                   │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │  Model Loader (reads metadata.json)                │   │ │
│  │  │    - Detects target EP from manifest               │   │ │
│  │  │    - Auto-fallback: CoreML → CPU on Mac            │   │ │
│  │  │    - Auto-fallback: TensorRT → CUDA → CPU on NVIDIA│   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │         ↓                                                   │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │  Session Manager (THIN wrapper over ORT)           │   │ │
│  │  │    - SessionOptions: graph_opt_level=ALL           │   │ │
│  │  │    - Provider selection: [Primary, Fallback, CPU]  │   │ │
│  │  │    - IOBinding: Device-resident tensors            │   │ │
│  │  │    - Thread tuning: intra_op, inter_op             │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │         ↓                                                   │ │
│  │  ┌──────────────────────────────────────────┐             │ │
│  │  │    Execution Provider Backends           │             │ │
│  │  ├──────────────────────────────────────────┤             │ │
│  │  │  CoreML (Apple Silicon)                  │             │ │
│  │  │  TensorRT (NVIDIA)                       │             │ │
│  │  │  CUDA (NVIDIA fallback)                  │             │ │
│  │  │  CPU (oneDNN / XNNPACK)                  │             │ │
│  │  └──────────────────────────────────────────┘             │ │
│  │         ↓                                                   │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │  LLM-Specific Optimizations                         │   │ │
│  │  │    - KV-cache on device (IOBinding)                 │   │ │
│  │  │    - Tight generate loop (no host copies)           │   │ │
│  │  │    - ORT GenAI integration (optional)               │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │         ↓                                                   │ │
│  │  Generated Output (text, embeddings)                       │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Package Structure

```
llama-pajamas/
├── quant/                           # Existing quantization pipeline
│   └── llama_pajamas_quant/
│       ├── converters/
│       │   ├── gguf.py             # ✅ Existing
│       │   ├── mlx.py              # ✅ Existing
│       │   └── onnx.py             # ← NEW
│       └── optimizers/              # ← NEW
│           ├── onnx_graph.py       # Graph-level optimizations
│           └── onnx_quant.py       # ONNX-specific quantization
│
├── run-onnx/                        # ← NEW package
│   ├── llama_pajamas_run_onnx/
│   │   ├── __init__.py
│   │   ├── loader.py               # Model loader
│   │   ├── session.py              # SessionOptions manager
│   │   └── backends/
│   │       ├── coreml_backend.py   # Apple Silicon
│   │       ├── tensorrt_backend.py # NVIDIA
│   │       ├── cuda_backend.py     # NVIDIA fallback
│   │       └── cpu_backend.py      # Universal fallback
│   └── pyproject.toml
│
└── run-core/                        # ✅ Existing shared runtime
    └── llama_pajamas_run_core/      # OpenAI API, server abstraction
```

---

## Part 2: Target Platforms & Execution Providers

### Primary Targets (Thin and Deep)

#### 1. Apple Silicon (M1/M2/M3/M4 Macs)

**Why ONNX on Mac when MLX exists?**
- **iOS deployment**: CoreML EP enables iPhone/iPad inference (MLX doesn't support iOS yet)
- **ANE acceleration**: Apple Neural Engine (16-core, up to 15.8 TOPS on A17) for INT8
- **Unified binaries**: Same ONNX model runs on Mac + iOS with CoreML EP

**Execution Provider**: CoreML EP
**Quantization**: QDQ INT8 (ANE-friendly)
**Expected Performance**: 50-70 tok/s (8B model, M3 Max)
**Advantages over MLX**:
- Runs on iOS (iPhone 15 Pro: 35-45 tok/s)
- Lower power consumption (ANE dedicated hardware)
- Unified deployment (Mac + iPhone + iPad)

**Settings**:
```python
coreml_opts = {
    "coreml_flags": 0,  # 1 = enable_on_subgraphs
    "preferred_memory_format": "NHWC",
}
so.intra_op_num_threads = 0
providers = [("CoreMLExecutionProvider", coreml_opts), ("CPUExecutionProvider", {})]
```

#### 2. Desktop NVIDIA GPUs (RTX 3060 → 4090)

**Why ONNX when GGUF works on CUDA?**
- **TensorRT compilation**: 15-25% faster than llama.cpp CUDA (compiled engines, kernel fusion)
- **FP16 Tensor Cores**: Better utilization than GGUF's mixed precision
- **Production serving**: ORT's IOBinding + multi-stream = better batching

**Execution Provider**: TensorRT EP (primary), CUDA EP (fallback)
**Quantization**: QDQ INT8 or FP16
**Expected Performance**: 110-140 tok/s (8B model, RTX 4090)
**Advantages over GGUF**:
- 15-25% faster (TensorRT compiled engines)
- Better batching for serving workloads
- INT8 Tensor Core acceleration

**Settings**:
```python
trt_opts = {
    "trt_fp16_enable": True,
    "trt_int8_enable": False,  # True if calibrated
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_cache",
    "trt_timing_cache_enable": True,
    "trt_max_workspace_size": 2<<30,  # 2GB
}
cuda_opts = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
}
so.intra_op_num_threads = 1
providers = [("TensorrtExecutionProvider", trt_opts),
             ("CUDAExecutionProvider", cuda_opts),
             ("CPUExecutionProvider", {})]
```

#### 3. Small NVIDIA GPUs (Jetson Nano/Orin, RTX 3050/3060)

**Why ONNX on edge devices?**
- **Optimized for small VRAM**: TensorRT INT8 fits 8B model in 4GB
- **Tight integration**: Jetpack SDK + TensorRT = maximum efficiency
- **Lower power**: INT8 kernels use less power than FP16

**Execution Provider**: TensorRT EP
**Quantization**: QDQ INT8 (mandatory for small VRAM)
**Expected Performance**:
- Jetson Orin: 25-35 tok/s (8B INT8)
- RTX 3060 (12GB): 65-85 tok/s (8B INT8)

**Settings**:
```python
trt_opts = {
    "trt_fp16_enable": False,  # Disable for INT8-only
    "trt_int8_enable": True,
    "trt_engine_cache_enable": True,
    "trt_max_workspace_size": 1<<30,  # 1GB (conserve VRAM)
}
# Same as desktop, but INT8 mandatory
```

### Secondary Targets (Future Phases)

**Mobile**:
- iOS: CoreML EP (via Mac workflow)
- Android: QNN EP (Snapdragon) + NNAPI EP (generic)

**Web**:
- ORT-Web: WASM (CPU) + WebGPU (GPU)

**Intel**:
- OpenVINO EP (CPU/iGPU/NPU)

**AMD**:
- DirectML EP (Windows) + ROCm EP (Linux, future)

---

## Part 3: Architecture-Aware ONNX Export

### Key Insight: Architecture Detection BEFORE Export

**Traditional ONNX workflow**:
```python
# Naive approach
model.export_to_onnx("model.onnx")  # Export FP32
ort.quantize_static("model.onnx", "model_q8.onnx", calibration_data)  # Quantize
```

**Llama-Pajamas approach**:
```python
# User specifies target architectures (pipeline doesn't detect)
target_specs = {
    "target_eps": ["TensorRT", "CoreML"],  # Generate for both
    "target_precisions": ["int8"],
    "optimization_hints": {
        # User provides hints about model architecture for optimizations
        "moe_experts": 128,      # If MoE model
        "gqa_ratio": 4,          # If GQA model (4:1 query:kv heads)
        "attention_type": "gqa", # "mha", "gqa", "mqa", "hybrid"
    }
}

# ONNX converter receives user-specified targets
onnx_converter = ONNXConverter()
onnx_converter.convert(
    model_id="Qwen/Qwen3-8B",
    target_specs=target_specs,  # ← User specifies, not detected
    calibration_data=calibration_data,
    output_dir="./models/qwen3-8b/onnx/"
)

# Generates:
# - ./models/qwen3-8b/onnx/tensorrt/int8/model.onnx
# - ./models/qwen3-8b/onnx/coreml/int8/model.onnx
```

**Why this matters**:
- **MoE models**: Expert-balanced calibration BEFORE ONNX quantization
- **GQA models**: Fuse KV-cache ops, optimize memory layout
- **Result**: 2-5% better quality vs naive ONNX export → quantize

### ONNX Export Pipeline

```python
# llama_pajamas_quant/converters/onnx.py

from typing import Dict, List, Optional, Any
from pathlib import Path
from datasets import Dataset

class ONNXConverter:
    """Convert HuggingFace models to optimized ONNX.

    This is a PIPELINE converter - users specify target architectures.
    Models will be distributed, so we don't detect at conversion time.
    """

    def convert(
        self,
        model_id: str,
        output_dir: Path,
        target_specs: Dict[str, Any],  # ← User specifies targets
        calibration_data: Optional[Dataset] = None,
    ):
        """
        Convert to EP-optimized ONNX for specified targets.

        USER SPECIFIES targets (not detected). This is a conversion pipeline
        for models that will be distributed - detection at conversion time
        couples the pipeline to specific models.

        Steps:
        1. Load model (FP32/FP16/BF16)
        2. Export to base ONNX (torch.onnx or Optimum)
        3. FOR EACH target_ep in target_specs:
           a. Apply EP-specific graph optimizations
           b. Apply optimization hints from target_specs
           c. Quantize with EP-specific strategy
           d. (Optional) Compile EP artifacts
           e. Save to output_dir/{ep}/{precision}/

        Args:
            model_id: HuggingFace model ID or local path
            output_dir: Output directory for ONNX files
            target_specs: User-specified conversion targets
                {
                    "target_eps": ["TensorRT", "CoreML", "CPU"],
                    "target_precisions": ["int8", "fp16"],
                    "optimization_hints": {
                        "moe_experts": 128,        # Optional: for MoE optimization
                        "gqa_ratio": 4,            # Optional: for GQA KV-cache
                        "attention_type": "gqa",   # Optional: attention pattern
                        "num_query_heads": 32,     # Optional: for attention fusion
                        "hidden_size": 4096,       # Optional: for graph opts
                    }
                }
            calibration_data: Calibration dataset for quantization

        Example:
            converter = ONNXConverter()
            converter.convert(
                model_id="Qwen/Qwen3-8B",
                output_dir="./models/qwen3-8b/onnx/",
                target_specs={
                    "target_eps": ["TensorRT", "CoreML"],
                    "target_precisions": ["int8"],
                    "optimization_hints": {
                        "gqa_ratio": 4,
                        "attention_type": "gqa",
                    }
                },
                calibration_data=calibration_dataset
            )
        """
        # Load model
        model = self._load_model(model_id)

        # Export base ONNX graph
        base_onnx = self._export_onnx(model, target_specs.get("optimization_hints", {}))

        # Generate EP-specific models
        for target_ep in target_specs["target_eps"]:
            for precision in target_specs["target_precisions"]:
                # Apply hints for optimizations
                optimized = self._optimize_graph(
                    base_onnx,
                    target_ep=target_ep,
                    hints=target_specs.get("optimization_hints", {})
                )

                # Quantize with EP-specific strategy
                quantized = self._quantize(
                    optimized,
                    target_ep=target_ep,
                    precision=precision,
                    calibration_data=calibration_data,
                    hints=target_specs.get("optimization_hints", {})
                )

                # Save
                ep_dir = output_dir / target_ep.lower() / precision
                ep_dir.mkdir(parents=True, exist_ok=True)
                self._save(quantized, ep_dir, target_ep, precision)
```

### Export Methods

**Option 1: Optimum (Recommended)**
```python
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Export
model = ORTModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    provider="CPUExecutionProvider",  # Export on CPU
)

# Quantize with architecture-aware calibration
qconfig = AutoQuantizationConfig.avx512_vnni(
    is_static=True,
    per_channel=True,
    operators_to_quantize=["MatMul", "Gemm", "Attention"],
)

quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir=output_dir,
    quantization_config=qconfig,
    calibration_tensors_range=calibration_ranges,  # From MoE-aware calibration
)
```

**Option 2: Native torch.onnx (More control)**
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# Export with dynamic axes for variable sequence length
torch.onnx.export(
    model,
    (input_ids, attention_mask, past_key_values),
    "model.onnx",
    export_params=True,
    opset_version=17,  # Latest for best EP support
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
)
```

### Graph-Level Optimizations (Pre-Quantization)

**llama_pajamas_quant/optimizers/onnx_graph.py**:

```python
class ONNXGraphOptimizer:
    """Graph-level optimizations before quantization.

    Receives user-provided hints (not detected architecture).
    """

    def optimize(
        self,
        onnx_model: onnx.ModelProto,
        target_ep: str,  # TensorRT, CoreML, etc.
        hints: Dict[str, Any]  # ← User-provided optimization hints
    ) -> onnx.ModelProto:
        """
        Apply hint-driven graph optimizations.

        Users provide hints about model architecture to enable optimizations.
        Pipeline doesn't detect - models will be distributed.

        Optimizations:
        1. Constant folding
        2. LayerNorm fusion (common pattern in transformers)
        3. Attention fusion:
           - If hints["attention_type"] == "gqa": Fuse KV-head indexing
           - If hints["attention_type"] == "mha": Fuse Q/K/V projections
        4. MoE-specific:
           - If hints["moe_experts"]: Router fusion + expert packing
        5. EP-specific:
           - TensorRT: Aggressive fusion
           - CoreML: ANE-compatible patterns

        Args:
            onnx_model: ONNX model graph
            target_ep: Target execution provider
            hints: User-provided optimization hints
                {
                    "num_query_heads": 32,     # Optional
                    "hidden_size": 4096,       # Optional
                    "attention_type": "gqa",   # Optional: "mha", "gqa", "mqa"
                    "moe_experts": 128,        # Optional: if MoE model
                    "gqa_ratio": 4,            # Optional: if GQA
                }
        """

        # ORT built-in optimizations
        from onnxruntime.transformers.optimizer import optimize_model

        optimized = optimize_model(
            onnx_model,
            model_type="bert",  # Use transformer optimizations
            num_heads=hints.get("num_query_heads", 32),
            hidden_size=hints.get("hidden_size", 4096),
            optimization_options={
                "enable_gelu_approximation": True,
                "enable_layer_norm_fusion": True,
                "enable_attention_fusion": True,
                "enable_skip_layer_norm_fusion": True,
            },
        )

        # Custom GQA optimization (if hint provided)
        if hints.get("attention_type") == "gqa":
            gqa_ratio = hints.get("gqa_ratio", 4)
            optimized = self._fuse_gqa_kv_indexing(optimized, gqa_ratio)

        # Custom MoE optimization (if hint provided)
        if hints.get("moe_experts"):
            num_experts = hints["moe_experts"]
            optimized = self._fuse_moe_routing(optimized, num_experts)

        # EP-specific optimizations
        if target_ep == "CoreML":
            # ANE-compatible patterns only
            optimized = self._apply_coreml_constraints(optimized)

        return optimized
```

### Quantization Strategies per EP

**llama_pajamas_quant/optimizers/onnx_quant.py**:

```python
class ONNXQuantizer:
    """EP-specific quantization strategies."""

    def quantize_for_tensorrt(
        self,
        model: onnx.ModelProto,
        calibration_data: Dataset,
        precision: str = "int8",
    ) -> onnx.ModelProto:
        """
        TensorRT-optimized quantization.

        Strategy:
        - QDQ format (QuantizeLinear/DequantizeLinear pairs)
        - Per-channel for Linear layers (better accuracy)
        - Per-tensor for Attention ops (TensorRT constraint)
        - Keep embeddings in FP16 (quality critical)
        - Keep output layer in FP16 (logits precision)
        """
        from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

        quantize_static(
            model_input="model.onnx",
            model_output="model_int8.onnx",
            calibration_data_reader=calibration_data,
            quant_format=QuantFormat.QDQ,  # TensorRT prefers QDQ
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={
                "EnableSubgraph": True,
                "ForceQuantizeNoInputCheck": False,
                "MatMulConstBOnly": True,
            },
            nodes_to_exclude=[
                "/model/embed_tokens/Gather",  # Keep embeddings FP16
                "/lm_head/MatMul",             # Keep output FP16
            ],
        )

    def quantize_for_coreml(
        self,
        model: onnx.ModelProto,
        calibration_data: Dataset,
    ) -> onnx.ModelProto:
        """
        CoreML/ANE-optimized quantization.

        Strategy:
        - QDQ INT8 symmetric (ANE requirement)
        - Per-tensor quantization (ANE doesn't support per-channel)
        - Avoid ops not supported on ANE (falls back to CPU/GPU)
        - Prefer NHWC layout
        """
        quantize_static(
            model_input="model.onnx",
            model_output="model_int8_coreml.onnx",
            calibration_data_reader=calibration_data,
            quant_format=QuantFormat.QDQ,
            per_channel=False,  # ANE limitation
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={
                "ActivationSymmetric": True,  # ANE requirement
                "WeightSymmetric": True,
            },
        )
```

---

## Part 4: ONNX Runtime Implementation

### Lightweight Runtime Design

**Philosophy**: Runtime is a THIN wrapper over ONNX Runtime, not a reimplementation.

```python
# llama_pajamas_run_onnx/session.py

class ONNXSession:
    """Lightweight session manager for ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        target_ep: str = "auto",  # auto, coreml, tensorrt, cuda, cpu
        hardware_config: Optional[dict] = None,
    ):
        """
        Initialize ONNX Runtime session.

        Args:
            model_path: Path to .onnx file
            target_ep: Execution provider (auto-detects if "auto")
            hardware_config: Optional config.json from auto-configure
        """
        import onnxruntime as ort

        # Auto-detect EP if not specified
        if target_ep == "auto":
            target_ep = self._detect_ep()

        # Create session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load hardware config if provided
        if hardware_config:
            so.intra_op_num_threads = hardware_config.get("n_threads", 0)
            so.inter_op_num_threads = hardware_config.get("inter_op_threads", 1)

        # Select providers
        providers = self._get_providers(target_ep)

        # Create session
        self.session = ort.InferenceSession(model_path, so, providers=providers)

        # Setup IOBinding for device-resident tensors
        self.io_binding = ort.IOBinding(self.session._sess)

    def _get_providers(self, target_ep: str) -> list:
        """Get provider list with fallbacks."""
        if target_ep == "coreml":
            return [
                ("CoreMLExecutionProvider", {"coreml_flags": 0}),
                ("CPUExecutionProvider", {}),
            ]
        elif target_ep == "tensorrt":
            return [
                ("TensorrtExecutionProvider", self._get_trt_opts()),
                ("CUDAExecutionProvider", self._get_cuda_opts()),
                ("CPUExecutionProvider", {}),
            ]
        elif target_ep == "cuda":
            return [
                ("CUDAExecutionProvider", self._get_cuda_opts()),
                ("CPUExecutionProvider", {}),
            ]
        else:  # cpu
            return [("CPUExecutionProvider", {"use_arena": True})]

    def generate(
        self,
        input_ids: np.ndarray,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        """
        Generate text (tight loop with device-resident KV-cache).

        Uses IOBinding to avoid host↔device copies.
        """
        # Bind inputs to device
        self.io_binding.bind_cpu_input("input_ids", input_ids)

        # Allocate outputs on device
        self.io_binding.bind_output("logits", "cuda")  # or "cpu"

        # Tight generation loop
        for _ in range(max_tokens):
            # Run inference (stays on device)
            self.session.run_with_iobinding(self.io_binding)

            # Get logits (still on device)
            logits = self.io_binding.get_outputs()[0]

            # Sample next token (on device if using CuPy)
            next_token = self._sample_token(logits, temperature)

            # Update input_ids
            input_ids = np.concatenate([input_ids, next_token], axis=1)

            # Rebind for next iteration
            self.io_binding.bind_cpu_input("input_ids", input_ids)

            if stream:
                yield next_token
```

### Backend Implementations

**llama_pajamas_run_onnx/backends/tensorrt_backend.py**:

```python
class TensorRTBackend:
    """TensorRT-specific optimizations."""

    def __init__(self):
        self.session = None

    def load_model(self, model_path: str, config_path: Optional[str] = None):
        """Load model with TensorRT optimizations."""
        import onnxruntime as ort

        # TensorRT provider options
        trt_opts = {
            "trt_fp16_enable": True,
            "trt_int8_enable": False,  # Load from config
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(Path(model_path).parent / "trt_cache"),
            "trt_timing_cache_enable": True,
            "trt_max_workspace_size": 2<<30,
        }

        # Load config if provided
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
                settings = config.get("settings", {})
                trt_opts["trt_int8_enable"] = settings.get("int8", False)
                trt_opts["trt_max_workspace_size"] = settings.get("workspace_size", 2<<30)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1

        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]

        self.session = ort.InferenceSession(model_path, so, providers=providers)
```

**llama_pajamas_run_onnx/backends/coreml_backend.py**:

```python
class CoreMLBackend:
    """CoreML/ANE-specific optimizations."""

    def load_model(self, model_path: str, config_path: Optional[str] = None):
        """Load model with CoreML optimizations."""
        import onnxruntime as ort

        coreml_opts = {
            "coreml_flags": 0,  # 1 = enable_on_subgraphs
            "preferred_memory_format": "NHWC",
        }

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # Let CoreML handle threading

        providers = [
            ("CoreMLExecutionProvider", coreml_opts),
            ("CPUExecutionProvider", {}),
        ]

        self.session = ort.InferenceSession(model_path, so, providers=providers)
```

---

## Part 5: Implementation Timeline (4 Weeks Post-MVP)

### Week 1: ONNX Export + TensorRT

**Goal**: Get TensorRT EP working on desktop NVIDIA

**Days 1-2: Export Infrastructure**
- Create `llama_pajamas_quant/converters/onnx.py`
- Implement Optimum-based export (FP16 → ONNX)
- Test export on Qwen3-8B
- Validate ONNX graph (opset 17, no errors)

**Deliverable**:
```bash
# User specifies targets and optimization hints (no detection)
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-eps TensorRT,CoreML \
  --target-precisions fp16 \
  --optimization-hints '{"attention_type": "gqa", "gqa_ratio": 4, "num_query_heads": 32}'

# Output (generates for BOTH target EPs):
# ./models/qwen3-8b/onnx/tensorrt/fp16/
#   ├── model.onnx
#   ├── metadata.json
# ./models/qwen3-8b/onnx/coreml/fp16/
#   ├── model.onnx
#   ├── metadata.json
```

**Days 3-4: Quantization + Graph Optimization**
- Implement `onnx_graph.py` (graph optimizations)
- Implement `onnx_quant.py` (TensorRT QDQ INT8)
- Use existing MoE-balanced calibration data
- Test INT8 quantization

**Deliverable**:
```bash
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-ep TensorRT \
  --precision int8 \
  --calibration-samples 512

# Output:
# ./models/qwen3-8b/onnx/tensorrt/int8/
#   ├── model_int8.onnx (~3.2GB)
#   ├── metadata.json
```

**Days 5-7: TensorRT Runtime**
- Create `llama_pajamas_run_onnx` package
- Implement `tensorrt_backend.py`
- Implement session manager with IOBinding
- Test on RTX 4070: target 100+ tok/s (INT8)

**Deliverable**:
```bash
llama-pajamas-run-onnx chat \
  --model ./models/qwen3-8b/onnx/tensorrt/int8/model_int8.onnx \
  --backend tensorrt

# Performance: 110+ tok/s (RTX 4070, batch=1)
```

**MILESTONE 1**: TensorRT INT8 working, faster than GGUF CUDA

---

### Week 2: CoreML + Apple Silicon

**Goal**: Get CoreML EP working on Mac, enable iOS deployment

**Days 8-9: CoreML Export**
- Implement CoreML-specific quantization (symmetric INT8)
- Test on M3 Max
- Validate ANE utilization (Activity Monitor → Neural Engine)

**Deliverable**:
```bash
llama-pajamas-quant convert \
  --model Qwen/Qwen3-8B \
  --output ./models/qwen3-8b \
  --formats onnx \
  --target-ep CoreML \
  --precision int8

# Output:
# ./models/qwen3-8b/onnx/coreml/int8/
#   ├── model_int8_coreml.onnx (~3.0GB)
#   ├── metadata.json
```

**Days 10-11: CoreML Runtime**
- Implement `coreml_backend.py`
- Test on M3 Max: target 50-70 tok/s
- Compare vs MLX (MLX should be 10-20% faster on Mac)

**Days 12: iOS Deployment Path (Documentation)**
- Document how to use CoreML ONNX on iOS
- Create Xcode example project (simple chat app)
- Test on iPhone 15 Pro: target 35-45 tok/s

**MILESTONE 2**: CoreML INT8 working on Mac + iOS deployment path

---

### Week 3: Jetson + Small GPUs

**Goal**: Optimize for edge devices

**Days 13-14: Jetson Optimization**
- Tune TensorRT for small VRAM (Jetson Orin: 8GB)
- INT8 mandatory (won't fit in FP16)
- Test on Jetson Orin: target 25-35 tok/s

**Days 15-16: Small Desktop GPUs**
- Tune for RTX 3060 (12GB)
- Compare INT8 vs FP16 (INT8 should be ~30% faster)
- Document optimal batch sizes

**Day 17: Edge Deployment Guide**
- Create EDGE_DEPLOYMENT.md
- Hardware-specific guides (Jetson, RTX 3050/3060)
- Power consumption analysis

**MILESTONE 3**: Edge deployment validated (Jetson + small GPUs)

---

### Week 4: Integration + Polish

**Days 18-19: Unified CLI**
- Update `llama-pajamas-run` to auto-select backend
- ONNX, GGUF, or MLX based on model format + available hardware
- Auto-configure for ONNX (detect EP, generate config)

**Days 20-21: Quality Validation**
- Run full benchmark suite on ONNX models
- Compare vs GGUF/MLX baselines
- Document quality/speed tradeoffs

**Days 22-23: Documentation**
- ONNX_README.md (comprehensive guide)
- Update MVP plan with ONNX section
- Create comparison table (ONNX vs GGUF vs MLX)

**Day 24: Buffer / Refinement**

**MILESTONE 4**: ONNX fully integrated, production-ready

---

## Part 6: INT4 and Lower Bit Quantization

### INT4 Support in ONNX Runtime

**Current State (2025)**:
- **ONNX Runtime supports INT4** via:
  1. **MatMulNBits operator** (INT4 weights, INT8 activations) - Windows DirectML, CPU
  2. **INT4 QDQ** (experimental) - Limited EP support
  3. **GPTQ format** via Optimum - Can export to ONNX INT4

**EP Support for INT4**:

```yaml
TensorRT:
  INT4: ✅ Supported (TensorRT 8.6+)
  Method: INT4 weights + FP16 activations
  Use case: Extreme memory constraints (3GB VRAM for 8B model)
  Performance: 1.5-2x faster than INT8 on Ampere/Ada GPUs
  Quality: ~5-8% loss vs FP16

CoreML:
  INT4: ❌ Not supported
  Limitation: ANE only supports INT8/FP16
  Workaround: Use INT8 (already smallest supported)

CPU (oneDNN):
  INT4: ✅ Supported (via MatMulNBits)
  Method: INT4 weights, INT8 activations
  Use case: Edge devices with <4GB RAM
  Performance: 30-40% faster than INT8
  Quality: ~6-10% loss vs FP16

DirectML (Windows):
  INT4: ✅ Supported
  Method: MatMulNBits operator
  Use case: Windows laptops, integrated GPUs
  Performance: 1.3-1.7x faster than INT8
```

### INT4 Implementation Plan

**Add to target_specs**:
```python
target_specs = {
    "target_eps": ["TensorRT", "CPU"],
    "target_precisions": ["int4", "int8", "fp16"],  # ← INT4 added
    "optimization_hints": {
        "gqa_ratio": 4,
        "attention_type": "gqa",
    }
}
```

**Quantization strategy per EP**:

```python
# llama_pajamas_quant/optimizers/onnx_quant.py

def quantize_int4_tensorrt(
    model: onnx.ModelProto,
    calibration_data: Dataset,
) -> onnx.ModelProto:
    """
    TensorRT INT4 quantization.

    Strategy:
    - INT4 weights (4-bit per weight)
    - FP16 activations (TensorRT recommendation)
    - Per-channel for Linear layers
    - Keep embeddings FP16 (quality critical)
    """
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

    quantize_static(
        model_input="model.onnx",
        model_output="model_int4.onnx",
        calibration_data_reader=calibration_data,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt4,      # ← INT4 weights
        activation_type=QuantType.QUInt8,  # Keep activations higher precision
        per_channel=True,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

def quantize_int4_cpu(
    model: onnx.ModelProto,
    calibration_data: Dataset,
) -> onnx.ModelProto:
    """
    CPU INT4 quantization (MatMulNBits).

    Strategy:
    - INT4 weights (block-wise quantization)
    - INT8 activations
    - Optimized for memory-constrained edge devices
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Use dynamic quantization with MatMulNBits
    quantize_dynamic(
        model_input="model.onnx",
        model_output="model_int4.onnx",
        weight_type=QuantType.QInt4,
        op_types_to_quantize=["MatMul"],
        extra_options={
            "MatMulNBits": True,  # Use INT4-optimized MatMul
            "block_size": 128,     # Block size for weight quantization
        },
    )
```

### Lower Bit Quantization (INT3, INT2, FP4)

**Experimental / Limited Support**:

```yaml
INT3 (3-bit):
  Status: ❌ Not supported in ONNX Runtime
  Alternative: Use GGUF IQ3_XXS (llama.cpp has best INT3 support)
  Why: ONNX operators don't support 3-bit packing natively

INT2 (2-bit):
  Status: ❌ Not supported in ONNX Runtime
  Alternative: Use GGUF IQ2_XXS or MLX 2-bit
  Why: Quality degradation too severe for ONNX use cases

FP4 (4-bit float):
  Status: ⚠️ Experimental (research only)
  EP Support: None in production
  Alternative: Stick with INT4 for production
```

**Recommendation for Space Constraints**:

```yaml
Extreme Space Constraints (<3GB):
  Primary: GGUF IQ2_XS/IQ3_XXS (llama.cpp excels here)
  Secondary: MLX 2-bit (Apple Silicon only)
  ONNX: Use INT4 TensorRT (smallest production-ready)

Moderate Constraints (3-6GB):
  Primary: ONNX INT4 TensorRT/CPU
  Secondary: GGUF Q4_K_M
  Tertiary: MLX 4-bit

Low Constraints (6GB+):
  Primary: ONNX INT8 (best quality/size trade-off)
  Secondary: GGUF Q4_K_M/Q5_K_M
```

### Size Comparison (8B Model)

| Format | Precision | Size | Tok/s (RTX 4090) | Quality Loss |
|--------|-----------|------|------------------|--------------|
| GGUF | IQ2_XS | 2.3GB | 145 | ~12-15% |
| MLX | 2-bit | 2.4GB | 90 (M3) | ~10-12% |
| GGUF | IQ3_XXS | 2.8GB | 140 | ~8-10% |
| **ONNX** | **INT4** | **3.2GB** | **135** | **~6-8%** |
| GGUF | Q4_K_M | 4.7GB | 120 | ~3-5% |
| ONNX | INT8 | 8.0GB | 130 | ~2-4% |

**Key Insight**: ONNX INT4 is the smallest **production-ready** option with good quality. For extreme compression, use GGUF IQ2_XS/IQ3_XXS (llama.cpp has better <4-bit support).

---

## Part 7: Future Phases (Vision, STT, Multimodal)

### Vision Models (Phase 5)

**Target Models & Use Cases**:

```yaml
Image Encoders:
  - CLIP (text-image similarity, embeddings)
  - DINOv2 (visual features, zero-shot)
  - SigLIP (improved CLIP)

Vision-Language Models:
  - Qwen3-VL (multimodal LLM with vision)
  - LLaVA (visual instruction following)
  - CogVLM (high-resolution vision)

Computer Vision:
  - YOLO v8/v9 (object detection)
  - SegmentAnything (segmentation)
  - Depth-Anything (monocular depth)
```

**Why ONNX Excels for Vision**:

```yaml
Mobile Deployment:
  - CoreML: ANE acceleration for Conv2D on iPhone
  - QNN: Snapdragon NPU for real-time inference
  - Android: 30+ FPS object detection on mid-range phones

Edge Devices:
  - Jetson: TensorRT INT8 for real-time (60 FPS YOLO)
  - Raspberry Pi: CPU-optimized Conv2D
  - Security cameras: Embedded NPUs

Production CV Pipelines:
  - Batch processing: Multiple images per inference
  - IoT: Low-power vision on edge
  - Automotive: Real-time vision (TensorRT)
```

**Pipeline Implementation**:

```python
# Vision-specific target_specs
target_specs = {
    "target_eps": ["TensorRT", "CoreML", "QNN"],
    "target_precisions": ["int8"],  # Vision tolerates INT8 well
    "optimization_hints": {
        "model_type": "vision_encoder",  # "vision_encoder", "vlm", "object_detection"
        "input_shape": [224, 224, 3],    # Static shape for mobile
        "batch_size": 1,                  # Mobile = batch 1, server = batch 16+
        "enable_dynamic_shapes": False,   # Mobile prefers static shapes
    }
}

# Vision-specific optimizations
class VisionONNXOptimizer:
    def optimize_for_vision(self, model, hints):
        """
        Vision-specific graph optimizations.

        - Conv + BatchNorm fusion
        - Conv + ReLU fusion
        - Depthwise separable Conv patterns (MobileNet)
        - Channel-last layout (NHWC) for mobile ANE/NPU
        """
        if hints.get("target_ep") == "CoreML":
            # ANE prefers NHWC, specific ops
            model = self._convert_to_nhwc(model)
            model = self._fuse_conv_bn_relu(model)

        if hints.get("target_ep") == "TensorRT":
            # Aggressive fusion for GPU
            model = self._fuse_conv_patterns(model)
            model = self._optimize_pooling(model)

        return model
```

**Example: CLIP Encoder for iPhone**:

```bash
llama-pajamas-quant convert \
  --model openai/clip-vit-base-patch32 \
  --output ./models/clip-vit/onnx/ \
  --formats onnx \
  --target-eps CoreML \
  --target-precisions int8 \
  --optimization-hints '{
    "model_type": "vision_encoder",
    "input_shape": [224, 224, 3],
    "batch_size": 1,
    "enable_dynamic_shapes": false
  }'

# Generates:
# - ./models/clip-vit/onnx/coreml/int8/vision_encoder.onnx
# - Size: ~80MB (vs 330MB FP32)
# - Performance: 12ms/image on iPhone 15 Pro (ANE)
```

**Timeline**: 2 weeks post-ONNX LLM
**Priority**: HIGH (Vision is ONNX's sweet spot - better than GGUF/MLX for CV)

---

### STT Models (Phase 6 - Whisper)

**Why Whisper on ONNX**:

```yaml
Mobile STT:
  - CoreML: On-device transcription (privacy)
  - iPhone: 15-20ms latency per chunk
  - Android: QNN for Snapdragon acceleration

Edge STT:
  - IoT devices: Low-power transcription
  - Smart speakers: Local processing
  - Security: On-premise audio processing

Production STT:
  - Batch transcription: Process 100s of files
  - Streaming: Real-time transcription with chunking
  - Multi-language: Support 99 languages efficiently
```

**Whisper-Specific Challenges**:

```yaml
Encoder-Decoder Architecture:
  - Encoder: CNN + Transformer (80-dim mel features → embeddings)
  - Decoder: Transformer (autoregressive text generation)

Split Strategy:
  Option 1: Single ONNX (encoder + decoder fused)
    - Pros: Simple deployment
    - Cons: Recompute encoder for each token
    - Use case: Short audio (<30s)

  Option 2: Separate ONNX (encoder.onnx, decoder.onnx)
    - Pros: Encode once, decode multiple tokens
    - Cons: More complex runtime
    - Use case: Long audio (>30s), streaming

Beam Search:
  - ONNX doesn't have native beam search
  - Options:
    1. Python beam search + ONNX decoder calls
    2. ORT GenAI (has beam search built-in)
    3. Greedy decoding (simpler, slightly worse quality)
```

**Pipeline Implementation**:

```python
# Whisper-specific target_specs
target_specs = {
    "target_eps": ["CoreML", "TensorRT", "CPU"],
    "target_precisions": ["int8"],
    "optimization_hints": {
        "model_type": "encoder_decoder_stt",
        "encoder_input_shape": [80, 3000],  # mel features: 80 bins, 30s audio
        "decoder_max_length": 448,           # Max output tokens
        "split_encoder_decoder": True,       # Generate 2 ONNX files
        "enable_kv_cache": True,             # KV-cache for decoder
    }
}

# Generates:
# - whisper_encoder.onnx (encode mel → embeddings)
# - whisper_decoder.onnx (decode embeddings → text tokens)
# - kv_cache config for decoder efficiency
```

**Example: Whisper Tiny for iPhone**:

```bash
llama-pajamas-quant convert \
  --model openai/whisper-tiny \
  --output ./models/whisper-tiny/onnx/ \
  --formats onnx \
  --target-eps CoreML \
  --target-precisions int8 \
  --optimization-hints '{
    "model_type": "encoder_decoder_stt",
    "split_encoder_decoder": true,
    "enable_kv_cache": true
  }'

# Generates:
# - whisper_tiny_encoder.onnx (~15MB INT8)
# - whisper_tiny_decoder.onnx (~25MB INT8)
# - Total: ~40MB (vs 150MB FP32)
# - Performance:
#   - Encoder: 50ms for 30s audio (iPhone 15 Pro, ANE)
#   - Decoder: 8ms/token (greedy, CoreML GPU)
#   - Real-time factor: 0.15x (6x faster than real-time)
```

**Runtime Integration**:

```python
# llama_pajamas_run_onnx/stt_runtime.py

class WhisperONNXRuntime:
    """Whisper STT runtime with split encoder/decoder."""

    def __init__(self, encoder_path, decoder_path, ep="CoreML"):
        self.encoder = ONNXSession(encoder_path, target_ep=ep)
        self.decoder = ONNXSession(decoder_path, target_ep=ep)

    def transcribe(self, audio_file: str, language: str = "en") -> str:
        # 1. Load audio, convert to mel-spectrogram
        mel_features = self._audio_to_mel(audio_file)

        # 2. Encode (once)
        encoder_hidden = self.encoder.run(mel_features)

        # 3. Decode (autoregressive with KV-cache)
        tokens = []
        kv_cache = None
        for _ in range(self.max_length):
            logits, kv_cache = self.decoder.run(
                encoder_hidden=encoder_hidden,
                input_ids=tokens[-1:],
                past_kv=kv_cache  # Reuse KV-cache
            )
            next_token = self._sample_token(logits)
            if next_token == END_TOKEN:
                break
            tokens.append(next_token)

        # 4. Decode tokens to text
        return self.tokenizer.decode(tokens)
```

**Timeline**: 2 weeks post-Vision
**Priority**: MEDIUM-HIGH (STT on mobile is valuable, but more complex than Vision)

---

### TTS Models (Phase 7 - VITS, Tacotron)

**Why TTS on ONNX**:

```yaml
Mobile TTS:
  - CoreML: On-device voice synthesis
  - Low latency: <100ms for sentence
  - Offline: No internet required

Edge TTS:
  - Smart assistants: Local voice output
  - Accessibility: Screen readers
  - IoT: Voice notifications

Production TTS:
  - Batch synthesis: Generate audiobooks
  - Custom voices: Fine-tuned models
  - Multi-language: Support 50+ languages
```

**TTS Pipeline (Simpler than STT)**:

```yaml
Architecture (VITS):
  Text → Phoneme Encoder → Duration Predictor → Mel Decoder → Vocoder → Audio

Split Strategy:
  Option 1: Full pipeline in single ONNX
    - Pros: Simple deployment
    - Use case: Mobile, edge devices

  Option 2: Split vocoder (text → mel, mel → audio)
    - Pros: Swap vocoders easily
    - Use case: Custom voices, quality tuning

Quantization:
  - INT8 works well for TTS (quality loss <3%)
  - Vocoder: Keep FP16 (audio quality sensitive)
  - Text encoder: INT8 (not quality critical)
```

**Example: VITS TTS for Android**:

```bash
llama-pajamas-quant convert \
  --model facebook/mms-tts-eng \
  --output ./models/vits-tts/onnx/ \
  --formats onnx \
  --target-eps QNN,CPU \
  --target-precisions int8 \
  --optimization-hints '{
    "model_type": "tts",
    "keep_vocoder_fp16": true
  }'

# Generates:
# - vits_tts_int8.onnx (~50MB)
# - Performance:
#   - Android (QNN): 80ms for 10-word sentence
#   - Real-time factor: 0.3x (3x faster than real-time)
```

**Timeline**: 2 weeks post-STT
**Priority**: MEDIUM (TTS less critical than Vision/STT for initial deployment)

---

### Multimodal Integration (Phase 8)

**Vision-Language Models (Qwen3-VL, LLaVA)**:

```yaml
Architecture:
  Vision Encoder (CLIP/SigLIP) → Projection → LLM

Split Strategy:
  - vision_encoder.onnx (separate ONNX)
  - projection.onnx (small adapter)
  - llm.onnx (reuse LLM ONNX from Phase 1-3)

Deployment:
  Pipeline: Image → Vision ONNX → Embeddings → LLM ONNX → Text

  Advantage:
    - Reuse LLM infrastructure
    - Mix-and-match vision encoders
    - Optimize each component separately

Example Flow:
  1. Load image (preprocessed to 224x224)
  2. Run vision_encoder.onnx → image_embeddings (CoreML, 15ms)
  3. Run projection.onnx → llm_embeddings (CPU, 2ms)
  4. Run llm.onnx with image_embeddings prefix (TensorRT, 500ms for 100 tokens)
  5. Total: ~520ms (real-time on iPhone/Android)
```

**Example: Qwen3-VL for Mobile**:

```bash
# Step 1: Convert vision encoder
llama-pajamas-quant convert \
  --model Qwen/Qwen3-VL-8B \
  --component vision_encoder \
  --formats onnx \
  --target-eps CoreML \
  --target-precisions int8

# Step 2: Convert projection adapter
llama-pajamas-quant convert \
  --model Qwen/Qwen3-VL-8B \
  --component projection \
  --formats onnx \
  --target-eps CPU

# Step 3: Reuse Qwen3-8B LLM ONNX (already have from Phase 1)

# Generates:
# - vision_encoder.onnx (~80MB)
# - projection.onnx (~5MB)
# - Reuse: qwen3-8b-int8.onnx (3.2GB)
# - Total: ~3.3GB (acceptable for mobile)
```

**Timeline**: 3 weeks post-TTS
**Priority**: HIGH (Multimodal is the future - images + text queries)

---

### Mobile Optimization (Phase 7)

**Android**:
- QNN EP (Snapdragon HTP/NPU)
- NNAPI EP (generic fallback)
- Calibration for INT8 (mandatory for NPU)

**iOS**:
- Already covered by CoreML (Phase 2)
- Optimize for iPhone 15/16 Pro (A17/A18)
- ANE utilization monitoring

**Timeline**: 3 weeks (Android + iOS optimization)

---

## Part 7: Hardware Profiles & Auto-Configuration

### Extend Existing Hardware Detection

**quant/config/hardware_profiles.json** (extend existing):

```json
{
  "apple_silicon_m3_64gb": {
    "display_name": "Apple M3 (64GB)",
    "onnx_ep": "CoreML",
    "onnx_settings": {
      "coreml_flags": 0,
      "intra_op_num_threads": 0,
      "precision": "int8",
      "expected_tokens_per_sec": 65
    }
  },
  "nvidia_rtx_4090": {
    "display_name": "NVIDIA RTX 4090 (24GB)",
    "onnx_ep": "TensorRT",
    "onnx_settings": {
      "trt_fp16_enable": true,
      "trt_int8_enable": false,
      "trt_max_workspace_size": 4294967296,
      "intra_op_num_threads": 1,
      "precision": "fp16",
      "expected_tokens_per_sec": 135
    }
  },
  "nvidia_jetson_orin": {
    "display_name": "NVIDIA Jetson Orin (8GB)",
    "onnx_ep": "TensorRT",
    "onnx_settings": {
      "trt_fp16_enable": false,
      "trt_int8_enable": true,
      "trt_max_workspace_size": 1073741824,
      "intra_op_num_threads": 1,
      "precision": "int8",
      "expected_tokens_per_sec": 30
    }
  }
}
```

### Auto-Configuration Flow

```bash
# Auto-configure for ONNX
llama-pajamas-run-onnx chat \
  --model ./models/qwen3-8b/onnx/ \
  --auto-configure \
  --verbose

# Detects hardware:
#   Hardware: Apple M3 Max (64GB)
#   EP: CoreML
#   Precision: INT8
#   Model: ./models/qwen3-8b/onnx/coreml/int8/model_int8_coreml.onnx
#   Expected: ~65 tok/s
```

---

## Part 8: Performance Expectations

### Comparison Table: ONNX vs GGUF vs MLX

| Hardware | Model | Format | Precision | Size | Tok/s | Notes |
|----------|-------|--------|-----------|------|-------|-------|
| **Apple M3 Max (64GB)** | Qwen3-8B | MLX | 4-bit | 4.3GB | 85 | ⭐ Best on Mac |
| | | GGUF | Q4_K_M | 4.7GB | 75 | Metal backend |
| | | ONNX | INT8 CoreML | 3.0GB | 65 | iOS-compatible |
| **RTX 4090 (24GB)** | Qwen3-8B | ONNX | INT8 TensorRT | 3.2GB | 135 | ⭐ Fastest |
| | | ONNX | FP16 TensorRT | 16GB | 125 | More VRAM |
| | | GGUF | Q4_K_M | 4.7GB | 110 | CUDA backend |
| **RTX 3060 (12GB)** | Qwen3-8B | ONNX | INT8 TensorRT | 3.2GB | 80 | ⭐ Best fit |
| | | GGUF | Q4_K_M | 4.7GB | 65 | CUDA backend |
| **Jetson Orin (8GB)** | Qwen3-8B | ONNX | INT8 TensorRT | 3.2GB | 30 | ⭐ Only fits INT8 |
| | | GGUF | Q4_K_M | 4.7GB | 22 | Slower, larger |
| **iPhone 15 Pro (6GB)** | Qwen3-8B | ONNX | INT8 CoreML | 3.0GB | 40 | ⭐ Only option |
| | | MLX | - | - | - | Not supported |
| | | GGUF | - | - | - | Not practical |

**Key Insights**:
- **ONNX INT8 is smallest**: 3.0-3.2GB (vs 4.3-4.7GB for MLX/GGUF)
- **TensorRT is fastest on NVIDIA**: 15-25% faster than GGUF CUDA
- **CoreML enables iOS**: Only viable option for on-device iPhone inference
- **Edge devices prefer ONNX**: Smaller size, better optimization

---

## Part 9: Success Criteria

### Functional Requirements

**Week 1 (TensorRT)**:
- [x] Export Qwen3-8B to ONNX FP16
- [x] Quantize to INT8 with architecture-aware calibration
- [x] Load and run on RTX 4070 (110+ tok/s)
- [x] TensorRT engine caching working

**Week 2 (CoreML)**:
- [x] Export Qwen3-8B to CoreML-optimized INT8
- [x] Run on M3 Max (50-70 tok/s)
- [x] Validate ANE utilization
- [x] iOS deployment guide

**Week 3 (Edge)**:
- [x] Run on Jetson Orin (25-35 tok/s)
- [x] Run on RTX 3060 (65-85 tok/s)
- [x] INT8 fits in small VRAM

**Week 4 (Integration)**:
- [x] Unified CLI supports ONNX
- [x] Auto-configure working
- [x] Quality validation (<5% loss)
- [x] Documentation complete

### Performance Targets

**TensorRT (RTX 4090)**:
- INT8: 130+ tok/s
- FP16: 120+ tok/s
- 15%+ faster than GGUF CUDA

**CoreML (M3 Max)**:
- INT8: 60+ tok/s
- 10-20% slower than MLX (acceptable for iOS compatibility)

**Jetson Orin**:
- INT8: 28+ tok/s
- Fits in 8GB VRAM

**iPhone 15 Pro**:
- INT8: 35+ tok/s
- Runs on device (no cloud)

### Quality Targets

**Perplexity** (vs baseline):
- TensorRT INT8: <5% increase
- CoreML INT8: <6% increase (ANE constraints)

**Accuracy** (benchmark suite):
- TensorRT INT8: >94% of FP16 baseline
- CoreML INT8: >92% of FP16 baseline

---

## Part 10: Why ONNX Complements (Doesn't Replace) GGUF/MLX

### The Three-Format Strategy

```yaml
Use GGUF when:
  - Rapid prototyping
  - CPU-only deployment
  - Flexibility over maximum performance
  - llama.cpp ecosystem preferred

Use MLX when:
  - Apple Silicon Mac deployment
  - Maximum performance on Mac
  - Pure Apple workflow

Use ONNX when:
  - Production serving (TensorRT batching)
  - Mobile deployment (iOS/Android)
  - Edge devices (Jetson, small GPUs)
  - Cloud-edge optimization
  - Vision/TTS models (future)
```

### Concrete Example Deployments

**Scenario 1: Mac-only development**
- Use MLX (fastest on Mac)
- Keep GGUF for testing on other platforms

**Scenario 2: Production serving on NVIDIA**
- Use ONNX TensorRT (15-25% faster than GGUF)
- Batch multiple requests efficiently

**Scenario 3: Mobile app (iOS)**
- Use ONNX CoreML (only viable option)
- Export once, deploy to Mac + iPhone + iPad

**Scenario 4: Edge device (Jetson Orin)**
- Use ONNX TensorRT INT8 (only fits in 8GB)
- Optimized for power efficiency

**Scenario 5: Multi-modal app (LLM + Vision)**
- Use ONNX for both LLM and CLIP
- Unified runtime, consistent deployment

---

## Part 11: Implementation Checklist

### Week 1: TensorRT
- [ ] `llama_pajamas_quant/converters/onnx.py` (export)
- [ ] `llama_pajamas_quant/optimizers/onnx_graph.py` (graph opts)
- [ ] `llama_pajamas_quant/optimizers/onnx_quant.py` (quantization)
- [ ] `llama_pajamas_run_onnx` package structure
- [ ] `llama_pajamas_run_onnx/backends/tensorrt_backend.py`
- [ ] Test on RTX 4070 (110+ tok/s)

### Week 2: CoreML
- [ ] CoreML-specific quantization (symmetric INT8)
- [ ] `llama_pajamas_run_onnx/backends/coreml_backend.py`
- [ ] Test on M3 Max (60+ tok/s)
- [ ] iOS deployment guide + Xcode example
- [ ] ANE utilization validation

### Week 3: Edge
- [ ] Jetson-specific tuning (INT8, small workspace)
- [ ] Test on Jetson Orin (28+ tok/s)
- [ ] Test on RTX 3060 (75+ tok/s)
- [ ] EDGE_DEPLOYMENT.md

### Week 4: Integration
- [ ] Update `llama-pajamas-run` CLI (auto-select backend)
- [ ] Extend hardware_profiles.json (ONNX settings)
- [ ] Quality validation (benchmark suite)
- [ ] ONNX_README.md
- [ ] Update MVP plan

---

## Conclusion

### What We're Building

**Llama-Pajamas ONNX is architecture-aware quantization + EP-specific optimization** for maximum performance per deployment target.

### Why It Matters

**Traditional ONNX workflow**: Export → quantize → hope it works everywhere

**Llama-Pajamas ONNX**: Architecture detection → MoE-balanced calibration → EP-specific optimization → 15-30% faster

### The 4-Week Plan Delivers

- ✅ TensorRT: 15-25% faster than GGUF on NVIDIA (Week 1)
- ✅ CoreML: iOS deployment enabled (Week 2)
- ✅ Edge: Jetson + small GPUs optimized (Week 3)
- ✅ Integration: Unified CLI, auto-configure (Week 4)

### After ONNX

**Scale horizontally**:
- Vision models (CLIP, Qwen3-VL)
- TTS/STT models (VITS, Whisper)
- Mobile optimization (Android QNN, iOS ANE)

**Scale vertically**:
- Advanced graph optimizations (custom fusions)
- Per-layer mixed precision
- ORT GenAI integration (device-side KV cache)

### Timeline Summary

```
Week 1-2 (MVP):    Qwen3-8B (GGUF + MLX) ✅
Week 3 (MVP):      GPT-OSS-20B (MoE)
Week 4-7 (ONNX):   TensorRT + CoreML + Edge + Integration
Week 8+ (Future):  Vision + TTS + Mobile
```

**Let's ship it** 🚀
