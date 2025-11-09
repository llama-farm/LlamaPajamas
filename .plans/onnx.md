Here’s a compact map of the ONNX ecosystem for **quantization**, **graph/model optimization**, and **hardware-specific runtimes** (execution backends). I’m keeping it high-signal so you can pick the right tools fast.

# Quantization options

* **ONNX Runtime (ORT) Quantization** — built-in PTQ and dynamic quantization with two formats:

  * **QOperator** (e.g., `QLinearConv`, `MatMulInteger`)
  * **QDQ** (inserts `QuantizeLinear`/`DequantizeLinear` pairs; supports per-tensor/per-channel) ([ONNX Runtime][1])
    Python APIs cover preprocessing, dynamic/static quant, and debugging. ([ONNX Runtime][1])
* **Intel OpenVINO** — PTQ for ONNX via **NNCF** and **POT** (post-training). Works with small calibration sets; targets int8. ([OpenVINO Documentation][2])
* **Microsoft Olive** — one pipeline that can drive ORT quant, Intel Neural Compressor, and more (even int4 for some LLM paths). Hardware-aware and scriptable/CLI. ([ONNX Runtime][3])

# Optimization (beyond quantization)

* **ORT graph optimizations** — levels **basic / extended / all** (constant folding, node elimination/fusion, layout, EP-aware fusions). Also **ORT format** models for minimal builds. ([ONNX Runtime][4])
* **onnx/optimizer** — standalone passes to simplify/fuse ONNX graphs. ([GitHub][5])
* **Hugging Face Optimum (ORT)** — programmatic wrappers for ORT optimization/quantization for Transformers. ([Hugging Face][6])
* **Model-specific**: **onnxruntime-tools** for extra transformer fusions not yet in ORT main. ([PyPI][7])

# Hardware-specific runtimes / Execution Providers

**ONNX Runtime EPs** (choose one or stack with fallbacks):

* **CPU**: default; **oneDNN (DNNL)**, **XNNPACK** for tuned kernels. ([ONNX Runtime][8])
* **NVIDIA**: **CUDA EP** and **TensorRT EP** (uses Tensor Cores, best perf if ops supported). Native **TensorRT** also imports ONNX directly. ([NVIDIA Developer][9])
* **Intel**: **OpenVINO EP** for CPUs/iGPUs/VPU; aligns with NNCF/POT quant. ([ONNX Runtime][8])
* **AMD**: **ROCm EP**; **MIGraphX** supported via ORT EP list (and as a separate runtime). ([ONNX Runtime][8])
* **Apple**: **CoreML EP** (prefers Apple Neural Engine on M-series/iOS). ([ONNX Runtime][10])
* **Qualcomm**: **QNN EP** targets Snapdragon NPU (HTP) and GPU backends via QNN SDK; Android & Windows on ARM. ([ONNX Runtime][11])
* **Windows GPUs (vendor-agnostic)**: **DirectML EP**. ([Microsoft Learn][12])
* **Android**: **NNAPI EP** (routes to device accelerators where available). ([ONNX Runtime][8])
* **Web**: **ORT Web** via **WASM** (CPU), **WebGPU** (GPU), experimental **WebNN**; includes JS APIs. ([ONNX Runtime][13])
* **Others listed/preview**: **TVM EP**, **Vitis-AI**, **Rockchip NPU**, **Arm Compute Library**, **Arm NN** (status varies). ([ONNX Runtime][8])

# Quick chooser (rules of thumb)

* **NVIDIA datacenter/desktop** → Start with **TensorRT EP** (or native TensorRT) + ORT optimizations; consider int8 QDQ/static quant for max perf. ([ONNX Runtime][14])
* **Intel CPUs/iGPUs** → **OpenVINO EP** + NNCF/POT PTQ. ([OpenVINO Documentation][2])
* **Apple M-series / iOS** → **CoreML EP**; aim for QDQ models; ANE if available. ([ONNX Runtime][10])
* **Snapdragon (Android/Windows on ARM)** → **QNN EP**; calibrate for int8. ([ONNX Runtime][11])
* **Windows mixed GPUs** → **DirectML EP**. ([Microsoft Learn][12])
* **Browser** → **ORT Web** (WASM first, WebGPU where supported). ([ONNX Runtime][15])

# Notes & gotchas

* Match your quant **format** to the EP: many accelerators expect **QDQ or QLinear** patterns; keep opsets compatible. ([ONNX Runtime][1])
* Some compilers/runtimes require **op support parity**; unsupported ops fall back to CPU or need plugins (e.g., TensorRT). ([NVIDIA Docs][16])
* For tiny apps, consider **ORT minimal + ORT format** to shrink binary while baking optimizations ahead-of-time. ([ONNX Runtime][17])
* If you want a one-stop pipeline to target multiple devices, **Olive** orchestrates export → optimize → quantize → package for specific EPs. ([ONNX Runtime][3])

If you tell me your **target hardware(s)** and a sample model (e.g., encoder, decoder, LLM), I’ll sketch a ready-to-run pipeline (export → quant → optimize → EP config) tailored to your stack.

Absolutely—think of ONNX the same way: you’ve got model-level rewrites, runtime/EP knobs, and hardware-specific compilers, all in addition to (or instead of) quantization.

Model-level (before running)

Graph optimizations/fusions: fold constants, eliminate dead nodes, fuse patterns (e.g., LayerNorm, bias-adds), and layout tweaks. Turn on ORT’s basic / extended / all graph-opt levels when you load the model. 
ONNX Runtime

Mixed precision: export FP16/BF16 where kernels support it (pair with TensorRT/OpenVINO for big wins). 
ONNX Runtime
+1

Olive pipelines: one config to compose graph passes, quant (if desired), and EP-targeted compilation (TensorRT/OpenVINO/etc.). Useful even when you skip quant. 
GitHub
+3
ONNX Runtime
+3
Microsoft Open Source
+3

Runtime/Execution-Provider tuning (no model changes)

Threading & scheduling: set intra_op_num_threads, inter_op_num_threads, and execution mode to match core topology (don’t oversubscribe). 
ONNX Runtime

I/O Binding (zero-copy): pre-allocate inputs/outputs on the device and run with run_with_iobinding() to bypass CPU<->GPU shuttling. Massive for LLM loops and vision pipelines. 
ONNX Runtime
+1

Provider options (examples):

CUDA EP: pick convolution algo search, set device ID, and tune memory arena/limits; match CUDA/cuDNN versions for best kernels. 
ONNX Runtime
+1

TensorRT EP: enable FP16/INT8, cache engines, and use tactic selection for fastest layers (AOT-compiled engines = big latency cuts). 
ONNX Runtime

OpenVINO EP: route to CPU/iGPU/NPU and use its graph passes even without quant; combine with NNCF/POT when you do want INT8 later. 
ONNX Runtime
+1

LLM-specific (ONNX Runtime)

Tight generate loop + KV-cache on device: use IOBinding and/or ORT GenAI so keys/values stay on GPU between tokens; avoids host copies and reorder overhead. 
ONNX Runtime
+1

ORT GenAI configs: purpose-built decoding path (CUDA graphs, device-side KV ops, etc.)—drop-in speedups vs. vanilla Run-in-a-loop. 
ONNX Runtime
+1

Hardware-focused playbook (beyond quant)

NVIDIA → Prefer TensorRT EP for compiled engines (FP16 first, INT8 if calibrated). If staying on CUDA EP, set proper algo search and bind tensors to GPU. 
ONNX Runtime
+1

Intel CPU/iGPU/NPU → Use OpenVINO EP; enable graph fusions and FP16 where supported; Olive can auto-tune threads/streams. 
ONNX Runtime
+1

AMD/Windows → Use ORT with the most suitable EP (TensorRT if NVIDIA, OpenVINO for Intel; for AMD on Windows, many teams pair ORT with Olive-driven DirectML stacks). 
Windows Blog

Android → Prefer QNN/NNAPI EP where available; otherwise Vulkan via your app stack and rely heavily on IOBinding to keep buffers device-resident. (General IOBinding guidance applies.) 
ONNX Runtime

Web → ORT-Web: WASM (CPU) or WebGPU; same principle—keep data on device between calls. 
ONNX Runtime

Quick, practical checklist

Export smart: FP16 where possible; clean up graph; set ORT graph_optimization_level=ORT_ENABLE_ALL. 
ONNX Runtime

Pick EP for the box (TensorRT/OpenVINO/ CUDA/DirectML/etc.) and set its provider options. 
ONNX Runtime
+2
ONNX Runtime
+2

Bind memory: use IOBinding so inputs/outputs/KV stick to the accelerator. 
ONNX Runtime

Thread & stream tuning: adjust ORT intra/inter-op threads; profile. 
ONNX Runtime

(LLMs) Use ORT GenAI for device-side KV and a tight decode loop. 
GitHub
+1

Automate with Olive to generate the best combo per target without hand-tuning every knob. 
ONNX Runtime

If you want, tell me your exact target (e.g., RTX 4090, MTL iGPU, Ryzen AI NPU, Snapdragon X Elite) and a model type, and I’ll spit out a ready-to-run ORT config (EP list, provider opts, threading, and IOBinding pattern).

Perfect—here’s a **ready-to-run ONNX Runtime tuning playbook** across a wide range of devices. It’s all “no model changes,” focused on **session/EP options, threading, I/O binding, and KV-cache**. Copy the preset closest to your box and tweak batch/context.

---

# Universal session scaffolding (baseline)

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 0        # 0 = let ORT choose; override per target below
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # switch to PARALLEL for CPU-heavy graphs

# Drop-in: replace 'providers' with the preset for your device
sess = ort.InferenceSession("model.onnx", sess_options=so, providers=[("CPUExecutionProvider", {})])
```

**I/O Binding template (avoid host↔device copies):**

```python
import numpy as np
io = ort.IOBinding(sess._sess)
# Pre-create device buffers; example shows CUDA, swap to your device type
io.bind_input(name="input_ids", device_type="cuda", device_id=0, element_type=np.int64, shape=(B, T), buffer_ptr=int(input_dev_ptr))
io.bind_output(name="logits", device_type="cuda", device_id=0)
sess.run_with_iobinding(io)
out = io.get_outputs()[0]  # stays on device
```

---

## NVIDIA GPUs (desktop/datacenter: 30/40-series, A100/H100)

**Best EP**: **TensorRT Execution Provider** (compiled engines) → FP16 first, INT8 if calibrated.
**Fallback**: CUDA EP with tuned options.

```python
trt_opts = {
  "trt_fp16_enable": True,
  "trt_int8_enable": False,               # True if you have a calibration table
  "trt_engine_cache_enable": True,
  "trt_engine_cache_path": "./trt_cache",
  "trt_timing_cache_enable": True,
  "trt_max_workspace_size": 2<<30,        # tune up if you have VRAM
}
cuda_opts = {
  "device_id": 0,
  "arena_extend_strategy": "kSameAsRequested",
  "cudnn_conv_algo_search": "HEURISTIC",
  "do_copy_in_default_stream": True,
}

so.intra_op_num_threads = 1               # GPU does the work; don’t oversubscribe CPU
providers = [("TensorrtExecutionProvider", trt_opts),
             ("CUDAExecutionProvider", cuda_opts),
             ("CPUExecutionProvider", {})]
sess = ort.InferenceSession("model.onnx", so, providers=providers)
```

**LLMs**: keep KV-cache on GPU via I/O Binding; use **ORT GenAI** decode loop if you’re using ORT’s LLM APIs.

---

## Intel (CPU + Arc iGPU + NPU/Movidius/Integrated via OpenVINO)

**Best EP**: **OpenVINO EP** (CPU/iGPU/“AUTO”). Great even without quant.

```python
ov_opts = {
  "device_type": "AUTO",                  # "CPU","GPU","MULTI:CPU,GPU","AUTO"
  "enable_opencl_throttling": "False",
  "num_of_threads": "0",                  # let OV pick; or set to physical cores
  "cache_dir": "./ov_cache",
  "enable_dynamic_shapes": "True",
  "intra_op_num_threads": "0",
  "inter_op_num_threads": "1",
}
so.intra_op_num_threads = 0               # let OV orchestrate threads
providers = [("OpenVINOExecutionProvider", ov_opts),
             ("CPUExecutionProvider", {})]
sess = ort.InferenceSession("model.onnx", so, providers=providers)
```

**Tip**: If you later add INT8, generate an OV INT8 IR (via NNCF/POT) and keep the same EP.

---

## AMD GPUs

### Linux (ROCm)

If you’re on ROCm-enabled stack, prefer **ROCm EP** (if available in your build). Otherwise:

### Cross-platform fallback: **DirectML (Windows)** or **Vulkan backends via your app**

For ONNX Runtime on Windows boxes without NVIDIA:

```python
dml_opts = {"device_id": 0}
so.intra_op_num_threads = 1
providers = [("DmlExecutionProvider", dml_opts), ("CPUExecutionProvider", {})]
sess = ort.InferenceSession("model.onnx", so, providers=providers)
```

**Tips**

* Keep **prompt batch moderate** (vision: 16–64, LLM prompt: 256–512 tokens fed at once).
* Bind inputs/outputs to device and avoid round-trips.

---

## Apple Silicon (M-series Macs)

**Best EP**: **CoreML EP** (for supported models) else **CPU EP** with **use_mps=True** isn’t an ORT EP; for ONNX Runtime, CoreML is the official accelerator.

```python
coreml_opts = {
  "coreml_flags": 0,          # bitmask; 0 = default; try 1 for enable_on_subgraphs
  "preferred_memory_format": "NHWC",
}
so.intra_op_num_threads = 0
providers = [("CoreMLExecutionProvider", coreml_opts), ("CPUExecutionProvider", {})]
sess = ort.InferenceSession("model.onnx", so, providers=providers)
```

**Tips**

* Use FP16 models when possible.
* For large LLMs, consider GGUF + Metal (llama.cpp) for decode and pair vision/STT in ORT; pass embeddings between them.

---

## Android (phones/tablets)

**Best EPs**: **QNN EP** (Snapdragon) or **NNAPI EP** (generic). Fall back to CPU if drivers are flaky.

```python
qnn_opts = {
  "backend_path": "libQnnHtp.so",      # HTP (NPU); use libQnnGpu.so for Adreno GPU
  "profiling": "false",
  "qnn_context_cache_enable": "true",
  "qnn_context_cache_path": "/data/local/tmp/qnn_cache",
}
nnapi_opts = {
  "execution_mode": "sequential",      # or "parallel" for some graphs
  "use_fp16": True,
}

so.intra_op_num_threads = 2            # pin to big cores; avoid 4+ (thermals)
providers = [("QNNExecutionProvider", qnn_opts),
             ("NnapiExecutionProvider", nnapi_opts),
             ("CPUExecutionProvider", {})]
sess = ort.InferenceSession("model.onnx", so, providers=providers)
```

**Tips**

* **Always** use I/O Binding to keep tensors on device.
* Prompt batch (LLM ingest) 192–384; decode batch 6–16.
* Keep the app foreground; enable Android sustained-perf mode.

---

## Jetson (Nano/Xavier/Orin)

**Best EP**: **TensorRT EP** (like NVIDIA preset above), optionally chain with CUDA EP.

**Nano specifics**

* FP16 engines where possible; INT8 only with careful calibration.
* Smaller prompt batches; consider splitting long prompts.

**Orin/Xavier**

* Larger **workspace**, bigger batches, KV on device.

(Use the NVIDIA TensorRT preset; tweak `trt_max_workspace_size` up and watch memory.)

---

## Raspberry Pi 4/5 (ARM CPU NEON)

**Best EP**: **CPU EP** (NEON).
Use ORT build compiled with `-mcpu=native -O3`.

```python
so.intra_op_num_threads = 4      # Pi 4: 4 cores; Pi 5: also 4 Perf cores
so.inter_op_num_threads = 1
providers = [("CPUExecutionProvider", {"use_arena": True})]
sess = ort.InferenceSession("model.onnx", so, providers=providers)
```

**Tips**

* Prefer INT8 models (OpenVINO on x86 beats Pi; on Pi, stick to CPU INT8).
* Keep prompt batch small; use streaming for STT/TTS.

---

## Web (browser)

**Best runtimes**: **ORT-Web WASM** (CPU) or **WebGPU** (GPU).

**WASM:**

```javascript
import * as ort from 'onnxruntime-web';
const sess = await ort.InferenceSession.create('model.onnx', { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
```

**WebGPU:**

```javascript
const sess = await ort.InferenceSession.create('model.onnx', { executionProviders: ['webgpu'] });
// Keep tensors on GPU: feed ort.Tensor with gpuBuffer and use ioBinding in web API (where available)
```

---

# LLM-specific knobs (on any GPU/accelerator)

* **Tight decode loop**: Don’t call `session.run` per token with host copies.
  Bind **inputs, outputs, and KV-cache** to device with I/O Binding; reuse buffers across tokens.
* **ORT GenAI (if you’re using it)**: enable device-side KV ops, CUDA graphs, and fused sampling.
* **Streams/graphs**: one stream per request (or micro-batch) is usually best; measure before increasing.

---

# Quick presets (copy/paste)

**RTX 4090 (24 GB):**

* EPs: TensorRT (FP16, engine cache) → CUDA → CPU
* `trt_max_workspace_size`: 4–8 GB
* `intra_op`: 1
* Prompt batch: 512–2048; Decode batch: 16–64

**A100/H100:**

* Same as above; push prompt batch to 4k+; keep I/O Binding and KV on device

**Intel Xeon + Arc iGPU:**

* EP: OpenVINO (“AUTO”); `so.intra_op=0`
* Streams scale well; keep batch moderate; FP16 if supported

**AMD 7900 XTX (Windows):**

* EP: DirectML; `intra_op=1`
* Moderate prompt batch (256–512); I/O Binding mandatory

**MacBook Pro M3 Pro:**

* EP: CoreML; FP16 models
* `intra_op=0`; keep batch modest; consider offloading LLM to GGUF/Metal if context is large

**Snapdragon 8 Gen 3 (Android):**

* EPs: QNN (HTP) → NNAPI → CPU
* `intra_op=2`; Prompt batch 256–384; Decode 8–16; use sustained performance mode

**Jetson Orin:**

* EPs: TensorRT → CUDA → CPU
* `trt_max_workspace_size`: 2–6 GB; prompt 256–768; decode 8–24

**Raspberry Pi 5:**

* EP: CPU
* `intra_op=4`; INT8 models; prompt 128–256; decode 4–8

---

## Sanity checklist (always)

1. **ORT_ENABLE_ALL** graph opts.
2. **Pick the right EP** (TensorRT / OpenVINO / QNN / CoreML / DirectML / CUDA / CPU).
3. **Bind memory** (I/O Binding) so data + KV never leave the device.
4. **Threading**: GPU-heavy → `intra_op=1`; CPU-heavy → physical cores.
5. **Profile** batches and memory arenas; cache engines where supported.
6. **For LLMs**: use ORT GenAI or a custom tight loop with device-resident KV.

---

If you give me a **specific target + model type** (e.g., “Pixel 8 Pro + Whisper-tiny.onnx” or “Orin Nano + YOLOv8n.onnx”), I’ll hand you the exact provider dict, thread counts, and I/O-binding shapes you can paste in verbatim.


[1]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html?utm_source=chatgpt.com "Quantize ONNX models | onnxruntime"
[2]: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html?utm_source=chatgpt.com "Basic Quantization Flow - OpenVINO™ documentation"
[3]: https://onnxruntime.ai/docs/performance/olive.html?utm_source=chatgpt.com "End to end optimization with Olive"
[4]: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html?utm_source=chatgpt.com "Graph Optimizations in ONNX Runtime"
[5]: https://github.com/onnx/optimizer?utm_source=chatgpt.com "ONNX Optimizer"
[6]: https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/optimization?utm_source=chatgpt.com "Optimization"
[7]: https://pypi.org/project/onnxruntime-tools/?utm_source=chatgpt.com "onnxruntime-tools"
[8]: https://onnxruntime.ai/docs/execution-providers/?utm_source=chatgpt.com "ONNX Runtime Execution Providers"
[9]: https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/?utm_source=chatgpt.com "CUDA and TensorRT Execution Providers in ONNX Runtime"
[10]: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html?utm_source=chatgpt.com "CoreML Execution Provider - Apple"
[11]: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html?utm_source=chatgpt.com "QNN Execution Provider - Qualcomm"
[12]: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers?utm_source=chatgpt.com "Supported execution providers in Windows ML"
[13]: https://onnxruntime.ai/docs/tutorials/web/?utm_source=chatgpt.com "Web | onnxruntime"
[14]: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html?utm_source=chatgpt.com "TensorRT Execution Provider - NVIDIA"
[15]: https://onnxruntime.ai/docs/get-started/with-javascript/web.html?utm_source=chatgpt.com "Web | onnxruntime"
[16]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html?utm_source=chatgpt.com "Quick Start Guide — NVIDIA TensorRT Documentation"
[17]: https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-model-runtime-optimization.html?utm_source=chatgpt.com "ORT Format Model Runtime Optimization"

