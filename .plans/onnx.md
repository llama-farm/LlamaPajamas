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
