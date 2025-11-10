Absolutely. You can skip ONNX entirely (or keep it just for portability) and run on **native, vendor-tuned runtimes**. Below is a pragmatic map—**what to use, when, and how**—plus minimal code/pipeline pointers. Think: fastest path to “shipping tokens/frames now.”

---

# Quick chooser (native runtimes)

| Hardware                               | Best native runtime(s)                                                       | Typical models                    | Why pick it                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Apple Silicon (M-series, iOS)**      | **MLX** (Python array/runtime), **Core ML** (coremltools)                    | LLMs, ViT/CLIP, Whisper           | MLX is dead-simple & fast on Mac; Core ML is the native graph runtime for apps (ANE use on iOS). |
| **NVIDIA (desktop/datacenter/Jetson)** | **TensorRT / TRT-LLM**, **cuDNN/cuBLAS** (via libraries), **TensorRT-Audio** | LLMs, YOLO/Detr, Whisper          | Kernel-fused, engine-compiled, FP16/INT8/FP8 paths; best throughput/latency.                     |
| **Intel CPU/iGPU/Movidius**            | **OpenVINO Runtime**                                                         | Vision, STT/TTS, LLMs (BF16/INT8) | Threading/streams, BF16 on Xeons, VPU offload; great server CPU perf.                            |
| **AMD GPU (Linux)**                    | **MIGraphX** (or ROCm + Triton kernels)                                      | Vision/LLM                        | Native graph compiler on ROCm; good when you live in AMD land.                                   |
| **Windows (vendor-agnostic GPU)**      | **DirectML**                                                                 | Vision, some LLMs                 | One API for AMD/Intel/NVIDIA; easy app integration.                                              |
| **Android (Snapdragon/iGPU)**          | **Qualcomm QNN SDK**, **NNAPI**                                              | Vision, Whisper, small LLM        | Targets HTP (NPU)/GPU; power-efficient mobile inference.                                         |
| **Web**                                | **WebGPU** (wgsl)                                                            | Vision, small LLMs                | Pure browser; no install—good demos and light clients.                                           |
| **CPU-only (Pi/edge boxes)**           | **oneDNN / XNNPACK / OpenVINO-CPU**                                          | Vision, TTS, tiny LLM             | Highly-tuned CPU kernels; predictable deployment.                                                |
| **GGUF path (multi-platform)**         | **llama.cpp / ggml backends (CUDA/Metal/ROCm/Vulkan)**                       | LLMs, some MLLM forks             | Best-in-class LLM decode loop with low-bit quant + device KV.                                    |

---

# When to choose native over ONNX

* **You need peak perf** (TensorRT/MLX/TRT-LLM often beat general runtimes).
* **You control a single target** (no need for “run anywhere”).
* **You want vendor features** (CUDA graphs, FP8, ANE/HTP/NPU paths).
* **You want fused, AOT-compiled engines** (TensorRT/OpenVINO caches).

If you still need **multi-target** delivery, keep an **ONNX export** as a fallback; you can maintain **two tracks**:

* Track A (native): MLX / TensorRT / OpenVINO build for production targets.
* Track B (portable): ONNX Runtime for everything else.

---

## Pipelines by device (copy-ready patterns)

### 1) Apple Silicon (Mac, iOS)

**Option A – MLX (fast dev, Python):**

* Convert weights via community scripts (e.g., LLaMA/Whisper → MLX).
* Use MLX array ops; it schedules Metal + AMX under the hood.

**LLM decode (MLX) sketch**

```python
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3-8B-instruct-4bit")
out = generate(model, tokenizer, prompt="Hello", max_tokens=256, temp=0.7)
print(out)
```

**Option B – Core ML (apps, ANE on iOS):**

* `coremltools` convert (PyTorch → Core ML), set **computeUnits=all**.
* Use **MLProgram** + weight FP16; fuse pre/post steps.

---

### 2) NVIDIA (desktop/datacenter/Jetson)

**TensorRT engine path**

* Export **directly from PyTorch** with **TensorRT-LLM** *or* via ONNX just to build an engine.
* Enable **FP16** first; add **INT8** with calibration for vision/STT; try **FP8** on H100 if supported.

**Build & run (Python, TRT):**

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# ... parse ONNX or use TRT-LLM builder ...
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6<<30)
config.set_flag(trt.BuilderFlag.FP16)
engine = builder.build_engine(network, config)
# serialize engine; later, create runtime + context, set bindings, enqueueV3
```

**TRT-LLM (LLMs)** gives a turnkey decode loop (CUDA graphs, paged KV-cache, tensor parallel).

**Jetson**: use `nvpmodel` + `jetson_clocks`, FP16 engines, moderate batch.

---

### 3) Intel (CPU/iGPU/VPU) – OpenVINO Runtime

**Direct compile & run**

```python
from openvino.runtime import Core
ie = Core()
model = ie.read_model("model.xml")        # or import PyTorch via OV frontend
compiled = ie.compile_model(model, "AUTO")# "CPU","GPU","MULTI:CPU,GPU"
infer = compiled.create_infer_request()
infer.infer({"input": input_blob})
```

* Use **AUTO** for device choice, **caching** for graphs, **streams** for throughput.
* BF16 (Xeon/Sapphire Rapids) + **INT8** (NNCF/POT) for big CPU wins.

---

### 4) AMD (ROCm) – MIGraphX

* Import ONNX or PyTorch FX, compile to **ROCm** kernels, run with MIGraphX runtime.
* Great when your fleet is AMD and you want native compilation.

---

### 5) Android – QNN / NNAPI

* **QNN SDK** converts models and targets **HTP (NPU)** or **Adreno GPU**.
* If you already have TFLite models, NNAPI provides a simple path.
* Keep **threads=2–3 big cores**; use sustained performance mode; keep tensors device-resident.

---

### 6) Web – WebGPU (no install)

* Compile shaders (WGSL) or use a framework (e.g., `webgpu` backends in libs).
* Keep tensors in GPU buffers; avoid CPU copies between steps.

---

## Modality playbook (vision / STT / TTS / LLM)

| Modality                        | Apple (MLX/Core ML) | NVIDIA (TensorRT)            | Intel (OpenVINO)       | Android (QNN/NNAPI) | Notes                                        |
| ------------------------------- | ------------------- | ---------------------------- | ---------------------- | ------------------- | -------------------------------------------- |
| **Vision** (YOLO/ViT)           | Core ML FP16        | TRT FP16/INT8 + engine cache | OV FP16/INT8 + streams | QNN HTP/Adreno      | AOT engines + fused post-proc where possible |
| **STT** (Whisper)               | MLX or Core ML FP16 | TRT FP16 (enc), INT8 (dec)   | OV INT8 enc, FP16 dec  | QNN/NNAPI INT8      | Stream chunks; mel on device                 |
| **TTS** (FastSpeech + HiFi-GAN) | Core ML FP16        | TRT FP16 (acoustic+vocoder)  | OV FP16/INT8           | NNAPI FP16/INT8     | Stream frames; ring buffers                  |
| **LLM**                         | MLX (4-bit OK)      | TRT-LLM FP16/INT8 + paged KV | OV BF16/INT8           | QNN small LLMs      | Keep KV on device; 1 stream/req              |

---

## Accuracy vs size (if you still need smaller)

* Prefer **native mixed precision** (FP16 weights/acts) first.
* Next, **INT8** with calibration/QAT for vision/STT/TTS parts.
* **INT4** is possible but usually needs QAT; if you mainly need LLM decode, consider **GGUF+llama.cpp** for the language part and keep **vision/STT/TTS** on native pipelines.

---

## Interop patterns (mixed runtimes)

It’s common to **mix**:

* **LLM** → GGUF + `llama.cpp` (Metal/CUDA) for decode loop
* **Vision/STT/TTS** → TensorRT/Core ML/OpenVINO natively
* **Glue** them with shared device buffers (or at least avoid unnecessary host copies).

Example:

```
camera → TensorRT (YOLO) → labels/embeddings
audio  → TensorRT/OpenVINO (Whisper) → text
text   → llama.cpp (GGUF) → response
resp   → TensorRT/OpenVINO (TTS) → audio out
```

---

## Practical “getting started” picks

* **Mac dev**: Start with **MLX** for LLMs + **Core ML** for vision/audio you’ll ship to iOS.
* **NVIDIA server**: Go **TensorRT/TRT-LLM** first; use ONNX only if you need a fallback.
* **Intel server**: **OpenVINO** end-to-end; great CPU perf and simple deployment.
* **Android**: **QNN** (Snapdragon) or **NNAPI** (generic), keep everything device-resident.

---

If you tell me 2–3 exact combos (e.g., *“M3 Pro + LLaMA-3-8B + CLIP”*, *“Jetson Orin + YOLOv8n + FastSpeech2”*, *“Pixel 8 Pro + Whisper-tiny streaming”*), I’ll give you **precise build flags, cache settings, stream/thread counts, and tiny code stubs** for each native runtime.




Awesome—here’s a compact, **production-oriented matrix** for native runtimes across your targets (Macs, Linux, Windows, AMD, NVIDIA, Jetson), by modality:

* **Small LLM (<1.7B)**
* **Larger LLM (≈8B)**
* **Vision (det/cls/seg)**
* **STT (Whisper-class)**
* **TTS (FastSpeech2 + HiFi-GAN or similar)**

Each row shows the **best native runtime**, **precision**, **batching**, **threading**, **buffer-sharing hint**, and **notes**. It’s “no model changes” (quant is optional).

---

## Macs (Apple Silicon, macOS)

| Modality          | Runtime                                     | Precision                       | Batch / Context              | Threads   | Device-buffer sharing                             | Notes                                            |
| ----------------- | ------------------------------------------- | ------------------------------- | ---------------------------- | --------- | ------------------------------------------------- | ------------------------------------------------ |
| Small LLM (<1.7B) | **MLX** (Python) or **Core ML**             | FP16 or 4-bit weights (MLX)     | prompt 256–512 / decode 8–16 | `intra=0` | MLX tensors stay on Metal; pass CPU text/ids only | MLX is dead-simple + fast; great for dev & tools |
| LLM 8B            | **GGUF + llama.cpp (Metal)** or **Core ML** | Q4_K_M (GGUF) or FP16 (Core ML) | 256–384 / 8–16               | `intra=0` | llama.cpp keeps **KV on GPU**; zero copies        | For long ctx, llama.cpp decode loop wins         |
| Vision            | **Core ML**                                 | FP16                            | N=32–128 @224²               | `intra=0` | Core ML inferences stay on ANE/GPU                | Convert with coremltools; NHWC preproc           |
| STT               | **Core ML** (or MLX for research)           | FP16                            | 10–30 s chunk / 10 ms hop    | `intra=0` | Keep mel on device; stream                        | iOS: **computeUnits=all**                        |
| TTS               | **Core ML**                                 | FP16 (INT8 for small)           | frame 50–100 ms              | `intra=0` | Use streaming output buffers                      | Favor Lite vocoder on mobile                     |

---

## NVIDIA (Linux/Windows desktop/datacenter)

| Modality  | Runtime                              | Precision                | Batch / Context  | Threads   | Device-buffer sharing                      | Notes                                         |
| --------- | ------------------------------------ | ------------------------ | ---------------- | --------- | ------------------------------------------ | --------------------------------------------- |
| Small LLM | **TensorRT-LLM** (or llama.cpp CUDA) | FP16; INT8 if calibrated | 256–1024 / 16–32 | `intra=1` | **TensorRT bindings** (GPU ptrs), reuse KV | Enable **engine+tactic cache**, big workspace |
| LLM 8B    | **TensorRT-LLM**                     | FP16 (INT8 if calib)     | 512–2048 / 16–64 | `intra=1` | KV on GPU; one stream/req                  | CUDA Graphs + paged KV                        |
| Vision    | **TensorRT**                         | FP16/INT8                | N=16–64 @640²    | `intra=1` | Bind input/output GPU ptrs                 | Fuse post-proc on GPU                         |
| STT       | **TensorRT**                         | FP16 enc, INT8 dec       | 15–30 s / 10 ms  | `intra=1` | Keep mel + logits on GPU                   | Streaming windowing on device                 |
| TTS       | **TensorRT**                         | FP16 (INT8 mixed)        | frame 50–100 ms  | `intra=1` | Ring buffers on GPU                        | Separate acoustic/vocoder engines OK          |

---

## AMD GPUs (Linux/Windows)

| Modality  | Runtime                            | Precision            | Batch / Context | Threads   | Device-buffer sharing      | Notes                             |
| --------- | ---------------------------------- | -------------------- | --------------- | --------- | -------------------------- | --------------------------------- |
| Small LLM | **llama.cpp ROCm** or **MIGraphX** | Q4_K_M (GGUF) / FP16 | 256–512 / 8–24  | `intra=1` | ROCm tensors; reuse KV     | Good sustained decode             |
| LLM 8B    | **llama.cpp ROCm** or **MIGraphX** | Q4_K_M / FP16        | 256–512 / 8–24  | `intra=1` | KV on GPU                  | Watch VRAM; mixed KV helps        |
| Vision    | **MIGraphX** (ROCm)                | FP16/INT8            | N=16–64 @640²   | `intra=1` | Device buffers via ROCm    | Prefer depthwise-friendly nets    |
| STT       | **MIGraphX** or CPU+OV fallback    | INT8/FP16            | 15–30 s / 10 ms | `intra=1` | Keep mel device-side       | If kernels missing, run on CPU OV |
| TTS       | **MIGraphX**                       | FP16                 | frame 50–100 ms | `intra=1` | Stream vocoder device-side | Consider Lite vocoder             |

---

## Linux (CPU-first servers)

| Modality  | Runtime                    | Precision          | Batch / Context | Threads              | Buffer sharing    | Notes                          |
| --------- | -------------------------- | ------------------ | --------------- | -------------------- | ----------------- | ------------------------------ |
| Small LLM | **OpenVINO Runtime (CPU)** | BF16/INT8          | 256–512 / 8–16  | **= physical cores** | OV remote tensors | Great throughput on Xeon       |
| LLM 8B    | **OpenVINO (CPU)**         | BF16/INT8 mix      | 256–512 / 8–24  | cores; NUMA pin      | KV in RAM         | Use streams; interleave memory |
| Vision    | **OpenVINO (CPU/iGPU)**    | INT8/FP16          | N=64–256 @224²  | cores                | Zero-copy blobs   | Cache compiled models          |
| STT       | **OpenVINO (CPU/iGPU)**    | INT8 enc, FP16 dec | 10–30 s / 10 ms | cores                | Stream            | Good for many small streams    |
| TTS       | **OpenVINO**               | FP16/INT8 mix      | frame 50–100 ms | cores                | Audio ring        | Real-time on modern CPUs       |

---

## Windows (vendor-agnostic GPU / CPU)

| Modality  | Runtime                              | Precision   | Batch / Context | Threads                   | Buffer sharing      | Notes                              |
| --------- | ------------------------------------ | ----------- | --------------- | ------------------------- | ------------------- | ---------------------------------- |
| Small LLM | **DirectML** (GPU) or OV (CPU)       | FP16 / INT8 | 256–512 / 8–16  | `intra=1` (GPU)           | D3D12 buffers (DML) | Moderate batch; warm once          |
| LLM 8B    | **DirectML** or TensorRT (if NVIDIA) | FP16        | 256–512 / 8–24  | `intra=1`                 | Device buffers      | Prefer vendor runtime if available |
| Vision    | **DirectML**                         | FP16/INT8   | N=8–32 @640²    | `intra=1`                 | D3D12 device ptrs   | Avoid overlay apps                 |
| STT       | **DirectML** or OV CPU               | FP16/INT8   | 10–30 s / 10 ms | `intra=1` GPU / cores CPU | Stream              | If driver gaps, run on OV CPU      |
| TTS       | **DirectML** or OV CPU               | FP16/INT8   | frame 50–100 ms | as above                  | Ring buffers        | Keep small/steady batches          |

---

## Jetson (Nano / Xavier / Orin)

| Modality  | Runtime                 | Precision          | Batch / Context | Threads   | Buffer sharing      | Notes                        |
| --------- | ----------------------- | ------------------ | --------------- | --------- | ------------------- | ---------------------------- |
| Small LLM | **llama.cpp CUDA**      | Q3/Q4_K_M          | 128–256 / 4–12  | `intra=1` | KV on GPU if fits   | Nano: tiny models only       |
| LLM 8B    | **TensorRT-LLM (Orin)** | FP16 (INT8 gated)  | 256–768 / 8–24  | `intra=1` | KV on GPU; 1 stream | `nvpmodel` + `jetson_clocks` |
| Vision    | **TensorRT**            | FP16/INT8          | N=16–32 @640²   | `intra=1` | Bind GPU ptrs       | Compile engines ahead; cache |
| STT       | **TensorRT**            | FP16 enc, INT8 dec | 10–20 s / 10 ms | `intra=1` | On-device mel       | Use smaller Whisper variants |
| TTS       | **TensorRT**            | FP16               | frame 50–100 ms | `intra=1` | Stream vocoder      | Cool & lock clocks for RT    |

---

New idea, how we will route dynamically in the future, not needed now.

# Router Model Selection: Multi-Stage (fast, CPU-only, 20–40 ms)

**Goal:** <50 ms overhead (ideally 20–40 ms), 95%+ accuracy, no GPU steal, cheap at scale.

### Stage 1 — Fast Feature Extraction (≈5–10 ms)

* Heuristics: content flags (image/audio/video), token/char counts, code fences, JSON hint, question mark, sentence count.
* Fast language ID (e.g., compact FastText), **regex-based code** detection.
* **Early exits** (≈50% of traffic):

  * `has_image` → **Vision** pipeline
  * `has_audio` → **STT/TTS** pipeline
  * `token_count<100 and max_tokens<200` → **Small LLM**
  * `security_level="airgap"` → CPU-only/OpenVINO node

### Stage 2 — Intent Classification (≈10–20 ms)

* **Model:** tiny DistilBERT-style (≈20–30M params) **INT8 CPU** or linear probe over 256-D embeddings.
* **Classes:** {chat, code, vision, stt, tts, tool-use, long-form, summarize, classify, route-only}
* **Target:** ~12 ms on 8-core x86, 95%+ accuracy on your intents.

### Stage 3 — Hardware Selection (≈5–10 ms)

* Rule set + small learned policy (bandit/logistic):

  * **LLM ≤1.7B**: CPU (OV BF16) if latency SLO ≤150 ms; else nearest GPU if free.
  * **LLM ≈8B**: NVIDIA→TensorRT-LLM; Apple→llama.cpp Metal; AMD→ROCm; Jetson→TRT-LLM.
  * **Vision**: GPU compiler (TRT/MIGraphX/Core ML).
  * **STT/TTS**: prefer GPU/NPU; fall back to CPU OV if queue long.
  * Respect **affinity** & **session reuse** to hit warm caches/engines.

### Zero-copy / shared-buffer rules (critical)

* **NVIDIA (TRT)**: allocate device buffers once; pass **raw CUDA pointers** to bindings each call; **reuse KV**.
* **OpenVINO**: use **remote tensors** / pre-allocated blobs; no host copies between stages on same device.
* **Core ML**: keep Core ML tensors in pipeline; avoid CPU pre/post when chaining.
* **llama.cpp**: decode loop keeps KV on device; feed token IDs from CPU only.
* **Cross-runtime bridge**: prefer **DLPack** (Torch/CuPy/TVM) to share GPU memory when mixing components.

---

## Minimal router policy (pseudo)

```python
def route(req, sysload):
    f = extract_heuristic_features(req)  # ~5ms
    # Stage 1 early exits
    if f['has_image']:      return "VISION", choose_device_gpu(sysload)
    if f['has_audio']:      return "STT" if req.text is None else "TTS", choose_audio_device(sysload)
    if f['token_count'] < 100 and req.max_tokens < 200:
        return "SMALL_LLM", choose_fast_cpu_or_gpu(sysload)

    # Stage 2 intent (~12ms)
    intent = tiny_intent_classifier(req.text)  # INT8 CPU

    # Stage 3 device policy (~5ms)
    if intent in {"vision"}:   return "VISION", best_gpu_engine("TRT/MIGraphX/CoreML", sysload)
    if intent in {"stt"}:      return "STT", best_audio_accel(sysload)
    if intent in {"tts"}:      return "TTS", best_audio_accel(sysload)

    # LLMs
    if intent in {"chat","summarize","classify"}:
        if needs_long_ctx(req) or high_quality(req):
            return "LLM_8B", best_llm_backend(sysload)   # TRT-LLM / Metal / ROCm
        else:
            return "SMALL_LLM", best_cpu_llm(sysload)    # OV BF16 / small GPU if idle
```

> **Latency budget:** Stage1(5–10) + Stage2(10–20) + Stage3(5–10) = **20–40 ms** typical.

---

## Practical defaults to start tomorrow

* **Mac (M-series):** LLM→**llama.cpp Metal**, Vision/STT/TTS→**Core ML**.
* **NVIDIA server:** everything compiled in **TensorRT / TRT-LLM**; keep **engine cache** on SSD.
* **Intel CPU farm:** all modalities in **OpenVINO** (BF16/INT8), many streams, **NUMA pinning**.
* **AMD fleet:** LLM→**llama.cpp ROCm**, Vision/STT/TTS→**MIGraphX**.
* **Jetson:** Vision/STT/TTS→**TensorRT**, LLM→**TRT-LLM** (Orin) or **llama.cpp** (Nano).

If you drop me 2–3 **exact combos** (e.g., “Orin Nano + YOLOv8n + FastSpeech2” and “Mac M3 Pro + 8B LLM + Whisper-tiny streaming”), I’ll hand back the **engine build flags, batch/stream counts, and buffer-binding snippets** you can paste in.

Absolutely—on the **edge** you optimize for *tight RAM/VRAM, thermals, power, and offline stability*. That changes **precisions, batching, threading, and routing**. Here’s a practical, copy-ready plan.

---

# What changes at the edge (principles)

* **Precision:** push to **INT8** by default; **INT4 / mixed** where kernels exist (keep sensitive ops FP16/INT8).
* **Batching:** small & steady. Big *prompt* batches only when you can prewarm; *decode/stream* batches tiny.
* **Threads:** cap to **physical big cores** (2–4 typical); avoid spikes → less throttling.
* **KV & buffers:** keep **device-resident** (or pinned host) and **reuse**; mmap models, warm once.
* **Thermal governor:** drop decode batch / switch model tier as temps rise.
* **Router:** more early exits + aggressive “small-first” routing; promote only if SLO not met.

---

# Edge modality defaults (use unless a device table below overrides)

| Modality                         | Precision target                                           | Runtime (native)                            | Typical batch/context                | Notes                                               |
| -------------------------------- | ---------------------------------------------------------- | ------------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| **Small LLM (<1.7B)**            | **INT4 weights + INT8 acts** (GGUF Q4_K_M)                 | llama.cpp (Metal/CUDA/ROCm/CPU)             | prompt **128–256** / decode **4–12** | KV on device; speculative/offline cache if possible |
| **LLM ~8B**                      | Edge-rare; if needed: **Q4_K_M**                           | TRT-LLM (Orin/NVIDIA), llama.cpp Metal/ROCm | prompt **256–384** / decode **8–16** | Consider split/hybrid or server fallback            |
| **Vision (det/cls)**             | **INT8** (per-channel), some layers **FP16**               | TensorRT / OpenVINO / MIGraphX / Core ML    | N=**8–16** @ 320–640²                | Fuse pre/post; NHWC; keep tensors on device         |
| **STT (Whisper-tiny/base-en)**   | **INT8 enc**, **INT8/FP16 dec**; (INT4 with QAT if needed) | TensorRT / OpenVINO / QNN / Core ML         | chunk **10–20s**, hop **10ms**       | Streamed mel; ring buffers                          |
| **TTS (FastSpeech2 + HiFi-GAN)** | **INT8** acoustic, **FP16** vocoder (or INT8 QAT)          | TensorRT / OpenVINO / Core ML               | frame **50–100ms**                   | Streaming playback; steady clocks                   |

---

# Device-specific edge presets (no model changes)

## Raspberry Pi 5 (8 GB, ARM NEON, CPU-only)

* Threads: **4/1**; Governor: performance; swap/zram on.
* **Small LLM:** llama.cpp **Q3_K/Q4_K_M**, 128–256 / 4–8; mmap+warm.
* **8B LLM:** not recommended → route to server.
* **Vision:** OpenVINO-CPU *or* ONNX-CPU INT8, N=8–16 @ 320–480².
* **STT:** OV-CPU INT8 enc / FP16 dec, 10–15s / 10ms; stream.
* **TTS:** OV-CPU INT8/FP16, 50–80ms frames.
* Notes: prefer lightweight backbones (MobileNetV3-S, YOLOv8n).

## Jetson **Nano** (4 GB)

* cuda cores are tight; keep VRAM <3.2 GB.
* **Small LLM:** llama.cpp CUDA **Q3_K** (≤3–4B distilled), 128 / 4–8; KV mostly on CPU if needed.
* **Vision:** TensorRT **FP16** (INT8 only if calibrated), N=8–12 @ 416–512².
* **STT:** TRT FP16 enc / INT8 dec, 10–15s / 10ms.
* **TTS:** TRT FP16; small vocoder.
* Notes: lock `jetson_clocks`; avoid large prompt spikes.

## Jetson **Orin Nano / Xavier NX**

* **Small LLM:** llama.cpp CUDA **Q4_K_M**, 256 / 8–12; KV on GPU.
* **8B LLM:** **TRT-LLM FP16** if 16 GB; prompt 256–384 / 8–16.
* **Vision:** TRT INT8 (calibrated) or FP16, N=16–32 @ 640².
* **STT/TTS:** TRT FP16/INT8 mixed; stream; ring buffers.
* Notes: `nvpmodel` max perf + `jetson_clocks`; big TRT workspace (2–4 GB).

## Intel NUC / small Xeon (CPU-first)

* Threads: **= physical cores**; NUMA pin on multi-socket; streams=cores/2.
* **Small LLM:** **OpenVINO BF16/INT8**, 256 / 8–12.
* **8B LLM:** OV **BF16/INT8 mix**, 256–384 / 8–16 (latency ↑).
* **Vision:** OV **INT8**, N=32–128 @ 224–512².
* **STT/TTS:** OV INT8/FP16; streaming.
* Notes: enable OV cache dir; interleave memory.

## AMD Ryzen Embedded / Mini-PC (RDNA/ROCm if Linux)

* **Small LLM:** llama.cpp **ROCm Q4_K_M**, 256 / 8–12.
* **8B LLM:** llama.cpp ROCm **Q4_K_M** (watch VRAM) or server fallback.
* **Vision:** **MIGraphX FP16/INT8**, N=16–32 @ 512².
* **STT/TTS:** MIGraphX FP16/INT8; stream.
* Notes: moderate batches; device-resident buffers via ROCm.

## NVIDIA desktop (low-power 3050/3060 or similar)

* **Small LLM:** llama.cpp CUDA **Q4_K_M**, 256 / 8–16, KV on GPU.
* **8B LLM:** **TRT-LLM FP16**, 256–512 / 12–24; engine cache on SSD.
* **Vision:** **TRT INT8/FP16**, N=16–32 @ 640².
* **STT/TTS:** TRT FP16 (INT8 where stable); stream.
* Notes: set persistence mode; large TRT workspace; one stream/request.

## macOS laptop (M-series) as portable edge

* **Small LLM:** **llama.cpp Metal Q4_K_M**, 256 / 8–12.
* **8B LLM:** llama.cpp Metal Q4_K_M (ok), or Core ML FP16 (short ctx).
* **Vision/STT/TTS:** **Core ML FP16**, N=32–64 @ 224–320²; stream audio.
* Notes: keep unified mem headroom; ANE engages via Core ML.

## Windows mini-PC (mixed GPUs)

* **Small LLM:** **DirectML** FP16 or OV-CPU; 256 / 8–12.
* **8B LLM:** TensorRT (if NVIDIA) else DirectML FP16 (latency ↑).
* **Vision/STT/TTS:** DirectML FP16 (or OV-CPU INT8); N=8–16 @ 512².
* Notes: warm once; avoid overlay hooks.

---

# Edge degradation ladders (automatic fallbacks)

**LLM:** 8B Q4 → small Q4 → small Q3 → remote.
**Vision:** 640² INT8 → 512² INT8 → 320² INT8 → CPU INT8.
**STT:** base-en INT8 → tiny INT8 → tiny INT8 enc/FP16 dec → server.
**TTS:** FP16 vocoder → INT8 vocoder (QAT) → parametric (LPCNet) → server.

---

# Router tweaks for edge (20–40 ms total)

* **Stage 1 (heuristics, 5–8 ms):** add device telemetry (temp, VRAM free, battery). Early exits favor *small* local models.
* **Stage 2 (intent, 10–15 ms):** INT8 tiny classifier; include “degrade_ok” label (summaries, short answers).
* **Stage 3 (policy, 5–10 ms):** cost/thermal-aware rules:

  * If **temp↑** or **battery <20%** → force small models / CPU.
  * If **latency SLO tight** and GPU idle → promote to GPU, else stay CPU.
  * If **RAM low** → downshift context/resolution before switching models.

**Zero-copy rules:** pre-allocate device/input/output/KV; reuse handles across requests; prefer DLPack/remote tensors when bridging runtimes.

---

# Edge hardening checklist

* ✅ **INT8 default**, **INT4** where QAT/mixed precision exists.
* ✅ **Pinned** big cores, threads ≤ 3, *no* oversubscription.
* ✅ **Warm-start** models; **cache** TRT/OV engines.
* ✅ **Steady small batches**; stream audio/video; avoid spikes.
* ✅ **Watchdog** to downshift on OOM/thermal/timeouts.
* ✅ **Offline** fallback prompts, token caches, and short-answer modes.

---

