Awesome—here’s a **hardware-by-hardware playbook** for speeding up **GGUF** inference *beyond* just quant selection. I assume llama.cpp/ggml family runtimes (Metal/CUDA/ROCm/Vulkan/WASM), but the ideas carry over to others.

---

# Universal quick wins (apply everywhere)

* **Right quant for task**: prefer **Q4_K_M/Q5_K_M** for quality, **Q3_K** for edge RAM, **K-blocks** over legacy Q4_0/Q5_0.
* **Prompt vs decode batching**: use **large prompt batch** (e.g., 256–1024) for ingestion, **small decode batch** (8–32).
* **Context & KV cache**: keep context just big enough; enable **KV cache compression/quant** if your build supports it; keep **KV in faster memory** when possible.
* **mmap + prefetch**: memory-map the model; warm it once to reduce page faults.
* **Threading**: threads ≈ physical cores; avoid hyper-thread oversubscription; pin threads to big cores (mobile).
* **Rope scaling**: only if you need long context; it costs some perf.
* **Speculative/prompt caching**: reuse shared prefixes; huge win for chat/apps.

---

## Apple Silicon (MacBook/Air/Pro, M-series)

**Backend**: Metal
**Quant**: Q4_K_M (default), Q5_K_M for quality; keep embeddings/projections in FP16 if you can (mixed precision).
**Tuning**

* Offload **as many layers as VRAM allows** (Metal “gpu-layers”).
* Set **threads = performance cores** (often 4–8) and leave efficiency cores mostly idle to prevent contention.
* Prompt batch 512–1024; decode batch 16–32.
* Keep **KV cache on GPU** when possible; fall back to CPU RAM for very long context.
  **Tips**: M2/M3 benefit from larger batch; watch unified-memory pressure (Activity Monitor).

---

## NVIDIA Desktop/Datacenter (RTX/GeForce/Jetson Orin)

**Backend**: CUDA (llama.cpp cuBLAS)
**Quant**: Q4_K_M/Q5_K_M; for LLMs ≥13B, consider **mixed KV (FP16) + Q4_K_M weights**.
**Tuning**

* **gpu-layers**: push until VRAM ~85–90% used at target context.
* Use **tensor split** for multi-GPU (even split unless one GPU is weaker).
* Prompt batch 512–2048 (desktop VRAM allows very large prompt batch); decode batch 16–64.
* Turn on **paged KV cache** only if you must extend context beyond VRAM.
  **Tips**: Keep clocks high (Prefer Max Perf), disable power throttling; on Linux, pin to a high-perf persistence mode.

---

## AMD GPU (Linux/Windows)

**Backend**: ROCm (Linux) if available; otherwise **Vulkan** backend for portability.
**Quant**: Q4_K_M/Q5_K_M; mixed precision for attention/out-proj if supported.
**Tuning**

* ROCm: similar to CUDA—maximize **gpu-layers** under VRAM headroom.
* Vulkan: prefer **moderate batch** (prompt 256–768); keep **decode batch** 8–24.
* Watch **PCIe throughput**; keep KV on GPU where possible.
  **Tips**: ROCm is best on RX 7000/MI series + modern kernels; on Windows, Vulkan tends to be more stable than OpenCL.

---

## Intel CPU (AVX2/AVX-512/VNNI)

**Backend**: CPU (ggml)
**Quant**: Q4_K_M for general use, **Q5_K_M** if you need accuracy, **Q3_K** for small RAM.
**Tuning**

* Build with **-march=native**; enable AVX-512/VNNI if your CPU has it.
* **Threads = physical cores**; try **NUMA pinning** on multi-socket (one process per socket, bind memory and threads).
* Prompt batch 256–512; decode batch 8–16.
* **mmap + mlock** if you have RAM headroom to keep pages hot.
  **Tips**: Large L3 (Xeon W/Server) loves bigger prompt batches; laptop parts run into power limits—cap threads slightly below core count.

---

## AMD CPU (Zen/Threadripper)

**Backend**: CPU (ggml)
**Quant**: Q4_K_M/Q5_K_M
**Tuning**

* Build with **-march=znver3** (or native Zen gen).
* **Threads = CCD big cores**; try **one NUMA domain per process** on Threadripper Pro.
* Prompt batch 256–768; decode batch 8–24.
  **Tips**: Zen 4 has great integer throughput; keep memory in **interleaved** mode for bandwidth.

---

## ARM CPU (Generic servers, Graviton, Ampere)

**Backend**: CPU (NEON/SVE)
**Quant**: Q3_K for small RAM, Q4_K_M when comfortable.
**Tuning**

* Compile with **NEON/SVE** enabled; set **threads = physical cores**.
* Prompt batch 128–512; decode batch 8–16.
  **Tips**: On Graviton, scale horizontally (multiple workers) rather than chasing single-instance max tokens/s.

---

## Jetson Nano (Maxwell, 4 GB) — *edge constrained*

**Backend**: CUDA, but VRAM & bandwidth limited
**Quant**: **Q3_K** or **Q4_K_M** small models (≤3B–7B distilled)
**Tuning**

* Offload only a **handful of layers**; most work will be CPU-side.
* Context 1–2k; prompt batch 64–128; decode batch 4–8.
* Keep **swap/zram** configured; avoid overcommitting VRAM.
  **Tips**: Prefer tiny models (Phi-2/3-mini, Distilled 3–7B) converted to GGUF; use **prompt caching** aggressively.

---

## Jetson Orin / Xavier (much better)

**Backend**: CUDA
**Quant**: Q4_K_M (or Q5_K_M if VRAM allows)
**Tuning**

* **gpu-layers** high (Orin can host many layers); KV on GPU; prompt batch 256–768; decode 8–24.
  **Tips**: Lock clocks (`nvpmodel`/`jetson_clocks`) and cool the device.

---

## Raspberry Pi 4/5

**Backend**: CPU (NEON)
**Quant**: **Q3_K** (often necessary), **Q4_K_M** for 2–3B models only.
**Tuning**

* 64-bit OS; compile with **NEON + -O3 -mcpu=native**.
* Threads = cores (4); prompt batch 64–128; decode batch 4–8.
* Context 512–1k; keep model on **fast SD/SSD**; enable **zram**.
  **Tips**: Use **speculative decoding** with a small draft model to recover tokens/s.

---

## Android (phones/tablets)

**Backend**: **Vulkan** (portable) or CPU NEON if GPU drivers are flaky
**Quant**: Q4_K_M for high-end; Q3_K for mid/low devices
**Tuning**

* Pin threads to **big cores**; limit to 2–4 threads to avoid thermal throttling.
* Prompt batch 128–384; decode 4–16.
* Short contexts (512–2k); prefer **short prompts + RAG** over long histories.
  **Tips**: Keep the app screen awake minimally; background throttling can cut perf in half.

---

## iOS / iPadOS

**Backend**: **Metal**
**Quant**: Q4_K_M; use **mixed precision** for attention if RAM allows.
**Tuning**

* Similar to Macs but with tighter RAM/thermal limits: prompt batch 128–384, decode 4–16; context 1–2k.
* Keep **KV on-device GPU** when possible; fall back to CPU for long contexts.
  **Tips**: Expect **thermal throttling** after a few minutes—adapt batch dynamically.

---

## Windows (any GPU)

**Backend**: Prefer **CUDA** (NVIDIA) / **ROCm** (Linux) / **Vulkan** (AMD or fallback)
**Quant**: Q4_K_M/Q5_K_M
**Tuning**

* If using vendor-agnostic paths (Vulkan/DirectML via other runtimes), stick to **moderate batch** and test op coverage before scaling.
* Watch **driver scheduling**; disable overlays that hook the GPU.

---

# Example tuning flow (copy/paste mindset)

1. **Pick model** sized for RAM/VRAM (e.g., 7B Q4_K_M for laptops, 13B Q4_K_M for 24 GB GPUs).
2. **Build** with the right backend (Metal/CUDA/ROCm/Vulkan) and `-march=native`.
3. **Find headroom**: start small context (e.g., 2k), increase **gpu-layers** until just below OOM.
4. **Batch sweep**:

   * Prompt batch: 256 → 1024; pick the peak tokens/s without OOM.
   * Decode batch: 8 → 32; pick best latency/throughput tradeoff.
5. **KV placement**: keep on GPU if possible; otherwise try **compressed/quantized KV**.
6. **Threading**: physical cores only; pin or set CPU affinity if needed.
7. **Cache**: enable prompt/speculative caching for repeated prefixes.

Extra for Android (Later)

Got it—here’s a **more granular Android tuning guide for GGUF** by chipset family. All assume **llama.cpp (Vulkan backend)**; fall back to **CPU/NEON** if drivers are flaky.

---

# Qualcomm Snapdragon (Adreno GPU)

**Targets:** 8 Gen 3/4 (flagship), 8 Gen 1/2, 7/6 series

**High-end (8 Gen 2/3/4, 12–16GB RAM)**

* **Quant:** Q4_K_M (7–13B), Q5_K_M if RAM allows
* **Threads:** 2–3 pinned to **big cores** (gold)
* **Batches:** prompt 256–512 (up to 768 if stable); decode 8–24
* **Context:** 1–2k (push 4k only if latency is okay)
* **Tips:** Enable “**Sustained performance mode**” (Android API) to reduce thermal throttling; keep screen brightness low

**Mid-range (7 Gen 1/3, 8–12GB)**

* **Quant:** Q4_K_M (≤7B), Q3_K for 13B
* **Threads:** 2 big cores
* **Batches:** prompt 192–384; decode 6–16
* **Context:** 1k–2k
* **Tips:** Prefer **short prompts + RAG**; watch unified memory pressure

**Lower/older (6 series / 855–888 with 6GB)**

* **Quant:** Q3_K (3–7B distilled)
* **Threads:** 2
* **Batches:** prompt 128–256; decode 4–8
* **Context:** 512–1k
* **Tips:** Consider **CPU-only** for stability if Adreno Vulkan drivers stutter

---

# MediaTek Dimensity (Mali/Immortalis GPU)

**Targets:** D9300/9200/9000 (flagship), 8200/8100 (mid), older G series

**High-end (D9300/9200/9000, 12–16GB)**

* **Quant:** Q4_K_M (7–13B)
* **Threads:** 2–3 big cores
* **Batches:** prompt 256–512; decode 8–20
* **Context:** 1–2k
* **Tips:** **Immortalis** drivers are decent; avoid huge prompt spikes (keep ≤512) to prevent thermal dips

**Mid-range (8200/8100, 8–12GB)**

* **Quant:** Q4_K_M (≤7B), else Q3_K
* **Threads:** 2
* **Batches:** prompt 192–384; decode 6–12
* **Context:** 1k–2k
* **Tips:** If you see Vulkan timeouts, drop prompt batch by 25–30%

**Older/entry**

* **Quant:** Q3_K (3–4B)
* **Threads:** 2
* **Batches:** prompt 96–192; decode 4–8
* **Context:** 512–1k

---

# Samsung Exynos (Mali or Xclipse/AMD RDNA)

**High-end (Exynos 2400, 12–16GB)**

* **Quant:** Q4_K_M (7–13B)
* **Threads:** 2–3 big cores
* **Batches:** prompt 256–448; decode 8–20
* **Context:** 1–2k
* **Tips:** Driver quality varies—test **smaller decode batches** first

**Mid/older Exynos**

* **Quant:** Q4_K_M (≤7B) or Q3_K
* **Threads:** 2
* **Batches:** prompt 160–320; decode 6–12
* **Context:** 1k–2k

---

# Google Tensor (Mali/Immortalis GPU)

**Targets:** Tensor G2/G3/G4 Pixels (8–16GB)

**Flagship Pixels (G3/G4, 12–16GB)**

* **Quant:** Q4_K_M (7–13B)
* **Threads:** 2–3 big cores
* **Batches:** prompt 224–448; decode 8–16
* **Context:** 1–2k
* **Tips:** Pixels throttle quickly—consider a **decode batch cap at 12** for sustained sessions

**Earlier Pixels (G2, 8–12GB)**

* **Quant:** Q4_K_M (≤7B), else Q3_K
* **Threads:** 2
* **Batches:** prompt 160–320; decode 6–12
* **Context:** 1k–2k

---

# Generic guidance (applies to all Android)

* **Thermals:** Use Android’s **SustainedPerformanceMode**, keep device cool, disable **Battery Saver**, and keep the app **foreground** (background throttling hurts by ~50%).
* **Threads:** 2–4 **max**; pin to **big cores**; avoid all-core blasts that trigger throttling.
* **Prompt vs decode:** Front-load work—**larger prompt batch, smaller decode batch**.
* **Context:** Prefer **512–2k**; for longer chats, compress history or switch to **RAG**.
* **Stability fallbacks:** If Vulkan is unstable, try **CPU/NEON**; reduce prompt batch in 20–25% steps before lowering decode batch.
* **Builds:** NDK build with `-O3 -ffast-math -march=armv8.2-a+fp16+dotprod` (if supported); enable **NEON**; link only needed ops to keep APK size down.

---

## Quick presets (copy/paste)

* **Snapdragon 8 Gen 3 (12–16GB)**: Q4_K_M (7–13B); threads **3**; prompt **384–512**; decode **12–20**; ctx **2k**.
* **Dimensity 9200 (12–16GB)**: Q4_K_M (7–13B); threads **3**; prompt **320–448**; decode **10–16**; ctx **2k**.
* **Tensor G3 (12GB)**: Q4_K_M (≤13B); threads **2–3**; prompt **288–416**; decode **8–14**; ctx **1–2k**.
* **Mid-tier (Snapdragon 7 Gen 1, 8–12GB)**: Q4_K_M (≤7B) / Q3_K; threads **2**; prompt **224–352**; decode **8–12**; ctx **1–2k**.
* **Entry (6GB RAM anything)**: Q3_K (3–4B); threads **2**; prompt **128–224**; decode **4–8**; ctx **512–1k**.

Want me to tailor exact flags and defaults for a specific phone (model + RAM)?


