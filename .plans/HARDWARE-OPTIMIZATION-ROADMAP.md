# Hardware Optimization Roadmap
**Focus**: Native, vendor-tuned runtimes for maximum performance

**Philosophy**: Skip universal formats (ONNX), go straight to hardware-optimized solutions

---

## Current State (v0.1.0 - MVP Complete)

| Format | Hardware Target | Status | Use Cases |
|--------|----------------|--------|-----------|
| **MLX** | Apple Silicon (M1/M2/M3) | ✅ Production | LLMs on Mac (4-bit, fast, unified memory) |
| **GGUF** | Universal (Metal/CUDA/ROCm/CPU) | ✅ Production | LLMs everywhere (Q4_K_M, llama.cpp) |
| **ONNX** | Cross-platform | ⚠️ PAUSED | Too universal, not hardware-optimized |

---

## Next Platforms (v0.2.0+)

### Decision Framework
- **Prioritize 10x improvements**, not 10% improvements
- **Focus on platforms where GGUF+MLX have gaps**
- **Multi-modal is the key differentiator** (vision, speech, LLMs together)

### Platform Analysis

#### 1. **Apple Silicon** - Multi-Modal (CoreML)
**Status**: ⏳ HIGH PRIORITY

**Current Gap**:
- ✅ MLX handles LLMs well
- ❌ Vision models (YOLO, ViT, CLIP) not optimized
- ❌ Speech models (Whisper, FastSpeech) not optimized
- ❌ No unified multi-modal pipeline

**Proposal: Add CoreML for Vision/Speech**
- **Target**: Vision (detection/classification/segmentation), STT (Whisper), TTS (FastSpeech2)
- **Precision**: FP16 (INT8 where calibrated)
- **Advantages**:
  - Apple Neural Engine (ANE) acceleration on iOS/Mac
  - Native integration with Metal
  - Zero-copy with MLX (unified memory)
  - iOS deployment path
- **Pipeline**:
  ```
  PyTorch/HF → CoreML (coremltools) → .mlpackage → ANE/GPU/CPU
  ```
- **Use Cases**:
  - Vision: YOLO detection, CLIP embeddings, ViT classification
  - STT: Whisper (streaming audio → text)
  - TTS: FastSpeech2 + HiFi-GAN (text → audio)
  - Multi-modal: CLIP + LLaVA/Qwen-VL vision-language models

**Implementation Priority**: **HIGH** (fills clear gap, huge Mac/iOS market)

---

#### 2. **NVIDIA** - Server/Desktop/Jetson (TensorRT + TensorRT-LLM)
**Status**: ⏳ MEDIUM PRIORITY

**Current Gap**:
- ✅ GGUF+CUDA works for LLMs
- ❓ Is TensorRT-LLM significantly better than GGUF+CUDA?
- ❌ Vision/STT/TTS not optimized (need TensorRT)

**Proposal: Add TensorRT for Vision/Speech, Evaluate TensorRT-LLM**
- **TensorRT (Vision/Speech)**: Clear win over GGUF
  - FP16/INT8/FP8 kernels
  - Engine caching (AOT compilation)
  - CUDA graphs for low latency
  - Targets: YOLO, Whisper, FastSpeech2

- **TensorRT-LLM (LLMs)**: Needs evaluation vs GGUF+CUDA
  - **Advantages over GGUF**:
    - Paged KV-cache (better memory efficiency)
    - In-flight batching (higher throughput)
    - FP8 support (H100/Ada)
    - Tensor parallelism (multi-GPU)
    - CUDA graphs (lower latency)
  - **Disadvantages**:
    - More complex setup than GGUF
    - Less mature quantization (vs GGUF's Q4_K_M)
    - Smaller community
  - **Decision**: Test on target hardware (RTX 4070, Jetson Orin, A100)
    - If TensorRT-LLM is <30% faster than GGUF+CUDA → **stick with GGUF**
    - If TensorRT-LLM is >50% faster or enables key features → **add TensorRT-LLM**

**Implementation Priority**:
- **TensorRT (Vision/Speech)**: **HIGH** (clear gap)
- **TensorRT-LLM**: **MEDIUM** (need to prove value over GGUF+CUDA)

---

#### 3. **Mobile/Edge** - ExecuTorch (iOS/Android/Jetson)
**Status**: ⏳ LOWER PRIORITY (unless clear user demand)

**Current Gap**:
- iOS: Can use CoreML (covered above)
- Android: No native optimization (ONNX was supposed to help via NNAPI)
- Jetson: Can use TensorRT (covered above)

**Proposal: ExecuTorch for true mobile optimization**
- **Target**: iOS (CoreML/MPS), Android (NNAPI/Hexagon), Jetson (CUDA)
- **Advantages**:
  - Native mobile runtimes (CoreML, NNAPI, QNN)
  - Low overhead (smaller than full PyTorch)
  - Quantization-aware export
- **Use Cases**:
  - On-device LLMs (privacy-preserving)
  - Low-latency vision/speech (no server round-trip)
  - Offline inference

**Implementation Priority**: **LOW** (CoreML covers iOS, Android market unclear)

---

#### 4. **AMD** - MIGraphX (Server/Desktop)
**Status**: ⏳ LOWER PRIORITY

**Current Gap**:
- ✅ GGUF+ROCm works for LLMs
- ❓ Is MIGraphX significantly better?

**Proposal: Stick with GGUF+ROCm unless AMD users demand native**
- MIGraphX has smaller ecosystem than TensorRT
- GGUF+ROCm is "good enough" for most AMD use cases
- Only add if we see significant AMD deployment

**Implementation Priority**: **LOW** (GGUF+ROCm covers 80%+ of AMD cases)

---

## Multi-Modal Strategy

### The Multi-Modal Gap
Current llama-pajamas only handles **LLMs**. Real-world applications need:
- **Vision**: Object detection, image classification, embeddings
- **Speech-to-Text**: Whisper streaming, multilingual
- **Text-to-Speech**: FastSpeech2, voice cloning
- **Vision-Language**: CLIP, LLaVA, Qwen-VL

### Platform-Specific Multi-Modal Paths

#### Apple Silicon (Mac/iOS)
```
┌─────────────────────────────────────────────┐
│ Multi-Modal Pipeline (Apple Silicon)        │
├─────────────────────────────────────────────┤
│                                             │
│  Vision (YOLO/ViT/CLIP)                     │
│    ↓ CoreML FP16 (ANE acceleration)         │
│    → embeddings/detections                  │
│                                             │
│  Speech-to-Text (Whisper)                   │
│    ↓ CoreML FP16 (streaming)                │
│    → transcription                          │
│                                             │
│  LLM (Qwen/LLaMA)                           │
│    ↓ MLX 4-bit (Metal)                      │
│    → text generation                        │
│                                             │
│  Text-to-Speech (FastSpeech2)               │
│    ↓ CoreML FP16                            │
│    → audio output                           │
│                                             │
│  All components share unified memory ✅     │
└─────────────────────────────────────────────┘
```

**Advantage**: Zero-copy between CoreML and MLX (unified memory)

#### NVIDIA (Desktop/Server/Jetson)
```
┌─────────────────────────────────────────────┐
│ Multi-Modal Pipeline (NVIDIA)                │
├─────────────────────────────────────────────┤
│                                             │
│  Vision (YOLO/Detr)                         │
│    ↓ TensorRT FP16/INT8 (CUDA)              │
│    → embeddings/detections                  │
│                                             │
│  Speech-to-Text (Whisper)                   │
│    ↓ TensorRT FP16 encoder, INT8 decoder    │
│    → transcription                          │
│                                             │
│  LLM (Qwen/LLaMA)                           │
│    ↓ Option A: GGUF+CUDA Q4_K_M             │
│    ↓ Option B: TensorRT-LLM FP16/INT8       │
│    → text generation                        │
│                                             │
│  Text-to-Speech (FastSpeech2)               │
│    ↓ TensorRT FP16/INT8 mixed               │
│    → audio output                           │
│                                             │
│  DLPack for zero-copy CUDA memory sharing ✅ │
└─────────────────────────────────────────────┘
```

**Advantage**: CUDA graphs + paged KV-cache + high throughput

---

## Router Concept (Future v0.3.0+)

### Purpose
Intelligently route requests to the optimal model + hardware based on:
- **Request type**: vision, speech, text, multi-modal
- **Hardware available**: Mac, NVIDIA, AMD, CPU
- **Resource constraints**: VRAM, thermal, latency SLO
- **Model size**: small (<1.7B), large (≈8B), vision, speech

### 3-Stage Router (20-40ms overhead)

#### Stage 1: Fast Heuristics (5-10ms)
```python
def stage1_early_exit(request, system_state):
    """CPU-only heuristics, zero GPU."""
    features = {
        'has_image': detect_image_data(request),
        'has_audio': detect_audio_data(request),
        'token_count': len(tokenize(request.text)),
        'has_code': detect_code_blocks(request.text),
        'language': fast_langid(request.text),
        'security_level': request.metadata.get('security_level'),
    }

    # Early exits (≈50% of traffic)
    if features['has_image']:
        return Route(pipeline='VISION', model='yolo-v8', device=select_gpu())

    if features['has_audio']:
        if request.text is None:
            return Route(pipeline='STT', model='whisper-base', device=select_gpu())
        else:
            return Route(pipeline='TTS', model='fastspeech2', device=select_gpu())

    if features['token_count'] < 100 and request.max_tokens < 200:
        return Route(pipeline='SMALL_LLM', model='qwen-1.7b', device=select_fast_device())

    if features['security_level'] == 'airgap':
        return Route(pipeline='LLM', model='qwen-8b', device='CPU')  # No GPU leak

    return None  # Continue to Stage 2
```

#### Stage 2: Intent Classification (10-20ms)
```python
def stage2_intent(request):
    """Tiny DistilBERT (20-30M params, INT8 CPU)."""
    embedding = tiny_encoder(request.text)  # 256-D
    intent = intent_classifier(embedding)   # ~12ms on 8-core x86

    # Intents: {chat, code, vision, stt, tts, tool-use,
    #           long-form, summarize, classify, multi-modal}

    return intent
```

#### Stage 3: Hardware Selection (5-10ms)
```python
def stage3_hardware_policy(intent, request, system_state):
    """Rule-based + learned policy (bandit/logistic)."""

    if intent in {'vision', 'stt', 'tts'}:
        # Prefer GPU for media processing
        return select_best_gpu_engine(
            available_runtimes=['TensorRT', 'CoreML', 'MIGraphX'],
            system_state=system_state
        )

    if intent in {'chat', 'code', 'summarize'}:
        if request.max_tokens > 1000 or 'high_quality' in request.metadata:
            # Large LLM needed
            return Route(
                pipeline='LLM_8B',
                model='qwen-8b-q4',
                device=select_llm_backend(system_state)  # TRT-LLM / MLX / GGUF+CUDA
            )
        else:
            # Small LLM sufficient
            return Route(
                pipeline='SMALL_LLM',
                model='qwen-1.7b-q4',
                device=select_fast_cpu_or_gpu(system_state)
            )

    if intent == 'multi-modal':
        # Complex routing: vision → embeddings → LLM
        return Route(
            pipeline='MULTI_MODAL',
            components=[
                ('vision', 'clip-vit', select_gpu()),
                ('llm', 'llava-1.5-7b', select_gpu())
            ]
        )
```

### Zero-Copy Memory Sharing (Critical)
```python
# Apple Silicon (CoreML + MLX)
coreml_output = vision_model(image)  # CoreML tensor (Metal buffer)
mlx_input = mx.array(coreml_output, copy=False)  # Zero-copy (unified memory)
llm_response = llm_generate(mlx_input)

# NVIDIA (TensorRT + TensorRT-LLM or GGUF)
import dlpack
trt_output = vision_model(image)  # TensorRT tensor (CUDA pointer)
cuda_ptr = to_dlpack(trt_output)
llm_input = from_dlpack(cuda_ptr)  # Zero-copy via DLPack
llm_response = llm_generate(llm_input)

# Keep KV-cache on device between requests
kv_cache = allocate_device_buffer(size=kv_max_size)  # Once
for request in stream:
    response = llm_generate(request, kv_cache=kv_cache)  # Reuse
```

---

## Recommended Roadmap

### Phase 1: Apple Multi-Modal (v0.2.0) - **8 weeks**
**Goal**: Complete Apple Silicon stack with vision + speech + LLM

#### Week 1-2: CoreML Vision
- [ ] Implement CoreML converter (PyTorch/HF → CoreML)
- [ ] Add YOLO-v8 detection pipeline
- [ ] Add CLIP embeddings
- [ ] Add ViT classification
- [ ] Test on Mac M1 64GB
- [ ] Optimize for ANE (FP16, NHWC layout)

#### Week 3-4: CoreML Speech
- [ ] Implement Whisper STT (streaming)
- [ ] Implement FastSpeech2 TTS
- [ ] Optimize audio preprocessing (mel on device)
- [ ] Test streaming performance

#### Week 5-6: Multi-Modal Integration
- [ ] Implement CLIP + MLX LLM pipeline
- [ ] Implement LLaVA/Qwen-VL (vision-language)
- [ ] Zero-copy buffers (CoreML → MLX)
- [ ] Unified CLI: `llama-pajamas-run multi-modal`

#### Week 7-8: Testing + Documentation
- [ ] End-to-end multi-modal examples
- [ ] Performance benchmarks (vision → LLM → TTS)
- [ ] iOS deployment guide
- [ ] Release v0.2.0

**Success Criteria**:
- ✅ Vision: YOLO-v8 @ 30+ FPS (640x640) on M3 Max
- ✅ STT: Whisper-base @ <100ms latency (10s chunk)
- ✅ TTS: FastSpeech2 @ real-time factor 0.3
- ✅ Multi-modal: CLIP → LLM pipeline @ <200ms total
- ✅ Zero-copy confirmed (no CPU transfers)

---

### Phase 2: NVIDIA Multi-Modal (v0.3.0) - **6 weeks**
**Goal**: TensorRT for vision/speech, evaluate TensorRT-LLM for LLMs

#### Week 1-2: TensorRT Vision
- [ ] Implement TensorRT converter
- [ ] Add YOLO-v8 detection
- [ ] Add CLIP embeddings
- [ ] Engine caching (SSD)
- [ ] FP16 + INT8 calibration

#### Week 3-4: TensorRT Speech
- [ ] Implement Whisper STT (TensorRT)
- [ ] Implement FastSpeech2 TTS
- [ ] Streaming audio pipelines
- [ ] CUDA graphs for low latency

#### Week 5: TensorRT-LLM Evaluation
- [ ] Test TensorRT-LLM vs GGUF+CUDA on RTX 4070
- [ ] Benchmark: throughput, latency, memory
- [ ] **Decision**: Add TRT-LLM or stick with GGUF?

#### Week 6: Integration + Release
- [ ] Multi-modal pipeline (TRT vision → GGUF/TRT-LLM)
- [ ] DLPack zero-copy
- [ ] Jetson Orin testing
- [ ] Release v0.3.0

**Success Criteria**:
- ✅ Vision: YOLO-v8 @ 60+ FPS (640x640) on RTX 4070
- ✅ STT: Whisper-base @ <50ms latency
- ✅ TensorRT-LLM decision made (add or skip)
- ✅ Multi-modal working end-to-end

---

### Phase 3: Router + Optimization (v0.4.0) - **4 weeks**
**Goal**: Intelligent routing between models and hardware

#### Week 1-2: Router Implementation
- [ ] Stage 1: Heuristics (5-10ms)
- [ ] Stage 2: Intent classifier (tiny DistilBERT, INT8 CPU)
- [ ] Stage 3: Hardware policy (rules + bandit)
- [ ] Telemetry: VRAM, temp, latency SLOs

#### Week 3: Hardware Profiles
- [ ] Auto-detect: Apple Silicon, NVIDIA, AMD, Intel
- [ ] Generate optimal configs per hardware
- [ ] Hardware-specific README sections

#### Week 4: Testing + Release
- [ ] Router latency benchmarks (<40ms)
- [ ] Multi-hardware testing
- [ ] Release v0.4.0

---

## Summary

### What We're Building (Priority Order)

1. **Apple Multi-Modal (v0.2.0)** - **HIGHEST PRIORITY**
   - CoreML for vision/speech
   - MLX for LLMs
   - Unified multi-modal pipelines
   - iOS deployment path
   - **Rationale**: Huge Mac/iOS market, clear gap in current stack

2. **NVIDIA Multi-Modal (v0.3.0)** - **HIGH PRIORITY**
   - TensorRT for vision/speech (clear win)
   - Evaluate TensorRT-LLM vs GGUF+CUDA for LLMs
   - Jetson support for edge
   - **Rationale**: NVIDIA is largest deployment target, need vision/speech optimization

3. **Router (v0.4.0)** - **MEDIUM PRIORITY**
   - Intelligent model + hardware selection
   - Fast (20-40ms overhead)
   - Thermal/VRAM aware
   - **Rationale**: Enables smart multi-model deployments

### What We're NOT Building (Yet)

- ❌ **ExecuTorch** - iOS covered by CoreML, Android market unclear
- ❌ **MIGraphX** - GGUF+ROCm is good enough for AMD
- ❌ **ONNX** - Too universal, doesn't provide hardware optimization
- ❌ **OpenVINO** - Intel CPU-only, niche market

### Decision Points

1. **TensorRT-LLM vs GGUF+CUDA**: Needs benchmarking on RTX 4070/A100/Jetson
   - If <30% faster → stick with GGUF
   - If >50% faster or enables key features → add TensorRT-LLM

2. **CoreML for LLMs**: Currently using MLX (faster), stick with MLX unless iOS App Store requires CoreML

3. **Mobile (Android)**: Wait for user demand before adding ExecuTorch/QNN

---

## Next Actions

1. **Immediate**: Start Phase 1 (Apple Multi-Modal)
   - Begin CoreML vision converter
   - YOLO-v8 as first target
   - Validate ANE acceleration on Mac

2. **Research**: TensorRT-LLM benchmarking
   - Set up test harness: RTX 4070
   - Compare GGUF+CUDA vs TensorRT-LLM
   - Decide by end of Phase 1

3. **Documentation**: Update MVP plan with multi-modal roadmap
