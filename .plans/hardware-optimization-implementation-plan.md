# Hardware-Optimized Runtime Configuration System

## Overview

Implement automatic hardware detection and generate optimized runtime configurations for different hardware platforms. Based on `.plans/runtime-optimizations.md`.

## Goals

1. **Auto-detect hardware** - CPU, GPU, RAM/VRAM, core topology
2. **Generate optimal configs** - Platform-specific settings for maximum performance
3. **Validate configurations** - Ensure configs don't exceed available resources
4. **Document per-platform** - Clear quickstart guides for each hardware type

## Architecture

### 1. Hardware Detection System (`scripts/detect_hardware.py`)

```python
class HardwareDetector:
    """Detect system hardware and capabilities."""

    def detect(self) -> HardwareProfile:
        """Returns detected hardware profile."""
        return HardwareProfile(
            platform="apple_silicon_m3_16gb",
            cpu_type="arm64",
            gpu_type="metal",
            ram_gb=16,
            vram_gb=16,  # unified memory
            cpu_cores_performance=8,
            cpu_cores_efficiency=4,
            capabilities=["metal", "neon", "fp16"]
        )
```

**Detection Methods:**
- **Platform**: `platform.system()`, `platform.machine()`
- **Apple Silicon**: Parse `sysctl hw.model`, detect M1/M2/M3, memory size
- **NVIDIA**: Run `nvidia-smi`, parse GPU model, VRAM, CUDA version
- **AMD GPU**: Run `rocm-smi` or `lspci | grep VGA`
- **CPU**: Parse `/proc/cpuinfo` (Linux), `sysctl` (Mac), `wmic` (Windows)
- **RAM**: `psutil.virtual_memory()`

### 2. Hardware Profiles Database (`config/hardware_profiles.json`)

```json
{
  "apple_silicon_m3_16gb": {
    "display_name": "Apple M3 (16GB)",
    "backend": "metal",
    "recommended_models": {
      "7-8B": {
        "precision": "Q4_K_M",
        "gpu_layers": -1,
        "prompt_batch": 512,
        "decode_batch": 16,
        "threads": 8,
        "context": 4096,
        "expected_tokens_per_sec": 70
      }
    }
  },
  "nvidia_rtx_4090": {
    "display_name": "NVIDIA RTX 4090 (24GB)",
    "backend": "cuda",
    "recommended_models": {
      "7-8B": {
        "precision": "Q4_K_M",
        "gpu_layers": -1,
        "prompt_batch": 2048,
        "decode_batch": 64,
        "threads": 8,
        "context": 8192,
        "expected_tokens_per_sec": 120
      },
      "13B": {
        "precision": "Q4_K_M",
        "gpu_layers": -1,
        "prompt_batch": 1024,
        "decode_batch": 32,
        "threads": 8,
        "context": 4096,
        "expected_tokens_per_sec": 60
      }
    }
  }
}
```

**Platforms to Support:**

1. **Apple Silicon**
   - M1: 8GB, 16GB
   - M2: 8GB, 16GB, 24GB
   - M3: 8GB, 16GB, 24GB, 32GB, 64GB (Max/Ultra)
   - M4: 16GB, 24GB, 32GB, 64GB+

2. **NVIDIA Desktop**
   - RTX 3060 (12GB)
   - RTX 3070/3080 (8-10GB)
   - RTX 3090/4080 (24GB)
   - RTX 4090 (24GB)
   - Tesla T4/V100/A100

3. **NVIDIA Edge**
   - Jetson Nano (4GB)
   - Jetson Xavier (8-32GB)
   - Jetson Orin (8-64GB)

4. **AMD GPU**
   - RX 7900 XT/XTX (20-24GB)
   - MI250/MI300 (datacenter)

5. **CPU-Only**
   - Intel: i5/i7/i9 (AVX2/AVX-512)
   - AMD: Ryzen/Threadripper (Zen 2/3/4)
   - ARM: Graviton, Ampere, generic

### 3. Configuration Generator (`scripts/generate_runtime_config.py`)

```python
def generate_config(
    hardware_profile: HardwareProfile,
    model_size: str,  # "7-8B", "13B", "30B+"
    precision: str = None,  # Auto-select if None
    use_case: str = "general"  # "general", "long_context", "speed", "quality"
) -> RuntimeConfig:
    """Generate optimized runtime config."""

    # Load hardware-specific template
    template = load_profile(hardware_profile.platform)

    # Adjust for use case
    if use_case == "long_context":
        template["context"] *= 2
        template["prompt_batch"] //= 2
    elif use_case == "speed":
        template["precision"] = "Q3_K_M"
        template["decode_batch"] *= 2
    elif use_case == "quality":
        template["precision"] = "Q5_K_M"

    # Validate against resources
    estimated_vram = estimate_vram_usage(model_size, template)
    if estimated_vram > hardware_profile.vram_gb:
        template = downgrade_config(template, hardware_profile)

    return RuntimeConfig(**template)
```

**Output Format:**

```json
{
  "backend": "metal",
  "model_path": "./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf",
  "settings": {
    "n_gpu_layers": -1,
    "n_threads": 8,
    "n_batch": 512,
    "n_ubatch": 16,
    "n_ctx": 4096,
    "rope_freq_base": 10000.0,
    "rope_freq_scale": 1.0
  },
  "metadata": {
    "hardware": "apple_silicon_m3_16gb",
    "model_size": "8B",
    "precision": "Q4_K_M",
    "expected_tokens_per_sec": 70,
    "estimated_vram_usage_gb": 5.2
  }
}
```

### 4. Validation System

```python
class ConfigValidator:
    """Validate runtime configs against hardware limits."""

    def validate(self, config: RuntimeConfig, hardware: HardwareProfile) -> ValidationResult:
        checks = []

        # VRAM check
        vram_needed = estimate_vram(config)
        if vram_needed > hardware.vram_gb * 0.9:
            checks.append(Warning("VRAM usage >90%, may OOM"))

        # Thread count check
        if config.threads > hardware.cpu_cores_performance:
            checks.append(Warning("Threads > physical cores"))

        # Thermal check (mobile/edge devices)
        if hardware.is_mobile and config.decode_batch > 16:
            checks.append(Warning("High decode batch may cause thermal throttling"))

        return ValidationResult(is_valid=True, warnings=checks)
```

### 5. CLI Tool

```bash
# Detect hardware and generate config
uv run python scripts/detect_hardware.py --output config.json

# Generate config for specific hardware + model
uv run python scripts/generate_runtime_config.py \
    --hardware apple_silicon_m3_16gb \
    --model-size 8B \
    --precision Q4_K_M \
    --use-case general \
    --output runtime_config.json

# Validate config
uv run python scripts/validate_config.py --config runtime_config.json
```

## Implementation Steps

### Phase 1: Detection & Profiles (2-3 days)

1. **Day 1: Hardware Detection**
   - [ ] Implement `HardwareDetector` class
   - [ ] Add platform-specific detection:
     - macOS: `sysctl`, parse M1/M2/M3/M4, memory
     - Linux: `nvidia-smi`, `rocm-smi`, `/proc/cpuinfo`
     - Windows: `wmic`, NVIDIA/AMD tools
   - [ ] Test on available hardware (Mac, Linux)

2. **Day 2: Hardware Profiles Database**
   - [ ] Create `config/hardware_profiles.json`
   - [ ] Add profiles for:
     - Apple Silicon (M1/M2/M3 variants)
     - NVIDIA Desktop (RTX 3060-4090)
     - NVIDIA Edge (Jetson Nano/Xavier/Orin)
     - AMD GPU (RX 7000, MI series)
     - CPU-only (Intel/AMD/ARM)
   - [ ] Document expected performance for each

3. **Day 3: Profile Matching**
   - [ ] Implement profile matching logic
   - [ ] Handle unknown hardware (fallback to conservative defaults)
   - [ ] Test detection → profile mapping

### Phase 2: Config Generation (2-3 days)

4. **Day 4: Configuration Generator**
   - [ ] Implement `RuntimeConfigGenerator` class
   - [ ] Add use-case templates:
     - General (balanced)
     - Long context (large context, smaller batches)
     - Speed (aggressive batching, lower precision)
     - Quality (higher precision, conservative settings)

5. **Day 5: VRAM Estimation**
   - [ ] Implement VRAM estimation formulas:
     - Model weights: `model_size_GB × precision_factor`
     - KV cache: `layers × context × hidden_dim × 2 × bytes_per_elem`
     - Overhead: 10-20% buffer
   - [ ] Add downgrade logic when VRAM insufficient

6. **Day 6: Validation System**
   - [ ] Implement `ConfigValidator`
   - [ ] Add checks: VRAM, thread count, thermal warnings
   - [ ] Generate actionable warnings/suggestions

### Phase 3: Integration & Documentation (1-2 days)

7. **Day 7: CLI Tools**
   - [ ] Create `scripts/detect_hardware.py`
   - [ ] Create `scripts/generate_runtime_config.py`
   - [ ] Create `scripts/validate_config.py`
   - [ ] Test end-to-end workflow

8. **Day 8: Documentation**
   - [ ] Add hardware-specific sections to README
   - [ ] Create quick-start per platform
   - [ ] Document expected performance
   - [ ] Add troubleshooting guide

### Phase 4: Runtime Integration (2-3 days)

9. **Day 9-10: Runtime Config Loading**
   - [ ] Update `llama-pajamas-run` to accept config JSON
   - [ ] Add `--auto-configure` flag (auto-detect + generate)
   - [ ] Test with different hardware profiles

10. **Day 11: Performance Validation**
    - [ ] Benchmark on different hardware
    - [ ] Validate expected tokens/sec matches reality
    - [ ] Tune profiles based on real measurements

## Deliverables

1. **Scripts:**
   - `scripts/detect_hardware.py` - Hardware detection
   - `scripts/generate_runtime_config.py` - Config generation
   - `scripts/validate_config.py` - Config validation

2. **Config Files:**
   - `config/hardware_profiles.json` - Hardware-specific presets

3. **Documentation:**
   - Hardware-specific README sections
   - Quick-start guides per platform
   - Performance expectations
   - Troubleshooting guide

4. **Runtime Integration:**
   - `llama-pajamas-run --auto-configure` support
   - Config JSON loading
   - Performance monitoring

## Expected Outcomes

1. **User Experience:**
   - One command to detect hardware and get optimal config
   - No manual tuning required
   - Clear performance expectations

2. **Performance:**
   - 10-30% improvement over default settings
   - Reduced OOM errors
   - Better thermal management on edge devices

3. **Coverage:**
   - Support 80%+ of common hardware
   - Graceful fallback for unknown hardware
   - Mobile support (Android/iOS) in future phase

## Example Usage

```bash
# Auto-configure and run
uv run python scripts/detect_hardware.py --output config.json
llama-pajamas-run \
    --model ./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf \
    --config config.json

# Or in one step
llama-pajamas-run \
    --model ./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf \
    --auto-configure

# Generate config for different use case
uv run python scripts/generate_runtime_config.py \
    --hardware auto \
    --model-size 8B \
    --use-case long_context \
    --output long_context_config.json
```

## Future Enhancements (Post-MVP)

1. **Mobile Support:**
   - Android chipset detection (Snapdragon/Dimensity/Exynos/Tensor)
   - iOS thermal management
   - Sustained performance mode

2. **Dynamic Tuning:**
   - Monitor performance metrics
   - Adjust batch sizes based on observed throughput
   - Thermal throttling detection and adaptation

3. **Multi-Model Optimization:**
   - Optimize for running multiple models simultaneously
   - Model switching with KV cache preservation

4. **Benchmark-Driven Tuning:**
   - Run micro-benchmarks on first run
   - Generate custom profile based on actual hardware performance
   - Store and reuse tuned profiles
