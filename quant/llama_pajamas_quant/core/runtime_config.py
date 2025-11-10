#!/usr/bin/env python3
"""Generate optimal runtime configuration for detected hardware.

Combines hardware detection with hardware profiles database to generate
optimal runtime configurations for llama-pajamas runtimes.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import hardware detector
sys.path.insert(0, str(Path(__file__).parent))
from .hardware import HardwareDetector, HardwareProfile


@dataclass
class RuntimeConfig:
    """Runtime configuration for llama-pajamas."""

    # Model info
    model_size: str  # "7-8B", "13B", "30B+"
    precision: str  # "Q4_K_M", "Q3_K_M", etc.

    # Backend
    backend: str  # "mlx", "cuda", "metal", "rocm", "cpu"

    # Core settings
    n_gpu_layers: int
    n_threads: int
    n_batch: int
    n_ubatch: int
    n_ctx: int

    # Metadata
    hardware_profile: str
    expected_tokens_per_sec: int
    notes: Optional[str] = None


class RuntimeConfigGenerator:
    """Generate runtime configs from hardware profiles."""

    def __init__(self, profiles_path: Optional[Path] = None):
        """Initialize with hardware profiles database."""
        if profiles_path is None:
            # Default to config/hardware_profiles.json
            profiles_path = Path(__file__).parent.parent / "config" / "hardware_profiles.json"

        with open(profiles_path) as f:
            self.profiles_db = json.load(f)

    def generate_config(
        self,
        hardware: HardwareProfile,
        model_size: str = "7-8B",
        precision: Optional[str] = None,
        use_case: str = "general"
    ) -> RuntimeConfig:
        """Generate optimal runtime config for hardware.

        Args:
            hardware: Detected hardware profile
            model_size: Model size category ("7-8B", "13B", "30B+")
            precision: Quantization precision (auto-selected if None)
            use_case: Use case ("general", "long_context", "speed", "quality")

        Returns:
            RuntimeConfig with optimal settings
        """
        # Load profile for this hardware
        profile = self.profiles_db.get(hardware.platform_id)

        if profile is None:
            # Fallback to generic profile
            profile = self.profiles_db.get("generic")
            platform_id = "generic"
            print(f"Warning: No profile for {hardware.platform_id}, using generic fallback", file=sys.stderr)
        else:
            platform_id = hardware.platform_id

        # Get model-specific settings
        model_profile = profile["recommended_models"].get(model_size)

        if model_profile is None:
            # Try to find closest match
            available_sizes = list(profile["recommended_models"].keys())
            if available_sizes:
                model_profile = profile["recommended_models"][available_sizes[0]]
                print(f"Warning: No profile for {model_size}, using {available_sizes[0]}", file=sys.stderr)
            else:
                raise ValueError(f"No model profiles available for {platform_id}")

        # Select precision
        if precision is None:
            # Use first recommended precision
            precision = model_profile["precision"][0]

        # Base config from profile
        config = RuntimeConfig(
            model_size=model_size,
            precision=precision,
            backend=profile["backend"],
            n_gpu_layers=model_profile["n_gpu_layers"],
            n_threads=model_profile["n_threads"],
            n_batch=model_profile["n_batch"],
            n_ubatch=model_profile["n_ubatch"],
            n_ctx=model_profile["n_ctx"],
            hardware_profile=platform_id,
            expected_tokens_per_sec=model_profile["expected_tokens_per_sec"],
            notes=model_profile.get("notes")
        )

        # Adjust for use case
        config = self._adjust_for_use_case(config, use_case, hardware)

        return config

    def _adjust_for_use_case(
        self,
        config: RuntimeConfig,
        use_case: str,
        hardware: HardwareProfile
    ) -> RuntimeConfig:
        """Adjust config for specific use case."""
        if use_case == "long_context":
            # Double context, reduce batch size
            config.n_ctx = min(config.n_ctx * 2, 16384)
            config.n_batch = max(config.n_batch // 2, 256)
            config.n_ubatch = max(config.n_ubatch // 2, 8)

        elif use_case == "speed":
            # Aggressive batching, lower precision
            config.n_batch = min(config.n_batch * 2, 2048)
            config.n_ubatch = min(config.n_ubatch * 2, 64)
            if config.precision == "Q4_K_M":
                config.precision = "Q3_K_M"

        elif use_case == "quality":
            # Higher precision, conservative settings
            if config.precision == "Q4_K_M":
                config.precision = "Q5_K_M"
            elif config.precision == "Q3_K_M":
                config.precision = "Q4_K_M"

        # Validate against hardware limits
        config = self._validate_config(config, hardware)

        return config

    def _validate_config(
        self,
        config: RuntimeConfig,
        hardware: HardwareProfile
    ) -> RuntimeConfig:
        """Validate config doesn't exceed hardware limits."""
        # Estimate VRAM usage (rough approximation)
        model_size_gb = {
            "7-8B": {"Q3_K_M": 3.8, "Q4_K_M": 4.7, "Q5_K_M": 5.5},
            "13B": {"Q3_K_M": 6.5, "Q4_K_M": 8.0, "Q5_K_M": 9.5},
            "30B+": {"Q3_K_M": 14, "Q4_K_M": 17, "Q5_K_M": 20}
        }

        model_gb = model_size_gb.get(config.model_size, {}).get(config.precision, 5.0)

        # KV cache estimate: ~0.5-1GB per 1k context for 8B model
        kv_cache_gb = (config.n_ctx / 1024) * 0.75 * (
            1.0 if "7-8B" in config.model_size else 1.5
        )

        total_vram_needed = model_gb + kv_cache_gb + 1.0  # +1GB overhead

        # Check if exceeds available VRAM
        if total_vram_needed > hardware.vram_gb * 0.9:
            print(f"Warning: Estimated VRAM usage ({total_vram_needed:.1f}GB) near limit ({hardware.vram_gb:.1f}GB)", file=sys.stderr)

            # Downgrade context if needed
            if config.n_ctx > 2048:
                config.n_ctx = max(config.n_ctx // 2, 2048)
                print(f"  Reduced context to {config.n_ctx}", file=sys.stderr)

        # Thread count check
        if config.n_threads > hardware.cpu_cores_performance:
            config.n_threads = hardware.cpu_cores_performance
            print(f"Warning: Reduced threads to {config.n_threads} (physical cores)", file=sys.stderr)

        return config

    def export_json(self, config: RuntimeConfig, output_path: Path):
        """Export config as JSON for runtime."""
        config_dict = {
            "model": {
                "size": config.model_size,
                "precision": config.precision
            },
            "backend": config.backend,
            "settings": {
                "n_gpu_layers": config.n_gpu_layers,
                "n_threads": config.n_threads,
                "n_batch": config.n_batch,
                "n_ubatch": config.n_ubatch,
                "n_ctx": config.n_ctx
            },
            "metadata": {
                "hardware_profile": config.hardware_profile,
                "expected_tokens_per_sec": config.expected_tokens_per_sec
            }
        }

        if config.notes:
            config_dict["metadata"]["notes"] = config.notes

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate optimal runtime configuration"
    )
    parser.add_argument(
        "--hardware",
        choices=["auto", "custom"],
        default="auto",
        help="Hardware detection mode"
    )
    parser.add_argument(
        "--model-size",
        default="7-8B",
        choices=["7-8B", "13B", "30B+"],
        help="Model size category"
    )
    parser.add_argument(
        "--precision",
        choices=["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
        help="Quantization precision (auto-selected if not specified)"
    )
    parser.add_argument(
        "--use-case",
        choices=["general", "long_context", "speed", "quality"],
        default="general",
        help="Use case optimization"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="json",
        help="Output format"
    )

    args = parser.parse_args()

    # Detect hardware
    if args.hardware == "auto":
        detector = HardwareDetector()
        hardware = detector.detect()
        print(f"Detected: {hardware.display_name}", file=sys.stderr)
    else:
        raise NotImplementedError("Custom hardware input not yet implemented")

    # Generate config
    generator = RuntimeConfigGenerator()
    config = generator.generate_config(
        hardware=hardware,
        model_size=args.model_size,
        precision=args.precision,
        use_case=args.use_case
    )

    # Output
    if args.format == "json":
        config_dict = {
            "model": {
                "size": config.model_size,
                "precision": config.precision
            },
            "backend": config.backend,
            "settings": {
                "n_gpu_layers": config.n_gpu_layers,
                "n_threads": config.n_threads,
                "n_batch": config.n_batch,
                "n_ubatch": config.n_ubatch,
                "n_ctx": config.n_ctx
            },
            "metadata": {
                "hardware_profile": config.hardware_profile,
                "expected_tokens_per_sec": config.expected_tokens_per_sec
            }
        }

        if config.notes:
            config_dict["metadata"]["notes"] = config.notes

        output = json.dumps(config_dict, indent=2)

        if args.output:
            generator.export_json(config, args.output)
            print(f"Config saved to {args.output}", file=sys.stderr)
        else:
            print(output)

    else:  # summary
        print(f"\nRuntime Configuration")
        print(f"=" * 50)
        print(f"Model: {config.model_size} @ {config.precision}")
        print(f"Backend: {config.backend}")
        print(f"Hardware: {config.hardware_profile}")
        print(f"\nSettings:")
        print(f"  GPU layers: {config.n_gpu_layers}")
        print(f"  Threads: {config.n_threads}")
        print(f"  Prompt batch: {config.n_batch}")
        print(f"  Decode batch: {config.n_ubatch}")
        print(f"  Context: {config.n_ctx}")
        print(f"\nExpected Performance:")
        print(f"  ~{config.expected_tokens_per_sec} tokens/sec")
        if config.notes:
            print(f"\nNotes: {config.notes}")


if __name__ == "__main__":
    main()
