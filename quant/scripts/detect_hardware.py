#!/usr/bin/env python3
"""Hardware detection for optimal runtime configuration.

Detects system hardware (CPU, GPU, RAM/VRAM, cores) and recommends
optimal runtime configurations for llama-pajamas runtimes.
"""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class HardwareProfile:
    """Detected hardware profile."""

    platform_id: str  # e.g., "apple_silicon_m3_16gb"
    display_name: str  # e.g., "Apple M3 (16GB)"

    # System info
    os_type: str  # darwin, linux, windows
    cpu_arch: str  # arm64, x86_64

    # CPU
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_performance: int  # Big cores (or all cores if not hybrid)
    cpu_cores_efficiency: int  # Little cores (0 if not hybrid)

    # Memory
    ram_gb: float
    vram_gb: float  # GPU VRAM or unified memory

    # GPU
    gpu_type: Optional[str]  # metal, cuda, rocm, vulkan, None
    gpu_model: Optional[str]
    gpu_compute_capability: Optional[str]  # For CUDA

    # Capabilities
    capabilities: List[str]  # e.g., ["metal", "neon", "fp16"]

    # Recommended backend
    recommended_backend: str  # mlx, cuda, metal, rocm, cpu


class HardwareDetector:
    """Detect system hardware and capabilities."""

    def __init__(self):
        self.os_type = platform.system().lower()
        self.cpu_arch = platform.machine().lower()

    def detect(self) -> HardwareProfile:
        """Detect hardware and return profile."""
        if self.os_type == "darwin":
            return self._detect_apple_silicon()
        elif self.os_type == "linux":
            return self._detect_linux()
        elif self.os_type == "windows":
            return self._detect_windows()
        else:
            return self._detect_generic()

    def _detect_apple_silicon(self) -> HardwareProfile:
        """Detect Apple Silicon Mac hardware."""
        # Get chip model
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True
            )
            cpu_model = result.stdout.strip()
        except:
            cpu_model = "Unknown Apple Silicon"

        # Parse M1/M2/M3/M4 and variant
        chip_name = "unknown"
        if "M1" in cpu_model:
            chip_name = "m1"
        elif "M2" in cpu_model:
            chip_name = "m2"
        elif "M3" in cpu_model:
            chip_name = "m3"
        elif "M4" in cpu_model:
            chip_name = "m4"

        # Get memory size
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True
            )
            ram_bytes = int(result.stdout.strip())
            ram_gb = ram_bytes / (1024**3)
        except:
            ram_gb = 16.0  # Default guess

        # Get core counts
        try:
            perfs = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, check=True
            )
            effs = subprocess.run(
                ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
                capture_output=True, text=True, check=True
            )
            perf_cores = int(perfs.stdout.strip())
            eff_cores = int(effs.stdout.strip())
            total_cores = perf_cores + eff_cores
        except:
            # Fallback
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True, text=True, check=True
                )
                total_cores = int(result.stdout.strip())
                perf_cores = total_cores
                eff_cores = 0
            except:
                total_cores = 8
                perf_cores = 8
                eff_cores = 0

        # Apple Silicon uses unified memory
        vram_gb = ram_gb

        # Determine platform ID
        ram_rounded = int(round(ram_gb))
        platform_id = f"apple_silicon_{chip_name}_{ram_rounded}gb"
        display_name = f"Apple {chip_name.upper()} ({ram_rounded}GB)"

        return HardwareProfile(
            platform_id=platform_id,
            display_name=display_name,
            os_type="darwin",
            cpu_arch="arm64",
            cpu_model=cpu_model,
            cpu_cores_physical=total_cores,
            cpu_cores_performance=perf_cores,
            cpu_cores_efficiency=eff_cores,
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            gpu_type="metal",
            gpu_model=cpu_model,  # Integrated GPU
            gpu_compute_capability=None,
            capabilities=["metal", "neon", "fp16"],
            recommended_backend="mlx"
        )

    def _detect_linux(self) -> HardwareProfile:
        """Detect Linux hardware (NVIDIA/AMD GPU or CPU)."""
        # Get CPU info
        cpu_model = "Unknown CPU"
        cpu_cores = 8
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":")[1].strip()
                        break

            result = subprocess.run(
                ["nproc"], capture_output=True, text=True, check=True
            )
            cpu_cores = int(result.stdout.strip())
        except:
            pass

        # Get RAM
        ram_gb = 16.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024**2)
                        break
        except:
            pass

        # Try to detect NVIDIA GPU
        nvidia_gpu = self._detect_nvidia_gpu()
        if nvidia_gpu:
            return nvidia_gpu

        # Try to detect AMD GPU
        amd_gpu = self._detect_amd_gpu()
        if amd_gpu:
            return amd_gpu

        # Fallback: CPU-only
        ram_rounded = int(round(ram_gb))
        platform_id = f"linux_cpu_{ram_rounded}gb"
        display_name = f"Linux CPU ({ram_rounded}GB)"

        capabilities = ["avx2"]
        if "avx512" in cpu_model.lower():
            capabilities.append("avx512")

        return HardwareProfile(
            platform_id=platform_id,
            display_name=display_name,
            os_type="linux",
            cpu_arch=self.cpu_arch,
            cpu_model=cpu_model,
            cpu_cores_physical=cpu_cores,
            cpu_cores_performance=cpu_cores,
            cpu_cores_efficiency=0,
            ram_gb=ram_gb,
            vram_gb=0.0,
            gpu_type=None,
            gpu_model=None,
            gpu_compute_capability=None,
            capabilities=capabilities,
            recommended_backend="cpu"
        )

    def _detect_nvidia_gpu(self) -> Optional[HardwareProfile]:
        """Try to detect NVIDIA GPU using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )

            line = result.stdout.strip().split("\n")[0]
            gpu_name, vram_str = line.split(",")
            gpu_name = gpu_name.strip()
            vram_mb = float(vram_str.strip().split()[0])
            vram_gb = vram_mb / 1024

            # Get compute capability
            compute_cap = None
            try:
                cap_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True
                )
                compute_cap = cap_result.stdout.strip()
            except:
                pass

            # Get CPU info for platform ID
            cpu_cores = 8
            ram_gb = 16.0
            try:
                result = subprocess.run(["nproc"], capture_output=True, text=True, check=True)
                cpu_cores = int(result.stdout.strip())

                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            ram_kb = int(line.split()[1])
                            ram_gb = ram_kb / (1024**2)
                            break
            except:
                pass

            # Determine GPU tier
            gpu_tier = "unknown"
            if "4090" in gpu_name or "4080" in gpu_name:
                gpu_tier = "rtx_4090"
            elif "3090" in gpu_name or "3080" in gpu_name:
                gpu_tier = "rtx_3090"
            elif "3070" in gpu_name or "3060" in gpu_name:
                gpu_tier = "rtx_3060"
            elif "A100" in gpu_name:
                gpu_tier = "tesla_a100"
            elif "V100" in gpu_name:
                gpu_tier = "tesla_v100"

            vram_rounded = int(round(vram_gb))
            platform_id = f"nvidia_{gpu_tier}_{vram_rounded}gb"
            display_name = f"NVIDIA {gpu_name} ({vram_rounded}GB)"

            cpu_model = "Unknown CPU"
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":")[1].strip()
                            break
            except:
                pass

            return HardwareProfile(
                platform_id=platform_id,
                display_name=display_name,
                os_type="linux",
                cpu_arch=self.cpu_arch,
                cpu_model=cpu_model,
                cpu_cores_physical=cpu_cores,
                cpu_cores_performance=cpu_cores,
                cpu_cores_efficiency=0,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                gpu_type="cuda",
                gpu_model=gpu_name,
                gpu_compute_capability=compute_cap,
                capabilities=["cuda", "avx2"],
                recommended_backend="cuda"
            )

        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _detect_amd_gpu(self) -> Optional[HardwareProfile]:
        """Try to detect AMD GPU using rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, check=True
            )

            # Parse output (very basic)
            gpu_name = "AMD GPU"
            for line in result.stdout.split("\n"):
                if "GPU" in line and ":" in line:
                    gpu_name = line.split(":")[1].strip()
                    break

            # Try to get VRAM
            vram_gb = 16.0  # Default guess
            try:
                vram_result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True, check=True
                )
                # Parse VRAM (implementation depends on rocm-smi output format)
                # This is a placeholder
                pass
            except:
                pass

            cpu_cores = 8
            ram_gb = 16.0
            try:
                result = subprocess.run(["nproc"], capture_output=True, text=True, check=True)
                cpu_cores = int(result.stdout.strip())

                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            ram_kb = int(line.split()[1])
                            ram_gb = ram_kb / (1024**2)
                            break
            except:
                pass

            vram_rounded = int(round(vram_gb))
            platform_id = f"amd_gpu_{vram_rounded}gb"
            display_name = f"AMD {gpu_name} ({vram_rounded}GB)"

            cpu_model = "Unknown CPU"
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":")[1].strip()
                            break
            except:
                pass

            return HardwareProfile(
                platform_id=platform_id,
                display_name=display_name,
                os_type="linux",
                cpu_arch=self.cpu_arch,
                cpu_model=cpu_model,
                cpu_cores_physical=cpu_cores,
                cpu_cores_performance=cpu_cores,
                cpu_cores_efficiency=0,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                gpu_type="rocm",
                gpu_model=gpu_name,
                gpu_compute_capability=None,
                capabilities=["rocm", "avx2"],
                recommended_backend="rocm"
            )

        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _detect_windows(self) -> HardwareProfile:
        """Detect Windows hardware."""
        # Placeholder - implement Windows detection using wmic
        return self._detect_generic()

    def _detect_generic(self) -> HardwareProfile:
        """Generic fallback detection."""
        cpu_cores = 4
        ram_gb = 8.0

        try:
            import multiprocessing
            cpu_cores = multiprocessing.cpu_count()
        except:
            pass

        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except:
            pass

        return HardwareProfile(
            platform_id="generic",
            display_name="Generic System",
            os_type=self.os_type,
            cpu_arch=self.cpu_arch,
            cpu_model="Unknown CPU",
            cpu_cores_physical=cpu_cores,
            cpu_cores_performance=cpu_cores,
            cpu_cores_efficiency=0,
            ram_gb=ram_gb,
            vram_gb=0.0,
            gpu_type=None,
            gpu_model=None,
            gpu_compute_capability=None,
            capabilities=[],
            recommended_backend="cpu"
        )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect hardware and output configuration"
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

    detector = HardwareDetector()
    profile = detector.detect()

    if args.format == "json":
        output = json.dumps(asdict(profile), indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
            print(f"Hardware profile saved to {args.output}")
        else:
            print(output)

    else:  # summary
        print(f"Hardware Profile: {profile.display_name}")
        print(f"Platform ID: {profile.platform_id}")
        print(f"\nSystem:")
        print(f"  OS: {profile.os_type}")
        print(f"  Architecture: {profile.cpu_arch}")
        print(f"\nCPU:")
        print(f"  Model: {profile.cpu_model}")
        print(f"  Performance cores: {profile.cpu_cores_performance}")
        if profile.cpu_cores_efficiency > 0:
            print(f"  Efficiency cores: {profile.cpu_cores_efficiency}")
        print(f"\nMemory:")
        print(f"  RAM: {profile.ram_gb:.1f} GB")
        if profile.vram_gb > 0:
            print(f"  VRAM: {profile.vram_gb:.1f} GB")
        print(f"\nGPU:")
        if profile.gpu_model:
            print(f"  Model: {profile.gpu_model}")
            print(f"  Type: {profile.gpu_type}")
            if profile.gpu_compute_capability:
                print(f"  Compute: {profile.gpu_compute_capability}")
        else:
            print(f"  None detected")
        print(f"\nRecommended Backend: {profile.recommended_backend}")
        print(f"Capabilities: {', '.join(profile.capabilities)}")


if __name__ == "__main__":
    main()
