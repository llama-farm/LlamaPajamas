#!/usr/bin/env python3
"""Auto-build llama.cpp with hardware-optimized settings.

Detects your hardware and configures llama.cpp with the best settings:
- macOS: Metal GPU acceleration
- Linux NVIDIA: CUDA support
- Linux AMD: ROCm/HIP support
- CPU fallback: Optimized CPU-only build
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class LlamaCppBuilder:
    """Smart builder for llama.cpp with hardware detection."""

    def __init__(self, llama_cpp_path: Optional[Path] = None):
        """Initialize builder.

        Args:
            llama_cpp_path: Path to llama.cpp directory
        """
        if llama_cpp_path is None:
            # Default to libs/llama.cpp from script location
            self.llama_cpp_path = Path(__file__).parent.parent.parent / "libs" / "llama.cpp"
        else:
            self.llama_cpp_path = Path(llama_cpp_path)

        if not self.llama_cpp_path.exists():
            raise ValueError(f"llama.cpp not found at {self.llama_cpp_path}")

        self.build_dir = self.llama_cpp_path / "build"
        self.system = platform.system()
        self.machine = platform.machine()

    def detect_hardware(self) -> dict:
        """Detect available hardware acceleration.

        Returns:
            Dict with hardware capabilities
        """
        hw = {
            "system": self.system,
            "machine": self.machine,
            "metal": False,
            "cuda": False,
            "rocm": False,
            "cpu_only": True,
        }

        print("\n" + "=" * 60)
        print("HARDWARE DETECTION")
        print("=" * 60)
        print(f"System: {self.system}")
        print(f"Architecture: {self.machine}")

        # macOS - Check for Metal
        if self.system == "Darwin":
            hw["metal"] = True
            hw["cpu_only"] = False
            print("✅ Metal GPU support detected (macOS)")

        # Linux - Check for NVIDIA CUDA
        elif self.system == "Linux":
            # Check for NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    hw["cuda"] = True
                    hw["cpu_only"] = False
                    print("✅ NVIDIA CUDA support detected")
                    # Extract GPU info
                    if "Tesla" in result.stdout or "RTX" in result.stdout or "GTX" in result.stdout:
                        gpu_lines = [line for line in result.stdout.split('\n') if 'MiB' in line]
                        if gpu_lines:
                            print(f"   GPU: {gpu_lines[0].strip()[:60]}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Check for AMD ROCm
            if not hw["cuda"]:
                try:
                    result = subprocess.run(
                        ["rocm-smi"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        hw["rocm"] = True
                        hw["cpu_only"] = False
                        print("✅ AMD ROCm support detected")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

        # CPU info
        if hw["cpu_only"]:
            print("ℹ️  No GPU detected, using CPU-only build")

        # Detect CPU features
        try:
            if self.system == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(f"   CPU: {result.stdout.strip()}")
            elif self.system == "Linux":
                result = subprocess.run(
                    ["lscpu"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Model name:' in line:
                            print(f"   CPU: {line.split(':')[1].strip()}")
                            break
        except:
            pass

        return hw

    def get_cmake_flags(self, hw: dict, force_cpu: bool = False) -> List[str]:
        """Get CMake flags based on hardware.

        Args:
            hw: Hardware detection dict
            force_cpu: Force CPU-only build

        Returns:
            List of CMake flags
        """
        flags = []

        if force_cpu:
            print("\n⚠️  Forcing CPU-only build (--cpu flag)")
            return flags

        # macOS Metal
        if hw["metal"]:
            flags.append("-DGGML_METAL=ON")
            print("\n🔧 Enabling Metal GPU acceleration")

        # NVIDIA CUDA
        elif hw["cuda"]:
            flags.append("-DGGML_CUDA=ON")
            print("\n🔧 Enabling NVIDIA CUDA acceleration")

            # Auto-detect CUDA architecture
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    compute_cap = result.stdout.strip().split('\n')[0].replace('.', '')
                    if compute_cap:
                        flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={compute_cap}")
                        print(f"   Detected CUDA compute capability: {compute_cap}")
            except:
                print("   Using default CUDA architectures")

        # AMD ROCm
        elif hw["rocm"]:
            flags.append("-DGGML_HIPBLAS=ON")
            print("\n🔧 Enabling AMD ROCm/HIP acceleration")

        # CPU optimizations
        else:
            print("\n🔧 Using CPU-only build with optimizations")

        return flags

    def build(self, force_cpu: bool = False, clean: bool = False) -> bool:
        """Build llama.cpp with optimal settings.

        Args:
            force_cpu: Force CPU-only build
            clean: Clean build directory first

        Returns:
            True if build succeeded
        """
        print("\n" + "=" * 60)
        print("BUILDING LLAMA.CPP")
        print("=" * 60)

        # Clean build if requested
        if clean and self.build_dir.exists():
            print(f"\n🗑️  Cleaning build directory: {self.build_dir}")
            import shutil
            shutil.rmtree(self.build_dir)

        # Create build directory
        self.build_dir.mkdir(exist_ok=True)

        # Detect hardware
        hw = self.detect_hardware()

        # Get CMake flags
        cmake_flags = self.get_cmake_flags(hw, force_cpu)

        # Configure with CMake
        print("\n" + "=" * 60)
        print("CMAKE CONFIGURATION")
        print("=" * 60)

        cmake_cmd = ["cmake", ".."] + cmake_flags

        print(f"\nRunning: {' '.join(cmake_cmd)}")

        try:
            result = subprocess.run(
                cmake_cmd,
                cwd=self.build_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✅ CMake configuration successful")
            if result.stdout and "--verbose" in sys.argv:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ CMake configuration failed!")
            print(f"Error: {e.stderr}")
            return False

        # Build
        print("\n" + "=" * 60)
        print("BUILDING BINARIES")
        print("=" * 60)
        print("\nThis may take a few minutes...")

        build_cmd = [
            "cmake",
            "--build", ".",
            "--config", "Release",
            "-j",  # Parallel build
        ]

        print(f"Running: {' '.join(build_cmd)}")

        try:
            result = subprocess.run(
                build_cmd,
                cwd=self.build_dir,
                check=True,
                capture_output=False,  # Show build output
                text=True,
            )
            print("\n✅ Build successful!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Build failed!")
            return False

        # Verify binaries
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)

        binaries = {
            "llama-imatrix": self.build_dir / "bin" / "llama-imatrix",
            "llama-quantize": self.build_dir / "bin" / "llama-quantize",
            "llama-cli": self.build_dir / "bin" / "llama-cli",
        }

        all_good = True
        for name, path in binaries.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✅ {name}: {size_mb:.1f} MB")
            else:
                print(f"❌ {name}: NOT FOUND")
                all_good = False

        if all_good:
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"\nllama.cpp built successfully at: {self.llama_cpp_path}")
            print(f"Binaries: {self.build_dir / 'bin'}")
        else:
            print("\n⚠️  Some binaries missing, build may have failed")

        return all_good


def main():
    parser = argparse.ArgumentParser(
        description="Auto-build llama.cpp with hardware-optimized settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect hardware and build
  python build_llama_cpp.py

  # Force CPU-only build
  python build_llama_cpp.py --cpu

  # Clean build from scratch
  python build_llama_cpp.py --clean

  # Custom llama.cpp path
  python build_llama_cpp.py --llama-cpp /path/to/llama.cpp

Hardware Detection:
  - macOS: Automatically enables Metal GPU
  - Linux + NVIDIA: Automatically enables CUDA
  - Linux + AMD: Automatically enables ROCm/HIP
  - Fallback: Optimized CPU-only build
        """
    )

    parser.add_argument(
        "--llama-cpp",
        type=Path,
        help="Path to llama.cpp directory (default: ../libs/llama.cpp)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only build (disable GPU acceleration)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose CMake output"
    )

    args = parser.parse_args()

    # Build
    builder = LlamaCppBuilder(llama_cpp_path=args.llama_cpp)
    success = builder.build(force_cpu=args.cpu, clean=args.clean)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
