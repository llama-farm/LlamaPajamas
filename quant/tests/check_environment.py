#!/usr/bin/env python3
"""
Environment Check for E2E Runtime Tests
Checks if your system can run various E2E tests.
"""

import sys
import platform
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 11:
        print("   ✅ Python version OK (3.11+)")
        return True
    else:
        print("   ❌ Python version too old (need 3.11+)")
        return False


def check_platform():
    """Check platform."""
    print("\n🖥️  Platform Check")
    system = platform.system()
    machine = platform.machine()
    print(f"   {system} on {machine}")

    capabilities = []

    if system == "Darwin":
        print("   ✅ macOS detected")
        if machine == "arm64":
            print("   ✅ Apple Silicon detected (MLX + CoreML available)")
            capabilities.extend(["mlx", "coreml", "onnx", "gguf"])
        else:
            print("   ⚠️  Intel Mac (limited to ONNX + GGUF)")
            capabilities.extend(["onnx", "gguf"])
    elif system == "Linux":
        print("   ✅ Linux detected")
        capabilities.extend(["onnx", "gguf"])
    elif system == "Windows":
        print("   ✅ Windows detected")
        capabilities.extend(["onnx", "gguf"])
    else:
        print(f"   ⚠️  Unknown platform: {system}")

    return capabilities


def check_nvidia_gpu():
    """Check for NVIDIA GPU."""
    print("\n🎮 NVIDIA GPU Check")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse GPU info
        if "NVIDIA" in result.stdout:
            lines = result.stdout.split("\n")
            for line in lines:
                if "Tesla" in line or "RTX" in line or "GTX" in line or "A100" in line or "H100" in line:
                    gpu_info = line.strip()
                    print(f"   ✅ NVIDIA GPU detected: {gpu_info[:60]}")
                    print("   ✅ TensorRT tests available")
                    return True

        print("   ✅ nvidia-smi available")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ No NVIDIA GPU detected (TensorRT tests unavailable)")
        return False


def check_docker():
    """Check Docker availability."""
    print("\n🐳 Docker Check")

    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ {result.stdout.strip()}")

        # Check docker compose
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"   ✅ {result.stdout.strip()}")
            return True
        except:
            print("   ⚠️  Docker Compose not available")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Docker not available (container tests unavailable)")
        return False


def check_dependencies():
    """Check Python dependencies."""
    print("\n📦 Python Dependencies Check")

    dependencies = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "ultralytics": "Ultralytics (YOLO)",
        "onnx": "ONNX",
        "onnxruntime": "ONNX Runtime",
        "huggingface_hub": "HuggingFace Hub",
        "numpy": "NumPy",
        "pillow": "Pillow (PIL)",
    }

    optional_dependencies = {
        "mlx": "MLX (Apple Silicon only)",
        "coremltools": "CoreML Tools (macOS only)",
        "tensorrt": "TensorRT (NVIDIA GPU only)",
    }

    available = []
    missing = []

    for package, name in dependencies.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
            available.append(package)
        except ImportError:
            print(f"   ❌ {name} - REQUIRED")
            missing.append(package)

    print("\n📦 Optional Dependencies")
    for package, name in optional_dependencies.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
            available.append(package)
        except ImportError:
            print(f"   ⚠️  {name} - optional")

    return len(missing) == 0


def check_disk_space():
    """Check available disk space."""
    print("\n💾 Disk Space Check")

    import shutil
    total, used, free = shutil.disk_usage("/")

    free_gb = free // (2**30)
    print(f"   Free space: {free_gb} GB")

    if free_gb >= 50:
        print(f"   ✅ Sufficient disk space (50+ GB)")
        return True
    else:
        print(f"   ⚠️  Low disk space (need 50+ GB, have {free_gb} GB)")
        return False


def generate_recommendations(capabilities, has_nvidia, has_docker):
    """Generate test recommendations."""
    print("\n" + "="*60)
    print("📋 TEST RECOMMENDATIONS")
    print("="*60)

    print("\nYou can run:")

    # LLM tests
    if "gguf" in capabilities:
        print("\n✅ LLM Tests:")
        print("   - GGUF runtime (CPU)")
        if "mlx" in capabilities:
            print("   - MLX runtime (Apple Silicon)")

    # Vision tests
    print("\n✅ Vision Tests:")
    if "onnx" in capabilities:
        print("   - ONNX runtime")
    if "coreml" in capabilities:
        print("   - CoreML runtime (macOS)")
    if has_nvidia:
        print("   - TensorRT runtime (NVIDIA GPU)")

    # Speech tests
    print("\n✅ Speech Tests:")
    if "onnx" in capabilities:
        print("   - ONNX runtime")
    if "coreml" in capabilities:
        print("   - CoreML runtime (macOS)")

    # Commands
    print("\n" + "="*60)
    print("🚀 RECOMMENDED COMMANDS")
    print("="*60)

    print("\n# Run all available tests:")
    cmd = "python run_e2e_tests.py"

    skip_flags = []
    if not has_nvidia:
        skip_flags.append("--skip-tensorrt")
    if "mlx" not in capabilities:
        skip_flags.append("--skip-mlx")
    if "coreml" not in capabilities:
        skip_flags.append("--skip-coreml")

    if skip_flags:
        cmd += " " + " ".join(skip_flags)

    print(cmd)

    # Docker
    if has_docker and has_nvidia:
        print("\n# Run TensorRT tests in Docker:")
        print("cd ../docker")
        print("docker compose -f docker-compose.tensorrt.yml run --rm tensorrt-e2e-test")

    print("\n" + "="*60)


def main():
    """Main entry point."""
    print("="*60)
    print("🧪 LlamaPajamas E2E Test Environment Check")
    print("="*60)

    # Run checks
    python_ok = check_python_version()
    capabilities = check_platform()
    has_nvidia = check_nvidia_gpu()
    has_docker = check_docker()
    deps_ok = check_dependencies()
    disk_ok = check_disk_space()

    # Generate recommendations
    generate_recommendations(capabilities, has_nvidia, has_docker)

    # Overall status
    print("\n" + "="*60)
    print("📊 OVERALL STATUS")
    print("="*60)

    if python_ok and deps_ok and disk_ok:
        print("\n✅ Your environment is ready for E2E tests!")
        print("\nNext steps:")
        print("1. cd quant/tests")
        print("2. python run_e2e_tests.py")
        return 0
    else:
        print("\n⚠️  Your environment has some issues:")
        if not python_ok:
            print("   - Upgrade Python to 3.11+")
        if not deps_ok:
            print("   - Install missing dependencies")
        if not disk_ok:
            print("   - Free up disk space")

        print("\nInstall dependencies:")
        print("pip install -e ./quant")
        print("pip install -e ./run")

        if "mlx" in capabilities:
            print("pip install -e ./run-mlx")
        if "coreml" in capabilities:
            print("pip install -e ./run-coreml")
        if has_nvidia:
            print("pip install -e ./run-tensorrt")

        print("pip install -e ./run-onnx")

        return 1


if __name__ == "__main__":
    sys.exit(main())
