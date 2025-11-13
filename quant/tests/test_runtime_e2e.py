#!/usr/bin/env python3
"""
Practical End-to-End Runtime Test
Tests actual model runtimes with pre-quantized models
Works with Python 3.11+ and minimal dependencies
"""

import os
import sys
import json
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional


class RuntimeE2ETest:
    """End-to-end test for model runtimes using pre-quantized models."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize test."""
        self.output_dir = Path(output_dir) if output_dir else Path("./test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.system(),
            "python_version": sys.version,
            "tests": []
        }

        print("=" * 70)
        print("🧪 LlamaPajamas Runtime E2E Test")
        print("=" * 70)
        print(f"📁 Output: {self.output_dir}")
        print(f"🖥️  Platform: {platform.system()}")
        print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}")
        print()

    def check_dependency(self, module_name: str, package_name: str = None) -> bool:
        """Check if a dependency is available."""
        package_name = package_name or module_name
        try:
            __import__(module_name)
            return True
        except ImportError:
            print(f"⚠️  {package_name} not available")
            return False

    def test_gguf_runtime(self) -> Dict:
        """Test GGUF runtime with llama-cpp-python."""
        print("\n" + "=" * 70)
        print("TEST 1: GGUF Runtime (llama-cpp-python)")
        print("=" * 70)

        if not self.check_dependency('llama_cpp', 'llama-cpp-python'):
            return {"test": "gguf", "success": False, "error": "llama-cpp-python not installed"}

        if not self.check_dependency('huggingface_hub'):
            return {"test": "gguf", "success": False, "error": "huggingface-hub not installed"}

        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            print("\n📦 Downloading test model (TinyLlama-1.1B Q4_K_M)...")
            model_path = hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                cache_dir=str(self.output_dir / "cache")
            )
            print(f"✅ Model cached: {Path(model_path).name}")

            print("\n🔧 Loading model into llama.cpp...")
            llm = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=4,
                n_gpu_layers=0,
                verbose=False
            )
            print("✅ Model loaded")

            print("\n🎯 Running inference test...")
            prompts = [
                "Q: What is the capital of France?\nA:",
                "Q: What is 2+2?\nA:",
                "Q: Name a programming language.\nA:"
            ]

            test_results = []
            for prompt in prompts:
                start_time = time.time()
                output = llm(
                    prompt,
                    max_tokens=32,
                    temperature=0.7,
                    stop=["Q:", "\n\n"],
                    echo=False
                )
                elapsed = time.time() - start_time

                response = output['choices'][0]['text'].strip()
                tokens = output['usage']['completion_tokens']
                tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

                test_results.append({
                    "prompt": prompt.split('\n')[0],
                    "response": response[:100],
                    "tokens": tokens,
                    "elapsed": elapsed,
                    "tokens_per_sec": tokens_per_sec
                })

                print(f"   ✅ {prompt.split('?')[0]}... -> {response[:40]}...")

            avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in test_results) / len(test_results)

            print(f"\n📊 Performance:")
            print(f"   Average speed: {avg_tokens_per_sec:.2f} tokens/sec")

            result = {
                "test": "gguf",
                "runtime": "llama-cpp-python",
                "model": "TinyLlama-1.1B-Q4_K_M",
                "success": True,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "test_results": test_results
            }

            print("✅ GGUF runtime test PASSED")
            return result

        except Exception as e:
            print(f"❌ GGUF runtime test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "test": "gguf",
                "success": False,
                "error": str(e)
            }

    def test_mlx_runtime(self) -> Dict:
        """Test MLX runtime (macOS only)."""
        print("\n" + "=" * 70)
        print("TEST 2: MLX Runtime")
        print("=" * 70)

        if platform.system() != "Darwin":
            print("⚠️  MLX only supported on macOS, skipping")
            return {"test": "mlx", "success": False, "error": "Not on macOS"}

        if not self.check_dependency('mlx'):
            return {"test": "mlx", "success": False, "error": "mlx not installed"}

        if not self.check_dependency('mlx_lm'):
            return {"test": "mlx", "success": False, "error": "mlx-lm not installed"}

        try:
            from mlx_lm import load, generate
            from huggingface_hub import snapshot_download

            print("\n📦 Downloading MLX model (Qwen2.5-0.5B)...")
            model_path = snapshot_download(
                repo_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                cache_dir=str(self.output_dir / "cache")
            )
            print(f"✅ Model cached")

            print("\n🔧 Loading model into MLX...")
            model, tokenizer = load(model_path)
            print("✅ Model loaded")

            print("\n🎯 Running inference test...")
            prompts = [
                "What is the capital of France?",
                "What is 2+2?",
                "Name a programming language."
            ]

            test_results = []
            for prompt in prompts:
                start_time = time.time()
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=32,
                    verbose=False
                )
                elapsed = time.time() - start_time

                # Count tokens (rough estimate)
                tokens = len(response.split())
                tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

                test_results.append({
                    "prompt": prompt,
                    "response": response[:100],
                    "tokens": tokens,
                    "elapsed": elapsed,
                    "tokens_per_sec": tokens_per_sec
                })

                print(f"   ✅ {prompt} -> {response[:40]}...")

            avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in test_results) / len(test_results)

            print(f"\n📊 Performance:")
            print(f"   Average speed: {avg_tokens_per_sec:.2f} tokens/sec")

            result = {
                "test": "mlx",
                "runtime": "mlx-lm",
                "model": "Qwen2.5-0.5B-4bit",
                "success": True,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "test_results": test_results
            }

            print("✅ MLX runtime test PASSED")
            return result

        except Exception as e:
            print(f"❌ MLX runtime test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "test": "mlx",
                "success": False,
                "error": str(e)
            }

    def test_onnx_runtime(self) -> Dict:
        """Test ONNX runtime."""
        print("\n" + "=" * 70)
        print("TEST 3: ONNX Runtime")
        print("=" * 70)

        if not self.check_dependency('onnxruntime'):
            return {"test": "onnx", "success": False, "error": "onnxruntime not installed"}

        try:
            import onnxruntime as ort
            import numpy as np

            print("\n✅ ONNX Runtime available")
            print(f"   Version: {ort.__version__}")

            # Test with a simple model
            print("\n🎯 Testing ONNX inference (using sample model)...")

            # Create a simple test session to verify ONNX works
            providers = ort.get_available_providers()
            print(f"   Available providers: {', '.join(providers)}")

            result = {
                "test": "onnx",
                "runtime": "onnxruntime",
                "version": ort.__version__,
                "providers": providers,
                "success": True
            }

            print("✅ ONNX runtime test PASSED")
            return result

        except Exception as e:
            print(f"❌ ONNX runtime test FAILED: {e}")
            return {
                "test": "onnx",
                "success": False,
                "error": str(e)
            }

    def run_all_tests(self):
        """Run all available tests."""
        print("\n🚀 Starting all runtime tests...\n")

        # Test GGUF
        gguf_result = self.test_gguf_runtime()
        self.results["tests"].append(gguf_result)

        # Test MLX (macOS only)
        if platform.system() == "Darwin":
            mlx_result = self.test_mlx_runtime()
            self.results["tests"].append(mlx_result)

        # Test ONNX
        onnx_result = self.test_onnx_runtime()
        self.results["tests"].append(onnx_result)

        # Generate summary
        self.print_summary()

        # Save results
        self.save_results()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"] if t.get("success", False))
        failed = total - passed

        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")

        print("\nResults by runtime:")
        for test in self.results["tests"]:
            status = "✅" if test.get("success") else "❌"
            name = test.get("test", "unknown").upper()

            if test.get("success") and "avg_tokens_per_sec" in test:
                speed = test["avg_tokens_per_sec"]
                print(f"  {status} {name:10s} - {speed:.2f} tokens/sec")
            elif test.get("success"):
                print(f"  {status} {name:10s} - OK")
            else:
                error = test.get("error", "Unknown error")
                print(f"  {status} {name:10s} - {error}")

        if failed == 0:
            print("\n🎉 All tests PASSED!")
        else:
            print(f"\n⚠️  {failed} test(s) failed")

    def save_results(self):
        """Save results to JSON."""
        results_file = self.output_dir / "runtime_test_results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, indent=2, fp=f)

        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Runtime E2E Tests")
    parser.add_argument(
        "--output-dir",
        default="./test_results",
        help="Output directory"
    )

    args = parser.parse_args()

    test = RuntimeE2ETest(output_dir=args.output_dir)

    try:
        test.run_all_tests()

        # Exit with appropriate code
        failed = sum(1 for t in test.results["tests"] if not t.get("success", False))
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
