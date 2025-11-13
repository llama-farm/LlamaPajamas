#!/usr/bin/env python3
"""
End-to-End Test for LLM Runtime
Tests: Qwen/Qwen2.5-1.5B-Instruct
Formats: MLX, GGUF
Quantizations: 8-bit, 4-bit, IQ2
"""

import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


class LLMRuntimeE2ETest:
    """End-to-end test for LLM model with multiple runtimes and quantizations."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        output_dir: Optional[str] = None,
        cleanup: bool = True
    ):
        """
        Initialize E2E test.

        Args:
            model_name: HuggingFace model name
            output_dir: Output directory (temp if None)
            cleanup: Whether to cleanup after test
        """
        self.model_name = model_name
        self.cleanup = cleanup

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="llm_e2e_"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": []
        }

        print(f"🧪 LLM E2E Test initialized")
        print(f"📦 Model: {model_name}")
        print(f"📁 Output: {self.output_dir}")
        print()

    def run_command(self, cmd: List[str], description: str) -> Dict:
        """Run a command and return results."""
        print(f"▶️  {description}")
        print(f"   Command: {' '.join(cmd)}")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.time() - start_time

            print(f"✅ Success ({elapsed:.2f}s)")
            return {
                "success": True,
                "elapsed": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed ({elapsed:.2f}s)")
            print(f"   Error: {e.stderr}")
            return {
                "success": False,
                "elapsed": elapsed,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "error": str(e)
            }

    def download_model(self) -> bool:
        """Download the base model from HuggingFace."""
        print("\n" + "="*60)
        print("STEP 1: Download Model")
        print("="*60)

        # Check if model already cached
        from huggingface_hub import snapshot_download

        try:
            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(self.output_dir / "hf_cache")
            )
            print(f"✅ Model downloaded: {model_path}")
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def export_gguf(self) -> bool:
        """Export model to GGUF format (FP16)."""
        print("\n" + "="*60)
        print("STEP 2a: Export to GGUF (FP16)")
        print("="*60)

        gguf_dir = self.output_dir / "gguf" / "fp16"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # Use llama-pajamas-quant to convert to GGUF
        cmd = [
            "python", "-m", "llama_pajamas_quant.cli.main",
            "quantize", "llm",
            "--model", self.model_name,
            "--formats", "gguf",
            "--output-dir", str(gguf_dir),
            "--precision", "fp16"
        ]

        result = self.run_command(cmd, "Export to GGUF FP16")

        if result["success"]:
            # Find the GGUF file
            gguf_files = list(gguf_dir.rglob("*.gguf"))
            if gguf_files:
                self.gguf_fp16_path = gguf_files[0]
                print(f"✅ GGUF file: {self.gguf_fp16_path}")
                return True

        return False

    def export_mlx(self) -> bool:
        """Export model to MLX format."""
        print("\n" + "="*60)
        print("STEP 2b: Export to MLX")
        print("="*60)

        mlx_dir = self.output_dir / "mlx" / "fp16"
        mlx_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "llama_pajamas_quant.cli.main",
            "quantize", "llm",
            "--model", self.model_name,
            "--formats", "mlx",
            "--output-dir", str(mlx_dir)
        ]

        result = self.run_command(cmd, "Export to MLX")

        if result["success"]:
            # MLX creates a directory with model files
            self.mlx_fp16_path = mlx_dir / Path(self.model_name).name
            if self.mlx_fp16_path.exists():
                print(f"✅ MLX model: {self.mlx_fp16_path}")
                return True

        return False

    def quantize_gguf(self, precision: str) -> Optional[Path]:
        """Quantize GGUF model to specific precision."""
        print(f"\n   Quantizing GGUF to {precision}...")

        output_dir = self.output_dir / "gguf" / precision
        output_dir.mkdir(parents=True, exist_ok=True)

        # For IQ quantizations, use the IQ workflow
        if precision.startswith("IQ") or precision.startswith("iq"):
            cmd = [
                "python", "-m", "llama_pajamas_quant.cli.main",
                "iq", "quantize",
                "--model", str(self.gguf_fp16_path),
                "--precision", precision,
                "--output-dir", str(output_dir)
            ]
        else:
            # Standard quantization
            from llama_pajamas_quant.quantizers.imatrix import IMatrixQuantizer
            quantizer = IMatrixQuantizer()

            try:
                output_file = quantizer.quantize_without_imatrix(
                    model_path=str(self.gguf_fp16_path),
                    output_dir=str(output_dir),
                    precision=precision
                )
                if output_file and Path(output_file).exists():
                    print(f"✅ Quantized to {precision}: {output_file}")
                    return Path(output_file)
            except Exception as e:
                print(f"❌ Quantization failed: {e}")
                return None

        result = self.run_command(cmd, f"Quantize GGUF to {precision}")

        if result["success"]:
            quantized_files = list(output_dir.rglob("*.gguf"))
            if quantized_files:
                return quantized_files[0]

        return None

    def quantize_mlx(self, bits: int) -> Optional[Path]:
        """Quantize MLX model to specific bit width."""
        print(f"\n   Quantizing MLX to {bits}-bit...")

        output_dir = self.output_dir / "mlx" / f"{bits}bit"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use mlx-lm for quantization
        cmd = [
            "python", "-m", "mlx_lm.convert",
            "--hf-path", self.model_name,
            "-q",
            "--q-bits", str(bits),
            "--mlx-path", str(output_dir)
        ]

        result = self.run_command(cmd, f"Quantize MLX to {bits}-bit")

        if result["success"]:
            if output_dir.exists():
                return output_dir

        return None

    def test_gguf_runtime(self, model_path: Path, name: str) -> Dict:
        """Test GGUF runtime with a model."""
        print(f"\n   Testing GGUF runtime: {name}...")

        try:
            from llama_pajamas_run.backends.gguf_backend import GGUFBackend

            backend = GGUFBackend(
                model_path=str(model_path),
                n_ctx=2048,
                n_gpu_layers=0  # CPU only for testing
            )

            # Simple generation test
            prompt = "What is the capital of France?"
            start_time = time.time()

            response = backend.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )

            elapsed = time.time() - start_time

            # Calculate tokens/sec
            tokens = len(response.split())
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

            print(f"✅ Runtime test passed")
            print(f"   Response: {response[:100]}...")
            print(f"   Speed: {tokens_per_sec:.2f} tokens/sec")

            return {
                "success": True,
                "elapsed": elapsed,
                "tokens": tokens,
                "tokens_per_sec": tokens_per_sec,
                "response": response
            }
        except Exception as e:
            print(f"❌ Runtime test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def test_mlx_runtime(self, model_path: Path, name: str) -> Dict:
        """Test MLX runtime with a model."""
        print(f"\n   Testing MLX runtime: {name}...")

        # Check if running on macOS
        if sys.platform != "darwin":
            print("⚠️  MLX only supported on macOS, skipping")
            return {"success": False, "error": "Not on macOS"}

        try:
            from llama_pajamas_run.backends.mlx_backend import MLXBackend

            backend = MLXBackend(model_path=str(model_path))

            # Simple generation test
            prompt = "What is the capital of France?"
            start_time = time.time()

            response = backend.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )

            elapsed = time.time() - start_time

            # Calculate tokens/sec
            tokens = len(response.split())
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

            print(f"✅ Runtime test passed")
            print(f"   Response: {response[:100]}...")
            print(f"   Speed: {tokens_per_sec:.2f} tokens/sec")

            return {
                "success": True,
                "elapsed": elapsed,
                "tokens": tokens,
                "tokens_per_sec": tokens_per_sec,
                "response": response
            }
        except Exception as e:
            print(f"❌ Runtime test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def evaluate_model(self, model_path: Path, format_type: str) -> Dict:
        """Run evaluation on a model."""
        print(f"\n   Running evaluation...")

        model_dir = model_path.parent

        cmd = [
            "python", "-m", "llama_pajamas_quant.cli.main",
            "evaluate", "llm",
            "--model-dir", str(model_dir),
            "--quick"  # Quick evaluation for testing
        ]

        result = self.run_command(cmd, "Evaluate model")

        # Try to load evaluation results
        eval_file = model_dir / "evaluation.json"
        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
                print(f"✅ Evaluation complete")
                print(f"   Score: {eval_data.get('overall_score', 'N/A')}")
                return {
                    "success": True,
                    "evaluation": eval_data
                }

        return result

    def run_full_test(self):
        """Run the complete end-to-end test."""
        print("\n" + "🚀 " * 30)
        print("STARTING LLM END-TO-END RUNTIME TEST")
        print("🚀 " * 30)

        # Step 1: Download model
        if not self.download_model():
            print("\n❌ Test failed at download step")
            return False

        # Step 2: Export to formats
        gguf_success = self.export_gguf()
        mlx_success = self.export_mlx()

        if not (gguf_success or mlx_success):
            print("\n❌ Test failed: No formats exported successfully")
            return False

        # Step 3: GGUF Quantizations and Tests
        if gguf_success:
            print("\n" + "="*60)
            print("STEP 3: GGUF Quantizations & Runtime Tests")
            print("="*60)

            gguf_tests = [
                ("Q8_0", "8-bit"),
                ("Q4_K_M", "4-bit"),
                ("IQ2_XS", "IQ2"),
            ]

            for precision, name in gguf_tests:
                print(f"\n--- Testing GGUF {name} ({precision}) ---")

                # Quantize
                quantized_path = self.quantize_gguf(precision)
                if not quantized_path:
                    print(f"⚠️  Skipping {name} - quantization failed")
                    continue

                # Test runtime
                runtime_result = self.test_gguf_runtime(quantized_path, name)

                # Evaluate (optional, can be slow)
                # eval_result = self.evaluate_model(quantized_path, "gguf")

                self.results["tests"].append({
                    "format": "GGUF",
                    "precision": precision,
                    "name": name,
                    "model_path": str(quantized_path),
                    "runtime": runtime_result
                })

        # Step 4: MLX Quantizations and Tests
        if mlx_success and sys.platform == "darwin":
            print("\n" + "="*60)
            print("STEP 4: MLX Quantizations & Runtime Tests")
            print("="*60)

            mlx_tests = [
                (8, "8-bit"),
                (4, "4-bit"),
            ]

            for bits, name in mlx_tests:
                print(f"\n--- Testing MLX {name} ({bits}-bit) ---")

                # Quantize
                quantized_path = self.quantize_mlx(bits)
                if not quantized_path:
                    print(f"⚠️  Skipping {name} - quantization failed")
                    continue

                # Test runtime
                runtime_result = self.test_mlx_runtime(quantized_path, name)

                self.results["tests"].append({
                    "format": "MLX",
                    "precision": f"{bits}bit",
                    "name": name,
                    "model_path": str(quantized_path),
                    "runtime": runtime_result
                })

        # Step 5: Save results
        self.save_results()

        # Step 6: Print summary
        self.print_summary()

        return True

    def save_results(self):
        """Save test results to JSON."""
        results_file = self.output_dir / "e2e_test_results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, indent=2, fp=f)

        print(f"\n💾 Results saved to: {results_file}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        total_tests = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"] if t.get("runtime", {}).get("success", False))

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {total_tests - passed}")
        print()

        print("Results by format:")
        for test in self.results["tests"]:
            status = "✅" if test.get("runtime", {}).get("success", False) else "❌"
            tokens_per_sec = test.get("runtime", {}).get("tokens_per_sec", 0)
            print(f"  {status} {test['format']:6s} {test['name']:10s} - {tokens_per_sec:.2f} tokens/sec")

        print()
        print(f"📁 Full results: {self.output_dir}")

    def cleanup_files(self):
        """Cleanup temporary files."""
        if self.cleanup and self.output_dir.exists():
            print(f"\n🧹 Cleaning up {self.output_dir}")
            shutil.rmtree(self.output_dir)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM E2E Runtime Test")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model to test"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (temp if not specified)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup files after test"
    )

    args = parser.parse_args()

    test = LLMRuntimeE2ETest(
        model_name=args.model,
        output_dir=args.output_dir,
        cleanup=not args.no_cleanup
    )

    try:
        success = test.run_full_test()

        if not args.no_cleanup:
            test.cleanup_files()

        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        if not args.no_cleanup:
            test.cleanup_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        if not args.no_cleanup:
            test.cleanup_files()
        sys.exit(1)


if __name__ == "__main__":
    main()
