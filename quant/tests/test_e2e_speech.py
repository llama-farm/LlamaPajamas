#!/usr/bin/env python3
"""
End-to-End Test for Speech/STT Model Runtime
Tests: Whisper (openai/whisper-tiny, whisper-base)
Formats: CoreML, ONNX
Quantizations: INT8, FP16
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
import platform
import numpy as np


class SpeechRuntimeE2ETest:
    """End-to-end test for Speech model with multiple runtimes and quantizations."""

    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
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
            self.output_dir = Path(tempfile.mkdtemp(prefix="speech_e2e_"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.system(),
            "tests": []
        }

        print(f"🧪 Speech E2E Test initialized")
        print(f"📦 Model: {model_name}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🖥️  Platform: {platform.system()}")
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
                check=True,
                timeout=600  # 10 minute timeout
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
        except subprocess.TimeoutExpired as e:
            print(f"❌ Timeout after {e.timeout}s")
            return {
                "success": False,
                "error": "Timeout"
            }

    def download_model(self) -> bool:
        """Download Whisper model from HuggingFace."""
        print("\n" + "="*60)
        print("STEP 1: Download Whisper Model")
        print("="*60)

        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(self.output_dir / "hf_cache")
            )

            self.pytorch_model_path = Path(model_path)
            print(f"✅ Model downloaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def export_coreml(self) -> Optional[Path]:
        """Export Whisper to CoreML format."""
        print("\n" + "="*60)
        print("STEP 2a: Export to CoreML")
        print("="*60)

        if platform.system() != "Darwin":
            print("⚠️  CoreML export only supported on macOS, skipping")
            return None

        coreml_dir = self.output_dir / "coreml" / "fp16"
        coreml_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use whisperkit for export (Apple's optimized Whisper for CoreML)
            # Or use exporters from llama_pajamas_quant
            from llama_pajamas_quant.exporters.unified import UnifiedExporter

            exporter = UnifiedExporter(
                model_name=self.model_name,
                output_dir=str(coreml_dir)
            )

            export_path = exporter.export_to_coreml(
                quantization="float16"
            )

            if export_path and Path(export_path).exists():
                print(f"✅ CoreML model: {export_path}")
                return Path(export_path)
            else:
                print(f"❌ Export failed")
                return None
        except Exception as e:
            print(f"❌ CoreML export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def quantize_coreml(self, model_path: Path) -> Optional[Path]:
        """Quantize CoreML Whisper model to INT8."""
        print(f"\n   Quantizing CoreML to INT8...")

        output_dir = self.output_dir / "coreml" / "int8"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from llama_pajamas_quant.quantizers.whisper_coreml import WhisperCoreMLQuantizer

            quantizer = WhisperCoreMLQuantizer()

            output_path = quantizer.quantize(
                model_path=str(model_path),
                output_dir=str(output_dir),
                precision="int8"
            )

            if output_path and Path(output_path).exists():
                print(f"✅ Quantized CoreML (INT8): {output_path}")
                return Path(output_path)
            else:
                print(f"❌ Quantization failed")
                return None
        except Exception as e:
            print(f"❌ CoreML quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def export_onnx(self) -> Optional[Path]:
        """Export Whisper to ONNX format."""
        print("\n" + "="*60)
        print("STEP 2b: Export to ONNX")
        print("="*60)

        onnx_dir = self.output_dir / "onnx" / "fp32"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        try:
            from llama_pajamas_quant.exporters.onnx_speech import ONNXSpeechExporter

            exporter = ONNXSpeechExporter(
                model_name=self.model_name,
                output_dir=str(onnx_dir)
            )

            export_paths = exporter.export()

            if export_paths:
                # Return encoder path (main component)
                encoder_path = export_paths.get("encoder")
                if encoder_path and Path(encoder_path).exists():
                    print(f"✅ ONNX model: {encoder_path}")
                    self.onnx_decoder_path = export_paths.get("decoder")
                    return Path(encoder_path)

            print(f"❌ Export failed")
            return None
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def quantize_onnx(self, model_path: Path) -> Optional[Path]:
        """Quantize ONNX Whisper model to INT8."""
        print(f"\n   Quantizing ONNX to INT8...")

        output_dir = self.output_dir / "onnx" / "int8"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            output_path = output_dir / f"{model_path.stem}_int8.onnx"

            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8
            )

            print(f"✅ Quantized ONNX (INT8): {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ ONNX quantization failed: {e}")
            return None

    def create_test_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> Path:
        """Create a test audio file."""
        audio_dir = self.output_dir / "test_audio"
        audio_dir.mkdir(exist_ok=True)

        audio_path = audio_dir / "test.wav"

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        # Save as WAV
        import scipy.io.wavfile as wavfile
        wavfile.write(str(audio_path), sample_rate, (audio * 32767).astype(np.int16))

        return audio_path

    def test_coreml_runtime(self, model_path: Path, name: str) -> Dict:
        """Test CoreML runtime with a Whisper model."""
        print(f"\n   Testing CoreML runtime: {name}...")

        if platform.system() != "Darwin":
            print("⚠️  CoreML only supported on macOS, skipping")
            return {"success": False, "error": "Not on macOS"}

        try:
            from llama_pajamas_run_coreml.backends.stt import CoreMLSTTBackend

            backend = CoreMLSTTBackend(model_path=str(model_path))

            # Create test audio
            audio_path = self.create_test_audio()

            # Run transcription
            start_time = time.time()
            result = backend.transcribe(str(audio_path))
            elapsed = time.time() - start_time

            # Calculate RTF (Real-Time Factor)
            audio_duration = 5.0  # We created 5 second audio
            rtf = elapsed / audio_duration

            print(f"✅ Runtime test passed")
            print(f"   Transcription: {result.get('text', '')[:50]}...")
            print(f"   RTF: {rtf:.2f}x (lower is better)")
            print(f"   Speed: {elapsed:.2f}s for {audio_duration}s audio")

            return {
                "success": True,
                "elapsed": elapsed,
                "rtf": rtf,
                "audio_duration": audio_duration,
                "text": result.get("text", "")
            }
        except Exception as e:
            print(f"❌ Runtime test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def test_onnx_runtime(self, model_path: Path, name: str) -> Dict:
        """Test ONNX runtime with a Whisper model."""
        print(f"\n   Testing ONNX runtime: {name}...")

        try:
            from llama_pajamas_run_onnx.backends.speech import ONNXSpeechBackend

            backend = ONNXSpeechBackend(
                encoder_path=str(model_path),
                decoder_path=str(self.onnx_decoder_path) if hasattr(self, 'onnx_decoder_path') else None
            )

            # Create test audio
            audio_path = self.create_test_audio()

            # Run transcription
            start_time = time.time()
            result = backend.transcribe(str(audio_path))
            elapsed = time.time() - start_time

            # Calculate RTF
            audio_duration = 5.0
            rtf = elapsed / audio_duration

            print(f"✅ Runtime test passed")
            print(f"   Transcription: {result.get('text', '')[:50]}...")
            print(f"   RTF: {rtf:.2f}x (lower is better)")
            print(f"   Speed: {elapsed:.2f}s for {audio_duration}s audio")

            return {
                "success": True,
                "elapsed": elapsed,
                "rtf": rtf,
                "audio_duration": audio_duration,
                "text": result.get("text", "")
            }
        except Exception as e:
            print(f"❌ Runtime test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def evaluate_model(self, model_path: Path, format_type: str) -> Dict:
        """Run evaluation on a speech model."""
        print(f"\n   Running evaluation...")

        model_dir = model_path.parent

        cmd = [
            "python", "-m", "llama_pajamas_quant.cli.main",
            "evaluate", "stt",
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
                print(f"   WER: {eval_data.get('wer', 'N/A')}")
                return {
                    "success": True,
                    "evaluation": eval_data
                }

        return result

    def run_full_test(self):
        """Run the complete end-to-end test."""
        print("\n" + "🚀 " * 30)
        print("STARTING SPEECH END-TO-END RUNTIME TEST")
        print("🚀 " * 30)

        # Step 1: Download model
        if not self.download_model():
            print("\n❌ Test failed at download step")
            return False

        # Step 2: Export to formats
        coreml_model = self.export_coreml()
        onnx_model = self.export_onnx()

        if not any([coreml_model, onnx_model]):
            print("\n❌ Test failed: No formats exported successfully")
            return False

        # Step 3: CoreML Tests
        if coreml_model:
            print("\n" + "="*60)
            print("STEP 3: CoreML Quantizations & Runtime Tests")
            print("="*60)

            # Test FP16
            print(f"\n--- Testing CoreML FP16 ---")
            runtime_result = self.test_coreml_runtime(coreml_model, "FP16")
            self.results["tests"].append({
                "format": "CoreML",
                "precision": "FP16",
                "model_path": str(coreml_model),
                "runtime": runtime_result
            })

            # Test INT8
            print(f"\n--- Testing CoreML INT8 ---")
            int8_model = self.quantize_coreml(coreml_model)
            if int8_model:
                runtime_result = self.test_coreml_runtime(int8_model, "INT8")
                self.results["tests"].append({
                    "format": "CoreML",
                    "precision": "INT8",
                    "model_path": str(int8_model),
                    "runtime": runtime_result
                })

        # Step 4: ONNX Tests
        if onnx_model:
            print("\n" + "="*60)
            print("STEP 4: ONNX Quantizations & Runtime Tests")
            print("="*60)

            # Test FP32
            print(f"\n--- Testing ONNX FP32 ---")
            runtime_result = self.test_onnx_runtime(onnx_model, "FP32")
            self.results["tests"].append({
                "format": "ONNX",
                "precision": "FP32",
                "model_path": str(onnx_model),
                "runtime": runtime_result
            })

            # Test INT8
            print(f"\n--- Testing ONNX INT8 ---")
            int8_model = self.quantize_onnx(onnx_model)
            if int8_model:
                runtime_result = self.test_onnx_runtime(int8_model, "INT8")
                self.results["tests"].append({
                    "format": "ONNX",
                    "precision": "INT8",
                    "model_path": str(int8_model),
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
            rtf = test.get("runtime", {}).get("rtf", 0)
            print(f"  {status} {test['format']:10s} {test['precision']:6s} - RTF: {rtf:.2f}x")

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

    parser = argparse.ArgumentParser(description="Speech E2E Runtime Test")
    parser.add_argument(
        "--model",
        default="openai/whisper-tiny",
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

    test = SpeechRuntimeE2ETest(
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
