#!/usr/bin/env python3
"""
End-to-End Test for Vision Model Runtime
Tests: YOLOv8n (ultralytics/yolov8n)
Formats: CoreML, ONNX, TensorRT
Quantizations: INT8, INT4, FP16
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


class VisionRuntimeE2ETest:
    """End-to-end test for Vision model with multiple runtimes and quantizations."""

    def __init__(
        self,
        model_name: str = "yolov8n",
        output_dir: Optional[str] = None,
        cleanup: bool = True,
        skip_tensorrt: bool = False
    ):
        """
        Initialize E2E test.

        Args:
            model_name: Model name (e.g., 'yolov8n', 'yolov8s')
            output_dir: Output directory (temp if None)
            cleanup: Whether to cleanup after test
            skip_tensorrt: Skip TensorRT tests (requires NVIDIA GPU)
        """
        self.model_name = model_name
        self.cleanup = cleanup
        self.skip_tensorrt = skip_tensorrt

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="vision_e2e_"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.system(),
            "tests": []
        }

        print(f"🧪 Vision E2E Test initialized")
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
        """Download YOLOv8 model from Ultralytics."""
        print("\n" + "="*60)
        print("STEP 1: Download YOLOv8 Model")
        print("="*60)

        try:
            from ultralytics import YOLO

            # Download model (will cache automatically)
            model = YOLO(f"{self.model_name}.pt")
            model_file = Path.home() / ".cache" / "ultralytics" / f"{self.model_name}.pt"

            if not model_file.exists():
                print(f"⚠️  Model file not found at {model_file}")
                # Try alternative path
                model_file = Path(f"{self.model_name}.pt")

            self.pytorch_model_path = model_file
            self.yolo_model = model

            print(f"✅ Model loaded: {model_file}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def export_coreml(self) -> Optional[Path]:
        """Export model to CoreML format."""
        print("\n" + "="*60)
        print("STEP 2a: Export to CoreML")
        print("="*60)

        if platform.system() != "Darwin":
            print("⚠️  CoreML export only supported on macOS, skipping")
            return None

        coreml_dir = self.output_dir / "coreml"
        coreml_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Export using Ultralytics
            export_path = self.yolo_model.export(
                format="coreml",
                half=True,  # FP16
                nms=True
            )

            if export_path and Path(export_path).exists():
                # Move to our directory
                target = coreml_dir / Path(export_path).name
                shutil.copy(export_path, target)
                print(f"✅ CoreML model: {target}")
                return target
            else:
                print(f"❌ Export failed")
                return None
        except Exception as e:
            print(f"❌ CoreML export failed: {e}")
            return None

    def quantize_coreml(self, model_path: Path, precision: str) -> Optional[Path]:
        """Quantize CoreML model."""
        print(f"\n   Quantizing CoreML to {precision}...")

        output_dir = self.output_dir / "coreml" / precision
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import coremltools as ct

            # Load model
            model = ct.models.MLModel(str(model_path))

            # Quantize based on precision
            if precision == "int8":
                # INT8 quantization
                op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    weight_threshold=512
                )
                config = ct.optimize.coreml.OptimizationConfig(
                    global_config=op_config
                )
            elif precision == "int4":
                # INT4 quantization (palettization)
                op_config = ct.optimize.coreml.OpPalettizerConfig(
                    mode="kmeans",
                    nbits=4
                )
                config = ct.optimize.coreml.OptimizationConfig(
                    global_config=op_config
                )
            else:
                print(f"⚠️  Unknown precision: {precision}")
                return None

            # Apply quantization
            quantized_model = ct.optimize.coreml.linear_quantize_weights(
                model,
                config=config
            )

            # Save
            output_path = output_dir / f"{self.model_name}_{precision}.mlpackage"
            quantized_model.save(str(output_path))

            print(f"✅ Quantized CoreML ({precision}): {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ CoreML quantization failed: {e}")
            return None

    def export_onnx(self) -> Optional[Path]:
        """Export model to ONNX format."""
        print("\n" + "="*60)
        print("STEP 2b: Export to ONNX")
        print("="*60)

        onnx_dir = self.output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Export using Ultralytics
            export_path = self.yolo_model.export(
                format="onnx",
                simplify=True,
                dynamic=False,
                opset=13
            )

            if export_path and Path(export_path).exists():
                # Move to our directory
                target = onnx_dir / f"{self.model_name}.onnx"
                shutil.copy(export_path, target)
                print(f"✅ ONNX model: {target}")
                return target
            else:
                print(f"❌ Export failed")
                return None
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return None

    def quantize_onnx(self, model_path: Path) -> Optional[Path]:
        """Quantize ONNX model to INT8."""
        print(f"\n   Quantizing ONNX to INT8...")

        output_dir = self.output_dir / "onnx" / "int8"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            output_path = output_dir / f"{self.model_name}_int8.onnx"

            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QUInt8
            )

            print(f"✅ Quantized ONNX (INT8): {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ ONNX quantization failed: {e}")
            return None

    def export_tensorrt(self) -> Optional[Path]:
        """Export model to TensorRT format."""
        print("\n" + "="*60)
        print("STEP 2c: Export to TensorRT")
        print("="*60)

        if self.skip_tensorrt:
            print("⚠️  TensorRT tests skipped (--skip-tensorrt)")
            return None

        # Check if NVIDIA GPU available
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True
            )
            print("✅ NVIDIA GPU detected")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  No NVIDIA GPU detected, skipping TensorRT")
            return None

        tensorrt_dir = self.output_dir / "tensorrt"
        tensorrt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Export using Ultralytics
            export_path = self.yolo_model.export(
                format="engine",
                half=True,  # FP16
                workspace=4,  # 4GB workspace
                device=0
            )

            if export_path and Path(export_path).exists():
                # Move to our directory
                target = tensorrt_dir / Path(export_path).name
                shutil.copy(export_path, target)
                print(f"✅ TensorRT engine: {target}")
                return target
            else:
                print(f"❌ Export failed")
                return None
        except Exception as e:
            print(f"❌ TensorRT export failed: {e}")
            return None

    def export_tensorrt_int8(self, calibration_data: Optional[str] = None) -> Optional[Path]:
        """Export model to TensorRT with INT8 precision."""
        print(f"\n   Exporting TensorRT INT8...")

        if self.skip_tensorrt:
            return None

        tensorrt_dir = self.output_dir / "tensorrt" / "int8"
        tensorrt_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Export with INT8
            export_path = self.yolo_model.export(
                format="engine",
                half=False,
                int8=True,
                data=calibration_data or "coco128.yaml",  # Calibration data
                workspace=4,
                device=0
            )

            if export_path and Path(export_path).exists():
                target = tensorrt_dir / Path(export_path).name
                shutil.copy(export_path, target)
                print(f"✅ TensorRT INT8 engine: {target}")
                return target
            else:
                print(f"❌ Export failed")
                return None
        except Exception as e:
            print(f"❌ TensorRT INT8 export failed: {e}")
            return None

    def test_coreml_runtime(self, model_path: Path, name: str) -> Dict:
        """Test CoreML runtime with a model."""
        print(f"\n   Testing CoreML runtime: {name}...")

        if platform.system() != "Darwin":
            print("⚠️  CoreML only supported on macOS, skipping")
            return {"success": False, "error": "Not on macOS"}

        try:
            from llama_pajamas_run_coreml.backends.vision import CoreMLVisionBackend
            import numpy as np
            from PIL import Image

            backend = CoreMLVisionBackend(model_path=str(model_path))

            # Create test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_image = Image.fromarray(test_image)

            # Run detection
            start_time = time.time()
            results = backend.detect(test_image, confidence=0.25)
            elapsed = time.time() - start_time

            fps = 1.0 / elapsed if elapsed > 0 else 0

            print(f"✅ Runtime test passed")
            print(f"   Detections: {len(results)}")
            print(f"   Speed: {fps:.2f} FPS ({elapsed*1000:.2f}ms)")

            return {
                "success": True,
                "elapsed": elapsed,
                "fps": fps,
                "detections": len(results)
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
        """Test ONNX runtime with a model."""
        print(f"\n   Testing ONNX runtime: {name}...")

        try:
            import onnxruntime as ort
            import numpy as np

            # Create session
            session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )

            # Get input shape
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape

            # Create test input
            if isinstance(input_shape[0], str):
                input_shape = [1, 3, 640, 640]  # Default YOLO shape

            test_input = np.random.randn(*input_shape).astype(np.float32)

            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: test_input})
            elapsed = time.time() - start_time

            fps = 1.0 / elapsed if elapsed > 0 else 0

            print(f"✅ Runtime test passed")
            print(f"   Output shapes: {[o.shape for o in outputs]}")
            print(f"   Speed: {fps:.2f} FPS ({elapsed*1000:.2f}ms)")

            return {
                "success": True,
                "elapsed": elapsed,
                "fps": fps,
                "output_shapes": [list(o.shape) for o in outputs]
            }
        except Exception as e:
            print(f"❌ Runtime test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def test_tensorrt_runtime(self, model_path: Path, name: str) -> Dict:
        """Test TensorRT runtime with a model."""
        print(f"\n   Testing TensorRT runtime: {name}...")

        if self.skip_tensorrt:
            return {"success": False, "error": "TensorRT skipped"}

        try:
            from llama_pajamas_run_tensorrt.backends.vision import TensorRTVisionBackend
            import numpy as np
            from PIL import Image

            backend = TensorRTVisionBackend(model_path=str(model_path))

            # Create test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_image = Image.fromarray(test_image)

            # Run detection
            start_time = time.time()
            results = backend.detect(test_image, confidence=0.25)
            elapsed = time.time() - start_time

            fps = 1.0 / elapsed if elapsed > 0 else 0

            print(f"✅ Runtime test passed")
            print(f"   Detections: {len(results)}")
            print(f"   Speed: {fps:.2f} FPS ({elapsed*1000:.2f}ms)")

            return {
                "success": True,
                "elapsed": elapsed,
                "fps": fps,
                "detections": len(results)
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
        """Run evaluation on a vision model."""
        print(f"\n   Running evaluation...")

        model_dir = model_path.parent

        cmd = [
            "python", "-m", "llama_pajamas_quant.cli.main",
            "evaluate", "vision",
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
                print(f"   mAP: {eval_data.get('mAP', 'N/A')}")
                return {
                    "success": True,
                    "evaluation": eval_data
                }

        return result

    def run_full_test(self):
        """Run the complete end-to-end test."""
        print("\n" + "🚀 " * 30)
        print("STARTING VISION END-TO-END RUNTIME TEST")
        print("🚀 " * 30)

        # Step 1: Download model
        if not self.download_model():
            print("\n❌ Test failed at download step")
            return False

        # Step 2: Export to formats
        coreml_model = self.export_coreml()
        onnx_model = self.export_onnx()
        tensorrt_model = self.export_tensorrt()

        if not any([coreml_model, onnx_model, tensorrt_model]):
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
            int8_model = self.quantize_coreml(coreml_model, "int8")
            if int8_model:
                runtime_result = self.test_coreml_runtime(int8_model, "INT8")
                self.results["tests"].append({
                    "format": "CoreML",
                    "precision": "INT8",
                    "model_path": str(int8_model),
                    "runtime": runtime_result
                })

            # Test INT4
            print(f"\n--- Testing CoreML INT4 ---")
            int4_model = self.quantize_coreml(coreml_model, "int4")
            if int4_model:
                runtime_result = self.test_coreml_runtime(int4_model, "INT4")
                self.results["tests"].append({
                    "format": "CoreML",
                    "precision": "INT4",
                    "model_path": str(int4_model),
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

        # Step 5: TensorRT Tests
        if tensorrt_model:
            print("\n" + "="*60)
            print("STEP 5: TensorRT Runtime Tests")
            print("="*60)

            # Test FP16
            print(f"\n--- Testing TensorRT FP16 ---")
            runtime_result = self.test_tensorrt_runtime(tensorrt_model, "FP16")
            self.results["tests"].append({
                "format": "TensorRT",
                "precision": "FP16",
                "model_path": str(tensorrt_model),
                "runtime": runtime_result
            })

            # Test INT8
            print(f"\n--- Testing TensorRT INT8 ---")
            int8_model = self.export_tensorrt_int8()
            if int8_model:
                runtime_result = self.test_tensorrt_runtime(int8_model, "INT8")
                self.results["tests"].append({
                    "format": "TensorRT",
                    "precision": "INT8",
                    "model_path": str(int8_model),
                    "runtime": runtime_result
                })

        # Step 6: Save results
        self.save_results()

        # Step 7: Print summary
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
            fps = test.get("runtime", {}).get("fps", 0)
            print(f"  {status} {test['format']:10s} {test['precision']:6s} - {fps:.2f} FPS")

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

    parser = argparse.ArgumentParser(description="Vision E2E Runtime Test")
    parser.add_argument(
        "--model",
        default="yolov8n",
        help="Model to test (yolov8n, yolov8s, etc.)"
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
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT tests (requires NVIDIA GPU)"
    )

    args = parser.parse_args()

    test = VisionRuntimeE2ETest(
        model_name=args.model,
        output_dir=args.output_dir,
        cleanup=not args.no_cleanup,
        skip_tensorrt=args.skip_tensorrt
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
