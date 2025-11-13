#!/usr/bin/env python3
"""
Main E2E Test Runner
Orchestrates all end-to-end runtime tests for LLM, Vision, and Speech models.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import platform


class E2ETestRunner:
    """Orchestrates all end-to-end tests."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        cleanup: bool = True,
        skip_tensorrt: bool = False,
        skip_mlx: bool = False,
        skip_coreml: bool = False
    ):
        """
        Initialize test runner.

        Args:
            output_dir: Output directory for all tests
            cleanup: Whether to cleanup after tests
            skip_tensorrt: Skip TensorRT tests
            skip_mlx: Skip MLX tests
            skip_coreml: Skip CoreML tests
        """
        self.cleanup = cleanup
        self.skip_tensorrt = skip_tensorrt
        self.skip_mlx = skip_mlx or (platform.system() != "Darwin")
        self.skip_coreml = skip_coreml or (platform.system() != "Darwin")

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("e2e_test_results") / time.strftime("%Y%m%d_%H%M%S")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.system(),
            "tests": {
                "llm": None,
                "vision": None,
                "speech": None
            },
            "summary": {}
        }

        print("=" * 70)
        print("🚀 LlamaPajamas End-to-End Runtime Test Suite")
        print("=" * 70)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Platform: {platform.system()}")
        print()

        if self.skip_mlx:
            print("⚠️  MLX tests will be skipped (not on macOS)")
        if self.skip_coreml:
            print("⚠️  CoreML tests will be skipped (not on macOS)")
        if self.skip_tensorrt:
            print("⚠️  TensorRT tests will be skipped")
        print()

    def run_test(self, test_name: str, test_script: str, args: List[str]) -> Dict:
        """Run a single test script."""
        print("\n" + "=" * 70)
        print(f"Running {test_name.upper()} Tests")
        print("=" * 70)

        test_output_dir = self.output_dir / test_name
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            test_script,
            "--output-dir", str(test_output_dir),
        ]

        if not self.cleanup:
            cmd.append("--no-cleanup")

        cmd.extend(args)

        print(f"▶️  Command: {' '.join(cmd)}")
        print()

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                check=True
            )
            elapsed = time.time() - start_time

            # Load results
            results_file = test_output_dir / "e2e_test_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    test_results = json.load(f)
            else:
                test_results = {}

            print(f"\n✅ {test_name.upper()} tests completed ({elapsed:.2f}s)")

            return {
                "success": True,
                "elapsed": elapsed,
                "results": test_results,
                "output_dir": str(test_output_dir)
            }

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {test_name.upper()} tests failed ({elapsed:.2f}s)")

            return {
                "success": False,
                "elapsed": elapsed,
                "error": str(e),
                "output_dir": str(test_output_dir)
            }

    def run_llm_tests(self):
        """Run LLM E2E tests."""
        test_script = str(Path(__file__).parent / "test_e2e_llm.py")

        args = []
        # Use smaller model for faster testing
        args.extend(["--model", "Qwen/Qwen2.5-1.5B-Instruct"])

        result = self.run_test("llm", test_script, args)
        self.results["tests"]["llm"] = result

    def run_vision_tests(self):
        """Run Vision E2E tests."""
        test_script = str(Path(__file__).parent / "test_e2e_vision.py")

        args = []
        args.extend(["--model", "yolov8n"])

        if self.skip_tensorrt:
            args.append("--skip-tensorrt")

        result = self.run_test("vision", test_script, args)
        self.results["tests"]["vision"] = result

    def run_speech_tests(self):
        """Run Speech E2E tests."""
        test_script = str(Path(__file__).parent / "test_e2e_speech.py")

        args = []
        args.extend(["--model", "openai/whisper-tiny"])

        result = self.run_test("speech", test_script, args)
        self.results["tests"]["speech"] = result

    def run_all_tests(self):
        """Run all E2E tests."""
        print("\n" + "🎯 " * 35)
        print("Starting All E2E Tests")
        print("🎯 " * 35 + "\n")

        total_start = time.time()

        # Run LLM tests
        self.run_llm_tests()

        # Run Vision tests
        self.run_vision_tests()

        # Run Speech tests
        self.run_speech_tests()

        total_elapsed = time.time() - total_start

        # Generate summary
        self.generate_summary(total_elapsed)

        # Save results
        self.save_results()

        # Print final summary
        self.print_final_summary()

    def generate_summary(self, total_elapsed: float):
        """Generate test summary."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for test_type, test_result in self.results["tests"].items():
            if not test_result or not test_result.get("success"):
                continue

            test_data = test_result.get("results", {})
            tests = test_data.get("tests", [])

            for test in tests:
                total_tests += 1
                if test.get("runtime", {}).get("success", False):
                    passed_tests += 1
                else:
                    failed_tests += 1

        self.results["summary"] = {
            "total_elapsed": total_elapsed,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }

    def save_results(self):
        """Save all results to JSON."""
        results_file = self.output_dir / "e2e_test_summary.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, indent=2, fp=f)

        print(f"\n💾 Summary saved to: {results_file}")

    def print_final_summary(self):
        """Print final test summary."""
        print("\n" + "=" * 70)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 70)

        summary = self.results["summary"]

        print(f"\nTotal elapsed time: {summary['total_elapsed']:.2f}s")
        print(f"Total tests run: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success rate: {summary['success_rate']:.1f}%")

        print("\n" + "-" * 70)
        print("Test Results by Type:")
        print("-" * 70)

        for test_type, test_result in self.results["tests"].items():
            if not test_result:
                print(f"\n{test_type.upper()}: SKIPPED")
                continue

            status = "✅ PASSED" if test_result.get("success") else "❌ FAILED"
            elapsed = test_result.get("elapsed", 0)
            print(f"\n{test_type.upper()}: {status} ({elapsed:.2f}s)")

            if test_result.get("success"):
                test_data = test_result.get("results", {})
                tests = test_data.get("tests", [])

                if tests:
                    print(f"  Individual tests:")
                    for test in tests:
                        test_status = "✅" if test.get("runtime", {}).get("success", False) else "❌"
                        format_name = test.get("format", "Unknown")
                        precision = test.get("precision", "Unknown")

                        # Get performance metric
                        runtime = test.get("runtime", {})
                        if "tokens_per_sec" in runtime:
                            metric = f"{runtime['tokens_per_sec']:.2f} tokens/s"
                        elif "fps" in runtime:
                            metric = f"{runtime['fps']:.2f} FPS"
                        elif "rtf" in runtime:
                            metric = f"{runtime['rtf']:.2f}x RTF"
                        else:
                            metric = "N/A"

                        print(f"    {test_status} {format_name:10s} {precision:6s} - {metric}")

        print("\n" + "=" * 70)
        print(f"📁 Full results: {self.output_dir}")
        print("=" * 70)

        # Overall status
        overall_success = all(
            test.get("success", False)
            for test in self.results["tests"].values()
            if test is not None
        )

        if overall_success:
            print("\n🎉 All tests PASSED!")
        else:
            print("\n⚠️  Some tests FAILED. Check logs for details.")

        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all LlamaPajamas E2E Runtime Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_e2e_tests.py

  # Run with custom output directory
  python run_e2e_tests.py --output-dir ./my_test_results

  # Skip TensorRT tests (if no NVIDIA GPU)
  python run_e2e_tests.py --skip-tensorrt

  # Skip cleanup to inspect results
  python run_e2e_tests.py --no-cleanup

  # Run specific tests only
  python run_e2e_tests.py --tests llm vision
        """
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for all tests (default: timestamped directory)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup files after tests"
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT tests (requires NVIDIA GPU)"
    )
    parser.add_argument(
        "--skip-mlx",
        action="store_true",
        help="Skip MLX tests (auto-detected on non-macOS)"
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Skip CoreML tests (auto-detected on non-macOS)"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["llm", "vision", "speech"],
        help="Run only specific tests (default: all)"
    )

    args = parser.parse_args()

    runner = E2ETestRunner(
        output_dir=args.output_dir,
        cleanup=not args.no_cleanup,
        skip_tensorrt=args.skip_tensorrt,
        skip_mlx=args.skip_mlx,
        skip_coreml=args.skip_coreml
    )

    try:
        if args.tests:
            # Run specific tests
            total_start = time.time()

            if "llm" in args.tests:
                runner.run_llm_tests()
            if "vision" in args.tests:
                runner.run_vision_tests()
            if "speech" in args.tests:
                runner.run_speech_tests()

            total_elapsed = time.time() - total_start
            runner.generate_summary(total_elapsed)
            runner.save_results()
            success = runner.print_final_summary()
        else:
            # Run all tests
            runner.run_all_tests()
            success = runner.results["summary"]["failed"] == 0

        sys.exit(0 if success else 1)

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
