#!/usr/bin/env python3
"""Run lm-eval benchmarks using our runtimes via OpenAI-compatible servers.

This script:
1. Starts MLX and GGUF servers (OpenAI-compatible)
2. Runs lm-eval against them using local-chat-completions backend
3. Stops the servers when done
"""

import subprocess
import time
import sys
import signal
import requests
from pathlib import Path


# Server configurations
MLX_PORT = 8080
GGUF_PORT = 8081


def start_mlx_server(model_path: str) -> subprocess.Popen:
    """Start MLX server (OpenAI-compatible)."""
    print(f"Starting MLX server on port {MLX_PORT}...")

    cmd = [
        "mlx_lm.server",
        "--model", model_path,
        "--port", str(MLX_PORT),
        "--host", "127.0.0.1"
    ]

    process = subprocess.Popen(
        ["python", "-m"] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    for i in range(30):
        try:
            response = requests.get(f"http://127.0.0.1:{MLX_PORT}/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ MLX server ready on port {MLX_PORT}")
                return process
        except:
            time.sleep(1)

    raise RuntimeError("MLX server failed to start")


def start_gguf_server(model_path: str) -> subprocess.Popen:
    """Start GGUF server (OpenAI-compatible via llama-cpp-python)."""
    print(f"Starting GGUF server on port {GGUF_PORT}...")

    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--port", str(GGUF_PORT),
        "--host", "127.0.0.1",
        "--n_gpu_layers", "-1"  # Use all GPU layers
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    for i in range(30):
        try:
            response = requests.get(f"http://127.0.0.1:{GGUF_PORT}/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ GGUF server ready on port {GGUF_PORT}")
                return process
        except:
            time.sleep(1)

    raise RuntimeError("GGUF server failed to start")


def run_lm_eval(base_url: str, output_dir: str, model_name: str):
    """Run lm-eval against a server."""
    print(f"\nRunning lm-eval benchmarks for {model_name}...")
    print(f"Server: {base_url}")
    print()

    cmd = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", f"base_url={base_url},num_concurrent=1",
        "--tasks", "mmlu_abstract_algebra,hellaswag,arc_easy,truthfulqa_mc2,gsm8k",
        "--apply_chat_template",  # Required for chat-completions endpoint
        "--num_fewshot", "5",
        "--limit", "100",
        "--output_path", output_dir,
        "--log_samples"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {model_name} benchmarks complete!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_name} benchmarks failed: {e}")
        return 1


def stop_server(process: subprocess.Popen, name: str):
    """Stop a server process."""
    print(f"Stopping {name} server...")
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
        print(f"✓ {name} server stopped")
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"✓ {name} server killed")


def main():
    """Run benchmarks with servers."""
    print()
    print("="*80)
    print("lm-eval Benchmarks with Our Runtimes")
    print("="*80)
    print()
    print("This will:")
    print("  1. Start MLX server (OpenAI-compatible)")
    print("  2. Run lm-eval benchmarks on MLX")
    print("  3. Stop MLX server")
    print("  4. Start GGUF server (OpenAI-compatible)")
    print("  5. Run lm-eval benchmarks on GGUF")
    print("  6. Stop GGUF server")
    print()
    print("Benchmarks:")
    print("  - MMLU, HellaSwag, ARC-Easy, TruthfulQA, GSM8K")
    print("  - 100 samples per task")
    print("  - 5-shot examples")
    print()
    print("Estimated time: 20-30 minutes total")
    print()

    mlx_path = "models/qwen3-8b/mlx"
    gguf_path = "models/qwen3-8b/gguf/b968826d9c46dd6066d109eabc6255188de91218_q4_k_m.gguf"
    output_dir = "models/qwen3-8b"

    mlx_server = None
    gguf_server = None

    try:
        # MLX benchmarks
        print("="*80)
        print("[1/2] MLX Model")
        print("="*80)
        print()

        mlx_server = start_mlx_server(mlx_path)
        mlx_result = run_lm_eval(
            f"http://127.0.0.1:{MLX_PORT}/v1",
            f"{output_dir}/lm_eval_mlx",
            "MLX"
        )
        stop_server(mlx_server, "MLX")
        mlx_server = None

        # GGUF benchmarks
        print()
        print("="*80)
        print("[2/2] GGUF Model")
        print("="*80)
        print()

        gguf_server = start_gguf_server(gguf_path)
        gguf_result = run_lm_eval(
            f"http://127.0.0.1:{GGUF_PORT}/v1",
            f"{output_dir}/lm_eval_gguf",
            "GGUF"
        )
        stop_server(gguf_server, "GGUF")
        gguf_server = None

        # Compare if both succeeded
        if mlx_result == 0 and gguf_result == 0:
            print()
            print("="*80)
            print("✅ All benchmarks complete!")
            print("="*80)
            print()
            print("Results saved to:")
            print(f"  - {output_dir}/lm_eval_mlx/results.json")
            print(f"  - {output_dir}/lm_eval_gguf/results.json")
            print()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up servers
        if mlx_server:
            stop_server(mlx_server, "MLX")
        if gguf_server:
            stop_server(gguf_server, "GGUF")


if __name__ == "__main__":
    sys.exit(main())
