"""Performance benchmarking for quantized models.

Measures:
- Tokens per second (tok/s)
- Memory usage
- Time to first token (TTFT)
- Overall throughput
"""

import sys
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List
import gc


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_mlx(model_path: str, num_tokens: int = 500) -> Dict[str, Any]:
    """Benchmark MLX model performance."""
    try:
        from mlx_lm import load, generate
        import mlx.core as mx

        print(f"Loading MLX model...")
        gc.collect()
        mem_before = get_memory_usage_mb()

        load_start = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - load_start

        mem_after = get_memory_usage_mb()
        mem_loaded = mem_after - mem_before

        # Warmup
        print("  Warming up...")
        _ = generate(model, tokenizer, "Test warmup", max_tokens=10, verbose=False)

        # Benchmark: Standard generation
        print(f"  Generating {num_tokens} tokens...")
        prompt = "Write a detailed explanation of how neural networks work, including backpropagation and gradient descent:"

        start = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=num_tokens, verbose=False)
        total_time = time.time() - start

        # Count tokens
        response_tokens = len(tokenizer.encode(response))
        tok_per_sec = response_tokens / total_time

        # Memory during generation
        mem_peak = get_memory_usage_mb()

        # Cleanup
        del model, tokenizer
        gc.collect()

        return {
            "backend": "MLX 4-bit",
            "load_time_seconds": load_time,
            "memory_loaded_mb": mem_loaded,
            "memory_peak_mb": mem_peak,
            "tokens_generated": response_tokens,
            "generation_time_seconds": total_time,
            "tokens_per_second": tok_per_sec,
            "prompt_tokens": len(tokenizer.encode(prompt)) if 'tokenizer' in locals() else 0,
        }

    except Exception as e:
        print(f"MLX benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_gguf(model_path: str, num_tokens: int = 500) -> Dict[str, Any]:
    """Benchmark GGUF model performance."""
    try:
        from llama_cpp import Llama

        print(f"Loading GGUF model...")
        gc.collect()
        mem_before = get_memory_usage_mb()

        load_start = time.time()
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # All layers to GPU
            verbose=False
        )
        load_time = time.time() - load_start

        mem_after = get_memory_usage_mb()
        mem_loaded = mem_after - mem_before

        # Warmup
        print("  Warming up...")
        _ = model("Test warmup", max_tokens=10, echo=False)

        # Benchmark: Standard generation
        print(f"  Generating {num_tokens} tokens...")
        prompt = "Write a detailed explanation of how neural networks work, including backpropagation and gradient descent:"

        start = time.time()
        response = model(prompt, max_tokens=num_tokens, temperature=0.7, echo=False)
        total_time = time.time() - start

        # Extract stats
        response_text = response["choices"][0]["text"]
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", num_tokens)

        tok_per_sec = completion_tokens / total_time

        # Memory during generation
        mem_peak = get_memory_usage_mb()

        # Cleanup
        del model
        gc.collect()

        return {
            "backend": "GGUF Q4_K_M",
            "load_time_seconds": load_time,
            "memory_loaded_mb": mem_loaded,
            "memory_peak_mb": mem_peak,
            "tokens_generated": completion_tokens,
            "generation_time_seconds": total_time,
            "tokens_per_second": tok_per_sec,
            "prompt_tokens": prompt_tokens,
        }

    except Exception as e:
        print(f"GGUF benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison(mlx_result: Dict, gguf_result: Dict):
    """Print performance comparison."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)

    # Model Loading
    print("\n📦 Model Loading:")
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Backend':<20} │ {'Load Time':<15} │ {'Memory Used':<20} │ {'Status':<15} │")
    print("├" + "─" * 78 + "┤")

    if mlx_result:
        status = "✅ PASS" if mlx_result["load_time_seconds"] < 10 else "⚠️  SLOW"
        print(f"│ {mlx_result['backend']:<20} │ {mlx_result['load_time_seconds']:>6.2f}s       │ {mlx_result['memory_loaded_mb']:>8.0f} MB          │ {status:<15} │")

    if gguf_result:
        status = "✅ PASS" if gguf_result["load_time_seconds"] < 30 else "⚠️  SLOW"
        print(f"│ {gguf_result['backend']:<20} │ {gguf_result['load_time_seconds']:>6.2f}s       │ {gguf_result['memory_loaded_mb']:>8.0f} MB          │ {status:<15} │")

    print("└" + "─" * 78 + "┘")

    # Throughput
    print("\n⚡ Generation Throughput:")
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Backend':<20} │ {'Tokens/sec':<15} │ {'vs Target':<20} │ {'Status':<15} │")
    print("├" + "─" * 78 + "┤")

    if mlx_result:
        target = 80  # MLX target from plan
        pct_of_target = (mlx_result["tokens_per_second"] / target) * 100
        status = "✅ PASS" if mlx_result["tokens_per_second"] >= target * 0.8 else "⚠️  CHECK"
        print(f"│ {mlx_result['backend']:<20} │ {mlx_result['tokens_per_second']:>6.1f} t/s     │ {pct_of_target:>6.1f}% of {target} t/s  │ {status:<15} │")

    if gguf_result:
        target = 70  # GGUF target from plan
        pct_of_target = (gguf_result["tokens_per_second"] / target) * 100
        status = "✅ PASS" if gguf_result["tokens_per_second"] >= target * 0.8 else "⚠️  CHECK"
        print(f"│ {gguf_result['backend']:<20} │ {gguf_result['tokens_per_second']:>6.1f} t/s     │ {pct_of_target:>6.1f}% of {target} t/s  │ {status:<15} │")

    print("└" + "─" * 78 + "┘")

    # Memory Usage
    print("\n💾 Memory Usage:")
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Backend':<20} │ {'Peak Memory':<15} │ {'vs Target':<20} │ {'Status':<15} │")
    print("├" + "─" * 78 + "┤")

    if mlx_result:
        target_mb = 1700  # ~1.7GB from plan
        status = "✅ PASS" if mlx_result["memory_peak_mb"] < target_mb * 3 else "⚠️  HIGH"
        print(f"│ {mlx_result['backend']:<20} │ {mlx_result['memory_peak_mb']:>8.0f} MB    │ {'Target: ~1700 MB':<20} │ {status:<15} │")

    if gguf_result:
        target_mb = 1900  # ~1.9GB from plan
        status = "✅ PASS" if gguf_result["memory_peak_mb"] < target_mb * 3 else "⚠️  HIGH"
        print(f"│ {gguf_result['backend']:<20} │ {gguf_result['memory_peak_mb']:>8.0f} MB    │ {'Target: ~1900 MB':<20} │ {status:<15} │")

    print("└" + "─" * 78 + "┘")

    # Summary
    print("\n🎯 Overall Assessment:")
    print(f"  ✅ Both backends operational")

    if mlx_result and gguf_result:
        faster = "MLX" if mlx_result["tokens_per_second"] > gguf_result["tokens_per_second"] else "GGUF"
        speedup = max(mlx_result["tokens_per_second"], gguf_result["tokens_per_second"]) / min(mlx_result["tokens_per_second"], gguf_result["tokens_per_second"])
        print(f"  ⚡ {faster} is {speedup:.1f}x faster")

        if mlx_result["tokens_per_second"] >= 64:  # 80% of 80 t/s target
            print(f"  ✅ MLX performance target met ({mlx_result['tokens_per_second']:.1f} t/s)")
        else:
            print(f"  ⚠️  MLX below target ({mlx_result['tokens_per_second']:.1f} t/s vs 80 t/s)")

        if gguf_result["tokens_per_second"] >= 56:  # 80% of 70 t/s target
            print(f"  ✅ GGUF performance target met ({gguf_result['tokens_per_second']:.1f} t/s)")
        else:
            print(f"  ⚠️  GGUF below target ({gguf_result['tokens_per_second']:.1f} t/s vs 70 t/s)")


def main():
    """Run performance benchmarks."""
    print("=" * 80)
    print("Llama-Pajamas Performance Benchmarking")
    print("=" * 80)

    # Find models (auto-discover from subdirectories)
    models_dir = Path(__file__).parent / "models" / "qwen3-8b"

    # Find MLX model (first subdirectory with config.json)
    mlx_subdirs = [d for d in (models_dir / "mlx").iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not mlx_subdirs:
        print(f"❌ No MLX model found in {models_dir / 'mlx'}")
        return 1
    mlx_path = mlx_subdirs[0]

    # Find GGUF model (search in subdirectories)
    gguf_files = list((models_dir / "gguf").glob("**/*.gguf"))
    if not gguf_files:
        print(f"❌ No GGUF model found in {models_dir / 'gguf'}")
        return 1

    gguf_path = str(gguf_files[0])

    print(f"\nModels:")
    print(f"  MLX:  {mlx_path}")
    print(f"  GGUF: {gguf_path}")
    print(f"\nHardware:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    num_tokens = 200  # Shorter for faster benchmarking

    # Benchmark MLX
    print("\n" + "─" * 80)
    print("PHASE 1: MLX Benchmark")
    print("─" * 80)
    mlx_result = benchmark_mlx(str(mlx_path), num_tokens)

    # Benchmark GGUF
    print("\n" + "─" * 80)
    print("PHASE 2: GGUF Benchmark")
    print("─" * 80)
    gguf_result = benchmark_gguf(gguf_path, num_tokens)

    # Compare results
    if mlx_result and gguf_result:
        print_comparison(mlx_result, gguf_result)
    elif mlx_result:
        print("\n⚠️  GGUF benchmark failed, showing MLX results only")
    elif gguf_result:
        print("\n⚠️  MLX benchmark failed, showing GGUF results only")
    else:
        print("\n❌ All benchmarks failed")
        return 1

    print("\n" + "=" * 80)
    print("Benchmarking Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
