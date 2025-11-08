#!/usr/bin/env python3
"""Quantize Qwen3-8B to 3-bit for both GGUF and MLX with benchmarking."""

import logging
from llama_pajamas_quant import Quantizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    """Run 3-bit quantization with benchmarks."""
    quantizer = Quantizer()

    result = quantizer.convert(
        model_path="Qwen/Qwen2.5-3B",  # Using 3B model for speed
        output_dir="./models/qwen3-3b-3bit",
        formats=["gguf", "mlx"],
        gguf_precision="Q3_K_M",  # 3-bit GGUF
        mlx_bits=3,               # 3-bit MLX
        mlx_mixed_precision=True,
        benchmark=True,           # Auto-benchmark!
    )

    print("\n" + "="*80)
    print("QUANTIZATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {result['output_dir']}")
    print(f"\nGGUF: {result['results']['gguf']['size_gb']:.2f} GB")
    print(f"MLX:  {result['results']['mlx']['size_gb']:.2f} GB")

    if result['benchmarks']:
        print("\nBenchmark Results:")
        if 'gguf' in result['benchmarks']:
            gguf_bench = result['benchmarks']['gguf']
            print(f"  GGUF: {gguf_bench['accuracy']:.1%} accuracy @ {gguf_bench['avg_time']:.2f}s/question")
        if 'mlx' in result['benchmarks']:
            mlx_bench = result['benchmarks']['mlx']
            print(f"  MLX:  {mlx_bench['accuracy']:.1%} accuracy @ {mlx_bench['avg_time']:.2f}s/question")

    print("="*80)

if __name__ == "__main__":
    main()
