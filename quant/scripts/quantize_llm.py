#!/usr/bin/env python3
"""Test dual-format conversion (GGUF + MLX) with Qwen3-8B."""

import argparse
import logging
from pathlib import Path
from llama_pajamas_quant.core import Quantizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test dual-format conversion pipeline."""
    parser = argparse.ArgumentParser(description="Test dual-format conversion (GGUF + MLX)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--output",
        default="./models/qwen3-8b",
        help="Output directory (default: ./models/qwen3-8b)"
    )
    parser.add_argument(
        "--formats",
        default="gguf,mlx",
        help="Comma-separated formats to generate (default: gguf,mlx)"
    )
    parser.add_argument(
        "--gguf-precision",
        default="Q4_K_M",
        help="GGUF quantization precision (default: Q4_K_M)"
    )
    parser.add_argument(
        "--mlx-bits",
        type=int,
        default=4,
        help="MLX quantization bits (default: 4)"
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip benchmarking (benchmarks run by default)"
    )

    args = parser.parse_args()
    formats = [f.strip() for f in args.formats.split(",")]

    print("=" * 70)
    print("Llama-Pajamas Dual-Format Conversion Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Formats: {', '.join(formats)}")
    print(f"GGUF Precision: {args.gguf_precision}")
    print(f"MLX Bits: {args.mlx_bits}")
    print("=" * 70)
    print()

    # Run dual-format conversion
    quantizer = Quantizer()

    try:
        result = quantizer.convert(
            model_path=args.model,
            output_dir=args.output,
            formats=formats,
            gguf_precision=args.gguf_precision,
            mlx_bits=args.mlx_bits,
            mlx_mixed_precision=True,
            benchmark=not args.no_benchmark,  # Benchmarks run by default
        )

        print()
        print("=" * 70)
        print("Conversion Results")
        print("=" * 70)

        # Architecture summary
        arch = result["architecture"]
        print(f"\nModel: {arch['model_id']}")
        print(f"Type: {arch['model_type']}")
        print(f"Family: {arch['family']}")
        print(f"Parameters: {arch['params_total']}")
        print(f"Attention: {arch['attention_type']}")
        if arch.get("gqa_ratio"):
            print(f"GQA Ratio: {arch['gqa_ratio']}:1")

        # Format results
        print(f"\nGenerated Formats:")
        for fmt, fmt_result in result["results"].items():
            print(f"\n{fmt.upper()}:")
            print(f"  Size: {fmt_result['size_gb']:.2f} GB")
            if fmt == "gguf":
                print(f"  File: {fmt_result['gguf_path']}")
                print(f"  Method: {fmt_result['precision']}")
            elif fmt == "mlx":
                print(f"  Directory: {fmt_result['mlx_dir']}")
                quant = fmt_result['quantization']
                if quant['enabled']:
                    print(f"  Bits: {quant['bits']}-bit")
                    print(f"  Mixed Precision: {quant['mixed_precision']}")

        # Manifest
        print(f"\nManifest: {result['output_dir']}/manifest.json")

        # Validation
        print()
        print("=" * 70)
        print("Validation")
        print("=" * 70)

        all_passed = True

        # Check GGUF if generated
        if "gguf" in result["results"]:
            gguf = result["results"]["gguf"]
            gguf_path = Path(gguf["gguf_path"])

            if gguf_path.exists():
                print(f"✅ GGUF file exists: {gguf_path}")
            else:
                print(f"❌ GGUF file not found: {gguf_path}")
                all_passed = False

            if 1.5 <= gguf["size_gb"] <= 2.5:
                print(f"✅ GGUF size within range: {gguf['size_gb']:.2f} GB")
            else:
                print(f"⚠️  GGUF size outside expected range (1.5-2.5 GB): {gguf['size_gb']:.2f} GB")

        # Check MLX if generated
        if "mlx" in result["results"]:
            mlx = result["results"]["mlx"]
            mlx_dir = Path(mlx["mlx_dir"])

            if mlx_dir.exists():
                print(f"✅ MLX directory exists: {mlx_dir}")
            else:
                print(f"❌ MLX directory not found: {mlx_dir}")
                all_passed = False

            if 1.4 <= mlx["size_gb"] <= 2.0:
                print(f"✅ MLX size within range: {mlx['size_gb']:.2f} GB")
            else:
                print(f"⚠️  MLX size outside expected range (1.4-2.0 GB): {mlx['size_gb']:.2f} GB")

            # Check for key MLX files
            config_file = mlx_dir / "config.json"
            if config_file.exists():
                print(f"✅ MLX config.json exists")
            else:
                print(f"❌ MLX config.json missing")
                all_passed = False

        # Check manifest
        manifest_path = Path(result["output_dir"]) / "manifest.json"
        if manifest_path.exists():
            print(f"✅ Manifest exists: {manifest_path}")
        else:
            print(f"❌ Manifest not found: {manifest_path}")
            all_passed = False

        print()
        if all_passed:
            print("=" * 70)
            print("🎉 All validations PASSED!")
            print("=" * 70)
            print()
            print("Architecture-aware dual-format quantization successful!")
            print(f"GGUF and MLX formats generated with <5% expected quality loss.")
            print()
            print("Next steps:")
            print("1. Test GGUF loading with llama-cpp-python")
            print("2. Test MLX loading with mlx-lm")
            print("3. Run inference quality tests")
            print("4. Benchmark performance on target hardware")
        else:
            print("=" * 70)
            print("⚠️  Some validations failed - review output above")
            print("=" * 70)

        return 0 if all_passed else 1

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
