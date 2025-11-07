#!/usr/bin/env python3
"""Test GGUF conversion with Qwen3-8B."""

import argparse
import logging
from pathlib import Path
from llama_pajamas_quant.core import ArchitectureDetector, ManifestGenerator
from llama_pajamas_quant.converters import GGUFConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test GGUF conversion pipeline."""
    parser = argparse.ArgumentParser(description="Test GGUF conversion")
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
        "--precision",
        default="Q4_K_M",
        help="Quantization precision (default: Q4_K_M)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download (use cached)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Llama-Pajamas GGUF Conversion Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Precision: {args.precision}")
    print("=" * 70)
    print()

    # Step 1: Detect architecture
    print("[1/4] Detecting model architecture...")
    detector = ArchitectureDetector()
    arch = detector.detect(args.model)
    print(arch)
    print()

    # Get quantization strategy
    strategy = arch.recommend_quantization()
    print(f"Recommended GGUF method: {strategy.get('gguf_method')}")
    print(f"Special handling: {', '.join(strategy.get('special_handling', []))}")
    print()

    # Step 2: Convert to GGUF
    print("[2/4] Converting to GGUF format...")
    print(f"This will download the model from HuggingFace if not cached.")
    print(f"Expected output size: ~1.9GB for Q4_K_M")
    print()

    converter = GGUFConverter()
    output_dir = Path(args.output)

    try:
        result = converter.convert(
            model_path=args.model,
            output_dir=output_dir,
            precision=args.precision,
            architecture_info=arch.to_dict(),
        )

        print()
        print("✅ GGUF conversion successful!")
        print(f"  File: {result['gguf_path']}")
        print(f"  Size: {result['size_gb']:.2f} GB")
        print(f"  Precision: {result['precision']}")
        print()

    except Exception as e:
        print(f"❌ GGUF conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Generate manifest
    print("[3/4] Generating manifest.json...")
    manifest_gen = ManifestGenerator()

    manifest = manifest_gen.generate(
        model_id=args.model,
        architecture_info=arch.to_dict(),
        formats=[result['metadata']],
        output_path=output_dir / "manifest.json",
    )

    print(f"✅ Manifest generated: {output_dir / 'manifest.json'}")
    print()

    # Step 4: Validation
    print("[4/4] Validation...")

    # Check file exists
    gguf_path = Path(result['gguf_path'])
    if not gguf_path.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return 1
    print(f"✅ GGUF file exists: {gguf_path}")

    # Check size is reasonable
    size_gb = result['size_gb']
    if size_gb < 1.5 or size_gb > 2.5:
        print(f"⚠️  Size outside expected range (1.5-2.5 GB): {size_gb:.2f} GB")
    else:
        print(f"✅ Size within expected range: {size_gb:.2f} GB")

    # Check manifest
    if manifest_gen.validate(manifest):
        print("✅ Manifest is valid")
    else:
        print("❌ Manifest validation failed")
        return 1

    print()
    print("=" * 70)
    print("🎉 All checks passed! GGUF conversion successful.")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Test loading with llama-cpp-python")
    print("2. Run inference to verify quality")
    print("3. Benchmark performance")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
