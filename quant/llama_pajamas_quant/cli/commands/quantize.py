"""Quantization commands."""

import argparse
import logging
from pathlib import Path

from ...core import Quantizer
from ..utils import print_section, format_size

logger = logging.getLogger(__name__)


def register_command(subparsers):
    """Register quantize command."""
    quant_parser = subparsers.add_parser('quantize', help='Quantize models')
    quant_subparsers = quant_parser.add_subparsers(dest='model_type')

    # LLM quantization
    llm_parser = quant_subparsers.add_parser('llm', help='Quantize LLM')
    llm_parser.add_argument('--model', required=True, help='Model ID or path')
    llm_parser.add_argument('--formats', default='gguf,mlx', help='Formats (comma-separated)')
    llm_parser.add_argument('--gguf-precision', default='Q4_K_M', help='GGUF precision (default: Q4_K_M)')
    llm_parser.add_argument('--mlx-bits', type=int, default=4, help='MLX bits (default: 4)')
    llm_parser.add_argument('--output', required=True, type=Path, help='Output directory')
    llm_parser.add_argument('--no-benchmark', action='store_true', help='Skip benchmarking')
    llm_parser.set_defaults(func=quantize_llm)

    # Vision quantization (placeholder)
    vision_parser = quant_subparsers.add_parser('vision', help='Quantize vision model')
    vision_parser.add_argument('--model', required=True, help='Model name')
    vision_parser.add_argument('--precision', required=True, choices=['int8', 'int4', 'fp16'])
    vision_parser.add_argument('--output', required=True, type=Path, help='Output directory')
    vision_parser.set_defaults(func=quantize_vision)

    # Speech quantization (placeholder)
    speech_parser = quant_subparsers.add_parser('speech', help='Quantize speech model')
    speech_parser.add_argument('--model', required=True, help='Model name')
    speech_parser.add_argument('--precision', required=True, choices=['int8', 'fp16'])
    speech_parser.add_argument('--output', required=True, type=Path, help='Output directory')
    speech_parser.set_defaults(func=quantize_speech)


def quantize_llm(args):
    """Quantize LLM."""
    print_section(f"Quantizing LLM: {args.model}")

    formats = [f.strip() for f in args.formats.split(',')]

    print(f"Formats: {', '.join(formats)}")
    print(f"GGUF precision: {args.gguf_precision}")
    print(f"MLX bits: {args.mlx_bits}")
    print(f"Output: {args.output}")

    quantizer = Quantizer()

    try:
        result = quantizer.convert(
            model_path=args.model,
            output_dir=str(args.output),
            formats=formats,
            gguf_precision=args.gguf_precision,
            mlx_bits=args.mlx_bits,
            mlx_mixed_precision=True,
            benchmark=not args.no_benchmark,
        )

        print_section("Success")
        print(f"✅ Quantized {args.model} to {', '.join(formats)}")

        for fmt, fmt_result in result["results"].items():
            print(f"\n{fmt.upper()}:")
            print(f"  Size: {fmt_result['size_gb']:.2f} GB")
            if fmt == "gguf":
                print(f"  File: {fmt_result['gguf_path']}")
            elif fmt == "mlx":
                print(f"  Directory: {fmt_result['mlx_dir']}")

        print(f"\nManifest: {result['output_dir']}/manifest.json")

        return 0
    except Exception as e:
        logger.error(f"Failed to quantize: {e}")
        return 1


def quantize_vision(args):
    """Quantize vision model."""
    logger.error("Vision quantization not yet implemented in CLI")
    logger.info("Use: llama-pajamas-quant export --backend coreml --precision int8")
    return 1


def quantize_speech(args):
    """Quantize speech model."""
    logger.error("Speech quantization not yet implemented in CLI")
    logger.info("Use: llama-pajamas-quant export --backend coreml --precision int8")
    return 1
