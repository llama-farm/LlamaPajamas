#!/usr/bin/env python3
"""Main CLI entry point for llama-pajamas-quant."""

import argparse
import sys
from pathlib import Path

from .utils import setup_logging
from .commands import quantize, iq, hardware, export, evaluate, batch, calibration


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='llama-pajamas-quant',
        description='Llama-Pajamas Quantization & Export Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize LLM to GGUF Q4_K_M
  llama-pajamas-quant quantize llm --model Qwen/Qwen3-8B --formats gguf --output ./models/

  # IQ quantization (extreme compression)
  llama-pajamas-quant iq quantize --model model.gguf --calibration data.txt --precision IQ2_XS

  # Generate domain-specific calibration data
  llama-pajamas-quant calibration generate --domain medical --output ./calibration_data --num-samples 200

  # Detect hardware
  llama-pajamas-quant hardware detect

  # Batch processing
  llama-pajamas-quant batch --config batch-config.yaml

For more help on a command, run:
  llama-pajamas-quant <command> --help
        """
    )

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (errors only)')
    parser.add_argument('--config', type=Path, help='Config file (YAML/JSON)')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register subcommands
    quantize.register_command(subparsers)
    iq.register_command(subparsers)
    calibration.add_calibration_parser(subparsers)
    hardware.register_command(subparsers)
    export.register_command(subparsers)
    evaluate.register_command(subparsers)
    batch.register_command(subparsers)

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        if args.verbose:
            raise
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
