"""IQ (Importance Quantization) commands."""

import argparse
import logging
from pathlib import Path

from ...quantizers.imatrix import IMatrixQuantizer
from ..utils import validate_path, print_section, format_size

logger = logging.getLogger(__name__)


def register_command(subparsers):
    """Register IQ command."""
    iq_parser = subparsers.add_parser(
        'iq',
        help='Importance quantization (IQ)',
        description='Importance quantization for optimal quality at low bit rates'
    )
    iq_subparsers = iq_parser.add_subparsers(dest='iq_command')

    # Generate imatrix
    gen_parser = iq_subparsers.add_parser(
        'generate-matrix',
        help='Generate importance matrix',
        description='Generate importance matrix from calibration data'
    )
    gen_parser.add_argument('--model', required=True, type=Path, help='Model path (GGUF, F16/Q4+)')
    gen_parser.add_argument('--calibration', required=True, type=Path, help='Calibration file (.txt)')
    gen_parser.add_argument('--output', required=True, type=Path, help='Output imatrix file')
    gen_parser.add_argument('--n-ctx', type=int, default=2048, help='Context length (default: 2048)')
    gen_parser.add_argument('--n-chunks', type=int, default=100, help='Number of chunks (default: 100)')
    gen_parser.set_defaults(func=generate_matrix)

    # Quantize with imatrix
    quant_parser = iq_subparsers.add_parser(
        'quantize',
        help='Quantize with importance matrix',
        description='Quantize model using importance matrix for optimal quality'
    )
    quant_parser.add_argument('--model', required=True, type=Path, help='Model path (GGUF, F16/Q4+)')
    quant_parser.add_argument('--output', required=True, type=Path, help='Output directory')
    quant_parser.add_argument(
        '--precision',
        default='IQ2_XS',
        choices=['IQ1_S', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ2_M', 'IQ3_XXS', 'IQ3_XS', 'IQ3_S', 'IQ3_M', 'IQ4_XS', 'IQ4_NL'],
        help='Precision (default: IQ2_XS)'
    )
    quant_parser.add_argument('--imatrix', type=Path, help='Existing imatrix file (skip generation)')
    quant_parser.add_argument('--calibration', type=Path, help='Calibration file (if generating imatrix)')
    quant_parser.add_argument('--n-ctx', type=int, default=2048, help='Context for imatrix generation')
    quant_parser.add_argument('--n-chunks', type=int, default=100, help='Chunks for imatrix generation')
    quant_parser.add_argument('--pure', action='store_true', help='Pure quantization (no mixed precision)')
    quant_parser.add_argument(
        '--quantize-output-tensor',
        action='store_true',
        help='Also quantize output tensor (smaller but lower quality)'
    )
    quant_parser.set_defaults(func=quantize_iq)

    # Generate calibration data
    cal_parser = iq_subparsers.add_parser(
        'generate-calibration',
        help='Generate calibration data',
        description='Generate calibration.txt from test prompts for imatrix generation'
    )
    cal_parser.add_argument('--output', type=Path, default=Path('calibration.txt'), help='Output file (default: calibration.txt)')
    cal_parser.add_argument('--num-samples', type=int, default=512, help='Number of samples (default: 512)')
    cal_parser.set_defaults(func=generate_calibration)

    # Run binary directly (advanced)
    run_parser = iq_subparsers.add_parser(
        'run-binary',
        help='Run llama.cpp binary directly (advanced)',
        description='Direct access to llama-imatrix or llama-quantize binaries'
    )
    run_parser.add_argument(
        'binary',
        choices=['llama-imatrix', 'llama-quantize'],
        help='Binary to run'
    )
    run_parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments to pass to binary')
    run_parser.set_defaults(func=run_binary)


def generate_matrix(args):
    """Generate importance matrix."""
    print_section("Generating Importance Matrix")

    # Validate inputs
    model_path = validate_path(args.model)
    calibration_file = validate_path(args.calibration)
    output_path = Path(args.output)

    # Initialize quantizer
    quantizer = IMatrixQuantizer()

    # Generate imatrix
    try:
        imatrix_file = quantizer.generate_imatrix(
            model_path=model_path,
            calibration_file=calibration_file,
            output_path=output_path,
            n_ctx=args.n_ctx,
            n_chunks=args.n_chunks,
        )

        print_section("Success")
        print(f"✅ Generated imatrix: {imatrix_file}")
        print(f"   Size: {format_size(imatrix_file.stat().st_size)}")
        print()
        print("You can now use this imatrix to quantize with different precisions:")
        print(f"  llama-pajamas-quant iq quantize --model {model_path} --imatrix {imatrix_file} --precision IQ2_XS")
        print(f"  llama-pajamas-quant iq quantize --model {model_path} --imatrix {imatrix_file} --precision IQ3_XS")

        return 0
    except Exception as e:
        logger.error(f"Failed to generate imatrix: {e}")
        return 1


def quantize_iq(args):
    """Quantize with importance matrix."""
    print_section(f"IQ Quantization ({args.precision})")

    # Validate inputs
    model_path = validate_path(args.model)
    output_dir = Path(args.output)

    # Initialize quantizer
    quantizer = IMatrixQuantizer()

    try:
        if args.imatrix:
            # Use existing imatrix
            print(f"Using existing imatrix: {args.imatrix}")
            imatrix_file = validate_path(args.imatrix)

            output_model = output_dir / f"model-{args.precision.lower()}.gguf"

            result = quantizer.quantize_with_imatrix(
                input_model=model_path,
                output_model=output_model,
                precision=args.precision,
                imatrix_file=imatrix_file,
                leave_output_tensor=not args.quantize_output_tensor,
                pure=args.pure,
            )

            print_section("Success")
            print(f"✅ Quantized to {args.precision}: {result}")
            print(f"   Size: {format_size(result.stat().st_size)}")

        else:
            # Full workflow: generate imatrix → quantize
            if not args.calibration:
                logger.error("Error: --calibration required if --imatrix not provided")
                return 1

            calibration_file = validate_path(args.calibration)

            result = quantizer.quantize_full_workflow(
                model_path=model_path,
                calibration_file=calibration_file,
                precision=args.precision,
                output_dir=output_dir,
                n_ctx=args.n_ctx,
                n_chunks=args.n_chunks,
                leave_output_tensor=not args.quantize_output_tensor,
                pure=args.pure,
            )

            print_section("Success")
            print(f"✅ Quantized to {args.precision}")
            print(f"   Model: {result['output_model']}")
            print(f"   IMatrix: {result['imatrix_file']}")
            print(f"   Size: {format_size(Path(result['output_model']).stat().st_size)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to quantize: {e}")
        return 1


def generate_calibration(args):
    """Generate calibration data."""
    print_section("Generating Calibration Data")

    from ...simple_benchmarks import TEST_PROMPTS

    output_path = Path(args.output)
    num_samples = min(args.num_samples, len(TEST_PROMPTS))

    print(f"Using {num_samples} samples from test prompts")
    print(f"Output: {output_path}")

    # Generate calibration text from test prompts
    calibration_text = []
    for i, prompt_data in enumerate(TEST_PROMPTS[:num_samples]):
        calibration_text.append(prompt_data['prompt'])

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n\n'.join(calibration_text))

    print_section("Success")
    print(f"✅ Generated calibration data: {output_path}")
    print(f"   Samples: {num_samples}")
    print(f"   Size: {format_size(output_path.stat().st_size)}")
    print()
    print("Next step: Generate importance matrix")
    print(f"  llama-pajamas-quant iq generate-matrix --model model.gguf --calibration {output_path} --output output.imatrix")

    return 0


def run_binary(args):
    """Run llama.cpp binary directly."""
    from ...tools.binary_wrapper import BinaryWrapper

    wrapper = BinaryWrapper()

    print(f"Running: {args.binary} {' '.join(args.args)}")

    try:
        wrapper.run(args.binary, args.args, check=True, capture_output=False)
        return 0
    except Exception as e:
        logger.error(f"Failed to run binary: {e}")
        return 1
