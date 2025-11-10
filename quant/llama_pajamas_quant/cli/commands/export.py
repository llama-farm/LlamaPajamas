"""Export commands (wrapper for unified exporter)."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_command(subparsers):
    """Register export command."""
    export_parser = subparsers.add_parser(
        'export',
        help='Export models to different backends',
        description='Unified model export (ONNX, CoreML, TensorRT, MLX)'
    )
    export_parser.add_argument('--model', required=True, help='Model name or path')
    export_parser.add_argument('--backend', required=True, choices=['onnx', 'coreml', 'tensorrt', 'mlx'], help='Export backend')
    export_parser.add_argument('--precision', required=True, help='Precision (fp32, fp16, int8, int4)')
    export_parser.add_argument('--output', required=True, type=Path, help='Output directory')
    export_parser.add_argument('--model-type', choices=['auto', 'vision', 'speech', 'llm'], default='auto', help='Model type')
    export_parser.set_defaults(func=export_model)


def export_model(args):
    """Export model (wrapper for lp-export)."""
    from ...exporters.unified import main as export_main
    import sys

    # Build args for unified exporter
    export_args = [
        '--model', args.model,
        '--backend', args.backend,
        '--precision', args.precision,
        '--output', str(args.output),
    ]

    if args.model_type != 'auto':
        export_args.extend(['--model-type', args.model_type])

    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = ['lp-export'] + export_args

    try:
        result = export_main()
        return result if result is not None else 0
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = old_argv
