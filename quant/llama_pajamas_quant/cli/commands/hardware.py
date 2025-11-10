"""Hardware detection commands."""

import argparse
import json
import logging
from pathlib import Path

from ...core.hardware import HardwareDetector
from ...core.runtime_config import RuntimeConfigGenerator
from ..utils import print_section

logger = logging.getLogger(__name__)


def register_command(subparsers):
    """Register hardware command."""
    hw_parser = subparsers.add_parser('hardware', help='Hardware detection and configuration')
    hw_subparsers = hw_parser.add_subparsers(dest='hw_command')

    # Detect hardware
    detect_parser = hw_subparsers.add_parser('detect', help='Detect system hardware')
    detect_parser.add_argument('--json', action='store_true', help='Output as JSON')
    detect_parser.set_defaults(func=detect_hardware)

    # Generate config
    config_parser = hw_subparsers.add_parser('config', help='Generate runtime configuration')
    config_parser.add_argument('--model-size', choices=['7-8B', '13B', '30B+'], default='7-8B', help='Model size category')
    config_parser.add_argument('--use-case', choices=['general', 'long_context', 'speed', 'quality'], default='general', help='Use case')
    config_parser.add_argument('--output', type=Path, help='Output file (default: runtime-config.json)')
    config_parser.set_defaults(func=generate_config)


def detect_hardware(args):
    """Detect system hardware."""
    print_section("Hardware Detection")

    detector = HardwareDetector()
    hardware = detector.detect()

    if args.json:
        from dataclasses import asdict
        print(json.dumps(asdict(hardware), indent=2))
    else:
        print(f"Platform: {hardware.display_name}")
        print(f"OS: {hardware.os_type}")
        print(f"CPU: {hardware.cpu_model}")
        print(f"  Physical cores: {hardware.cpu_cores_physical}")
        print(f"  Performance cores: {hardware.cpu_cores_performance}")
        if hardware.cpu_cores_efficiency > 0:
            print(f"  Efficiency cores: {hardware.cpu_cores_efficiency}")
        print(f"Memory: {hardware.ram_gb:.1f} GB RAM")
        if hardware.vram_gb != hardware.ram_gb:
            print(f"  {hardware.vram_gb:.1f} GB VRAM")
        if hardware.gpu_type:
            print(f"GPU: {hardware.gpu_type.upper()}")
            if hardware.gpu_model:
                print(f"  {hardware.gpu_model}")
        print(f"\nRecommended backend: {hardware.recommended_backend}")
        print(f"Capabilities: {', '.join(hardware.capabilities)}")

    return 0


def generate_config(args):
    """Generate runtime configuration."""
    print_section("Generating Runtime Configuration")

    detector = HardwareDetector()
    hardware = detector.detect()

    print(f"Detected: {hardware.display_name}")
    print(f"Model size: {args.model_size}")
    print(f"Use case: {args.use_case}")

    generator = RuntimeConfigGenerator()
    config = generator.generate_config(
        hardware=hardware,
        model_size=args.model_size,
        use_case=args.use_case
    )

    output_path = args.output or Path("runtime-config.json")
    generator.export_json(config, output_path)

    print(f"\n✅ Generated config: {output_path}")
    print(f"   Backend: {config.backend}")
    print(f"   GPU layers: {config.n_gpu_layers}")
    print(f"   Threads: {config.n_threads}")
    print(f"   Context: {config.n_ctx}")

    return 0
