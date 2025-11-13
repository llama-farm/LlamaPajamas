"""CLI commands for calibration data generation."""

import argparse
from pathlib import Path
import sys

from llama_pajamas_quant.calibration import (
    generate_domain_calibration,
    get_military_seed_examples,
    get_military_domain_description,
    get_medical_seed_examples,
    get_medical_domain_description,
    get_tone_analysis_seed_examples,
    get_tone_analysis_domain_description,
    save_military_calibration,
    save_medical_calibration,
    save_tone_analysis_calibration,
    save_tool_calling_calibration,
    save_summarization_calibration,
    save_rag_calibration,
)


DOMAIN_CONFIGS = {
    "military": {
        "description": get_military_domain_description,
        "examples": get_military_seed_examples,
        "save_func": save_military_calibration,
    },
    "medical": {
        "description": get_medical_domain_description,
        "examples": get_medical_seed_examples,
        "save_func": save_medical_calibration,
    },
    "tone_analysis": {
        "description": get_tone_analysis_domain_description,
        "examples": get_tone_analysis_seed_examples,
        "save_func": save_tone_analysis_calibration,
    },
    "tool_calling": {
        "save_func": save_tool_calling_calibration,
    },
    "summarization": {
        "save_func": save_summarization_calibration,
    },
    "rag": {
        "save_func": save_rag_calibration,
    },
}


def add_calibration_parser(subparsers):
    """Add calibration generation command to the CLI.

    Args:
        subparsers: Argparse subparsers object
    """
    parser = subparsers.add_parser(
        "calibration",
        help="Generate and manage calibration data",
        description="Generate domain-specific calibration data for IQ quantization",
    )

    sub = parser.add_subparsers(dest="calibration_command", required=True)

    # Generate subcommand
    gen_parser = sub.add_parser(
        "generate",
        help="Generate synthetic calibration data",
        description="Generate synthetic calibration data using cloud LLMs",
    )
    gen_parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["military", "medical", "tone_analysis"],
        help="Domain for calibration data generation",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for calibration data",
    )
    gen_parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=200,
        help="Number of samples to generate (default: 200, recommended: 150-300)",
    )
    gen_parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider to use (default: anthropic)",
    )
    gen_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LLM provider (reads from environment if not provided)",
    )
    gen_parser.add_argument(
        "--custom-purpose",
        type=str,
        help="Custom purpose description (overrides default domain description)",
    )
    gen_parser.add_argument(
        "--custom-examples",
        type=str,
        nargs="+",
        help="Custom example prompts (overrides default domain examples)",
    )

    # Export subcommand (export built-in seed data)
    export_parser = sub.add_parser(
        "export",
        help="Export built-in calibration data",
        description="Export built-in seed calibration data to files",
    )
    export_parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=list(DOMAIN_CONFIGS.keys()),
        help="Domain to export",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file path",
    )

    # List subcommand
    list_parser = sub.add_parser(
        "list",
        help="List available calibration domains",
        description="List all available calibration domains and their descriptions",
    )

    parser.set_defaults(func=handle_calibration)


def handle_calibration(args):
    """Handle calibration generation commands.

    Args:
        args: Parsed command-line arguments
    """
    if args.calibration_command == "generate":
        return generate_calibration(args)
    elif args.calibration_command == "export":
        return export_calibration(args)
    elif args.calibration_command == "list":
        return list_domains(args)


def generate_calibration(args):
    """Generate synthetic calibration data.

    Args:
        args: Parsed command-line arguments
    """
    domain = args.domain
    output_dir = args.output
    num_samples = args.num_samples
    provider = args.provider
    api_key = args.api_key

    print(f"\n{'='*70}")
    print(f"Generating Synthetic Calibration Data")
    print(f"{'='*70}")
    print(f"Domain: {domain}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"Provider: {provider}")
    print(f"{'='*70}\n")

    # Get domain config
    if domain not in DOMAIN_CONFIGS:
        print(f"Error: Unknown domain '{domain}'")
        return 1

    config = DOMAIN_CONFIGS[domain]

    # Get purpose and examples
    if args.custom_purpose:
        purpose = args.custom_purpose
        print(f"Using custom purpose description")
    else:
        purpose = config["description"]()
        print(f"Using built-in domain description")

    if args.custom_examples:
        examples = args.custom_examples
        print(f"Using {len(examples)} custom examples")
    else:
        examples = config["examples"]()
        print(f"Using {len(examples)} built-in seed examples")

    print()

    try:
        output_path = generate_domain_calibration(
            domain=domain,
            purpose=purpose,
            examples=examples,
            output_dir=output_dir,
            num_samples=num_samples,
            provider=provider,
            api_key=api_key,
        )

        print(f"\n{'='*70}")
        print(f"✓ Success!")
        print(f"{'='*70}")
        print(f"Generated: {output_path}")
        print(f"Metadata: {output_path.parent / f'{output_path.stem}_metadata.json'}")
        print(f"\nYou can now use this calibration data with:")
        print(f"  llama-pajamas-quant iq quantize --calibration {output_path} ...")
        print(f"{'='*70}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Error generating calibration data")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        print(f"\nMake sure you have:")
        if provider == "anthropic":
            print(f"  - Set ANTHROPIC_API_KEY environment variable or use --api-key")
            print(f"  - Installed anthropic: pip install anthropic")
        elif provider == "openai":
            print(f"  - Set OPENAI_API_KEY environment variable or use --api-key")
            print(f"  - Installed openai: pip install openai")
        print(f"{'='*70}\n")
        return 1


def export_calibration(args):
    """Export built-in calibration data.

    Args:
        args: Parsed command-line arguments
    """
    domain = args.domain
    output_path = args.output

    print(f"\n{'='*70}")
    print(f"Exporting Built-in Calibration Data")
    print(f"{'='*70}")
    print(f"Domain: {domain}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    config = DOMAIN_CONFIGS.get(domain)
    if not config:
        print(f"Error: Unknown domain '{domain}'")
        return 1

    save_func = config["save_func"]

    try:
        save_func(output_path)

        print(f"\n{'='*70}")
        print(f"✓ Success!")
        print(f"{'='*70}")
        print(f"Exported to: {output_path}")
        print(f"\nYou can now use this calibration data with:")
        print(f"  llama-pajamas-quant iq quantize --calibration {output_path} ...")
        print(f"{'='*70}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Error exporting calibration data")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        return 1


def list_domains(args):
    """List available calibration domains.

    Args:
        args: Parsed command-line arguments
    """
    print(f"\n{'='*70}")
    print(f"Available Calibration Domains")
    print(f"{'='*70}\n")

    # Built-in domains with seed data
    print("Built-in Domains (seed data available):")
    print("-" * 70)

    domains_with_seeds = [
        ("tool_calling", "Tool calling, function invocation, API interactions, and structured outputs"),
        ("summarization", "Text summarization, compression, and abstraction tasks"),
        ("rag", "Retrieval-augmented generation with context-question pairs"),
        ("military", "Military, defense, tactical planning, and operations"),
        ("medical", "Medical, healthcare, clinical diagnosis, and treatment"),
        ("tone_analysis", "Sentiment analysis, emotional tone detection, and communication style"),
    ]

    for domain, description in domains_with_seeds:
        print(f"\n{domain}")
        print(f"  {description}")

    # Synthetic generation
    print("\n\n" + "=" * 70)
    print("Synthetic Generation (cloud LLM required):")
    print("-" * 70)
    print("\nGenerate custom calibration data for any domain:")
    print("  llama-pajamas-quant calibration generate \\")
    print("    --domain <domain> \\")
    print("    --output <dir> \\")
    print("    --num-samples 200")

    print("\n\nAvailable for synthetic generation:")
    for domain in ["military", "medical", "tone_analysis"]:
        print(f"  - {domain}")

    print("\n" + "=" * 70)
    print("\nExamples:")
    print("-" * 70)
    print("\n1. Export built-in tool calling data:")
    print("   llama-pajamas-quant calibration export \\")
    print("     --domain tool_calling \\")
    print("     --output ./calibration_data/tool_calling.txt")

    print("\n2. Generate synthetic medical data:")
    print("   llama-pajamas-quant calibration generate \\")
    print("     --domain medical \\")
    print("     --output ./calibration_data \\")
    print("     --num-samples 250 \\")
    print("     --provider anthropic")

    print("\n3. Generate with custom examples:")
    print("   llama-pajamas-quant calibration generate \\")
    print("     --domain tone_analysis \\")
    print("     --output ./calibration_data \\")
    print("     --custom-examples 'Analyze this...' 'Detect tone in...' \\")
    print("     --num-samples 200")

    print("\n" + "=" * 70 + "\n")

    return 0
