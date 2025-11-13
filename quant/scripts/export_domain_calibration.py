#!/usr/bin/env python3
"""Export all domain-specific calibration data to files.

This script exports built-in seed calibration data for all available domains.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_quant.calibration import (
    save_tool_calling_calibration,
    save_summarization_calibration,
    save_rag_calibration,
    save_military_calibration,
    save_medical_calibration,
    save_tone_analysis_calibration,
)


def main():
    """Export all domain calibration data."""
    # Output directory
    output_dir = Path(__file__).parent.parent / "calibration_data" / "domains"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Exporting Domain-Specific Calibration Data")
    print("=" * 70)
    print(f"Output directory: {output_dir}\n")

    # Export each domain
    domains = [
        ("tool_calling", save_tool_calling_calibration),
        ("summarization", save_summarization_calibration),
        ("rag", save_rag_calibration),
        ("military", save_military_calibration),
        ("medical", save_medical_calibration),
        ("tone_analysis", save_tone_analysis_calibration),
    ]

    for domain_name, save_func in domains:
        output_file = output_dir / f"calibration_{domain_name}.txt"
        print(f"\nExporting {domain_name}...")
        try:
            save_func(output_file)
            size_kb = output_file.stat().st_size / 1024
            print(f"  ✓ Saved to {output_file} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Create a combined "general" domain file
    print(f"\nCreating combined 'general' domain...")
    general_file = output_dir / "calibration_general.txt"

    from llama_pajamas_quant.calibration import (
        get_tool_calling_calibration_text,
        get_summarization_calibration_text,
        get_rag_calibration_text,
    )

    combined_text = (
        get_tool_calling_calibration_text() + '\n' +
        get_summarization_calibration_text() + '\n' +
        get_rag_calibration_text()
    )

    with open(general_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    size_kb = general_file.stat().st_size / 1024
    print(f"  ✓ Saved to {general_file} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print("Export Complete!")
    print("=" * 70)
    print(f"\nAll calibration files are in: {output_dir}")
    print("\nYou can use these files with:")
    print("  llama-pajamas-quant iq quantize --calibration <file> ...")
    print("\nOr use the --domain flag:")
    print("  llama-pajamas-quant iq quantize --domain medical ...")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
