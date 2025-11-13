#!/usr/bin/env python3
"""Direct export by executing calibration module files."""

import sys
from pathlib import Path

# Load calibration modules directly
quant_dir = Path(__file__).parent.parent
calibration_dir = quant_dir / "llama_pajamas_quant" / "calibration"

output_dir = quant_dir / "calibration_data" / "domains"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Exporting Domain Calibration Data")
print("=" * 70)
print(f"Output: {output_dir}\n")

domains = {
    "tool_calling": calibration_dir / "tool_calling.py",
    "summarization": calibration_dir / "summarization.py",
    "rag": calibration_dir / "rag.py",
    "military": calibration_dir / "military.py",
    "medical": calibration_dir / "medical.py",
    "tone_analysis": calibration_dir / "tone_analysis.py",
}

for domain_name, module_path in domains.items():
    print(f"Processing {domain_name}...")

    try:
        # Read the module file
        with open(module_path, 'r') as f:
            module_code = f.read()

        # Execute it in a namespace
        namespace = {}
        exec(module_code, namespace)

        # Get the calibration data
        calibration_var = f"{domain_name.upper().replace('_', '_')}_CALIBRATION"
        if calibration_var not in namespace:
            # Try variations
            if "TOOL_CALLING_CALIBRATION" in namespace:
                calibration_var = "TOOL_CALLING_CALIBRATION"
            elif "SUMMARIZATION_CALIBRATION" in namespace:
                calibration_var = "SUMMARIZATION_CALIBRATION"
            elif "RAG_CALIBRATION" in namespace:
                calibration_var = "RAG_CALIBRATION"
            elif "MILITARY_CALIBRATION" in namespace:
                calibration_var = "MILITARY_CALIBRATION"
            elif "MEDICAL_CALIBRATION" in namespace:
                calibration_var = "MEDICAL_CALIBRATION"
            elif "TONE_ANALYSIS_CALIBRATION" in namespace:
                calibration_var = "TONE_ANALYSIS_CALIBRATION"

        calibration_data = namespace[calibration_var]

        # Write to file
        output_file = output_dir / f"calibration_{domain_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Handle RAG format (list of dicts)
            if domain_name == "rag" and isinstance(calibration_data, list) and len(calibration_data) > 0 and isinstance(calibration_data[0], dict):
                formatted = []
                for i, item in enumerate(calibration_data):
                    formatted.append(
                        f"Context {i+1}:\n{item['context']}\n\n"
                        f"Question: {item['question']}"
                    )
                f.write('\n\n'.join(formatted))
            else:
                f.write('\n'.join(calibration_data))

        size_kb = output_file.stat().st_size / 1024
        print(f"  ✓ {len(calibration_data)} samples -> {output_file.name} ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"  ✗ Error: {e}")

# Create general domain (combined)
print(f"\nCreating general (combined) domain...")
try:
    general_samples = []

    for domain in ["tool_calling", "summarization", "rag"]:
        filepath = output_dir / f"calibration_{domain}.txt"
        if filepath.exists():
            with open(filepath, 'r') as f:
                general_samples.extend(f.read().strip().split('\n'))

    general_file = output_dir / "calibration_general.txt"
    with open(general_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(general_samples))

    size_kb = general_file.stat().st_size / 1024
    print(f"  ✓ {len(general_samples)} samples -> {general_file.name} ({size_kb:.1f} KB)")

except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)
print("Export Complete!")
print("=" * 70)
print(f"\nFiles saved to: {output_dir}")
print("\nUsage:")
print("  llama-pajamas-quant iq quantize --domain medical ...")
print("  llama-pajamas-quant iq quantize --calibration <file> ...")
print("=" * 70 + "\n")
