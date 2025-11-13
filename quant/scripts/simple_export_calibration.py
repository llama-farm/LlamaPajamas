#!/usr/bin/env python3
"""Simple export of domain calibration data (no dependencies)."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct imports
try:
    from llama_pajamas_quant.calibration.tool_calling import save_tool_calling_calibration
    from llama_pajamas_quant.calibration.summarization import save_summarization_calibration
    from llama_pajamas_quant.calibration.rag import save_rag_calibration
    from llama_pajamas_quant.calibration.military import save_military_calibration
    from llama_pajamas_quant.calibration.medical import save_medical_calibration
    from llama_pajamas_quant.calibration.tone_analysis import save_tone_analysis_calibration

    # Create output directory
    output_dir = Path(__file__).parent.parent / "calibration_data" / "domains"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting domain calibration data...")

    domains = [
        ("tool_calling", save_tool_calling_calibration),
        ("summarization", save_summarization_calibration),
        ("rag", save_rag_calibration),
        ("military", save_military_calibration),
        ("medical", save_medical_calibration),
        ("tone_analysis", save_tone_analysis_calibration),
    ]

    for name, save_func in domains:
        filepath = output_dir / f"calibration_{name}.txt"
        save_func(filepath)

    print(f"\nAll files saved to: {output_dir}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
