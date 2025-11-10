#!/usr/bin/env python3
"""
Update Whisper model manifests to include INT8 quantized formats.
"""

import json
from pathlib import Path

def get_model_size(model_path: Path) -> tuple[int, float]:
    """Get total size of .mlpackage directory."""
    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    return total_size, total_size / (1024 * 1024)

def update_manifest(manifest_path: Path):
    """Add INT8 format entry to manifest."""
    print(f"\n📝 Updating: {manifest_path.parent.name}/manifest.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check if INT8 already exists
    existing_precisions = {fmt['precision'] for fmt in manifest.get('formats', [])}
    if 'int8' in existing_precisions:
        print("   ✅ INT8 format already exists, skipping")
        return

    # Get INT8 encoder path and size
    int8_path = manifest_path.parent / "coreml" / "int8" / "encoder.mlpackage"
    if not int8_path.exists():
        print(f"   ⚠️ INT8 encoder not found: {int8_path}")
        return

    size_bytes, size_mb = get_model_size(int8_path)

    # Create INT8 format entry
    int8_entry = {
        "format": "coreml",
        "component": "encoder",
        "precision": "int8",
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_mb, 1),
        "compatible_backends": ["coreml"],
        "runtime_requirements": "llama-pajamas-run-coreml >= 0.2.0",
        "hardware_requirements": "Apple Silicon (M1, M2, M3, M4) or iOS 15+",
        "compute_units": "ALL (CPU + GPU + ANE)",
        "optimized_for_ane": True,
        "path": "coreml/int8/encoder.mlpackage",
        "quantization": {
            "method": "linear_symmetric",
            "weight_bits": 8,
            "activation_bits": 16
        }
    }

    # Add INT8 entry
    manifest['formats'].append(int8_entry)

    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"   ✅ Added INT8 format ({size_mb:.1f} MB)")

def main():
    models_dir = Path(__file__).parent.parent / "models"

    # Update all whisper model manifests
    for model_name in ["whisper-tiny", "whisper-base", "whisper-small"]:
        manifest_path = models_dir / model_name / "manifest.json"
        if manifest_path.exists():
            update_manifest(manifest_path)
        else:
            print(f"⚠️ Manifest not found: {manifest_path}")

    print("\n✅ All manifests updated!")

if __name__ == "__main__":
    main()
