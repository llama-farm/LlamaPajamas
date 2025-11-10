#!/usr/bin/env python3
"""Update manifest.json files to include INT8 quantized models."""

import json
from pathlib import Path


def update_manifest(model_dir: Path, int8_size_mb: float):
    """Add INT8 entry to manifest.json."""
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"⚠️  Manifest not found: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check if INT8 already exists
    int8_exists = any(
        fmt.get("format") == "coreml" and fmt.get("precision") == "int8"
        for fmt in manifest.get("formats", [])
    )

    if int8_exists:
        print(f"✓ INT8 already in manifest: {model_dir.name}")
        return

    # Add INT8 format
    int8_format = {
        "format": "coreml",
        "precision": "int8",
        "file_size_bytes": int(int8_size_mb * 1024 * 1024),
        "file_size_mb": round(int8_size_mb, 1),
        "compatible_backends": ["coreml"],
        "runtime_requirements": "llama-pajamas-run-coreml >= 0.1.0",
        "hardware_requirements": "Apple Silicon (M1, M2, M3, M4) or iOS 15+",
        "compute_units": "ALL (CPU + GPU + ANE)",
        "optimized_for_ane": True,
        "path": "coreml/int8/model.mlpackage"
    }

    manifest["formats"].append(int8_format)

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Added INT8 to manifest: {model_dir.name} ({int8_size_mb:.1f} MB)")


def main():
    models_dir = Path(__file__).parent.parent / "models"

    # Model sizes from quantization output
    model_sizes = {
        "yolo-v8n": 0.1,
        "vit-base": 83.0,
        "clip-vit-base": 83.8,
    }

    for model_name, size_mb in model_sizes.items():
        model_dir = models_dir / model_name
        update_manifest(model_dir, size_mb)

    print("\n✅ All manifests updated!")


if __name__ == "__main__":
    main()
