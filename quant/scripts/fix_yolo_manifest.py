#!/usr/bin/env python3
"""Remove INT8 entry from YOLO manifest (quantization failed)."""

import json
from pathlib import Path


def main():
    manifest_path = Path(__file__).parent.parent / "models" / "yolo-v8n" / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Remove INT8 format
    manifest["formats"] = [
        fmt for fmt in manifest["formats"]
        if not (fmt.get("format") == "coreml" and fmt.get("precision") == "int8")
    ]

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Removed INT8 from YOLO manifest (pipeline-type quantization not supported)")


if __name__ == "__main__":
    main()
