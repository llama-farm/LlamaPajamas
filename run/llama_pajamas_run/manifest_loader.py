"""Manifest loader for Llama-Pajamas quantized models."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_manifest(model_path: str) -> Dict[str, Any]:
    """Load manifest.json from quantized model directory.

    Args:
        model_path: Path to model directory containing manifest.json

    Returns:
        Dictionary containing manifest data

    Raises:
        FileNotFoundError: If manifest.json not found
        ValueError: If manifest is invalid
    """
    model_dir = Path(model_path)
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}. "
            "Make sure this is a Llama-Pajamas quantized model directory."
        )

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid manifest.json: {e}")

    # Validate required fields
    required_fields = ["model_id", "architecture", "formats"]
    missing_fields = [f for f in required_fields if f not in manifest]
    if missing_fields:
        raise ValueError(f"Manifest missing required fields: {missing_fields}")

    logger.info(f"Loaded manifest for {manifest['model_id']}")
    logger.info(f"Available formats: {', '.join(manifest['formats'].keys())}")

    return manifest


def get_format_path(manifest: Dict[str, Any], format_name: str, model_dir: Path) -> Optional[Path]:
    """Get the file/directory path for a specific format.

    Args:
        manifest: Loaded manifest dictionary
        format_name: Format name ('gguf' or 'mlx')
        model_dir: Model directory path

    Returns:
        Path to format file/directory, or None if not available
    """
    if format_name not in manifest.get("formats", {}):
        return None

    format_data = manifest["formats"][format_name]

    if format_name == "gguf":
        # GGUF stores the file path
        gguf_path = format_data.get("file")
        if gguf_path:
            return model_dir / gguf_path
    elif format_name == "mlx":
        # MLX stores the directory
        mlx_dir = format_data.get("directory")
        if mlx_dir:
            return model_dir / mlx_dir

    return None
