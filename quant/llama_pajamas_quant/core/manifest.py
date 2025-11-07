"""Manifest generator for quantized model artifacts."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ManifestGenerator:
    """Generate manifest.json for quantized model artifacts."""

    def __init__(self):
        """Initialize manifest generator."""
        pass

    def generate(
        self,
        model_id: str,
        architecture_info: Dict[str, Any],
        formats: List[Dict[str, Any]],
        output_path: Path,
        validation_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate manifest.json for quantized model.

        Args:
            model_id: HuggingFace model ID or name
            architecture_info: Architecture information from detector
            formats: List of format metadata (GGUF, MLX, etc.)
            output_path: Path to write manifest.json
            validation_results: Optional validation results

        Returns:
            Manifest dictionary
        """
        manifest = {
            "version": "1.0",
            "model_id": model_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "llama_pajamas_version": "0.1.0",
            "architecture": architecture_info,
            "formats": formats,
        }

        if validation_results:
            manifest["validation"] = validation_results

        # Write manifest
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Generated manifest: {output_path}")
        return manifest

    def add_format(
        self,
        manifest: Dict[str, Any],
        format_type: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add a format to existing manifest.

        Args:
            manifest: Existing manifest dictionary
            format_type: Format type (gguf, mlx, etc.)
            metadata: Format-specific metadata

        Returns:
            Updated manifest
        """
        format_entry = {
            "type": format_type,
            **metadata
        }

        if "formats" not in manifest:
            manifest["formats"] = []

        manifest["formats"].append(format_entry)
        return manifest

    def load(self, path: Path) -> Dict[str, Any]:
        """Load manifest from file.

        Args:
            path: Path to manifest.json

        Returns:
            Manifest dictionary

        Raises:
            FileNotFoundError: If manifest doesn't exist
            json.JSONDecodeError: If manifest is invalid JSON
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with open(path, "r") as f:
            manifest = json.load(f)

        logger.info(f"Loaded manifest: {path}")
        return manifest

    def validate(self, manifest: Dict[str, Any]) -> bool:
        """Validate manifest structure.

        Args:
            manifest: Manifest dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["version", "model_id", "architecture", "formats"]

        for field in required_fields:
            if field not in manifest:
                logger.error(f"Missing required field: {field}")
                return False

        if not isinstance(manifest["formats"], list):
            logger.error("'formats' must be a list")
            return False

        if len(manifest["formats"]) == 0:
            logger.error("'formats' list is empty")
            return False

        for fmt in manifest["formats"]:
            if "type" not in fmt:
                logger.error("Format missing 'type' field")
                return False

        return True
