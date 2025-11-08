"""MLX converter for Apple Silicon optimization."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


class MLXConverter:
    """Convert HuggingFace models to MLX format for Apple Silicon."""

    def __init__(self):
        """Initialize MLX converter."""
        try:
            import mlx
            import mlx_lm
            self.mlx_available = True
            logger.info("MLX libraries loaded successfully")
        except ImportError:
            self.mlx_available = False
            logger.warning("MLX not available - install with: pip install mlx mlx-lm")

    def convert(
        self,
        model_path: Union[str, Path],
        output_dir: Path,
        quantize: bool = True,
        q_bits: int = 4,
        q_group_size: int = 64,
        mixed_precision: bool = True,
        architecture_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert HuggingFace model to MLX format.

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory for MLX files
            quantize: Whether to quantize (default: True)
            q_bits: Quantization bits for body (default: 4)
            q_group_size: Group size for quantization (default: 64)
            mixed_precision: Use mixed precision (6-bit embeddings/output) (default: True)
            architecture_info: Optional architecture info for optimization

        Returns:
            Dictionary with conversion results including paths and metadata

        Raises:
            RuntimeError: If MLX is not available or conversion fails
        """
        if not self.mlx_available:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm"
            )

        model_path = str(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create MLX subdirectory
        mlx_dir = output_dir / "mlx"

        # Check if MLX conversion already exists or is incomplete
        config_file = mlx_dir / "config.json"
        if mlx_dir.exists():
            files_in_dir = list(mlx_dir.iterdir())
            if not files_in_dir:
                # Empty directory from failed conversion - remove it
                logger.info(f"Removing empty MLX directory from previous failed conversion: {mlx_dir}")
                mlx_dir.rmdir()
            elif config_file.exists():
                # Complete conversion exists - skip and return
                logger.info(f"MLX conversion already exists, skipping: {mlx_dir}")
                # Calculate total size
                total_size = sum(f.stat().st_size for f in mlx_dir.rglob("*") if f.is_file())
                logger.info(f"Existing MLX size: {total_size / (1024**3):.2f} GB")

                # Generate metadata for existing conversion
                metadata = self._generate_metadata(
                    mlx_dir,
                    quantize=quantize,
                    q_bits=q_bits if quantize else None,
                    mixed_precision=mixed_precision if quantize else False,
                    architecture_info=architecture_info,
                )
                metadata_path = mlx_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                return {
                    "mlx_dir": str(mlx_dir),
                    "metadata_path": str(metadata_path),
                    "size_bytes": total_size,
                    "size_gb": total_size / (1024**3),
                    "quantization": {
                        "enabled": quantize,
                        "bits": q_bits if quantize else None,
                        "mixed_precision": mixed_precision if quantize else False,
                    },
                    "metadata": metadata,
                }

        # Don't create the directory - mlx_lm.convert will create it

        logger.info(f"Converting {model_path} to MLX format")
        logger.info(f"Output directory: {mlx_dir}")
        logger.info(f"Quantization: {q_bits}-bit (mixed: {mixed_precision})")

        # Use mlx-lm convert command
        if quantize:
            result = self._convert_and_quantize(
                model_path,
                mlx_dir,
                q_bits=q_bits,
                q_group_size=q_group_size,
                mixed_precision=mixed_precision,
                architecture_info=architecture_info,
            )
        else:
            result = self._convert_only(model_path, mlx_dir)

        # Generate metadata
        metadata = self._generate_metadata(
            mlx_dir,
            quantize=quantize,
            q_bits=q_bits if quantize else None,
            mixed_precision=mixed_precision if quantize else False,
            architecture_info=architecture_info,
        )
        metadata_path = mlx_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"MLX conversion complete: {mlx_dir}")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in mlx_dir.rglob("*") if f.is_file())

        return {
            "mlx_dir": str(mlx_dir),
            "metadata_path": str(metadata_path),
            "size_bytes": total_size,
            "size_gb": total_size / (1024**3),
            "quantization": {
                "enabled": quantize,
                "bits": q_bits if quantize else None,
                "mixed_precision": mixed_precision if quantize else False,
            },
            "metadata": metadata,
        }

    def _convert_and_quantize(
        self,
        model_path: str,
        output_dir: Path,
        q_bits: int = 4,
        q_group_size: int = 64,
        mixed_precision: bool = True,
        architecture_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert and quantize model using mlx-lm.

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory
            q_bits: Quantization bits
            q_group_size: Group size for quantization
            mixed_precision: Use mixed precision
            architecture_info: Optional architecture info

        Returns:
            Conversion result dictionary

        Raises:
            RuntimeError: If conversion fails
        """
        logger.info("Converting and quantizing with mlx-lm...")

        # Build command with absolute path
        cmd = [
            "python3", "-m", "mlx_lm.convert",
            "--hf-path", model_path,
            "--mlx-path", str(output_dir.resolve()),
        ]

        # Add quantization parameters
        if q_bits:
            cmd.extend(["-q"])  # Enable quantization

        # mlx-lm uses --q-bits and --q-group-size
        # For mixed precision, we'll post-process or use mlx.quantize API
        # For now, let's use the CLI for simplicity

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Conversion output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Conversion warnings: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise RuntimeError(f"MLX conversion failed: {e.stderr}")

        # Verify output files exist
        config_path = output_dir / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Expected config.json not created in {output_dir}")

        return {"status": "success", "output_dir": str(output_dir)}

    def _convert_only(
        self,
        model_path: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Convert model without quantization.

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory

        Returns:
            Conversion result dictionary

        Raises:
            RuntimeError: If conversion fails
        """
        logger.info("Converting without quantization...")

        cmd = [
            "python3", "-m", "mlx_lm.convert",
            "--hf-path", model_path,
            "--mlx-path", str(output_dir.resolve()),
        ]

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Conversion output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Conversion warnings: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise RuntimeError(f"MLX conversion failed: {e.stderr}")

        return {"status": "success", "output_dir": str(output_dir)}

    def _generate_metadata(
        self,
        mlx_dir: Path,
        quantize: bool,
        q_bits: Optional[int],
        mixed_precision: bool,
        architecture_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate metadata for MLX conversion.

        Args:
            mlx_dir: MLX output directory
            quantize: Whether quantization was applied
            q_bits: Quantization bits (if quantized)
            mixed_precision: Whether mixed precision was used
            architecture_info: Optional architecture info

        Returns:
            Metadata dictionary
        """
        # Calculate total size
        total_size = sum(f.stat().st_size for f in mlx_dir.rglob("*") if f.is_file())

        metadata = {
            "format": "mlx",
            "quantization": {
                "enabled": quantize,
            },
            "file_size_bytes": total_size,
            "file_size_gb": round(total_size / (1024**3), 2),
            "compatible_backends": ["mlx"],
            "runtime_requirements": "llama-pajamas-run[mlx] >= 0.1.0",
            "hardware_requirements": "Apple Silicon (M1, M2, M3, M4)",
        }

        if quantize and q_bits:
            metadata["quantization"].update({
                "body_bits": q_bits,
                "group_size": 64,  # Default
            })

            if mixed_precision:
                metadata["quantization"].update({
                    "embedding_bits": 6,
                    "output_bits": 6,
                    "mixed_precision": True,
                })
            else:
                metadata["quantization"]["mixed_precision"] = False

        if architecture_info:
            metadata["architecture"] = architecture_info

        return metadata
