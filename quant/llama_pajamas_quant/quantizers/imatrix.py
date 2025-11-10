"""Importance quantization (IQ) for optimal quality at low bit rates."""

import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ..tools.binary_wrapper import BinaryWrapper

logger = logging.getLogger(__name__)


class IMatrixQuantizer:
    """Orchestrate importance quantization workflow."""

    def __init__(self, llama_cpp_path: Optional[Path] = None):
        """Initialize IMatrix quantizer.

        Args:
            llama_cpp_path: Path to llama.cpp directory (optional)
        """
        self.binary_wrapper = BinaryWrapper(llama_cpp_path)
        self.imatrix_bin = self.binary_wrapper.get_binary("llama-imatrix")
        self.quantize_bin = self.binary_wrapper.get_binary("llama-quantize")

        logger.info(f"Using llama-imatrix: {self.imatrix_bin}")
        logger.info(f"Using llama-quantize: {self.quantize_bin}")

    def generate_imatrix(
        self,
        model_path: Path,
        calibration_file: Path,
        output_path: Path,
        n_ctx: int = 2048,
        n_chunks: int = 100,
        n_gpu_layers: int = 99,
    ) -> Path:
        """Generate importance matrix from calibration data.

        Args:
            model_path: Path to GGUF model file (F16 or high precision)
            calibration_file: Path to calibration.txt file
            output_path: Path to save imatrix file
            n_ctx: Context length for processing
            n_chunks: Number of chunks to process from calibration data
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)

        Returns:
            Path to generated imatrix file
        """
        logger.info("=" * 60)
        logger.info("GENERATING IMPORTANCE MATRIX")
        logger.info("=" * 60)
        logger.info(f"Model: {model_path}")
        logger.info(f"Calibration: {calibration_file}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Context: {n_ctx}, Chunks: {n_chunks}")

        cmd = [
            str(self.imatrix_bin),
            "-m", str(model_path),
            "-f", str(calibration_file),
            "-o", str(output_path),
            "-ngl", str(n_gpu_layers),
            "-c", str(n_ctx),
            "--chunks", str(n_chunks),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
            )

            if output_path.exists():
                logger.info(f"✅ Successfully generated imatrix: {output_path}")
                return output_path
            else:
                raise RuntimeError("Imatrix generation completed but output file not found")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating imatrix: {e}")
            raise

    def quantize_with_imatrix(
        self,
        input_model: Path,
        output_model: Path,
        precision: str,
        imatrix_file: Path,
        leave_output_tensor: bool = True,
        pure: bool = False,
    ) -> Path:
        """Quantize model using importance matrix.

        Args:
            input_model: Path to input GGUF model (F16 or high precision)
            output_model: Path to save quantized model
            precision: Quantization precision (IQ2_XS, IQ2_XXS, IQ2_S, etc.)
            imatrix_file: Path to importance matrix file
            leave_output_tensor: Keep output tensor at high precision
            pure: Use pure quantization (no mixed precision)

        Returns:
            Path to quantized model
        """
        logger.info("=" * 60)
        logger.info("QUANTIZING WITH IMPORTANCE MATRIX")
        logger.info("=" * 60)
        logger.info(f"Input: {input_model}")
        logger.info(f"Output: {output_model}")
        logger.info(f"Precision: {precision}")
        logger.info(f"IMatrix: {imatrix_file}")

        # Ensure output directory exists
        output_model.parent.mkdir(parents=True, exist_ok=True)

        # Build command - flags must come BEFORE positional arguments
        cmd = [str(self.quantize_bin)]

        # Add flags first
        if leave_output_tensor:
            cmd.append("--leave-output-tensor")
        if pure:
            cmd.append("--pure")

        # Add imatrix flag
        cmd.extend(["--imatrix", str(imatrix_file)])

        # Add positional arguments last
        cmd.extend([
            str(input_model),
            str(output_model),
            precision,
        ])

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
            )

            if output_model.exists():
                size_mb = output_model.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Successfully quantized model: {output_model}")
                logger.info(f"Size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")
                return output_model
            else:
                raise RuntimeError("Quantization completed but output file not found")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error quantizing model: {e}")
            raise

    def quantize_full_workflow(
        self,
        model_path: Path,
        calibration_file: Path,
        precision: str,
        output_dir: Path,
        n_ctx: int = 2048,
        n_chunks: int = 100,
        leave_output_tensor: bool = True,
        pure: bool = False,
    ) -> Dict[str, Any]:
        """Run full IQ workflow: generate imatrix → quantize.

        Args:
            model_path: Path to GGUF model file (F16 or high precision)
            calibration_file: Path to calibration.txt file
            precision: Quantization precision (IQ2_XS, IQ2_XXS, etc.)
            output_dir: Output directory for imatrix and quantized model
            n_ctx: Context length for imatrix generation
            n_chunks: Number of chunks for imatrix generation
            leave_output_tensor: Keep output tensor at high precision
            pure: Use pure quantization

        Returns:
            Dict with paths and metadata
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate imatrix
        model_name = model_path.stem
        imatrix_file = output_dir / f"{model_name}.imatrix"

        logger.info("\n" + "=" * 60)
        logger.info("FULL IQ WORKFLOW")
        logger.info("=" * 60)
        logger.info(f"Step 1/2: Generating importance matrix...")

        self.generate_imatrix(
            model_path=model_path,
            calibration_file=calibration_file,
            output_path=imatrix_file,
            n_ctx=n_ctx,
            n_chunks=n_chunks,
        )

        # Quantize with imatrix
        logger.info(f"\nStep 2/2: Quantizing to {precision}...")
        output_model = output_dir / f"model-{precision.lower()}.gguf"

        self.quantize_with_imatrix(
            input_model=model_path,
            output_model=output_model,
            precision=precision,
            imatrix_file=imatrix_file,
            leave_output_tensor=leave_output_tensor,
            pure=pure,
        )

        # Save metadata
        metadata = {
            "model_path": str(model_path),
            "precision": precision,
            "imatrix_file": str(imatrix_file),
            "output_model": str(output_model),
            "calibration_file": str(calibration_file),
            "n_ctx": n_ctx,
            "n_chunks": n_chunks,
            "leave_output_tensor": leave_output_tensor,
            "pure": pure,
        }

        metadata_path = output_dir / "iq_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✅ Saved metadata to {metadata_path}")

        logger.info("\n" + "=" * 60)
        logger.info("IQ WORKFLOW COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Quantized model: {output_model}")
        logger.info(f"Precision: {precision}")
        logger.info(f"IMatrix: {imatrix_file}")

        return metadata

    def save_metadata(
        self,
        output_dir: Path,
        model_id: str,
        precision: str,
        imatrix_file: Path,
        calibration_metadata: dict,
    ):
        """Save quantization metadata.

        Args:
            output_dir: Output directory
            model_id: Model identifier
            precision: Quantization precision
            imatrix_file: Path to imatrix file
            calibration_metadata: Calibration data metadata
        """
        metadata = {
            "model_id": model_id,
            "quantization": {
                "precision": precision,
                "method": "importance_quantization",
                "imatrix_file": str(imatrix_file),
            },
            "calibration": calibration_metadata,
        }

        metadata_path = output_dir / "quantization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
