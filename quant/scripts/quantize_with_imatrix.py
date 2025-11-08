#!/usr/bin/env python3
"""Quantize model to IQ2_XS with importance matrix for optimal quality.

This script orchestrates the full importance quantization workflow:
1. Build calibration data (if not already built)
2. Generate importance matrix using llama-imatrix
3. Quantize to IQ2_XS (or other IQ formats) using the imatrix
4. Validate output

For best quality with IQ formats (IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, etc.),
an importance matrix is essential.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
import json

# Import our smart builder
try:
    from build_llama_cpp import LlamaCppBuilder
except ImportError:
    # If running from different directory
    sys.path.insert(0, str(Path(__file__).parent))
    from build_llama_cpp import LlamaCppBuilder


class IMatrixQuantizer:
    """Orchestrate importance quantization workflow."""

    def __init__(self, llama_cpp_path: Optional[Path] = None):
        """Initialize quantizer.

        Args:
            llama_cpp_path: Path to llama.cpp directory
        """
        if llama_cpp_path is None:
            # Default to libs/llama.cpp
            self.llama_cpp_path = Path(__file__).parent.parent.parent / "libs" / "llama.cpp"
        else:
            self.llama_cpp_path = Path(llama_cpp_path)

        if not self.llama_cpp_path.exists():
            raise ValueError(f"llama.cpp not found at {self.llama_cpp_path}")

        # Key binaries
        self.imatrix_bin = self.llama_cpp_path / "build" / "bin" / "llama-imatrix"
        self.quantize_bin = self.llama_cpp_path / "build" / "bin" / "llama-quantize"

        # Auto-build llama.cpp if binaries don't exist
        if not self.imatrix_bin.exists() or not self.quantize_bin.exists():
            print(f"\n⚠️  llama.cpp binaries not found, building now...")
            self._build_llama_cpp()

        # Verify binaries exist after build
        if not self.imatrix_bin.exists():
            raise ValueError(f"llama-imatrix not found at {self.imatrix_bin} even after build")
        if not self.quantize_bin.exists():
            raise ValueError(f"llama-quantize not found at {self.quantize_bin} even after build")

        print(f"Using llama.cpp at: {self.llama_cpp_path}")

    def _build_llama_cpp(self):
        """Build llama.cpp binaries with hardware-optimized settings."""
        builder = LlamaCppBuilder(llama_cpp_path=self.llama_cpp_path)
        success = builder.build()

        if not success:
            raise RuntimeError("Failed to build llama.cpp with hardware-optimized settings")

    def generate_imatrix(
        self,
        model_path: Path,
        calibration_file: Path,
        output_path: Path,
        n_ctx: int = 2048,
        n_chunks: int = 100,
        verbosity: int = 1,
    ) -> Path:
        """Generate importance matrix from calibration data.

        Args:
            model_path: Path to GGUF model file (F16 or high precision)
            calibration_file: Path to calibration.txt file
            output_path: Path to save imatrix file
            n_ctx: Context length for processing
            n_chunks: Number of chunks to process from calibration data
            verbosity: Verbosity level (0-2)

        Returns:
            Path to generated imatrix file
        """
        print("\n" + "=" * 60)
        print("GENERATING IMPORTANCE MATRIX")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Calibration: {calibration_file}")
        print(f"Output: {output_path}")
        print(f"Context: {n_ctx}, Chunks: {n_chunks}")

        cmd = [
            str(self.imatrix_bin),
            "-m", str(model_path),
            "-f", str(calibration_file),
            "-o", str(output_path),
            "-ngl", "99",  # Offload all layers to GPU if available
            "-c", str(n_ctx),
            "--chunks", str(n_chunks),
            "-v", str(verbosity),
        ]

        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show live output
                text=True,
            )

            if output_path.exists():
                print(f"\n✅ Successfully generated imatrix: {output_path}")
                return output_path
            else:
                raise RuntimeError("Imatrix generation completed but output file not found")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error generating imatrix: {e}")
            sys.exit(1)

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
        print("\n" + "=" * 60)
        print("QUANTIZING WITH IMPORTANCE MATRIX")
        print("=" * 60)
        print(f"Input: {input_model}")
        print(f"Output: {output_model}")
        print(f"Precision: {precision}")
        print(f"IMatrix: {imatrix_file}")

        cmd = [
            str(self.quantize_bin),
            str(input_model),
            str(output_model),
            precision,
            "--imatrix", str(imatrix_file),
        ]

        if leave_output_tensor:
            cmd.append("--leave-output-tensor")
        if pure:
            cmd.append("--pure")

        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
            )

            if output_model.exists():
                size_mb = output_model.stat().st_size / (1024 * 1024)
                print(f"\n✅ Successfully quantized model: {output_model}")
                print(f"Size: {size_mb:.1f} MB")
                return output_model
            else:
                raise RuntimeError("Quantization completed but output file not found")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error quantizing model: {e}")
            sys.exit(1)

    def save_metadata(
        self,
        output_dir: Path,
        model_id: str,
        precision: str,
        imatrix_file: Path,
        calibration_metadata: dict,
    ):
        """Save quantization metadata."""
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

        print(f"\nSaved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize model with importance matrix for optimal quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow: Build calibration → Generate imatrix → Quantize to IQ2_XS
  python quantize_with_imatrix.py \\
      --model /path/to/model-f16.gguf \\
      --output ./models/qwen3-8b/gguf/IQ2_XS/ \\
      --precision IQ2_XS \\
      --calibration-data ./calibration_data/calibration.txt

  # Use existing imatrix
  python quantize_with_imatrix.py \\
      --model /path/to/model-f16.gguf \\
      --output ./models/qwen3-8b/gguf/IQ2_XS/ \\
      --precision IQ2_XS \\
      --imatrix ./qwen3-8b.imatrix

  # Generate imatrix only (for reuse with different precisions)
  python quantize_with_imatrix.py \\
      --model /path/to/model-f16.gguf \\
      --calibration-data ./calibration_data/calibration.txt \\
      --imatrix-only \\
      --output-imatrix ./qwen3-8b.imatrix
        """
    )

    # Input/output paths
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Input model (GGUF, F16 or high precision)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for quantized model"
    )

    # Quantization options
    parser.add_argument(
        "--precision",
        default="IQ2_XS",
        help="Quantization precision (IQ2_XS, IQ2_XXS, IQ2_S, IQ3_XXS, etc.) (default: IQ2_XS)"
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="Use pure quantization (no mixed precision)"
    )
    parser.add_argument(
        "--quantize-output-tensor",
        action="store_true",
        help="Also quantize output tensor (lower quality but smaller)"
    )

    # IMatrix options
    parser.add_argument(
        "--imatrix",
        type=Path,
        help="Path to existing importance matrix file"
    )
    parser.add_argument(
        "--imatrix-only",
        action="store_true",
        help="Only generate imatrix, don't quantize"
    )
    parser.add_argument(
        "--output-imatrix",
        type=Path,
        help="Path to save generated imatrix (default: <output>/imatrix.dat)"
    )

    # Calibration options
    parser.add_argument(
        "--calibration-data",
        type=Path,
        help="Path to calibration.txt file (required if --imatrix not provided)"
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context length for imatrix generation (default: 2048)"
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=100,
        help="Number of chunks to process for imatrix (default: 100)"
    )

    # Other options
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        help="Path to llama.cpp directory (default: ../libs/llama.cpp)"
    )
    parser.add_argument(
        "--model-id",
        default="unknown",
        help="Model identifier for metadata"
    )

    args = parser.parse_args()

    # Validation
    if not args.imatrix and not args.calibration_data:
        parser.error("Either --imatrix or --calibration-data must be provided")

    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")

    if args.calibration_data and not args.calibration_data.exists():
        parser.error(f"Calibration data not found: {args.calibration_data}")

    # Initialize quantizer
    quantizer = IMatrixQuantizer(llama_cpp_path=args.llama_cpp)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine imatrix path
    if args.imatrix:
        imatrix_file = args.imatrix
        if not imatrix_file.exists():
            parser.error(f"IMatrix file not found: {imatrix_file}")
    else:
        # Generate imatrix
        if args.output_imatrix:
            imatrix_file = args.output_imatrix
        else:
            model_name = args.model.stem
            imatrix_file = args.output / f"{model_name}.imatrix"

        imatrix_file = quantizer.generate_imatrix(
            model_path=args.model,
            calibration_file=args.calibration_data,
            output_path=imatrix_file,
            n_ctx=args.n_ctx,
            n_chunks=args.n_chunks,
        )

    # If imatrix-only mode, we're done
    if args.imatrix_only:
        print("\n" + "=" * 60)
        print("IMATRIX GENERATION COMPLETE")
        print("=" * 60)
        print(f"\nIMatrix saved to: {imatrix_file}")
        print("\nYou can now use this imatrix to quantize with different precisions:")
        print(f"  --imatrix {imatrix_file} --precision IQ2_XS")
        print(f"  --imatrix {imatrix_file} --precision IQ2_XXS")
        return

    # Quantize model
    output_model = args.output / f"{args.model.stem}-{args.precision.lower()}.gguf"

    quantized_model = quantizer.quantize_with_imatrix(
        input_model=args.model,
        output_model=output_model,
        precision=args.precision,
        imatrix_file=imatrix_file,
        leave_output_tensor=not args.quantize_output_tensor,
        pure=args.pure,
    )

    # Load calibration metadata if available
    calibration_metadata = {}
    if args.calibration_data:
        metadata_file = args.calibration_data.parent / "calibration_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                calibration_metadata = json.load(f)

    # Save metadata
    quantizer.save_metadata(
        output_dir=args.output,
        model_id=args.model_id,
        precision=args.precision,
        imatrix_file=imatrix_file,
        calibration_metadata=calibration_metadata,
    )

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nQuantized model: {quantized_model}")
    print(f"Precision: {args.precision}")
    print(f"IMatrix: {imatrix_file}")
    print(f"\nNext step: Benchmark on held-out evaluation set")


if __name__ == "__main__":
    main()
