"""GGUF converter using llama.cpp."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union
import shutil

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class GGUFConverter:
    """Convert HuggingFace models to GGUF format using llama.cpp."""

    def __init__(self, llama_cpp_path: Optional[Path] = None):
        """Initialize GGUF converter.

        Args:
            llama_cpp_path: Path to llama.cpp directory. If None, uses libs/llama.cpp
        """
        if llama_cpp_path is None:
            # Default to libs/llama.cpp relative to project root
            self.llama_cpp_path = Path(__file__).parent.parent.parent.parent / "libs" / "llama.cpp"
        else:
            self.llama_cpp_path = Path(llama_cpp_path)

        if not self.llama_cpp_path.exists():
            raise ValueError(f"llama.cpp not found at {self.llama_cpp_path}")

        self.convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        self.quantize_bin = self.llama_cpp_path / "build" / "bin" / "llama-quantize"

        logger.info(f"Initialized GGUFConverter with llama.cpp at {self.llama_cpp_path}")

    def convert(
        self,
        model_path: Union[str, Path],
        output_dir: Path,
        precision: str = "Q4_K_M",
        architecture_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert HuggingFace model to GGUF format.

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory for GGUF files
            precision: Quantization precision (Q4_K_M, Q5_K_M, Q6_K, etc.)
            architecture_info: Optional architecture info for optimization

        Returns:
            Dictionary with conversion results including paths and metadata

        Raises:
            RuntimeError: If conversion fails
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create GGUF subdirectory
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {model_path} to GGUF format")
        logger.info(f"Output directory: {gguf_dir}")

        # Download model from HuggingFace if it's a model ID
        local_model_path = self._download_model(model_path, output_dir)

        # Calculate final output path
        model_name = local_model_path if isinstance(local_model_path, str) else local_model_path.name
        if precision.upper() != "F16":
            final_gguf_path = gguf_dir / f"{model_name.replace('/', '_')}_{precision.lower()}.gguf"
        else:
            final_gguf_path = gguf_dir / f"{model_name.replace('/', '_')}_f16.gguf"
        final_gguf_path = final_gguf_path.resolve()

        # Check if final file already exists
        if final_gguf_path.exists():
            logger.info(f"Final GGUF already exists, skipping conversion: {final_gguf_path}")
            logger.info(f"File size: {final_gguf_path.stat().st_size / (1024**3):.2f} GB")

            # Generate metadata
            metadata = self._generate_metadata(final_gguf_path, precision, architecture_info)
            metadata_path = gguf_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return {
                "gguf_path": str(final_gguf_path),
                "metadata_path": str(metadata_path),
                "size_bytes": final_gguf_path.stat().st_size,
                "size_gb": final_gguf_path.stat().st_size / (1024**3),
                "precision": precision,
                "metadata": metadata,
            }

        # Step 1: Convert HuggingFace to GGUF (FP16)
        fp16_gguf = self._convert_hf_to_gguf(local_model_path, gguf_dir)

        # Step 2: Quantize to target precision
        if precision.upper() != "F16":
            quantized_gguf = self._quantize_gguf(fp16_gguf, precision, architecture_info)

            # Remove FP16 version to save space
            if fp16_gguf.exists() and fp16_gguf != quantized_gguf:
                logger.info(f"Removing intermediate FP16 file: {fp16_gguf}")
                fp16_gguf.unlink()

            final_gguf = quantized_gguf
        else:
            final_gguf = fp16_gguf

        # Step 3: Generate metadata
        metadata = self._generate_metadata(final_gguf, precision, architecture_info)
        metadata_path = gguf_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"GGUF conversion complete: {final_gguf}")
        logger.info(f"File size: {final_gguf.stat().st_size / (1024**3):.2f} GB")

        return {
            "gguf_path": str(final_gguf),
            "metadata_path": str(metadata_path),
            "size_bytes": final_gguf.stat().st_size,
            "size_gb": final_gguf.stat().st_size / (1024**3),
            "precision": precision,
            "metadata": metadata,
        }

    def _download_model(self, model_path: Union[str, Path], output_dir: Path) -> Path:
        """Download model from HuggingFace if it's a model ID, otherwise return the path.

        Args:
            model_path: HuggingFace model ID or local path
            output_dir: Output directory for downloaded models

        Returns:
            Path to local model directory
        """
        model_path_str = str(model_path)

        # If it's a local path that exists, return it
        if Path(model_path).exists():
            logger.info(f"Using local model at {model_path}")
            return Path(model_path)

        # If it's a HuggingFace model ID (contains / and doesn't exist locally)
        if "/" in model_path_str and not Path(model_path).exists():
            logger.info(f"Downloading model from HuggingFace: {model_path_str}")
            cache_dir = output_dir / "hf_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            local_path = snapshot_download(
                repo_id=model_path_str,
                cache_dir=str(cache_dir),
                local_dir=str(output_dir / "hf_model"),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model downloaded to {local_path}")
            return Path(local_path)

        # Otherwise, assume it's a local path even if it doesn't exist (will fail later)
        return Path(model_path)

    def _convert_hf_to_gguf(self, model_path: Path, output_dir: Path) -> Path:
        """Convert HuggingFace model to FP16 GGUF.

        Args:
            model_path: Path to HuggingFace model
            output_dir: Output directory

        Returns:
            Path to generated FP16 GGUF file

        Raises:
            RuntimeError: If conversion fails
        """
        if not self.convert_script.exists():
            raise RuntimeError(f"Conversion script not found: {self.convert_script}")

        # Output filename
        model_name = model_path if isinstance(model_path, str) else model_path.name
        output_file = output_dir / f"{model_name.replace('/', '_')}_f16.gguf"
        output_file = output_file.resolve()  # Convert to absolute path

        # Skip if file already exists
        if output_file.exists():
            logger.info(f"FP16 GGUF already exists, skipping conversion: {output_file}")
            logger.info(f"File size: {output_file.stat().st_size / (1024**3):.2f} GB")
            return output_file

        logger.info(f"Converting to FP16 GGUF: {output_file}")

        # Run conversion script
        cmd = [
            "python3",
            str(self.convert_script),
            str(model_path),
            "--outfile", str(output_file),
            "--outtype", "f16",
        ]

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.llama_cpp_path),
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Conversion output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Conversion warnings: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise RuntimeError(f"HF to GGUF conversion failed: {e.stderr}")

        if not output_file.exists():
            raise RuntimeError(f"Expected output file not created: {output_file}")

        return output_file

    def _quantize_gguf(
        self,
        input_gguf: Path,
        precision: str,
        architecture_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Quantize GGUF to target precision.

        Args:
            input_gguf: Path to FP16 GGUF file
            precision: Target precision (Q4_K_M, Q5_K_M, etc.)
            architecture_info: Optional architecture info for optimization

        Returns:
            Path to quantized GGUF file

        Raises:
            RuntimeError: If quantization fails
        """
        if not self.quantize_bin.exists():
            raise RuntimeError(
                f"Quantization binary not found: {self.quantize_bin}\n"
                "Please build llama.cpp first: make llama-quantize"
            )

        # Output filename
        output_file = input_gguf.parent / input_gguf.name.replace("_f16.gguf", f"_{precision.lower()}.gguf")

        # Skip if file already exists
        if output_file.exists():
            logger.info(f"Quantized GGUF already exists, skipping: {output_file}")
            logger.info(f"File size: {output_file.stat().st_size / (1024**3):.2f} GB")
            return output_file

        logger.info(f"Quantizing to {precision}: {output_file}")

        # Build command
        cmd = [
            str(self.quantize_bin),
            str(input_gguf),
            str(output_file),
            precision.upper(),
        ]

        # Add architecture-specific optimizations
        if architecture_info:
            # For future: add special handling based on architecture
            # e.g., --imatrix for importance matrix, --leave-output-tensor for embeddings
            pass

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Quantization output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Quantization warnings: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Quantization failed: {e.stderr}")
            raise RuntimeError(f"GGUF quantization failed: {e.stderr}")

        if not output_file.exists():
            raise RuntimeError(f"Expected output file not created: {output_file}")

        return output_file

    def _generate_metadata(
        self,
        gguf_path: Path,
        precision: str,
        architecture_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate metadata for GGUF file.

        Args:
            gguf_path: Path to GGUF file
            precision: Quantization precision
            architecture_info: Optional architecture info

        Returns:
            Metadata dictionary
        """
        file_size = gguf_path.stat().st_size

        metadata = {
            "format": "gguf",
            "method": precision.upper(),
            "file_path": gguf_path.name,
            "file_size_bytes": file_size,
            "file_size_gb": round(file_size / (1024**3), 2),
            "compatible_backends": ["cpu", "cuda", "metal", "rocm"],
            "runtime_requirements": "llama-pajamas-run >= 0.1.0",
        }

        if architecture_info:
            metadata["architecture"] = architecture_info

        return metadata
