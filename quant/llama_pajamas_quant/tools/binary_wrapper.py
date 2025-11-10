"""Wrapper for llama.cpp binary tools."""

from pathlib import Path
from typing import Optional
import subprocess
import logging

logger = logging.getLogger(__name__)


class BinaryWrapper:
    """Wrapper for accessing llama.cpp binaries."""

    def __init__(self, llama_cpp_path: Optional[Path] = None):
        """Initialize binary wrapper.

        Args:
            llama_cpp_path: Path to llama.cpp directory (optional)
        """
        if llama_cpp_path is None:
            # Try multiple locations in order of preference
            root = Path(__file__).parent.parent.parent.parent

            # 1. Check bin/ symlinks (preferred)
            bin_dir = root / "bin"
            if bin_dir.exists() and (bin_dir / "llama-imatrix").exists():
                self.bin_dir = bin_dir
                logger.info(f"Using llama.cpp binaries from: {bin_dir}")
            else:
                # 2. Fall back to libs/llama.cpp/build/bin
                self.bin_dir = root / "libs" / "llama.cpp" / "build" / "bin"
                logger.info(f"Using llama.cpp binaries from: {self.bin_dir}")
        else:
            self.bin_dir = Path(llama_cpp_path) / "build" / "bin"

        # Auto-build llama.cpp if binaries don't exist
        if not self.bin_dir.exists() or not (self.bin_dir / "llama-quantize").exists():
            logger.warning("llama.cpp binaries not found, attempting to build...")
            self._auto_build()

    def _auto_build(self):
        """Auto-build llama.cpp if binaries are missing."""
        from ..core.llama_cpp_builder import LlamaCppBuilder

        logger.info("Building llama.cpp with hardware-optimized settings...")
        builder = LlamaCppBuilder()
        success = builder.build()

        if not success:
            raise RuntimeError("Failed to build llama.cpp")

    def get_binary(self, name: str) -> Path:
        """Get path to binary.

        Args:
            name: Binary name (e.g., "llama-imatrix", "llama-quantize")

        Returns:
            Path to binary

        Raises:
            FileNotFoundError: If binary not found
        """
        binary = self.bin_dir / name

        if not binary.exists():
            raise FileNotFoundError(
                f"Binary '{name}' not found at {binary}. "
                f"Please build llama.cpp first: cd quant && uv run python scripts/build_llama_cpp.py"
            )

        return binary

    def run(
        self,
        binary_name: str,
        args: list,
        check: bool = True,
        capture_output: bool = False,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """Run binary with arguments.

        Args:
            binary_name: Name of binary to run
            args: Arguments to pass
            check: Raise exception on non-zero exit code
            capture_output: Capture stdout/stderr
            **kwargs: Additional arguments to subprocess.run

        Returns:
            CompletedProcess object
        """
        binary = self.get_binary(binary_name)
        cmd = [str(binary)] + [str(arg) for arg in args]

        logger.debug(f"Running: {' '.join(cmd)}")

        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            **kwargs
        )
