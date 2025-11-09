"""ONNX Runtime session manager for llama-pajamas.

This module provides a high-level interface for loading and running
ONNX models with different execution providers (CoreML, TensorRT, CPU).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .backends.base import Backend, GenerationConfig, GenerationResult
from .backends.coreml_backend import CoreMLBackend

logger = logging.getLogger(__name__)


class ONNXSession:
    """High-level session manager for ONNX Runtime.

    This class automatically selects the appropriate backend based on:
    1. User-specified execution provider
    2. Model manifest metadata
    3. Available hardware

    Example:
        >>> from llama_pajamas_run_onnx import ONNXSession
        >>> from pathlib import Path
        >>>
        >>> # Auto-detect backend from manifest
        >>> session = ONNXSession.from_manifest(
        ...     manifest_path=Path("models/qwen3-8b/onnx/manifest.json"),
        ...     ep="CoreML",  # Or "TensorRT", "CPU"
        ...     precision="int8"
        ... )
        >>>
        >>> # Generate text
        >>> result = session.generate("Hello, world!")
        >>> print(result.text)
        >>> print(f"Speed: {result.tokens_per_second:.1f} tok/s")
    """

    def __init__(self, backend: Backend):
        """Initialize ONNX session with a backend.

        Args:
            backend: Backend instance (CoreML, TensorRT, CPU).
        """
        self.backend = backend
        self.backend._load_model()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        ep: str,
        precision: str = "fp16",
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> "ONNXSession":
        """Create session from ONNX manifest.

        Args:
            manifest_path: Path to manifest.json.
            ep: Execution provider (CoreML, TensorRT, CPU).
            precision: Target precision (fp16, int8, int4).
            provider_options: Optional provider-specific options.

        Returns:
            ONNXSession instance.

        Raises:
            FileNotFoundError: If manifest or model not found.
            ValueError: If EP/precision combination not available.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        logger.info(f"Loaded manifest: {manifest['model_id']}")
        logger.info(f"Available variants: {len(manifest['variants'])}")

        # Find matching variant
        variant = None
        for v in manifest["variants"]:
            if v["ep"] == ep and v["precision"] == precision:
                variant = v
                break

        if variant is None:
            available = [(v["ep"], v["precision"]) for v in manifest["variants"]]
            raise ValueError(
                f"Variant {ep}@{precision} not found in manifest. "
                f"Available: {available}"
            )

        # Get model path
        model_path = manifest_path.parent / variant["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Selected variant: {ep} @ {precision} ({variant['size_mb']:.1f} MB)")
        logger.info(f"Model path: {model_path}")

        # Create backend
        backend = cls._create_backend(
            ep=ep,
            model_path=model_path,
            provider_options=provider_options,
        )

        return cls(backend)

    @classmethod
    def from_model_path(
        cls,
        model_path: Path,
        ep: str,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> "ONNXSession":
        """Create session directly from model path.

        Args:
            model_path: Path to ONNX model file.
            ep: Execution provider (CoreML, TensorRT, CPU).
            provider_options: Optional provider-specific options.

        Returns:
            ONNXSession instance.
        """
        backend = cls._create_backend(
            ep=ep,
            model_path=model_path,
            provider_options=provider_options,
        )
        return cls(backend)

    @staticmethod
    def _create_backend(
        ep: str,
        model_path: Path,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> Backend:
        """Create backend for execution provider.

        Args:
            ep: Execution provider name.
            model_path: Path to model.
            provider_options: Optional provider options.

        Returns:
            Backend instance.

        Raises:
            ValueError: If EP not supported.
        """
        if ep == "CoreML":
            return CoreMLBackend(
                model_path=model_path,
                provider_options=provider_options,
            )
        # Future: TensorRT, CPU, etc.
        # elif ep == "TensorRT":
        #     return TensorRTBackend(model_path, provider_options)
        # elif ep == "CPU":
        #     return CPUBackend(model_path, provider_options)
        else:
            raise ValueError(
                f"Unsupported execution provider: {ep}. "
                "Supported: CoreML (more coming soon)"
            )

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from prompt.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Returns:
            GenerationResult with generated text and metadata.
        """
        return self.backend.generate(prompt, config)

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ):
        """Generate text with streaming (token-by-token).

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Yields:
            Generated tokens as they're produced.
        """
        yield from self.backend.generate_stream(prompt, config)

    def benchmark(
        self,
        prompt: str = "The quick brown fox jumps over the lazy dog",
        num_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark generation performance.

        Args:
            prompt: Test prompt.
            num_tokens: Number of tokens to generate.

        Returns:
            Dictionary with benchmark results.
        """
        return self.backend.benchmark(prompt, num_tokens)


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python session.py <manifest.json> <ep> [precision] [prompt]")
        print("  python session.py <model.onnx> <ep> [prompt]")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.name == "manifest.json":
        # Load from manifest
        ep = sys.argv[2] if len(sys.argv) > 2 else "CoreML"
        precision = sys.argv[3] if len(sys.argv) > 3 else "fp16"
        prompt = sys.argv[4] if len(sys.argv) > 4 else "Hello, how are you?"

        session = ONNXSession.from_manifest(path, ep, precision)
    else:
        # Load from model path
        ep = sys.argv[2] if len(sys.argv) > 2 else "CoreML"
        prompt = sys.argv[3] if len(sys.argv) > 3 else "Hello, how are you?"

        session = ONNXSession.from_model_path(path, ep)

    # Generate
    config = GenerationConfig(max_tokens=50, temperature=0.7)
    result = session.generate(prompt, config)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result.text}")
    print(f"Speed: {result.tokens_per_second:.1f} tok/s")
