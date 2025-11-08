"""Model loader that orchestrates config, manifest, and backend selection."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional

from .config import RuntimeConfig
from .manifest_loader import load_manifest, get_format_path
from .backends import GGUFBackend, MLXBackend

logger = logging.getLogger(__name__)


class ModelLoader:
    """High-level model loader with OpenAI-compatible API."""

    def __init__(self, config: RuntimeConfig):
        """Initialize model loader with runtime configuration.

        Args:
            config: Runtime configuration specifying backend and model path
        """
        self.config = config
        self.backend = None
        self.manifest = None
        self._loaded = False

    def load(self) -> None:
        """Load model using specified backend."""
        if self._loaded:
            logger.warning("Model already loaded")
            return

        model_path = Path(self.config.model_path)

        # Try to load manifest.json (optional for direct file paths)
        manifest_path = model_path / "manifest.json" if model_path.is_dir() else None
        if manifest_path and manifest_path.exists():
            self.manifest = load_manifest(str(model_path))
            format_path = get_format_path(self.manifest, self.config.backend, model_path)
            if not format_path:
                raise ValueError(
                    f"Format '{self.config.backend}' not available in manifest. "
                    f"Available: {list(self.manifest.get('formats', {}).keys())}"
                )
            actual_model_path = format_path
            logger.info(f"Using format '{self.config.backend}' from manifest: {actual_model_path}")
        else:
            # Direct path to model file/directory
            actual_model_path = model_path
            logger.info(f"Loading model directly from: {actual_model_path}")

        # Instantiate backend
        if self.config.backend == "gguf":
            self.backend = GGUFBackend()
            self.backend.load_model(
                str(actual_model_path),
                n_ctx=self.config.n_ctx or 4096,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=self.config.verbose,
            )
        elif self.config.backend == "mlx":
            self.backend = MLXBackend()
            self.backend.load_model(str(actual_model_path))
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        self._loaded = True
        logger.info(f"Model loaded successfully with {self.config.backend} backend")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Override config max_tokens
            temperature: Override config temperature
            top_p: Override config top_p
            stop: List of stop sequences
            stream: Enable streaming output

        Returns:
            Generated text (or iterator if streaming)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            stop=stop,
            stream=stream,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Generate OpenAI-compatible chat completion.

        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Override config max_tokens
            temperature: Override config temperature
            top_p: Override config top_p
            stop: List of stop sequences
            stream: Enable streaming output

        Returns:
            OpenAI-compatible chat completion response (or iterator if streaming)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.backend.chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            stop=stop,
            stream=stream,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.backend.count_tokens(text)

    def unload(self) -> None:
        """Unload model and free resources."""
        if self.backend:
            self.backend.unload()
            self.backend = None
            self._loaded = False
            logger.info("Model unloaded")

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
