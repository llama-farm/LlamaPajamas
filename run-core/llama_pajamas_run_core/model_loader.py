"""Model loader with OpenAI-compatible API."""

from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional

from .config import RuntimeConfig
from .manifest_loader import load_manifest
from .backends import Backend


class ModelLoader:
    """High-level model loader with OpenAI-compatible API.

    Backend is injected by the specific runtime package
    (llama-pajamas-run-mlx or llama-pajamas-run-gguf).
    """

    def __init__(self, config: RuntimeConfig, backend: Backend):
        """Initialize model loader with runtime configuration and backend.

        Args:
            config: Runtime configuration
            backend: Backend implementation (MLX or GGUF)
        """
        self.config = config
        self.backend = backend
        self.manifest = None
        self._loaded = False

    def load(self) -> None:
        """Load model using provided backend."""
        if self._loaded:
            return

        model_path = Path(self.config.model_path)

        # Try to load manifest.json (optional for direct file paths)
        manifest_path = model_path / "manifest.json" if model_path.is_dir() else None
        if manifest_path and manifest_path.exists():
            self.manifest = load_manifest(str(model_path))
            # Use manifest to find actual model path
            # (This is optional - backends can also work with direct paths)

        # Load model via backend
        self.backend.load_model(
            str(model_path),
            n_ctx=self.config.n_ctx or 4096,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
        )

        self._loaded = True

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
            prompt: Input text
            max_tokens: Max tokens to generate (uses config.max_tokens if None)
            temperature: Sampling temperature (uses config.temperature if None)
            top_p: Nucleus sampling parameter (uses config.top_p if None)
            stop: Stop sequences
            stream: Whether to stream response

        Returns:
            Generated text or iterator of text chunks
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
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream response

        Returns:
            OpenAI-format response or iterator of response chunks
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
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.backend.count_tokens(text)

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._loaded:
            self.backend.unload()
            self._loaded = False

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
