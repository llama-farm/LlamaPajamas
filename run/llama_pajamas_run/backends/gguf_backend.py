"""GGUF backend using llama-cpp-python."""

import time
from typing import Dict, Any, List, Iterator, Optional
from .base import Backend


class GGUFBackend(Backend):
    """GGUF inference backend using llama-cpp-python."""

    def __init__(self):
        """Initialize GGUF backend."""
        self.model = None
        self._model_path = None

    def load_model(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, verbose: bool = False, **kwargs) -> None:
        """Load GGUF model using llama-cpp-python.

        Args:
            model_path: Path to .gguf file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            verbose: Enable verbose logging
            **kwargs: Additional llama-cpp-python parameters
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-pajamas-run[cuda] or uv add llama-cpp-python"
            )

        self._model_path = model_path
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            **kwargs
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate text from prompt."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if stream:
            return self._stream_generate(prompt, max_tokens, temperature, top_p, stop)

        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )
        return output["choices"][0]["text"]

    def _stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]) -> Iterator[str]:
        """Stream generation token by token."""
        output_stream = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=True,
        )

        for chunk in output_stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Generate OpenAI-compatible chat completion."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
        )

        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        tokens = self.model.tokenize(text.encode("utf-8"))
        return len(tokens)

    def unload(self) -> None:
        """Unload model and free resources."""
        if self.model:
            del self.model
            self.model = None
