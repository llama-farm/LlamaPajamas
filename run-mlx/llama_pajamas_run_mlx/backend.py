"""MLX backend for Apple Silicon inference."""

import time
from typing import Dict, Any, List, Iterator, Optional
from llama_pajamas_run_core.backends import Backend


class MLXBackend(Backend):
    """MLX inference backend for Apple Silicon."""

    def __init__(self):
        """Initialize MLX backend."""
        self.model = None
        self.tokenizer = None
        self._model_path = None

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load MLX model.

        Args:
            model_path: Path to MLX model directory
            **kwargs: Additional mlx-lm parameters
        """
        try:
            from mlx_lm import load
        except ImportError:
            raise RuntimeError(
                "mlx-lm not installed. Install with: "
                "pip install llama-pajamas-run[mlx] or uv add mlx mlx-lm"
            )

        self._model_path = model_path
        self.model, self.tokenizer = load(model_path)

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
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from mlx_lm import generate as mlx_generate

        if stream:
            return self._stream_generate(prompt, max_tokens, temperature, top_p)

        # MLX generate function has a simpler API
        # Note: temperature and top_p are not directly supported in current MLX version
        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response

    def _stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Iterator[str]:
        """Stream generation token by token."""
        from mlx_lm import generate as mlx_generate

        # MLX streaming returns tokens
        # Note: temperature and top_p are not directly supported in current MLX version
        for token in mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        ):
            yield token

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
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)

        # Generate response
        response_text = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=False,
        )

        # Format as OpenAI-compatible response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-pajamas-mlx",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self.count_tokens(prompt),
                "completion_tokens": self.count_tokens(response_text),
                "total_tokens": self.count_tokens(prompt + response_text),
            },
        }

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to single prompt.

        Uses a simple template. In production, should use model-specific chat template.
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def unload(self) -> None:
        """Unload model and free resources."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
