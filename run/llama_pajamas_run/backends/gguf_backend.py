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

    def load_model(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1,
                   n_threads: Optional[int] = None, n_batch: int = 512, n_ubatch: int = 16,
                   verbose: bool = False, config_path: Optional[str] = None, **kwargs) -> None:
        """Load GGUF model using llama-cpp-python.

        Args:
            model_path: Path to .gguf file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_threads: Number of CPU threads (None = auto-detect)
            n_batch: Prompt processing batch size
            n_ubatch: Decode (generation) batch size
            verbose: Enable verbose logging
            config_path: Path to runtime config JSON (overrides other params)
            **kwargs: Additional llama-cpp-python parameters
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-pajamas-run[cuda] or uv add llama-cpp-python"
            )

        # Load config if provided
        if config_path:
            import json
            from pathlib import Path
            with open(Path(config_path)) as f:
                config = json.load(f)
                settings = config.get("settings", {})
                n_ctx = settings.get("n_ctx", n_ctx)
                n_gpu_layers = settings.get("n_gpu_layers", n_gpu_layers)
                n_threads = settings.get("n_threads", n_threads)
                n_batch = settings.get("n_batch", n_batch)
                n_ubatch = settings.get("n_ubatch", n_ubatch)

                if verbose:
                    metadata = config.get("metadata", {})
                    print(f"Loaded config for: {metadata.get('hardware_profile', 'unknown')}")
                    print(f"Expected performance: ~{metadata.get('expected_tokens_per_sec', '?')} tokens/sec")

        self._model_path = model_path
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
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
        enable_thinking: Optional[bool] = None,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Generate OpenAI-compatible chat completion.

        Uses the model's built-in chat template from metadata.

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Enable streaming
            enable_thinking: Control <think> tags (True=allow, False=disable, None=model default)
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # For Qwen3-style thinking control, inject a system message
        final_messages = messages.copy()
        if enable_thinking is False:
            # Add system instruction to skip thinking
            no_thinking_instruction = {
                "role": "system",
                "content": "Respond directly and concisely without showing your reasoning process or using <think> tags."
            }
            # Insert at beginning if no system message, or append to existing
            if not final_messages or final_messages[0].get("role") != "system":
                final_messages.insert(0, no_thinking_instruction)
            else:
                # Append to existing system message
                final_messages[0]["content"] += "\n\n" + no_thinking_instruction["content"]

        response = self.model.create_chat_completion(
            messages=final_messages,
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
