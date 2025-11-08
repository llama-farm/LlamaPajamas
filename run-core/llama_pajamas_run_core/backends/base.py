"""Base backend interface for model inference."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Iterator


class Backend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load model from path.

        Args:
            model_path: Path to model file/directory
            **kwargs: Backend-specific parameters
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            stream: Enable streaming output

        Returns:
            Generated text (or iterator if streaming)
        """
        pass

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Generate chat completion response.

        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            stream: Enable streaming output

        Returns:
            OpenAI-compatible chat completion response (or iterator if streaming)
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass
