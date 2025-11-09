"""Base backend interface for ONNX Runtime.

All ONNX Runtime backends (CoreML, TensorRT, CPU, etc.) inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = greedy, 1.0 = random).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        repetition_penalty: Penalty for repeating tokens.
        stream: Whether to stream tokens as they're generated.
    """

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stream: bool = False


@dataclass
class GenerationResult:
    """Result of text generation.

    Attributes:
        text: Generated text.
        tokens: Generated token IDs.
        num_tokens: Number of tokens generated.
        tokens_per_second: Generation speed (tokens/sec).
        metadata: Additional metadata (e.g., logprobs, finish_reason).
    """

    text: str
    tokens: List[int]
    num_tokens: int
    tokens_per_second: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Abstract base class for ONNX Runtime backends.

    Each backend (CoreML, TensorRT, CPU) implements this interface
    to provide a consistent API for model loading and inference.
    """

    def __init__(
        self,
        model_path: Path,
        execution_provider: str,
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize backend.

        Args:
            model_path: Path to ONNX model file.
            execution_provider: ONNX Runtime execution provider name.
            provider_options: Optional provider-specific options.
        """
        self.model_path = Path(model_path)
        self.execution_provider = execution_provider
        self.provider_options = provider_options or {}

        # Will be set by _load_model()
        self.session = None
        self.tokenizer = None

    @abstractmethod
    def _load_model(self):
        """Load ONNX model and create inference session.

        This method should:
        1. Create ONNX Runtime InferenceSession with appropriate EP
        2. Load tokenizer
        3. Set up any EP-specific configuration
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Generate text with streaming (token-by-token).

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Yields:
            Generated tokens as they're produced.
        """
        pass

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
            Dictionary with benchmark results:
                - tokens_per_second: Generation speed
                - latency_ms: Time to first token
                - total_time_ms: Total generation time
        """
        import time

        config = GenerationConfig(
            max_tokens=num_tokens,
            temperature=0.0,  # Greedy for consistent benchmarks
            stream=False,
        )

        start = time.time()
        result = self.generate(prompt, config)
        total_time = time.time() - start

        return {
            "tokens_per_second": result.tokens_per_second,
            "total_time_ms": total_time * 1000,
            "num_tokens": result.num_tokens,
            "prompt": prompt,
        }

    def __enter__(self):
        """Context manager entry."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources
        if self.session:
            del self.session
        self.session = None
