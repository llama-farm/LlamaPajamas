"""Performance benchmarking utilities for model inference."""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""

    prompt_tokens: int
    generated_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    time_to_first_token_seconds: Optional[float] = None
    backend: str = ""
    model_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "time_to_first_token_seconds": round(self.time_to_first_token_seconds, 3)
            if self.time_to_first_token_seconds
            else None,
            "backend": self.backend,
            "model_path": self.model_path,
        }

    def __repr__(self) -> str:
        """Pretty string representation."""
        lines = [
            "=" * 60,
            "Benchmark Results",
            "=" * 60,
            f"Backend:              {self.backend}",
            f"Model:                {self.model_path}",
            f"Prompt tokens:        {self.prompt_tokens}",
            f"Generated tokens:     {self.generated_tokens}",
            f"Total time:           {self.total_time_seconds:.3f}s",
            f"Tokens/second:        {self.tokens_per_second:.2f}",
        ]
        if self.time_to_first_token_seconds:
            lines.append(f"Time to first token:  {self.time_to_first_token_seconds:.3f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


def benchmark_generation(
    loader: ModelLoader,
    prompt: str = "Write a detailed explanation of how neural networks work, including backpropagation.",
    num_tokens: int = 200,
    warmup_runs: int = 1,
) -> BenchmarkResult:
    """Benchmark text generation performance.

    Args:
        loader: Loaded ModelLoader instance
        prompt: Prompt to use for generation
        num_tokens: Number of tokens to generate
        warmup_runs: Number of warmup runs before benchmarking

    Returns:
        BenchmarkResult with performance metrics
    """
    if not loader._loaded:
        raise RuntimeError("Model not loaded")

    # Warmup
    logger.info(f"Running {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _ = loader.generate(prompt, max_tokens=50, temperature=0.0)

    # Count prompt tokens
    prompt_tokens = loader.count_tokens(prompt)

    # Benchmark
    logger.info(f"Benchmarking generation of {num_tokens} tokens...")
    start_time = time.time()
    output = loader.generate(prompt, max_tokens=num_tokens, temperature=0.7)
    end_time = time.time()

    # Count generated tokens
    generated_tokens = loader.count_tokens(output)
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_seconds=total_time,
        tokens_per_second=tokens_per_second,
        backend=loader.config.backend,
        model_path=loader.config.model_path,
    )

    logger.info(f"\n{result}")
    return result


def benchmark_streaming(
    loader: ModelLoader,
    prompt: str = "Write a detailed explanation of how neural networks work, including backpropagation.",
    num_tokens: int = 200,
) -> BenchmarkResult:
    """Benchmark streaming generation performance with time-to-first-token.

    Args:
        loader: Loaded ModelLoader instance
        prompt: Prompt to use for generation
        num_tokens: Number of tokens to generate

    Returns:
        BenchmarkResult with performance metrics including TTFT
    """
    if not loader._loaded:
        raise RuntimeError("Model not loaded")

    # Count prompt tokens
    prompt_tokens = loader.count_tokens(prompt)

    # Benchmark streaming
    logger.info(f"Benchmarking streaming generation of {num_tokens} tokens...")
    start_time = time.time()
    first_token_time = None
    generated_text = ""

    for i, chunk in enumerate(loader.generate(prompt, max_tokens=num_tokens, stream=True, temperature=0.7)):
        if i == 0:
            first_token_time = time.time() - start_time
        generated_text += chunk

    end_time = time.time()

    # Count generated tokens
    generated_tokens = loader.count_tokens(generated_text)
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_seconds=total_time,
        tokens_per_second=tokens_per_second,
        time_to_first_token_seconds=first_token_time,
        backend=loader.config.backend,
        model_path=loader.config.model_path,
    )

    logger.info(f"\n{result}")
    return result


def benchmark_chat(
    loader: ModelLoader,
    messages: Optional[list] = None,
    num_tokens: int = 200,
    warmup_runs: int = 1,
) -> BenchmarkResult:
    """Benchmark chat completion performance.

    Args:
        loader: Loaded ModelLoader instance
        messages: Chat messages (default: simple question)
        num_tokens: Number of tokens to generate
        warmup_runs: Number of warmup runs before benchmarking

    Returns:
        BenchmarkResult with performance metrics
    """
    if not loader._loaded:
        raise RuntimeError("Model not loaded")

    if messages is None:
        messages = [
            {"role": "user", "content": "Explain how transformers work in machine learning."}
        ]

    # Warmup
    logger.info(f"Running {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _ = loader.chat(messages, max_tokens=50, temperature=0.0)

    # Count prompt tokens (approximate from messages)
    prompt_text = " ".join([m["content"] for m in messages])
    prompt_tokens = loader.count_tokens(prompt_text)

    # Benchmark
    logger.info(f"Benchmarking chat completion with {num_tokens} token target...")
    start_time = time.time()
    response = loader.chat(messages, max_tokens=num_tokens, temperature=0.7)
    end_time = time.time()

    # Extract response text
    response_text = response["choices"][0]["message"]["content"]
    generated_tokens = loader.count_tokens(response_text)
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_seconds=total_time,
        tokens_per_second=tokens_per_second,
        backend=loader.config.backend,
        model_path=loader.config.model_path,
    )

    logger.info(f"\n{result}")
    return result
