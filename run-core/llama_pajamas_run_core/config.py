"""Runtime configuration for Llama-Pajamas."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    """Configuration for model inference runtime.

    Backend is determined by which runtime package is used
    (llama-pajamas-run-mlx or llama-pajamas-run-gguf).
    """

    model_path: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # GGUF-specific options (ignored by MLX)
    n_ctx: Optional[int] = None
    n_gpu_layers: int = -1

    # Server options
    host: str = "127.0.0.1"
    port: int = 8000

    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
