"""Runtime configuration for Llama-Pajamas."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RuntimeConfig:
    """Configuration for model inference runtime.

    Attributes:
        backend: Backend to use ('gguf' or 'mlx')
        model_path: Path to model directory containing manifest.json
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        n_ctx: Context window size (default: from model config)
        n_gpu_layers: Number of layers to offload to GPU (GGUF only, -1 = all)
        verbose: Enable verbose logging
    """

    backend: Literal["gguf", "mlx"]
    model_path: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    n_ctx: Optional[int] = None
    n_gpu_layers: int = -1  # GGUF only: -1 = all layers on GPU
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.backend not in ("gguf", "mlx"):
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'gguf' or 'mlx'")

        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
