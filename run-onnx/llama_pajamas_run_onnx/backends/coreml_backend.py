"""CoreML backend for ONNX Runtime on Apple Silicon.

This backend uses the CoreML execution provider to leverage the
Apple Neural Engine (ANE) for accelerated inference on Mac.

Requirements:
- macOS with Apple Silicon (M1/M2/M3)
- ONNX Runtime with CoreML EP
- Model quantized with INT8 symmetric (ANE requirement)

Performance targets:
- M3 Max: 50-70 tok/s for Qwen3-8B INT8
- M2 Max: 40-60 tok/s for Qwen3-8B INT8
- M1 Max: 30-50 tok/s for Qwen3-8B INT8
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .base import Backend, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class CoreMLBackend(Backend):
    """CoreML backend for Apple Silicon inference.

    This backend uses ONNX Runtime's CoreML execution provider to
    run models on the Apple Neural Engine.

    Example:
        >>> from llama_pajamas_run_onnx.backends import CoreMLBackend
        >>> from pathlib import Path
        >>>
        >>> backend = CoreMLBackend(
        ...     model_path=Path("models/qwen3-8b/onnx/CoreML/int8/model.onnx"),
        ...     provider_options={"use_ane": True}
        ... )
        >>> backend._load_model()
        >>> result = backend.generate("Hello, world!")
        >>> print(result.text)
        >>> print(f"Speed: {result.tokens_per_second:.1f} tok/s")
    """

    def __init__(
        self,
        model_path: Path,
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CoreML backend.

        Args:
            model_path: Path to ONNX model file.
            provider_options: Optional CoreML-specific options.
                - use_ane: Whether to use Apple Neural Engine (default: True)
                - compute_units: "CPU_AND_NE" | "CPU_ONLY" | "ALL" (default: "ALL")
        """
        # Default CoreML options
        default_options = {
            "use_ane": True,  # Use Apple Neural Engine
            "compute_units": "ALL",  # Use all available compute units
        }
        if provider_options:
            default_options.update(provider_options)

        super().__init__(
            model_path=model_path,
            execution_provider="CoreMLExecutionProvider",
            provider_options=default_options,
        )

        # CoreML-specific state
        self.input_names = None
        self.output_names = None

    def _load_model(self):
        """Load ONNX model with CoreML execution provider.

        This method:
        1. Creates ONNX Runtime session with CoreML EP
        2. Loads tokenizer from model directory
        3. Caches input/output names for inference
        """
        logger.info(f"Loading ONNX model: {self.model_path}")

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Configure CoreML provider
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    # ANE optimization
                    "use_ane": self.provider_options.get("use_ane", True),
                    "compute_units": self.provider_options.get("compute_units", "ALL"),
                },
            ),
            "CPUExecutionProvider",  # Fallback
        ]

        logger.info(f"Creating session with providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")

        try:
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )

            # Log which provider is actually being used
            actual_providers = self.session.get_providers()
            logger.info(f"Active providers: {actual_providers}")

            if "CoreMLExecutionProvider" not in actual_providers:
                logger.warning(
                    "CoreML EP not active! Falling back to CPU. "
                    "Make sure you're on macOS with Apple Silicon."
                )

            # Get input/output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            logger.info(f"Inputs: {self.input_names}")
            logger.info(f"Outputs: {self.output_names}")

            # Load tokenizer (should be in parent directory of ONNX model)
            # Structure: models/qwen3-8b/onnx/CoreML/int8/model.onnx
            # Tokenizer: models/qwen3-8b/ (root)
            tokenizer_path = self._find_tokenizer_path()
            logger.info(f"Loading tokenizer from: {tokenizer_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _find_tokenizer_path(self) -> Path:
        """Find tokenizer directory from ONNX model path.

        Returns:
            Path to tokenizer directory.
        """
        # Try to find tokenizer in parent directories
        # models/qwen3-8b/onnx/CoreML/int8/model.onnx → models/qwen3-8b/
        current = self.model_path.parent

        for _ in range(5):  # Search up to 5 levels
            # Check if tokenizer files exist
            if (current / "tokenizer_config.json").exists() or (current / "tokenizer.json").exists():
                return current
            current = current.parent

        # Fallback: assume it's a HuggingFace model ID
        logger.warning(f"Tokenizer not found in parent directories of {self.model_path}")
        logger.warning("Please ensure tokenizer files are in the model root directory")
        raise FileNotFoundError(f"Tokenizer not found for {self.model_path}")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from prompt using CoreML.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Returns:
            GenerationResult with generated text and metadata.
        """
        if config is None:
            config = GenerationConfig()

        if self.session is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        logger.info(f"Generating with CoreML (max_tokens={config.max_tokens})...")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)

        # Generate tokens
        generated_tokens = []
        start_time = time.time()

        for _ in range(config.max_tokens):
            # Run inference
            ort_inputs = {
                "input_ids": input_ids,
            }

            # Add attention mask if needed
            if "attention_mask" in self.input_names:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
                ort_inputs["attention_mask"] = attention_mask

            outputs = self.session.run(self.output_names, ort_inputs)

            # Get logits (last token)
            logits = outputs[0][0, -1, :]  # Shape: [vocab_size]

            # Sample next token
            if config.temperature == 0.0:
                # Greedy sampling
                next_token = np.argmax(logits)
            else:
                # Temperature sampling
                logits = logits / config.temperature
                probs = self._softmax(logits)

                # Top-p sampling
                if config.top_p < 1.0:
                    next_token = self._sample_top_p(probs, config.top_p)
                else:
                    next_token = np.random.choice(len(probs), p=probs)

            next_token = int(next_token)
            generated_tokens.append(next_token)

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

            # Append to input_ids for next iteration
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

        end_time = time.time()

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_per_second = len(generated_tokens) / (end_time - start_time)

        logger.info(f"Generated {len(generated_tokens)} tokens at {tokens_per_second:.1f} tok/s")

        return GenerationResult(
            text=generated_text,
            tokens=generated_tokens,
            num_tokens=len(generated_tokens),
            tokens_per_second=tokens_per_second,
            metadata={
                "prompt": prompt,
                "config": config,
            },
        )

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
        if config is None:
            config = GenerationConfig()

        if self.session is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)

        # Generate tokens with streaming
        for _ in range(config.max_tokens):
            # Run inference
            ort_inputs = {
                "input_ids": input_ids,
            }

            if "attention_mask" in self.input_names:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
                ort_inputs["attention_mask"] = attention_mask

            outputs = self.session.run(self.output_names, ort_inputs)

            # Get next token
            logits = outputs[0][0, -1, :]

            if config.temperature == 0.0:
                next_token = np.argmax(logits)
            else:
                logits = logits / config.temperature
                probs = self._softmax(logits)
                next_token = (
                    self._sample_top_p(probs, config.top_p)
                    if config.top_p < 1.0
                    else np.random.choice(len(probs), p=probs)
                )

            next_token = int(next_token)

            # Decode and yield token
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            yield token_text

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

            # Append to input_ids
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.

        Args:
            logits: Logit values.

        Returns:
            Probability distribution.
        """
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)

    @staticmethod
    def _sample_top_p(probs: np.ndarray, top_p: float) -> int:
        """Sample from top-p (nucleus) distribution.

        Args:
            probs: Probability distribution.
            top_p: Cumulative probability threshold.

        Returns:
            Sampled token index.
        """
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff index
        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = min(cutoff_idx + 1, len(sorted_indices))  # Include at least one token

        # Sample from truncated distribution
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)  # Renormalize

        return int(np.random.choice(top_indices, p=top_probs))


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python coreml_backend.py <model.onnx> [prompt]")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"

    # Create backend
    backend = CoreMLBackend(model_path)

    # Load model
    backend._load_model()

    # Generate
    config = GenerationConfig(max_tokens=50, temperature=0.7)
    result = backend.generate(prompt, config)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result.text}")
    print(f"Speed: {result.tokens_per_second:.1f} tok/s")

    # Benchmark
    bench_result = backend.benchmark(num_tokens=100)
    print(f"\nBenchmark: {bench_result['tokens_per_second']:.1f} tok/s")
