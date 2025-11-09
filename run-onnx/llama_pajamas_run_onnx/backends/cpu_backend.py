"""Optimized CPU backend for ONNX Runtime (universal fallback).

This backend provides CPU inference with multi-threading optimizations
for systems without dedicated accelerators.

Requirements:
- ONNX Runtime (CPU build)
- Model with INT8 or INT4 quantization for best performance

Performance targets (Qwen3-1.7B INT8):
- AMD Ryzen 9 7950X (16 cores): 30-40 tok/s
- Intel i9-13900K (24 cores): 25-35 tok/s
- Apple M3 Max (CPU only): 20-30 tok/s
- Raspberry Pi 5 (4 cores): 3-5 tok/s

Optimizations:
- Multi-threading (intra_op_num_threads set to physical cores)
- INT8/INT4 MatMulNBits operators for quantized models
- Memory arena management
- Sequential execution mode for better cache locality
"""

import logging
import multiprocessing
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .base import Backend, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


def get_physical_cores() -> int:
    """Get number of physical CPU cores (excluding hyper-threading).

    Returns:
        Number of physical cores, or total cores if detection fails.
    """
    try:
        # Try to get physical cores (works on Linux/macOS)
        import psutil
        return psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
    except (ImportError, AttributeError):
        # Fallback: use total cores
        return multiprocessing.cpu_count()


class CPUBackend(Backend):
    """Optimized CPU backend for universal inference.

    This backend uses ONNX Runtime's CPU execution provider with
    multi-threading optimizations for systems without GPU/NPU.

    Example:
        >>> from llama_pajamas_run_onnx.backends import CPUBackend
        >>> from pathlib import Path
        >>>
        >>> backend = CPUBackend(
        ...     model_path=Path("models/qwen3-1.7b/onnx/CPU/int8/model.onnx"),
        ...     provider_options={"intra_op_num_threads": 8}
        ... )
        >>> backend._load_model()
        >>> result = backend.generate("Hello, world!")
        >>> print(f"Speed: {result.tokens_per_second:.1f} tok/s")
    """

    def __init__(
        self,
        model_path: Path,
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CPU backend.

        Args:
            model_path: Path to ONNX model file (preferably INT8 or INT4 quantized).
            provider_options: Optional CPU-specific options.
                - intra_op_num_threads: Number of threads for intra-op parallelism
                  (default: physical cores)
                - inter_op_num_threads: Number of threads for inter-op parallelism (default: 1)
                - use_arena: Enable memory arena (default: True)
        """
        # Detect physical cores
        physical_cores = get_physical_cores()

        # Default CPU options from onnx.md plan
        default_options = {
            "intra_op_num_threads": physical_cores,  # Use all physical cores
            "inter_op_num_threads": 1,  # Sequential execution usually better for LLMs
            "use_arena": True,  # Memory arena for faster allocations
        }
        if provider_options:
            default_options.update(provider_options)

        super().__init__(
            model_path=model_path,
            execution_provider="CPUExecutionProvider",
            provider_options=default_options,
        )

        # CPU-specific state
        self.input_names = None
        self.output_names = None
        self.physical_cores = physical_cores

    def _load_model(self):
        """Load ONNX model with optimized CPU execution provider.

        This method:
        1. Configures multi-threading for CPU inference
        2. Creates ONNX Runtime session with CPU EP
        3. Loads tokenizer from model directory
        4. Logs performance tips
        """
        logger.info(f"Loading ONNX model for CPU: {self.model_path}")
        logger.info(f"Detected {self.physical_cores} physical CPU cores")

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Threading configuration
        intra_threads = self.provider_options.get("intra_op_num_threads", self.physical_cores)
        inter_threads = self.provider_options.get("inter_op_num_threads", 1)

        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Better for LLMs

        logger.info(f"Threading: intra_op={intra_threads}, inter_op={inter_threads}")

        # Configure CPU provider
        cpu_options = {
            "use_arena": self.provider_options.get("use_arena", True),
        }

        providers = [("CPUExecutionProvider", cpu_options)]

        logger.info(f"Creating session with CPU EP...")

        try:
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )

            # Get input/output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            logger.info(f"Inputs: {self.input_names}")
            logger.info(f"Outputs: {self.output_names}")

            # Load tokenizer
            tokenizer_path = self._find_tokenizer_path()
            logger.info(f"Loading tokenizer from: {tokenizer_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✅ Model loaded successfully with CPU")

            # Performance tips
            if "int8" in str(self.model_path).lower() or "int4" in str(self.model_path).lower():
                logger.info("💡 INT8/INT4 quantized model detected - optimal for CPU inference")
            else:
                logger.warning(
                    "⚠️  FP16/FP32 model detected - consider using INT8 or INT4 "
                    "quantization for 2-4x CPU speedup"
                )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _find_tokenizer_path(self) -> Path:
        """Find tokenizer directory from ONNX model path.

        Returns:
            Path to tokenizer directory.
        """
        # Same logic as other backends
        current = self.model_path.parent

        for _ in range(5):  # Search up to 5 levels
            if (current / "tokenizer_config.json").exists() or (current / "tokenizer.json").exists():
                return current
            current = current.parent

        raise FileNotFoundError(f"Tokenizer not found for {self.model_path}")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from prompt using CPU.

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

        logger.info(f"Generating with CPU (max_tokens={config.max_tokens})...")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)

        # Generate tokens
        generated_tokens = []
        start_time = time.time()

        for _ in range(config.max_tokens):
            # Run inference
            ort_inputs = {"input_ids": input_ids}

            if "attention_mask" in self.input_names:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
                ort_inputs["attention_mask"] = attention_mask

            outputs = self.session.run(self.output_names, ort_inputs)

            # Get logits (last token)
            logits = outputs[0][0, -1, :]  # Shape: [vocab_size]

            # Sample next token
            if config.temperature == 0.0:
                next_token = np.argmax(logits)
            else:
                logits = logits / config.temperature
                probs = self._softmax(logits)

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
                "provider": "CPU",
                "physical_cores": self.physical_cores,
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
            ort_inputs = {"input_ids": input_ids}

            if "attention_mask" in self.input_names:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
                ort_inputs["attention_mask"] = attention_mask

            outputs = self.session.run(self.output_names, ort_inputs)
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
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    @staticmethod
    def _sample_top_p(probs: np.ndarray, top_p: float) -> int:
        """Sample from top-p (nucleus) distribution."""
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)

        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = min(cutoff_idx + 1, len(sorted_indices))

        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)

        return int(np.random.choice(top_indices, p=top_probs))


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python cpu_backend.py <model.onnx> [prompt] [num_threads]")
        print("\nExample:")
        print("  python cpu_backend.py models/qwen3-1.7b/onnx/CPU/int8/model.onnx")
        print("  python cpu_backend.py models/qwen3-1.7b/onnx/CPU/int8/model.onnx 'Hello' 8")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"
    num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else get_physical_cores()

    # Create backend
    backend = CPUBackend(
        model_path,
        provider_options={"intra_op_num_threads": num_threads},
    )

    # Load model
    backend._load_model()

    # Generate
    config = GenerationConfig(max_tokens=50, temperature=0.7)
    result = backend.generate(prompt, config)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result.text}")
    print(f"Speed: {result.tokens_per_second:.1f} tok/s")
    print(f"Cores used: {num_threads}")

    # Benchmark
    print("\nRunning benchmark...")
    bench_result = backend.benchmark(num_tokens=100)
    print(f"Benchmark: {bench_result['tokens_per_second']:.1f} tok/s")
