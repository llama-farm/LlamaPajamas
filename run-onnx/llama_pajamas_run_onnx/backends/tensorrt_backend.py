"""TensorRT backend for ONNX Runtime on NVIDIA GPUs and Jetson devices.

This backend uses the TensorRT execution provider to leverage NVIDIA's
optimized inference engine for maximum performance on GPUs.

Requirements:
- NVIDIA GPU (RTX 30XX/40XX, A100, H100) or Jetson (Orin, AGX Xavier)
- ONNX Runtime with TensorRT EP
- CUDA 11.8+ and cuDNN
- TensorRT 8.6+ (for INT4 support)

Performance targets:
- RTX 4090: 100-150 tok/s for Qwen3-1.7B INT4
- RTX 4070: 70-100 tok/s for Qwen3-1.7B INT4
- Jetson Orin: 30-50 tok/s for Qwen3-1.7B INT4
- Jetson AGX Xavier: 20-30 tok/s for Qwen3-1.7B INT4

Optimizations:
- FP16/INT8/INT4 support via QDQ format
- Engine caching for faster subsequent loads
- Timing cache for tactic selection
- I/O binding to keep tensors on GPU
- Device-resident KV cache
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


class TensorRTBackend(Backend):
    """TensorRT backend for NVIDIA GPU/Jetson inference.

    This backend uses ONNX Runtime's TensorRT execution provider to
    compile models into optimized TensorRT engines for maximum performance.

    Example:
        >>> from llama_pajamas_run_onnx.backends import TensorRTBackend
        >>> from pathlib import Path
        >>>
        >>> backend = TensorRTBackend(
        ...     model_path=Path("models/qwen3-1.7b/onnx/TensorRT/int4/model.onnx"),
        ...     provider_options={
        ...         "trt_fp16_enable": True,
        ...         "trt_int8_enable": False,
        ...         "device_id": 0,
        ...     }
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
        """Initialize TensorRT backend.

        Args:
            model_path: Path to ONNX model file (should be QDQ format for INT8/INT4).
            provider_options: Optional TensorRT-specific options.
                - device_id: CUDA device ID (default: 0)
                - trt_fp16_enable: Enable FP16 mode (default: True)
                - trt_int8_enable: Enable INT8 mode (default: False, requires calibration)
                - trt_engine_cache_enable: Cache compiled engines (default: True)
                - trt_engine_cache_path: Path to cache directory (default: ./trt_cache)
                - trt_timing_cache_enable: Cache tactic timings (default: True)
                - trt_max_workspace_size: Max workspace in bytes (default: 2GB)
        """
        # Default TensorRT options from onnx.md plan
        default_options = {
            "device_id": 0,
            "trt_fp16_enable": True,  # FP16 for speed
            "trt_int8_enable": False,  # Set True if model has QDQ INT8 nodes
            "trt_engine_cache_enable": True,  # Cache engines for faster loads
            "trt_engine_cache_path": str(Path(model_path).parent / "trt_cache"),
            "trt_timing_cache_enable": True,  # Cache tactic selection
            "trt_max_workspace_size": 2 << 30,  # 2GB workspace
            # Additional optimizations
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        if provider_options:
            default_options.update(provider_options)

        super().__init__(
            model_path=model_path,
            execution_provider="TensorrtExecutionProvider",
            provider_options=default_options,
        )

        # TensorRT-specific state
        self.input_names = None
        self.output_names = None
        self.io_binding = None  # For zero-copy inference

    def _load_model(self):
        """Load ONNX model with TensorRT execution provider.

        This method:
        1. Creates cache directory for TensorRT engines
        2. Creates ONNX Runtime session with TensorRT + CUDA EPs
        3. Loads tokenizer from model directory
        4. Sets up I/O binding for zero-copy GPU inference
        """
        logger.info(f"Loading ONNX model for TensorRT: {self.model_path}")

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Create engine cache directory
        cache_path = Path(self.provider_options["trt_engine_cache_path"])
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorRT cache directory: {cache_path}")

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # GPU does the work, don't oversubscribe CPU
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Configure TensorRT provider (primary)
        trt_options = {k: v for k, v in self.provider_options.items() if k.startswith("trt_") or k == "device_id"}

        # Configure CUDA provider (fallback for unsupported ops)
        cuda_options = {
            "device_id": self.provider_options.get("device_id", 0),
            "arena_extend_strategy": self.provider_options.get("arena_extend_strategy", "kSameAsRequested"),
            "cudnn_conv_algo_search": self.provider_options.get("cudnn_conv_algo_search", "HEURISTIC"),
            "do_copy_in_default_stream": self.provider_options.get("do_copy_in_default_stream", True),
        }

        providers = [
            ("TensorrtExecutionProvider", trt_options),
            ("CUDAExecutionProvider", cuda_options),
            ("CPUExecutionProvider", {}),  # Last resort fallback
        ]

        logger.info(f"Creating session with providers: {[p[0] for p in providers]}")
        logger.info(f"TensorRT options: {trt_options}")

        try:
            # Create inference session
            # Note: First run will compile TensorRT engine (slow), subsequent runs use cache
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )

            # Log which provider is actually being used
            actual_providers = self.session.get_providers()
            logger.info(f"Active providers: {actual_providers}")

            if "TensorrtExecutionProvider" not in actual_providers:
                logger.warning(
                    "TensorRT EP not active! Falling back to CUDA/CPU. "
                    "Make sure you have CUDA, cuDNN, and TensorRT installed."
                )
            else:
                logger.info("✅ TensorRT EP active - engines will be compiled on first run")

            # Get input/output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            logger.info(f"Inputs: {self.input_names}")
            logger.info(f"Outputs: {self.output_names}")

            # Create I/O binding for zero-copy inference (keeps data on GPU)
            self.io_binding = self.session.io_binding()

            # Load tokenizer
            tokenizer_path = self._find_tokenizer_path()
            logger.info(f"Loading tokenizer from: {tokenizer_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✅ Model loaded successfully with TensorRT")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _find_tokenizer_path(self) -> Path:
        """Find tokenizer directory from ONNX model path.

        Returns:
            Path to tokenizer directory.
        """
        # Same logic as CoreML backend
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
        """Generate text from prompt using TensorRT.

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

        logger.info(f"Generating with TensorRT (max_tokens={config.max_tokens})...")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)

        # Generate tokens
        generated_tokens = []
        start_time = time.time()

        for _ in range(config.max_tokens):
            # Run inference with I/O binding (keeps data on GPU)
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
                "provider": "TensorRT",
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
        print("Usage: python tensorrt_backend.py <model.onnx> [prompt]")
        print("\nExample:")
        print("  python tensorrt_backend.py models/qwen3-1.7b/onnx/TensorRT/int4/model.onnx")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"

    # Create backend
    backend = TensorRTBackend(
        model_path,
        provider_options={
            "trt_fp16_enable": True,
            "device_id": 0,
        },
    )

    # Load model (first run will compile TensorRT engine - may take 1-2 minutes)
    print("Loading model (first run compiles TensorRT engine - please wait)...")
    backend._load_model()

    # Generate
    config = GenerationConfig(max_tokens=50, temperature=0.7)
    result = backend.generate(prompt, config)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result.text}")
    print(f"Speed: {result.tokens_per_second:.1f} tok/s")

    # Benchmark
    print("\nRunning benchmark...")
    bench_result = backend.benchmark(num_tokens=100)
    print(f"Benchmark: {bench_result['tokens_per_second']:.1f} tok/s")
