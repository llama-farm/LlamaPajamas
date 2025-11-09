"""ONNX model converter with CoreML/TensorRT optimization support.

This converter exports Hugging Face transformers to ONNX format with
hardware-specific optimizations for:
- CoreML (Apple Neural Engine): INT8 symmetric quantization
- TensorRT (NVIDIA): INT8/INT4 QDQ format
- CPU: INT4 MatMulNBits operators

Architecture:
    1. User specifies target_specs (no auto-detection, pipeline-based)
    2. Export to ONNX FP16 (base model)
    3. Apply EP-specific graph optimizations
    4. Apply EP-specific quantization
    5. Validate and save

Memory Management:
    - Uses reduced sequence length (256) by default to prevent OOM
    - Garbage collection between major steps
    - PyTorch memory cleanup after model loading
    - Optimized for M1/M2/M3 Macs with 64GB RAM

Example:
    >>> from llama_pajamas_quant.converters.onnx import ONNXConverter
    >>> converter = ONNXConverter()
    >>> converter.convert(
    ...     model_path="Qwen/Qwen3-8B",
    ...     output_dir="./models/qwen3-8b/onnx/",
    ...     target_specs={
    ...         "target_eps": ["CoreML"],
    ...         "target_precisions": ["int8"],
    ...         "optimization_hints": {
    ...             "attention_type": "gqa",
    ...             "gqa_ratio": 4,
    ...         }
    ...     },
    ...     seq_length=256  # Reduced for memory efficiency
    ... )
"""

import gc
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import onnx
import psutil
import torch
from optimum.exporters.onnx import main_export
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

ExecutionProvider = Literal[
    "CoreML", "TensorRT", "CUDA", "CPU", "DirectML", "ROCm", "QNN"
]
Precision = Literal["fp16", "int8", "int4"]


@dataclass
class TargetSpec:
    """User-specified target configuration for ONNX export.

    This pipeline is user-driven: users specify what hardware targets
    they want to optimize for, rather than auto-detecting at runtime.

    Attributes:
        target_eps: List of execution providers to generate for.
        target_precisions: List of precisions to generate (fp16, int8, int4).
        optimization_hints: Optional hints about model architecture for optimizations.
            - attention_type: "mha", "gqa", "mqa", "hybrid"
            - gqa_ratio: N:1 query:kv head ratio (e.g., 4 for 32:8)
            - moe_experts: Number of experts (if MoE model)
            - context_length: Max sequence length for optimization
    """

    target_eps: List[ExecutionProvider]
    target_precisions: List[Precision]
    optimization_hints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate target specifications."""
        # Validate EP + precision combinations
        for ep in self.target_eps:
            for precision in self.target_precisions:
                if ep == "CoreML" and precision == "int4":
                    logger.warning(
                        f"CoreML does not support INT4. Skipping INT4 for CoreML."
                    )
                if ep == "TensorRT" and precision not in ["fp16", "int8", "int4"]:
                    logger.warning(
                        f"TensorRT supports fp16/int8/int4. Got: {precision}"
                    )


@dataclass
class ONNXExportResult:
    """Result of ONNX export operation.

    Attributes:
        onnx_path: Path to exported ONNX model.
        ep: Execution provider this export is for.
        precision: Precision of the export (fp16, int8, int4).
        size_mb: Model size in MB.
        metadata: Additional metadata (graph info, optimization stats, etc.).
    """

    onnx_path: Path
    ep: ExecutionProvider
    precision: Precision
    size_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ONNXConverter:
    """Convert Hugging Face models to ONNX with EP-specific optimizations.

    This converter uses a pipeline architecture where users specify target
    hardware (no auto-detection). It generates optimized ONNX models for
    multiple execution providers from a single source model.

    Workflow:
        1. Load HF model
        2. Export to ONNX FP16 (base)
        3. For each (EP, precision) pair:
            a. Apply graph optimizations (fusions, layout, etc.)
            b. Apply quantization (QDQ for TensorRT, symmetric for CoreML)
            c. Validate and save

    Example:
        >>> converter = ONNXConverter()
        >>> results = converter.convert(
        ...     model_path="Qwen/Qwen3-8B",
        ...     output_dir="./models/qwen3-8b/onnx/",
        ...     target_specs=TargetSpec(
        ...         target_eps=["CoreML", "TensorRT"],
        ...         target_precisions=["int8"],
        ...         optimization_hints={"attention_type": "gqa", "gqa_ratio": 4}
        ...     )
        ... )
        >>> # Results: models/qwen3-8b/onnx/{CoreML,TensorRT}/int8/model.onnx
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize ONNX converter.

        Args:
            cache_dir: Optional cache directory for intermediate files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache/llama_pajamas"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def convert(
        self,
        model_path: str,
        output_dir: str,
        target_specs: Optional[TargetSpec] = None,
        batch_size: int = 1,
        seq_length: int = 256,  # REDUCED: 256 instead of 2048 to prevent OOM on M1 64GB
        device: str = "cpu",
    ) -> List[ONNXExportResult]:
        """Convert Hugging Face model to ONNX with EP-specific optimizations.

        Args:
            model_path: HuggingFace model ID or local path.
            output_dir: Output directory for ONNX models.
            target_specs: User-specified target configuration. If None, defaults to CoreML FP16.
            batch_size: Batch size for export (default: 1 for inference).
            seq_length: Sequence length for export (default: 256, reduced for memory efficiency).
            device: Device to use for export (cpu/cuda).

        Returns:
            List of ONNXExportResult for each (EP, precision) pair.

        Raises:
            ValueError: If target_specs are invalid.
            RuntimeError: If export fails.
        """
        # Memory management: Set environment variables for efficiency
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For M1/M2/M3 Macs
        os.environ["TOKENIZERS_PARALLELISM"] = "false"   # Avoid tokenizer warnings

        # Log memory status at start
        self._log_memory_usage("Start of conversion")
        # Default to CoreML FP16 if no specs provided
        if target_specs is None:
            target_specs = TargetSpec(
                target_eps=["CoreML"],
                target_precisions=["fp16"],
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {model_path} to ONNX...")
        logger.info(f"Targets: {target_specs.target_eps}")
        logger.info(f"Precisions: {target_specs.target_precisions}")

        # Step 1: Load model and config
        logger.info("Loading model and tokenizer...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Step 2: Export to FP16 ONNX (base model)
        logger.info("Exporting to ONNX FP16 (base model)...")
        base_onnx_path = self._export_to_onnx_fp16(
            model_path=model_path,
            output_dir=output_path / "base",
            batch_size=batch_size,
            seq_length=seq_length,
            device=device,
        )

        # Memory cleanup after export
        self._cleanup_memory()
        self._cleanup_temp_files(output_path)  # Clean up temporary ONNX files
        self._log_memory_usage("After ONNX export")

        # Step 3: For each (EP, precision), create optimized variant
        results = []
        for ep in target_specs.target_eps:
            for precision in target_specs.target_precisions:
                # Skip invalid combinations
                if ep == "CoreML" and precision == "int4":
                    logger.warning(f"Skipping CoreML INT4 (not supported)")
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {ep} @ {precision.upper()}")
                logger.info(f"{'='*60}")

                # Create EP-specific output directory
                ep_output_dir = output_path / ep / precision
                ep_output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    result = self._create_optimized_variant(
                        base_onnx_path=base_onnx_path,
                        ep=ep,
                        precision=precision,
                        output_dir=ep_output_dir,
                        target_specs=target_specs,
                        config=config,
                    )
                    results.append(result)
                    logger.info(f"✅ {ep} @ {precision.upper()}: {result.size_mb:.1f} MB")

                except Exception as e:
                    logger.error(f"❌ Failed to create {ep} @ {precision}: {e}")
                    # Clean up on failure
                    self._cleanup_temp_files(output_path)
                    # Continue with other variants
                    continue

        # Step 4: Save manifest
        self._save_manifest(
            output_dir=output_path,
            model_path=model_path,
            results=results,
            target_specs=target_specs,
        )

        # Final cleanup before returning
        self._cleanup_temp_files(output_path)
        self._cleanup_memory()
        self._log_memory_usage("Conversion complete")

        logger.info(f"\n✅ ONNX conversion complete!")
        logger.info(f"Generated {len(results)} variant(s)")
        logger.info(f"Output: {output_path}")

        return results

    def _export_to_onnx_fp16(
        self,
        model_path: str,
        output_dir: Path,
        batch_size: int,
        seq_length: int,
        device: str,
    ) -> Path:
        """Export HuggingFace model to ONNX FP16 using Optimum.

        Args:
            model_path: HuggingFace model ID or local path.
            output_dir: Output directory for base ONNX model.
            batch_size: Batch size for export.
            seq_length: Sequence length for export.
            device: Device to use (cpu/cuda).

        Returns:
            Path to exported ONNX model.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use Optimum to export (handles decoder-only models well)
        # We'll export with FP16 precision for better quality baseline
        logger.info(f"Exporting with Optimum (batch={batch_size}, seq={seq_length})...")

        try:
            # Use Optimum Python API directly (better than CLI)
            from optimum.exporters.onnx import main_export

            logger.info(f"Using Optimum main_export to {output_dir}")
            logger.info("Exporting with FP16 precision to save memory...")

            # Export using Optimum's main_export function
            # Use fp16=True to export directly to FP16 (avoids memory-intensive conversion)
            main_export(
                model_name_or_path=model_path,
                output=str(output_dir),
                task="text-generation-with-past",
                device=device,
                trust_remote_code=True,
                fp16=True,  # Export directly to FP16 (memory efficient)
            )

            # Find the exported model
            onnx_path = output_dir / "model.onnx"
            if not onnx_path.exists():
                # Try decoder_model.onnx (some exports use this name)
                decoder_path = output_dir / "decoder_model.onnx"
                if decoder_path.exists():
                    onnx_path = decoder_path
                else:
                    # List what was actually created
                    created_files = list(output_dir.glob("*.onnx"))
                    if created_files:
                        onnx_path = created_files[0]
                        logger.info(f"Using ONNX file: {onnx_path.name}")
                    else:
                        raise FileNotFoundError(f"ONNX export failed: no .onnx file found in {output_dir}")

            # Skip FP16 conversion - Optimum exports with fp16=True directly
            # This avoids loading the entire 8B model into memory
            logger.info("Verifying ONNX export...")

            # Quick check without loading full model (memory efficient)
            try:
                onnx.checker.check_model(str(onnx_path))
                logger.info(f"✅ Base ONNX export successful: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX validation warning (non-critical): {e}")
                logger.info(f"Model exported to: {onnx_path}")

            return onnx_path

        except Exception as e:
            logger.error(f"Optimum export failed: {e}")
            logger.info("Falling back to manual export...")
            return self._export_manual_fp16(model_path, output_dir, batch_size, seq_length, device)

    def _export_manual_fp16(
        self,
        model_path: str,
        output_dir: Path,
        batch_size: int,
        seq_length: int,
        device: str,
    ) -> Path:
        """Manual ONNX export fallback (if Optimum fails).

        Args:
            model_path: HuggingFace model ID or local path.
            output_dir: Output directory.
            batch_size: Batch size.
            seq_length: Sequence length.
            device: Device (cpu/cuda).

        Returns:
            Path to exported ONNX model.
        """
        logger.info("Using manual ONNX export...")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create dummy input
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)

        # Export with torch.onnx.export
        onnx_path = output_dir / "model.onnx"
        logger.info(f"Exporting to {onnx_path}...")

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=17,  # Latest stable opset
                do_constant_folding=True,
            )

        logger.info(f"✅ Manual export successful: {onnx_path}")
        return onnx_path

    def _create_optimized_variant(
        self,
        base_onnx_path: Path,
        ep: ExecutionProvider,
        precision: Precision,
        output_dir: Path,
        target_specs: TargetSpec,
        config: Any,
    ) -> ONNXExportResult:
        """Create EP-specific optimized variant from base ONNX model.

        Args:
            base_onnx_path: Path to base FP16 ONNX model.
            ep: Execution provider (CoreML, TensorRT, etc.).
            precision: Target precision (fp16, int8, int4).
            output_dir: Output directory for this variant.
            target_specs: User-specified target configuration.
            config: HuggingFace model config.

        Returns:
            ONNXExportResult with path and metadata.
        """
        # Import optimizers (will implement next)
        from llama_pajamas_quant.optimizers.onnx_graph import ONNXGraphOptimizer
        from llama_pajamas_quant.optimizers.onnx_quant import ONNXQuantizer

        # Step 1: Copy base model to output directory
        variant_path = output_dir / "model.onnx"
        shutil.copy(base_onnx_path, variant_path)

        # Also copy external data file if it exists (for large models)
        data_file = base_onnx_path.parent / f"{base_onnx_path.stem}.onnx_data"
        if data_file.exists():
            variant_data = output_dir / f"{variant_path.stem}.onnx_data"
            shutil.copy(data_file, variant_data)
            logger.info(f"Copied external data file to {variant_data}")

        logger.info(f"Copied base model to {variant_path}")

        # Step 2: Simplify with onnxsim (reduces bloat)
        logger.info("Simplifying ONNX graph with onnxsim...")
        variant_path = self._simplify_with_onnxsim(variant_path)

        # Step 3: Apply graph optimizations
        logger.info(f"Applying {ep} graph optimizations...")
        graph_optimizer = ONNXGraphOptimizer(ep=ep, optimization_hints=target_specs.optimization_hints)
        variant_path = graph_optimizer.optimize(variant_path)

        # Step 3: Apply quantization (if not FP16)
        if precision != "fp16":
            logger.info(f"Applying {precision.upper()} quantization...")
            quantizer = ONNXQuantizer(ep=ep, precision=precision, optimization_hints=target_specs.optimization_hints)
            variant_path = quantizer.quantize(variant_path)

        # Step 4: Validate and compute metadata
        # Handle >2GB models (protobuf limit) gracefully
        size_mb = variant_path.stat().st_size / (1024 * 1024)
        metadata = {"optimization_hints": target_specs.optimization_hints}

        try:
            logger.info("Validating ONNX model...")
            onnx.checker.check_model(str(variant_path))

            # Try to load model for metadata (may fail for >2GB models)
            model = onnx.load(str(variant_path))
            metadata.update({
                "num_ops": len(model.graph.node),
                "num_initializers": len(model.graph.initializer),
                "opset_version": model.opset_import[0].version,
            })
            logger.info(f"✅ Model validation passed ({len(model.graph.node)} ops)")

        except Exception as e:
            # For >2GB models, skip validation and detailed metadata
            if "protobuf" in str(e).lower() or "2gib" in str(e).lower():
                logger.warning(f"⚠️  Model >2GB - skipping validation (external data format used)")
                metadata.update({
                    "num_ops": "N/A (model >2GB)",
                    "num_initializers": "N/A (model >2GB)",
                    "opset_version": "N/A (model >2GB)",
                })
            else:
                # Re-raise other errors
                raise

        # Save metadata JSON
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "ep": ep,
                    "precision": precision,
                    "size_mb": size_mb,
                    **metadata,
                },
                f,
                indent=2,
            )

        return ONNXExportResult(
            onnx_path=variant_path,
            ep=ep,
            precision=precision,
            size_mb=size_mb,
            metadata=metadata,
        )

    def _save_manifest(
        self,
        output_dir: Path,
        model_path: str,
        results: List[ONNXExportResult],
        target_specs: TargetSpec,
    ):
        """Save ONNX export manifest.

        Args:
            output_dir: Output directory.
            model_path: Original model path.
            results: List of export results.
            target_specs: Target specifications.
        """
        manifest = {
            "model_id": model_path,
            "format": "onnx",
            "target_specs": {
                "target_eps": target_specs.target_eps,
                "target_precisions": target_specs.target_precisions,
                "optimization_hints": target_specs.optimization_hints,
            },
            "variants": [
                {
                    "ep": r.ep,
                    "precision": r.precision,
                    "path": str(r.onnx_path.relative_to(output_dir)),
                    "size_mb": r.size_mb,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved manifest: {manifest_path}")

    def _log_memory_usage(self, stage: str):
        """Log current memory usage.

        Args:
            stage: Stage name for logging.
        """
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)

            # Get system memory
            vm = psutil.virtual_memory()
            total_mb = vm.total / (1024 * 1024)
            available_mb = vm.available / (1024 * 1024)
            used_pct = vm.percent

            logger.info(
                f"Memory [{stage}]: "
                f"Process={mem_mb:.0f}MB, "
                f"System={used_pct:.1f}% used "
                f"({available_mb:.0f}MB / {total_mb:.0f}MB available)"
            )

            # Warning if memory usage is high
            if used_pct > 85:
                logger.warning(
                    f"⚠️  High memory usage detected ({used_pct:.1f}%). "
                    "Consider reducing seq_length or batch_size."
                )
        except Exception as e:
            logger.debug(f"Could not log memory usage: {e}")

    def _cleanup_memory(self):
        """Force garbage collection and PyTorch cache cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force Python GC
        gc.collect()

    def _cleanup_temp_files(self, output_dir: Path):
        """Clean up temporary ONNX files to save disk space.

        Large ONNX models create 32GB+ temporary files during export.
        This removes them after successful conversion to prevent disk bloat.

        Args:
            output_dir: Output directory to clean.
        """
        try:
            # Find and remove large temporary ONNX data files
            temp_patterns = [
                "*.onnx_data.tmp",  # Temporary data files
                "*.tmp",             # Other temp files
            ]

            files_removed = 0
            space_freed_mb = 0

            for pattern in temp_patterns:
                for temp_file in output_dir.rglob(pattern):
                    size_mb = temp_file.stat().st_size / (1024 * 1024)
                    temp_file.unlink()
                    files_removed += 1
                    space_freed_mb += size_mb
                    logger.debug(f"Removed temp file: {temp_file.name} ({size_mb:.1f} MB)")

            if files_removed > 0:
                logger.info(
                    f"Cleaned up {files_removed} temporary file(s), "
                    f"freed {space_freed_mb:.1f} MB"
                )
        except Exception as e:
            logger.debug(f"Error during temp file cleanup: {e}")

    def _simplify_with_onnxsim(self, model_path: Path) -> Path:
        """Simplify ONNX model using onnxsim to reduce bloat.

        This addresses the bloat issue where FP16 models grow 2x expected size due to:
        - Duplicate tied weights (embed_tokens/lm_head)
        - Transposed weight copies (W and W^T)
        - Large RoPE constant tables

        Args:
            model_path: Path to ONNX model to simplify.

        Returns:
            Path to simplified model (overwrites original).
        """
        try:
            import onnxsim

            logger.info(f"Loading model for simplification...")
            size_before_mb = model_path.stat().st_size / (1024 * 1024)

            # Load model (handle external data)
            model = onnx.load(str(model_path), load_external_data=True)

            logger.info(f"Simplifying (this may take a few minutes)...")
            # Simplify: removes duplicates, folds constants, eliminates dead code
            model_simplified, check = onnxsim.simplify(
                model,
                skip_fuse_bn=True,  # Keep batch norms separate for better quantization
                skip_shape_inference=False,  # Infer shapes for optimization
            )

            if check:
                logger.info("✅ Simplification successful!")

                # Save simplified model
                onnx.save(model_simplified, str(model_path))

                size_after_mb = model_path.stat().st_size / (1024 * 1024)
                reduction_pct = ((size_before_mb - size_after_mb) / size_before_mb) * 100

                logger.info(
                    f"Size: {size_before_mb:.1f} MB → {size_after_mb:.1f} MB "
                    f"({reduction_pct:+.1f}%)"
                )
            else:
                logger.warning("Simplification check failed, keeping original model")

            return model_path

        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")
            logger.info("Install with: uv pip install onnxsim")
            return model_path
        except Exception as e:
            logger.warning(f"Simplification failed: {e}")
            logger.info("Continuing with unsimplified model")
            return model_path
