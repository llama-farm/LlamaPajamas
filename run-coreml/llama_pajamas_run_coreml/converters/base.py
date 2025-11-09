"""Base converter for PyTorch → CoreML conversion.

Follows the same architecture pattern as quant/converters with:
- Model detection and validation
- Configuration management
- Optimization for Apple Neural Engine (ANE)
- Error handling and logging
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json

logger = logging.getLogger(__name__)


class CoreMLConverterConfig:
    """Configuration for CoreML conversion."""

    def __init__(
        self,
        precision: str = "float16",
        compute_units: str = "ALL",
        minimum_deployment_target: str = "iOS15",
        optimize_for_ane: bool = True,
        **kwargs
    ):
        """Initialize converter config.

        Args:
            precision: Model precision (float32, float16, mixed)
            compute_units: Compute units (ALL, CPU_AND_GPU, CPU_ONLY, CPU_AND_NE)
            minimum_deployment_target: Minimum iOS/macOS version (iOS15, iOS16, macOS12, etc.)
            optimize_for_ane: Apply ANE-specific optimizations (FP16, NHWC layout)
            **kwargs: Additional model-specific options
        """
        self.precision = precision
        self.compute_units = compute_units
        self.minimum_deployment_target = minimum_deployment_target
        self.optimize_for_ane = optimize_for_ane
        self.extra_options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "precision": self.precision,
            "compute_units": self.compute_units,
            "minimum_deployment_target": self.minimum_deployment_target,
            "optimize_for_ane": self.optimize_for_ane,
            **self.extra_options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreMLConverterConfig":
        """Create config from dictionary."""
        return cls(**data)


class CoreMLConverter(ABC):
    """Base class for PyTorch → CoreML converters."""

    def __init__(self, config: Optional[CoreMLConverterConfig] = None):
        """Initialize converter.

        Args:
            config: Converter configuration
        """
        self.config = config or CoreMLConverterConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def convert(
        self,
        model_name_or_path: str,
        output_dir: Union[str, Path],
        **kwargs
    ) -> Path:
        """Convert PyTorch model to CoreML.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            output_dir: Output directory for .mlpackage
            **kwargs: Model-specific conversion options

        Returns:
            Path to .mlpackage directory
        """
        pass

    def _get_compute_units(self):
        """Get CoreML compute units enum."""
        try:
            import coremltools as ct

            compute_units_map = {
                "ALL": ct.ComputeUnit.ALL,
                "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
                "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
                "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,  # CPU + Neural Engine
            }
            return compute_units_map.get(
                self.config.compute_units, ct.ComputeUnit.ALL
            )
        except ImportError:
            self.logger.error("coremltools not installed")
            raise

    def _apply_ane_optimizations(self, model):
        """Apply Apple Neural Engine optimizations.

        ANE prefers:
        - FP16 precision (not FP32)
        - NHWC layout (not NCHW)
        - Specific operator patterns (avoid unsupported ops)

        Args:
            model: CoreML model to optimize

        Returns:
            Optimized CoreML model
        """
        if not self.config.optimize_for_ane:
            return model

        self.logger.info("Applying ANE optimizations...")

        try:
            import coremltools as ct
            from coremltools.optimize.coreml import (
                OpPalettizerConfig,
                OptimizationConfig,
                palettize_weights,
            )

            # FP16 precision (ANE prefers FP16)
            if self.config.precision == "float16":
                self.logger.info("Converting weights to FP16...")
                model = ct.models.neural_network.quantization_utils.quantize_weights(
                    model, nbits=16
                )

            self.logger.info("✅ ANE optimizations applied")
            return model

        except Exception as e:
            self.logger.warning(f"ANE optimization failed: {e}")
            self.logger.warning("Continuing with non-optimized model")
            return model

    def _save_metadata(self, output_dir: Path, metadata: Dict[str, Any]):
        """Save conversion metadata.

        Args:
            output_dir: Output directory
            metadata: Metadata dictionary
        """
        metadata_path = output_dir / "conversion_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata: {metadata_path}")

    def _validate_model(self, model_path: Path) -> bool:
        """Validate converted CoreML model.

        Args:
            model_path: Path to .mlpackage

        Returns:
            True if valid, False otherwise
        """
        try:
            import coremltools as ct

            self.logger.info(f"Validating CoreML model: {model_path}")
            model = ct.models.MLModel(str(model_path))
            spec = model.get_spec()

            self.logger.info(f"Model type: {spec.WhichOneof('Type')}")
            self.logger.info(f"Inputs: {[inp.name for inp in spec.description.input]}")
            self.logger.info(
                f"Outputs: {[out.name for out in spec.description.output]}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
