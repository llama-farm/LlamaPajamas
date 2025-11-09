"""Vision model converters (PyTorch → CoreML).

Supports:
- YOLO-v8: Object detection
- CLIP: Image embeddings
- ViT: Image classification
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import torch
import numpy as np

from .base import CoreMLConverter, CoreMLConverterConfig

logger = logging.getLogger(__name__)


class YOLOv8Converter(CoreMLConverter):
    """Convert YOLO-v8 models to CoreML.

    Handles detection pipeline:
    1. Download YOLO-v8 from ultralytics
    2. Export to CoreML with preprocessing/postprocessing
    3. Optimize for ANE (FP16, NHWC)
    """

    def convert(
        self,
        model_name_or_path: str = "yolov8n.pt",
        output_dir: Union[str, Path] = "./models/yolo-v8n-coreml",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs
    ) -> Path:
        """Convert YOLO-v8 to CoreML.

        Args:
            model_name_or_path: YOLO model path or name (yolov8n.pt, yolov8s.pt, etc.)
            output_dir: Output directory
            input_size: Input image size (width, height)
            **kwargs: Additional options

        Returns:
            Path to .mlpackage
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Converting YOLO-v8: {model_name_or_path}")
        self.logger.info(f"Input size: {input_size}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            # Import ultralytics
            from ultralytics import YOLO

            # Load YOLO model
            self.logger.info("Loading YOLO model...")
            model = YOLO(model_name_or_path)

            # Export to CoreML
            self.logger.info("Exporting to CoreML...")
            export_path = model.export(
                format="coreml",
                imgsz=input_size[0],  # ultralytics uses square images
                nms=True,  # Include NMS in model
                half=self.config.precision == "float16",  # FP16 for ANE
            )

            self.logger.info(f"Exported to: {export_path}")

            # Move to output directory
            import shutil
            final_path = output_dir / "model.mlpackage"
            if final_path.exists():
                shutil.rmtree(final_path)
            shutil.move(str(export_path), str(final_path))

            # Validate model
            if self._validate_model(final_path):
                self.logger.info("✅ YOLO-v8 conversion successful")
            else:
                raise RuntimeError("Model validation failed")

            # Save metadata
            metadata = {
                "model_type": "yolo-v8",
                "source": model_name_or_path,
                "input_size": input_size,
                "precision": self.config.precision,
                "compute_units": self.config.compute_units,
                "optimize_for_ane": self.config.optimize_for_ane,
            }
            self._save_metadata(output_dir, metadata)

            return final_path

        except ImportError:
            self.logger.error("ultralytics not installed: pip install ultralytics")
            raise
        except Exception as e:
            self.logger.error(f"YOLO-v8 conversion failed: {e}")
            raise


class CLIPConverter(CoreMLConverter):
    """Convert CLIP vision encoder to CoreML.

    Handles embedding pipeline:
    1. Download CLIP from HuggingFace/OpenAI
    2. Extract vision encoder only
    3. Export to CoreML with preprocessing
    4. Optimize for ANE (FP16, NHWC)
    """

    def convert(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        output_dir: Union[str, Path] = "./models/clip-vit-base-coreml",
        input_size: Tuple[int, int] = (224, 224),
        **kwargs
    ) -> Path:
        """Convert CLIP vision encoder to CoreML.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            output_dir: Output directory
            input_size: Input image size (width, height)
            **kwargs: Additional options

        Returns:
            Path to .mlpackage
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Converting CLIP: {model_name_or_path}")
        self.logger.info(f"Input size: {input_size}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            import coremltools as ct
            from transformers import CLIPModel, CLIPProcessor
            from PIL import Image

            # Load CLIP model
            self.logger.info("Loading CLIP model...")
            model = CLIPModel.from_pretrained(model_name_or_path)
            processor = CLIPProcessor.from_pretrained(model_name_or_path)

            # Extract vision encoder only
            vision_model = model.vision_model
            vision_model.eval()

            # Wrapper to extract just the pooled output (not dict)
            class CLIPVisionWrapper(torch.nn.Module):
                def __init__(self, vision_model):
                    super().__init__()
                    self.vision_model = vision_model

                def forward(self, x):
                    # Returns just the pooled output tensor
                    output = self.vision_model(x)
                    return output.pooler_output  # Shape: (batch, embed_dim)

            wrapped_model = CLIPVisionWrapper(vision_model)
            wrapped_model.eval()

            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

            # Trace model
            self.logger.info("Tracing vision encoder...")
            traced_model = torch.jit.trace(wrapped_model, dummy_input)

            # Convert to CoreML
            self.logger.info("Converting to CoreML...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.ImageType(
                        name="image",
                        shape=(1, 3, input_size[0], input_size[1]),
                        scale=1.0 / 255.0,  # Normalize to [0, 1]
                        bias=[0, 0, 0],
                    )
                ],
                outputs=[ct.TensorType(name="embedding")],
                compute_units=self._get_compute_units(),
                minimum_deployment_target=ct.target.macOS12,
            )

            # Apply ANE optimizations
            mlmodel = self._apply_ane_optimizations(mlmodel)

            # Save model
            final_path = output_dir / "model.mlpackage"
            mlmodel.save(str(final_path))

            # Validate model
            if self._validate_model(final_path):
                self.logger.info("✅ CLIP conversion successful")
            else:
                raise RuntimeError("Model validation failed")

            # Save metadata
            metadata = {
                "model_type": "clip",
                "source": model_name_or_path,
                "input_size": input_size,
                "embedding_dim": model.config.projection_dim,
                "precision": self.config.precision,
                "compute_units": self.config.compute_units,
                "optimize_for_ane": self.config.optimize_for_ane,
            }
            self._save_metadata(output_dir, metadata)

            return final_path

        except ImportError as e:
            self.logger.error(f"Missing dependency: {e}")
            self.logger.error(
                "Install with: pip install transformers torch pillow"
            )
            raise
        except Exception as e:
            self.logger.error(f"CLIP conversion failed: {e}")
            raise


class ViTConverter(CoreMLConverter):
    """Convert Vision Transformer (ViT) to CoreML.

    Handles classification pipeline:
    1. Download ViT from HuggingFace
    2. Export to CoreML with preprocessing
    3. Optimize for ANE (FP16, NHWC)
    """

    def convert(
        self,
        model_name_or_path: str = "google/vit-base-patch16-224",
        output_dir: Union[str, Path] = "./models/vit-base-coreml",
        input_size: Tuple[int, int] = (224, 224),
        **kwargs
    ) -> Path:
        """Convert ViT to CoreML.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            output_dir: Output directory
            input_size: Input image size (width, height)
            **kwargs: Additional options

        Returns:
            Path to .mlpackage
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Converting ViT: {model_name_or_path}")
        self.logger.info(f"Input size: {input_size}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            import coremltools as ct
            from transformers import ViTForImageClassification, ViTImageProcessor
            from PIL import Image

            # Load ViT model
            self.logger.info("Loading ViT model...")
            model = ViTForImageClassification.from_pretrained(model_name_or_path)
            processor = ViTImageProcessor.from_pretrained(model_name_or_path)
            model.eval()

            # Wrapper to extract just the logits (not dict)
            class ViTWrapper(torch.nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.vit_model = vit_model

                def forward(self, x):
                    # Returns just the logits tensor
                    output = self.vit_model(x)
                    return output.logits  # Shape: (batch, num_classes)

            wrapped_model = ViTWrapper(model)
            wrapped_model.eval()

            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

            # Trace model
            self.logger.info("Tracing ViT model...")
            traced_model = torch.jit.trace(wrapped_model, dummy_input)

            # Convert to CoreML
            self.logger.info("Converting to CoreML...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.ImageType(
                        name="image",
                        shape=(1, 3, input_size[0], input_size[1]),
                        scale=1.0 / 255.0,  # Normalize to [0, 1]
                        bias=[0, 0, 0],
                    )
                ],
                outputs=[ct.TensorType(name="logits")],
                compute_units=self._get_compute_units(),
                minimum_deployment_target=ct.target.macOS12,
                convert_to="mlprogram",  # Use ML Program (better ANE support)
            )

            # Apply ANE optimizations
            mlmodel = self._apply_ane_optimizations(mlmodel)

            # Save model
            final_path = output_dir / "model.mlpackage"
            mlmodel.save(str(final_path))

            # Validate model
            if self._validate_model(final_path):
                self.logger.info("✅ ViT conversion successful")
            else:
                raise RuntimeError("Model validation failed")

            # Save metadata
            metadata = {
                "model_type": "vit",
                "source": model_name_or_path,
                "input_size": input_size,
                "num_classes": model.config.num_labels,
                "precision": self.config.precision,
                "compute_units": self.config.compute_units,
                "optimize_for_ane": self.config.optimize_for_ane,
                "labels": list(model.config.id2label.values()),
            }
            self._save_metadata(output_dir, metadata)

            return final_path

        except ImportError as e:
            self.logger.error(f"Missing dependency: {e}")
            self.logger.error(
                "Install with: pip install transformers torch pillow"
            )
            raise
        except Exception as e:
            self.logger.error(f"ViT conversion failed: {e}")
            raise


# Convenience functions
def convert_yolo(
    model_name: str = "yolov8n.pt",
    output_dir: Union[str, Path] = "./models/yolo-v8n-coreml",
    config: Optional[CoreMLConverterConfig] = None,
    **kwargs
) -> Path:
    """Convert YOLO-v8 to CoreML.

    Args:
        model_name: YOLO model name
        output_dir: Output directory
        config: Converter configuration
        **kwargs: Additional options

    Returns:
        Path to .mlpackage
    """
    converter = YOLOv8Converter(config or CoreMLConverterConfig())
    return converter.convert(model_name, output_dir, **kwargs)


def convert_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    output_dir: Union[str, Path] = "./models/clip-vit-base-coreml",
    config: Optional[CoreMLConverterConfig] = None,
    **kwargs
) -> Path:
    """Convert CLIP vision encoder to CoreML.

    Args:
        model_name: HuggingFace model ID
        output_dir: Output directory
        config: Converter configuration
        **kwargs: Additional options

    Returns:
        Path to .mlpackage
    """
    converter = CLIPConverter(config or CoreMLConverterConfig())
    return converter.convert(model_name, output_dir, **kwargs)


def convert_vit(
    model_name: str = "google/vit-base-patch16-224",
    output_dir: Union[str, Path] = "./models/vit-base-coreml",
    config: Optional[CoreMLConverterConfig] = None,
    **kwargs
) -> Path:
    """Convert ViT to CoreML.

    Args:
        model_name: HuggingFace model ID
        output_dir: Output directory
        config: Converter configuration
        **kwargs: Additional options

    Returns:
        Path to .mlpackage
    """
    converter = ViTConverter(config or CoreMLConverterConfig())
    return converter.convert(model_name, output_dir, **kwargs)
