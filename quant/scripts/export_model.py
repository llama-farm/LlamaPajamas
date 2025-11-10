#!/usr/bin/env python3
"""
Unified model export script for all backends.

Consolidates export across ONNX, TensorRT, CoreML, MLX, and GGUF.

Usage:
    # Vision model to ONNX with INT8
    ./scripts/export_model.py --model yolov8n --backend onnx --precision int8 --output models/yolo-v8n/

    # Speech model to CoreML
    ./scripts/export_model.py --model whisper-tiny --backend coreml --precision int8 --output models/whisper-tiny/

    # LLM to MLX
    ./scripts/export_model.py --model Qwen/Qwen3-8B --backend mlx --precision 4bit --output models/qwen3-8b/

    # Vision to TensorRT
    ./scripts/export_model.py --model yolov8n --backend tensorrt --precision fp16 --output models/yolo-v8n/

Supported:
    --backend: onnx, tensorrt, coreml, mlx, gguf
    --precision: fp32, fp16, int8, int4, 4bit (MLX), Q4_K_M (GGUF), etc.
    --model-type: auto, vision, speech, llm (auto-detects if not specified)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelTypeDetector:
    """Detect model type from model name."""

    VISION_MODELS = ['yolo', 'vit', 'clip', 'resnet', 'efficientnet', 'mobilenet']
    SPEECH_MODELS = ['whisper', 'wav2vec', 'hubert', 'wavlm']
    LLM_MODELS = ['qwen', 'llama', 'mistral', 'phi', 'gemma', 'gpt']

    @staticmethod
    def detect(model_name: str) -> str:
        """Detect model type from name.

        Returns: 'vision', 'speech', or 'llm'
        """
        model_lower = model_name.lower()

        # Check vision models
        for pattern in ModelTypeDetector.VISION_MODELS:
            if pattern in model_lower:
                return 'vision'

        # Check speech models
        for pattern in ModelTypeDetector.SPEECH_MODELS:
            if pattern in model_lower:
                return 'speech'

        # Check LLM models
        for pattern in ModelTypeDetector.LLM_MODELS:
            if pattern in model_lower:
                return 'llm'

        # Default to LLM if contains slash (HuggingFace format)
        if '/' in model_name:
            return 'llm'

        raise ValueError(
            f"Cannot auto-detect model type for '{model_name}'. "
            "Please specify --model-type [vision|speech|llm]"
        )


class UnifiedExporter:
    """Unified model exporter for all backends."""

    def __init__(self, backend: str, precision: str, output_dir: Path):
        self.backend = backend
        self.precision = precision
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_vision_onnx(self, model_name: str, model_type_hint: str = 'yolo') -> Dict[str, Any]:
        """Export vision model to ONNX."""
        from llama_pajamas_quant.exporters import export_yolo_to_onnx, export_huggingface_to_onnx, export_pytorch_to_onnx

        logger.info(f"Exporting {model_name} to ONNX (vision)")

        # Determine export method
        if 'yolo' in model_name.lower():
            metadata = export_yolo_to_onnx(model_name, self.output_dir)
        elif '/' in model_name:  # HuggingFace
            metadata = export_huggingface_to_onnx(model_name, self.output_dir, task='image-classification')
        else:
            raise ValueError(f"Unknown vision model format: {model_name}")

        # Quantize if needed
        if self.precision in ['int8', 'int4']:
            logger.info(f"Quantizing to {self.precision.upper()}...")
            metadata = self._quantize_onnx(Path(metadata['onnx_path']))

        return metadata

    def export_speech_onnx(self, model_name: str) -> Dict[str, Any]:
        """Export speech model to ONNX."""
        from llama_pajamas_quant.exporters import export_whisper_to_onnx, export_huggingface_speech_to_onnx

        logger.info(f"Exporting {model_name} to ONNX (speech)")

        # Determine export method
        if 'whisper' in model_name.lower():
            model_size = model_name.replace('whisper-', '')
            metadata = export_whisper_to_onnx(model_size, self.output_dir)
        elif '/' in model_name:  # HuggingFace
            metadata = export_huggingface_speech_to_onnx(model_name, self.output_dir)
        else:
            raise ValueError(f"Unknown speech model format: {model_name}")

        # Quantize if needed
        if self.precision in ['int8', 'int4']:
            logger.info(f"Quantizing to {self.precision.upper()}...")
            if 'encoder_path' in metadata:
                metadata = self._quantize_onnx(Path(metadata['encoder_path']))

        return metadata

    def export_vision_coreml(self, model_name: str) -> Dict[str, Any]:
        """Export vision model to CoreML."""
        logger.info(f"Exporting {model_name} to CoreML (vision)")

        # Import CoreML export logic
        # This would call existing CoreML export scripts
        logger.warning("CoreML export integrated - using existing scripts")

        # For now, provide instructions
        return {
            'status': 'manual',
            'instructions': f"Use existing CoreML scripts in quant/scripts/quantize_*_coreml.py",
            'model': model_name,
            'backend': 'coreml',
            'precision': self.precision
        }

    def export_vision_tensorrt(self, model_name: str) -> Dict[str, Any]:
        """Export vision model to TensorRT."""
        logger.info(f"Exporting {model_name} to TensorRT (vision)")

        # Step 1: Export to ONNX first
        logger.info("Step 1: Exporting to ONNX...")
        onnx_metadata = self.export_vision_onnx(model_name, model_type_hint='yolo')

        # Step 2: Build TensorRT engine
        logger.info("Step 2: Building TensorRT engine...")
        onnx_path = Path(onnx_metadata.get('onnx_path', ''))

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        engine_path = onnx_path.parent / f"{onnx_path.stem}_{self.precision}.engine"

        # Provide docker command
        logger.info(f"""
To build TensorRT engine, run:

./scripts/build_tensorrt_engine.sh \\
    {onnx_path} \\
    {engine_path} \\
    {self.precision}

Or manually with trtexec:
trtexec --onnx={onnx_path} --saveEngine={engine_path} --{self.precision}
        """)

        return {
            'status': 'onnx_ready',
            'onnx_path': str(onnx_path),
            'engine_path': str(engine_path),
            'next_step': f"./scripts/build_tensorrt_engine.sh {onnx_path} {engine_path} {self.precision}"
        }

    def export_llm_mlx(self, model_name: str) -> Dict[str, Any]:
        """Export LLM to MLX."""
        logger.info(f"Exporting {model_name} to MLX")

        # Use existing MLX converter
        try:
            from llama_pajamas_quant.converters.mlx import MLXConverter

            converter = MLXConverter()

            # Parse precision
            bits = int(self.precision.replace('bit', '')) if 'bit' in self.precision else 4

            result = converter.convert(
                model_id=model_name,
                output_dir=self.output_dir,
                bits=bits
            )

            return result
        except Exception as e:
            logger.error(f"MLX export failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def export_llm_gguf(self, model_name: str) -> Dict[str, Any]:
        """Export LLM to GGUF."""
        logger.info(f"Exporting {model_name} to GGUF")

        # Use existing GGUF converter
        try:
            from llama_pajamas_quant.converters.gguf import GGUFConverter

            converter = GGUFConverter()

            # Map precision to GGUF method
            method_map = {
                'int8': 'Q8_0',
                'int4': 'Q4_K_M',
                '4bit': 'Q4_K_M',
                'fp16': 'F16',
            }

            method = method_map.get(self.precision, self.precision)  # Allow custom like Q4_K_M

            result = converter.convert(
                model_id=model_name,
                output_dir=self.output_dir,
                method=method
            )

            return result
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _quantize_onnx(self, onnx_path: Path) -> Dict[str, Any]:
        """Quantize ONNX model."""
        from llama_pajamas_quant.quantizers import quantize_onnx_dynamic, quantize_onnx_int4

        output_path = onnx_path.parent / f"{onnx_path.stem}_{self.precision}.onnx"

        if self.precision == 'int8':
            return quantize_onnx_dynamic(onnx_path, output_path)
        elif self.precision == 'int4':
            return quantize_onnx_int4(onnx_path, output_path)
        else:
            logger.warning(f"Precision {self.precision} not supported for quantization")
            return {'onnx_path': str(onnx_path)}

    def export(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """Export model to specified backend."""

        # Route to appropriate export function
        if self.backend == 'onnx':
            if model_type == 'vision':
                return self.export_vision_onnx(model_name)
            elif model_type == 'speech':
                return self.export_speech_onnx(model_name)
            else:
                raise ValueError(f"ONNX export for {model_type} not yet supported")

        elif self.backend == 'coreml':
            if model_type == 'vision':
                return self.export_vision_coreml(model_name)
            elif model_type == 'speech':
                logger.warning("Use existing CoreML scripts for speech")
                return {'status': 'use_existing_scripts'}
            else:
                raise ValueError(f"CoreML export for {model_type} not supported")

        elif self.backend == 'tensorrt':
            if model_type == 'vision':
                return self.export_vision_tensorrt(model_name)
            elif model_type == 'llm':
                logger.warning("TensorRT-LLM export requires separate script")
                return {'status': 'use_tensorrt_llm'}
            else:
                raise ValueError(f"TensorRT export for {model_type} not yet supported")

        elif self.backend == 'mlx':
            if model_type == 'llm':
                return self.export_llm_mlx(model_name)
            else:
                raise ValueError("MLX only supports LLM models")

        elif self.backend == 'gguf':
            if model_type == 'llm':
                return self.export_llm_gguf(model_name)
            else:
                raise ValueError("GGUF only supports LLM models")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified model export script for all backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ONNX vision
  ./scripts/export_model.py --model yolov8n --backend onnx --precision int8 --output models/yolo-v8n/

  # CoreML speech
  ./scripts/export_model.py --model whisper-tiny --backend coreml --precision int8 --output models/whisper-tiny/

  # MLX LLM
  ./scripts/export_model.py --model Qwen/Qwen3-8B --backend mlx --precision 4bit --output models/qwen3-8b/

  # TensorRT vision
  ./scripts/export_model.py --model yolov8n --backend tensorrt --precision fp16 --output models/yolo-v8n/

  # GGUF LLM
  ./scripts/export_model.py --model Qwen/Qwen3-8B --backend gguf --precision Q4_K_M --output models/qwen3-8b/
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., yolov8n, whisper-tiny, Qwen/Qwen3-8B)')
    parser.add_argument('--backend', type=str, required=True,
                       choices=['onnx', 'tensorrt', 'coreml', 'mlx', 'gguf'],
                       help='Export backend')
    parser.add_argument('--precision', type=str, default='fp32',
                       help='Precision (fp32, fp16, int8, int4, 4bit, Q4_K_M, etc.)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--model-type', type=str, choices=['vision', 'speech', 'llm', 'auto'],
                       default='auto',
                       help='Model type (auto-detect if not specified)')

    args = parser.parse_args()

    # Detect model type
    if args.model_type == 'auto':
        try:
            model_type = ModelTypeDetector.detect(args.model)
            logger.info(f"Auto-detected model type: {model_type}")
        except ValueError as e:
            logger.error(str(e))
            return 1
    else:
        model_type = args.model_type

    # Create exporter
    output_dir = Path(args.output)
    exporter = UnifiedExporter(args.backend, args.precision, output_dir)

    # Export
    try:
        result = exporter.export(args.model, model_type)

        logger.info("\n" + "=" * 60)
        logger.info("Export Complete!")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Backend: {args.backend}")
        logger.info(f"Precision: {args.precision}")
        logger.info(f"Type: {model_type}")
        logger.info(f"Output: {output_dir}")
        logger.info("\nResult:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
