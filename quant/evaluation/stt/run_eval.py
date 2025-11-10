#!/usr/bin/env python3
"""
Evaluate CoreML Whisper STT models from quant/models directory.

This script follows the same pattern as vision evaluations:
- Models are in quant/models/{model-name}/coreml/{precision}/
- Results are saved alongside the model
- Comparisons are generated at the top level

Metrics:
- WER (Word Error Rate): Primary accuracy metric
- Latency: Time to transcribe (ms)
- RTF (Real-time Factor): Processing time / audio duration
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_run_coreml.backends.stt import CoreMLSTTBackend
from llama_pajamas_run_core.utils.audio_utils import (
    load_audio,
    calculate_wer,
    calculate_rtf,
    normalize_text,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_models(base_dir: Path) -> List[Dict[str, Any]]:
    """Find all CoreML Whisper models in quant/models structure."""
    models = []

    for model_dir in base_dir.glob("whisper-*/"):
        if not model_dir.is_dir():
            continue

        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Find CoreML encoder models
        for fmt in manifest.get("formats", []):
            if fmt["format"] == "coreml" and fmt.get("component") == "encoder":
                encoder_path = model_dir / fmt["path"]
                if encoder_path.exists():
                    models.append({
                        "name": manifest["model_name"],
                        "encoder_path": encoder_path,
                        "precision": fmt["precision"],
                        "task": manifest["task"],
                        "manifest": manifest,
                        "format_info": fmt,
                        "model_dir": model_dir,
                    })

    return models


def evaluate_stt_model(
    model_info: Dict[str, Any],
    audio_samples: List[Dict[str, Any]],
    audio_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single STT model and save results."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating: {model_info['name']} ({model_info['precision']})")
    logger.info(f"Encoder: {model_info['encoder_path']}")
    logger.info(f"{'='*70}")

    # Initialize backend
    backend = CoreMLSTTBackend()

    # Extract model name (tiny, base, small)
    model_name = model_info['name'].replace('whisper-', '')

    try:
        backend.load_model(
            str(model_info['encoder_path']),
            model_name=model_name
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {}

    # Evaluate on all audio samples
    results = []
    total_wer = 0.0
    total_latency = 0.0
    total_audio_duration = 0.0
    total_processing_time = 0.0

    for sample in audio_samples:
        audio_path = audio_dir / sample["audio_file"]
        reference_text = sample["text"]

        logger.info(f"Processing: {sample['id']}")

        try:
            # Load audio
            audio = load_audio(str(audio_path), sample_rate=16000)
            audio_duration = len(audio) / 16000

            # Transcribe
            start_time = time.time()
            transcription = backend.transcribe(audio, sample_rate=16000)
            processing_time = (time.time() - start_time) * 1000  # ms

            # Calculate WER
            wer = calculate_wer(reference_text, transcription.text)

            # Calculate RTF
            rtf = calculate_rtf(processing_time / 1000, audio_duration)

            result = {
                "sample_id": sample["id"],
                "reference": reference_text,
                "hypothesis": transcription.text,
                "wer": wer,
                "latency_ms": processing_time,
                "audio_duration_sec": audio_duration,
                "rtf": rtf,
            }

            results.append(result)

            total_wer += wer
            total_latency += processing_time
            total_audio_duration += audio_duration
            total_processing_time += processing_time / 1000

            logger.info(f"  WER: {wer:.3f}, Latency: {processing_time:.1f}ms, RTF: {rtf:.3f}")

        except Exception as e:
            logger.error(f"Failed to process {sample['id']}: {e}")
            continue

    # Calculate aggregate metrics
    num_samples = len(results)
    if num_samples == 0:
        logger.warning("No samples were successfully processed")
        return {}

    avg_wer = total_wer / num_samples
    avg_latency = total_latency / num_samples
    overall_rtf = total_processing_time / total_audio_duration

    evaluation_result = {
        "task": "speech_to_text",
        "num_samples": num_samples,
        "performance": {
            "avg_wer": round(avg_wer, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "rtf": round(overall_rtf, 3),
            "total_audio_sec": round(total_audio_duration, 2),
            "total_processing_sec": round(total_processing_time, 2),
        },
        "model": {
            "name": model_info["name"],
            "format": "coreml",
            "precision": model_info["precision"],
            "size_mb": model_info["format_info"]["file_size_mb"],
            "hardware": model_info["format_info"]["hardware_requirements"],
            "component": "encoder",
        },
        "samples": results,
    }

    # Save evaluation.json
    eval_path = model_info["encoder_path"].parent / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation_result, f, indent=2)

    logger.info(f"\n📊 Results Summary:")
    logger.info(f"   Avg WER: {avg_wer:.3f} ({avg_wer*100:.1f}% error rate)")
    logger.info(f"   Avg Latency: {avg_latency:.1f}ms")
    logger.info(f"   RTF: {overall_rtf:.3f} ({'faster' if overall_rtf < 1 else 'slower'} than real-time)")
    logger.info(f"   Saved: {eval_path}")

    return evaluation_result


def generate_comparison_report(
    all_results: List[Dict[str, Any]],
    models_dir: Path,
) -> None:
    """Generate EVALUATION_REPORT.md for each model."""
    # Group by model name
    by_model = {}
    for result in all_results:
        model_name = result["model"]["name"]
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(result)

    # Generate report for each model
    for model_name, results in by_model.items():
        model_dir = models_dir / model_name
        report_path = model_dir / "EVALUATION_REPORT.md"

        with open(report_path, "w") as f:
            f.write(f"# {model_name} STT Evaluation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Formats Evaluated:** {len(results)}\n")
            f.write(f"**Task:** Speech-to-Text (Whisper)\n\n")

            f.write("## Performance Summary\n\n")
            f.write("| Precision | WER | Avg Latency (ms) | RTF | Size (MB) |\n")
            f.write("|-----------|-----|------------------|-----|----------|\n")

            for result in results:
                perf = result["performance"]
                model = result["model"]
                f.write(f"| {model['precision']} | "
                       f"{perf['avg_wer']:.3f} | {perf['avg_latency_ms']:.1f} | "
                       f"{perf['rtf']:.3f} | {model['size_mb']:.1f} |\n")

            f.write("\n## Metrics Explained\n\n")
            f.write("- **WER (Word Error Rate)**: Lower is better. 0.0 = perfect, 1.0 = completely wrong\n")
            f.write("- **Latency**: Time to transcribe in milliseconds\n")
            f.write("- **RTF (Real-time Factor)**: Processing time / audio duration\n")
            f.write("  - RTF < 1.0: Faster than real-time (good for streaming)\n")
            f.write("  - RTF = 1.0: Real-time\n")
            f.write("  - RTF > 1.0: Slower than real-time\n\n")

            f.write("## Sample Results\n\n")
            for result in results:
                f.write(f"### {result['model']['format']} ({result['model']['precision']})\n\n")

                # Show first 3 samples
                for sample in result.get("samples", [])[:3]:
                    f.write(f"**Sample**: {sample['sample_id']}\n")
                    f.write(f"- Reference: `{sample['reference']}`\n")
                    f.write(f"- Hypothesis: `{sample['hypothesis']}`\n")
                    f.write(f"- WER: {sample['wer']:.3f}\n")
                    f.write(f"- Latency: {sample['latency_ms']:.1f}ms\n\n")

            f.write("---\n\n")
            f.write("*Generated by `evaluation/stt/run_eval.py`*\n")

        logger.info(f"Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoreML Whisper STT models from quant/models directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../quant/models",
        help="Base models directory (default: ../quant/models)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Audio directory (default: evaluation/stt/audio)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset JSON file (default: evaluation/stt/dataset.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Evaluate specific model only (e.g., whisper-tiny)",
    )

    args = parser.parse_args()

    # Resolve paths
    models_dir = Path(args.models_dir).resolve()

    if args.audio_dir:
        audio_dir = Path(args.audio_dir).resolve()
    else:
        audio_dir = Path(__file__).parent / "audio"

    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
    else:
        dataset_path = Path(__file__).parent / "dataset.json"

    # Check paths
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        logger.info(f"Run: uv run python evaluation/stt/download_audio.py")
        sys.exit(1)

    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    audio_samples = dataset["samples"]
    logger.info(f"Loaded {len(audio_samples)} audio samples from {dataset_path}")

    # Find models
    logger.info(f"Searching for models in: {models_dir}")
    all_models = find_models(models_dir)

    if args.model:
        all_models = [m for m in all_models if m["name"] == args.model]

    if not all_models:
        logger.error("No models found!")
        sys.exit(1)

    logger.info(f"Found {len(all_models)} models to evaluate")

    # Evaluate all models
    all_results = []
    for model_info in all_models:
        result = evaluate_stt_model(
            model_info,
            audio_samples,
            audio_dir,
        )
        if result:
            all_results.append(result)

    # Generate comparison reports
    if all_results:
        generate_comparison_report(all_results, models_dir)
        logger.info(f"\n{'='*70}")
        logger.info("Evaluation complete!")
        logger.info(f"Evaluated {len(all_results)} models")
        logger.info(f"{'='*70}")
    else:
        logger.warning("No models were successfully evaluated")


if __name__ == "__main__":
    main()
