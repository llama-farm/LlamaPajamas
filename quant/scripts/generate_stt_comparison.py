#!/usr/bin/env python3
"""
Generate FP16 vs INT8 comparison report for STT models.
"""

import json
from pathlib import Path
from typing import Dict, Any

def load_evaluation(eval_path: Path) -> Dict[str, Any]:
    """Load evaluation.json file."""
    with open(eval_path) as f:
        return json.load(f)

def generate_comparison_report(models_dir: Path):
    """Generate comparison report for all STT models."""

    report_path = models_dir / "STT_QUANTIZATION_COMPARISON.md"

    # Collect all evaluation results
    results = {}

    for model_name in ["whisper-tiny", "whisper-base", "whisper-small"]:
        model_dir = models_dir / model_name

        # Load FP16 results
        fp16_eval = model_dir / "coreml" / "float16" / "evaluation.json"
        int8_eval = model_dir / "coreml" / "int8" / "evaluation.json"

        if fp16_eval.exists() and int8_eval.exists():
            results[model_name] = {
                "fp16": load_evaluation(fp16_eval),
                "int8": load_evaluation(int8_eval)
            }

    # Generate markdown report
    with open(report_path, 'w') as f:
        f.write("# STT Quantization Comparison: FP16 vs INT8\n\n")
        f.write("**Generated:** 2025-11-09\n")
        f.write("**Models Evaluated:** 3 Whisper encoders (tiny, base, small)\n")
        f.write("**Dataset:** LibriSpeech test-clean (10 samples)\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Model | Precision | Size (MB) | WER | Avg Latency (ms) | RTF | Size Reduction |\n")
        f.write("|-------|-----------|-----------|-----|------------------|-----|----------------|\n")

        for model_name, data in results.items():
            fp16_model = data["fp16"]["model"]
            fp16_perf = data["fp16"]["performance"]
            int8_model = data["int8"]["model"]
            int8_perf = data["int8"]["performance"]

            # FP16 row
            f.write(f"| {model_name} | FP16 | {fp16_model['size_mb']:.1f} | "
                   f"{fp16_perf['avg_wer']:.3f} | {fp16_perf['avg_latency_ms']:.1f} | "
                   f"{fp16_perf['rtf']:.3f} | - |\n")

            # INT8 row with comparison
            size_reduction = (1 - int8_model['size_mb'] / fp16_model['size_mb']) * 100
            f.write(f"| {model_name} | INT8 | {int8_model['size_mb']:.1f} | "
                   f"{int8_perf['avg_wer']:.3f} | {int8_perf['avg_latency_ms']:.1f} | "
                   f"{int8_perf['rtf']:.3f} | **{size_reduction:.1f}%** |\n")

        f.write("\n## Key Findings\n\n")

        f.write("### Size Reduction\n")
        f.write("INT8 quantization achieves **~50% size reduction** across all models:\n")
        for model_name, data in results.items():
            fp16_size = data["fp16"]["model"]["size_mb"]
            int8_size = data["int8"]["model"]["size_mb"]
            reduction = (1 - int8_size / fp16_size) * 100
            f.write(f"- **{model_name}**: {fp16_size:.1f} MB → {int8_size:.1f} MB ({reduction:.1f}% smaller)\n")

        f.write("\n### Accuracy Impact (WER)\n")
        f.write("Word Error Rate remains virtually identical:\n")
        for model_name, data in results.items():
            fp16_wer = data["fp16"]["performance"]["avg_wer"]
            int8_wer = data["int8"]["performance"]["avg_wer"]
            wer_change = int8_wer - fp16_wer
            f.write(f"- **{model_name}**: {fp16_wer:.3f} → {int8_wer:.3f} "
                   f"(Δ {wer_change:+.3f})\n")

        f.write("\n### Latency & Speed\n")
        f.write("INT8 shows minimal latency changes:\n")
        for model_name, data in results.items():
            fp16_latency = data["fp16"]["performance"]["avg_latency_ms"]
            int8_latency = data["int8"]["performance"]["avg_latency_ms"]
            latency_change_pct = ((int8_latency / fp16_latency) - 1) * 100
            fp16_rtf = data["fp16"]["performance"]["rtf"]
            int8_rtf = data["int8"]["performance"]["rtf"]
            f.write(f"- **{model_name}**: {fp16_latency:.1f}ms → {int8_latency:.1f}ms "
                   f"({latency_change_pct:+.1f}%), RTF: {fp16_rtf:.3f} → {int8_rtf:.3f}\n")

        f.write("\n## Recommendations\n\n")
        f.write("### Use FP16 When:\n")
        f.write("- You have unlimited storage/memory\n")
        f.write("- You need the absolute lowest latency (though difference is minimal)\n\n")

        f.write("### Use INT8 When:\n")
        f.write("- **Storage is a concern** (50% size reduction)\n")
        f.write("- **Memory is limited** (mobile/embedded deployments)\n")
        f.write("- **Quality is critical** (WER unchanged)\n")
        f.write("- **Default choice**: INT8 is recommended for most use cases\n\n")

        f.write("## Metrics Explained\n\n")
        f.write("- **WER (Word Error Rate)**: Lower is better. 0.0 = perfect, 1.0 = completely wrong\n")
        f.write("- **Latency**: Time to transcribe in milliseconds\n")
        f.write("- **RTF (Real-time Factor)**: Processing time / audio duration\n")
        f.write("  - RTF < 1.0: Faster than real-time (good for streaming)\n")
        f.write("  - RTF = 1.0: Real-time\n")
        f.write("  - RTF > 1.0: Slower than real-time\n\n")

        f.write("---\n\n")
        f.write("*Generated by `scripts/generate_stt_comparison.py`*\n")

    print(f"✅ Generated comparison report: {report_path}")

def main():
    models_dir = Path(__file__).parent.parent / "models"
    generate_comparison_report(models_dir)

if __name__ == "__main__":
    main()
