#!/usr/bin/env python3
"""Evaluate baseline (original) Qwen3-8B model for comparison."""

import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from llama_pajamas_quant.evaluator import ModelEvaluator, STANDARD_PROMPTS


def evaluate_baseline():
    """Evaluate the original Qwen3-8B model."""
    print("=" * 80)
    print("Baseline Evaluation - Original Qwen3-8B (FP16/BF16)")
    print("=" * 80)
    print()

    model_id = "Qwen/Qwen3-8B"  # Original model

    print(f"Loading model: {model_id}")
    print("(Using transformers directly - no quantization)")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("✓ Model loaded")
    print()

    # Create evaluator
    print("Creating evaluator with GPT-5 nano...")
    evaluator = ModelEvaluator(judge_model="gpt-5-nano", prompts=STANDARD_PROMPTS)
    print("✓ Evaluator ready")
    print()

    print("=" * 80)
    print(f"Evaluating Baseline Model ({len(STANDARD_PROMPTS)} prompts)")
    print("=" * 80)
    print()

    results = []
    for i, eval_prompt in enumerate(STANDARD_PROMPTS, 1):
        print(f"[{i}/{len(STANDARD_PROMPTS)}] {eval_prompt.category}: {eval_prompt.prompt[:50]}...")

        # Generate response
        inputs = tokenizer(eval_prompt.prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        gen_time = time.time() - start_time

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Get judge scores
        scores = evaluator.judge.evaluate_response(
            prompt=eval_prompt.prompt,
            response=response,
            category=eval_prompt.category,
            expected_qualities=eval_prompt.expected_qualities,
        )

        print(f"  ✓ Quality: {scores['quality_score']:.1f}/10 ({gen_time:.2f}s)")

        results.append({
            "prompt": eval_prompt.prompt,
            "category": eval_prompt.category,
            "response": response,
            "gen_time": gen_time,
            "scores": scores,
        })

    # Calculate averages
    avg_quality = sum(r["scores"]["quality_score"] for r in results) / len(results)
    avg_accuracy = sum(r["scores"]["accuracy_score"] for r in results) / len(results)
    avg_coherence = sum(r["scores"]["coherence_score"] for r in results) / len(results)
    avg_relevance = sum(r["scores"]["relevance_score"] for r in results) / len(results)
    avg_gen_time = sum(r["gen_time"] for r in results) / len(results)

    print()
    print("=" * 80)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Precision: FP16")
    print()
    print(f"Average Quality:   {avg_quality:.1f}/10")
    print(f"Average Accuracy:  {avg_accuracy:.1f}/10")
    print(f"Average Coherence: {avg_coherence:.1f}/10")
    print(f"Average Relevance: {avg_relevance:.1f}/10")
    print(f"Average Gen Time:  {avg_gen_time:.2f}s")
    print("=" * 80)
    print()

    # Load quantized results for comparison
    mlx_eval_path = Path("models/qwen3-8b/evaluation_mlx.json")
    gguf_eval_path = Path("models/qwen3-8b/evaluation_gguf.json")

    if mlx_eval_path.exists() and gguf_eval_path.exists():
        import json

        with open(mlx_eval_path) as f:
            mlx_data = json.load(f)
        with open(gguf_eval_path) as f:
            gguf_data = json.load(f)

        print("=" * 80)
        print("COMPARISON: Baseline vs Quantized Models")
        print("=" * 80)
        print()
        print(f"{'Metric':<20} {'Baseline':<15} {'MLX 4-bit':<15} {'GGUF Q4_K_M':<15}")
        print("-" * 80)

        mlx_quality = mlx_data["aggregate_scores"]["quality"]
        gguf_quality = gguf_data["aggregate_scores"]["quality"]
        print(f"{'Quality':<20} {avg_quality:<15.1f} {mlx_quality:<15.1f} {gguf_quality:<15.1f}")

        mlx_accuracy = mlx_data["aggregate_scores"]["accuracy"]
        gguf_accuracy = gguf_data["aggregate_scores"]["accuracy"]
        print(f"{'Accuracy':<20} {avg_accuracy:<15.1f} {mlx_accuracy:<15.1f} {gguf_accuracy:<15.1f}")

        mlx_coherence = mlx_data["aggregate_scores"]["coherence"]
        gguf_coherence = gguf_data["aggregate_scores"]["coherence"]
        print(f"{'Coherence':<20} {avg_coherence:<15.1f} {mlx_coherence:<15.1f} {gguf_coherence:<15.1f}")

        mlx_time = mlx_data["aggregate_scores"]["generation_time"]
        gguf_time = gguf_data["aggregate_scores"]["generation_time"]
        print(f"{'Speed (s/prompt)':<20} {avg_gen_time:<15.2f} {mlx_time:<15.2f} {gguf_time:<15.2f}")

        print()
        print("=" * 80)
        print("QUALITY DEGRADATION")
        print("=" * 80)
        mlx_degradation = ((avg_quality - mlx_quality) / avg_quality * 100) if avg_quality > 0 else 0
        gguf_degradation = ((avg_quality - gguf_quality) / avg_quality * 100) if avg_quality > 0 else 0

        print(f"MLX 4-bit:     {mlx_degradation:+.1f}% quality loss")
        print(f"GGUF Q4_K_M:   {gguf_degradation:+.1f}% quality loss")
        print("=" * 80)
        print()

    # Save baseline results
    import json
    from datetime import datetime

    baseline_result = {
        "model_name": model_id,
        "model_path": model_id,
        "quantization_format": "baseline",
        "quantization_config": {"precision": "fp16"},
        "aggregate_scores": {
            "accuracy": avg_accuracy,
            "coherence": avg_coherence,
            "relevance": avg_relevance,
            "quality": avg_quality,
            "generation_time": avg_gen_time,
        },
        "prompt_results": [
            {
                "prompt": r["prompt"],
                "category": r["category"],
                "model_response": r["response"],
                "generation_time_seconds": r["gen_time"],
                "accuracy_score": r["scores"]["accuracy_score"],
                "coherence_score": r["scores"]["coherence_score"],
                "relevance_score": r["scores"]["relevance_score"],
                "quality_score": r["scores"]["quality_score"],
                "judge_reasoning": r["scores"]["reasoning"],
                "timestamp": datetime.now().isoformat(),
            }
            for r in results
        ],
        "evaluation_timestamp": datetime.now().isoformat(),
        "judge_model": "gpt-5-nano",
    }

    output_path = "models/qwen3-8b/evaluation_baseline.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline_result, f, indent=2)

    print(f"💾 Baseline evaluation saved to: {output_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(evaluate_baseline())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
