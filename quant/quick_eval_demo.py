#!/usr/bin/env python3
"""Quick evaluation demo - single prompt to show how it works."""

import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
from llama_pajamas_quant.evaluator import ModelEvaluator, EvaluationPrompt

def main():
    print("=" * 80)
    print("Quick Evaluation Demo - GPT-5 Nano LLM-as-Judge")
    print("=" * 80)
    print()

    # Use just ONE simple prompt for speed
    quick_prompts = [
        EvaluationPrompt(
            prompt="Write a Python function to calculate the factorial of a number:",
            category="code",
            expected_qualities=["correct", "efficient", "readable"]
        )
    ]

    # Create evaluator
    print("Creating evaluator with GPT-5 nano...")
    evaluator = ModelEvaluator(judge_model="gpt-5-nano", prompts=quick_prompts)
    print("✓ Ready")
    print()

    # Evaluate MLX
    mlx_path = Path("models/qwen3-8b/mlx")
    if mlx_path.exists():
        print("=" * 80)
        print("Testing MLX Evaluation (1 prompt)")
        print("=" * 80)
        print(f"Model: {mlx_path}")
        print()

        from mlx_lm import load, generate

        print("Loading MLX model...")
        model, tokenizer = load(str(mlx_path))
        print("✓ Model loaded")
        print()

        prompt = quick_prompts[0].prompt
        print(f"Prompt: {prompt}")
        print()

        print("Generating response...")
        start = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
        gen_time = time.time() - start
        print(f"✓ Generated in {gen_time:.2f}s")
        print()

        print(f"Response preview:")
        print(f"  {response[:200]}...")
        print()

        print("Sending to GPT-5 nano for judging...")
        scores = evaluator.judge.evaluate_response(
            prompt=prompt,
            response=response,
            category="code",
            expected_qualities=["correct", "efficient", "readable"],
        )
        print("✓ Judging complete")
        print()

        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"  Accuracy:  {scores['accuracy_score']:.1f}/10")
        print(f"  Coherence: {scores['coherence_score']:.1f}/10")
        print(f"  Relevance: {scores['relevance_score']:.1f}/10")
        print(f"  Quality:   {scores['quality_score']:.1f}/10")
        print()
        print("Judge's Reasoning:")
        print(f"  {scores['reasoning']}")
        print("=" * 80)
        print()
    else:
        print(f"❌ MLX model not found: {mlx_path}")

    print("✅ Demo complete!")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
