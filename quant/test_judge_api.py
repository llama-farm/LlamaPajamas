#!/usr/bin/env python3
"""Quick test of LLM-as-judge API integration.

Tests that OpenAI API key is configured and gpt-5-nano works correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_openai_connection():
    """Test OpenAI API connection and gpt-5-nano availability."""
    print("=" * 80)
    print("Testing OpenAI API Connection")
    print("=" * 80)
    print()

    # Check API key
    api_key = os.getenv("OPEN_AI_KEY")
    if not api_key:
        print("❌ OPEN_AI_KEY not found in environment")
        print("   Please set it in .env file or environment variables")
        return False

    print(f"✓ API key found: {api_key[:20]}...")
    print()

    # Test LLM judge
    from llama_pajamas_quant.evaluator import LLMJudge

    print("Creating LLM judge with gpt-5-nano...")
    judge = LLMJudge(model="gpt-5-nano")
    print("✓ Judge created")
    print()

    # Test evaluation
    print("Testing evaluation with sample prompt...")
    print()

    test_prompt = "Write a Python function to calculate the factorial of a number:"
    test_response = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

    print(f"Prompt: {test_prompt}")
    print(f"Response: {test_response}")
    print()
    print("Sending to gpt-5-nano for evaluation...")
    print()

    try:
        result = judge.evaluate_response(
            prompt=test_prompt,
            response=test_response,
            category="code",
            expected_qualities=["correct", "efficient", "readable"],
        )

        print("✅ Evaluation successful!")
        print()
        print("Scores:")
        print(f"  Accuracy:  {result['accuracy_score']:.1f}/10")
        print(f"  Coherence: {result['coherence_score']:.1f}/10")
        print(f"  Relevance: {result['relevance_score']:.1f}/10")
        print(f"  Quality:   {result['quality_score']:.1f}/10")
        print()
        print("Judge Reasoning:")
        print(f"  {result['reasoning']}")
        print()

        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    success = test_openai_connection()
    print()
    print("=" * 80)

    if success:
        print("✅ All tests passed! Ready to use LLM-as-judge evaluation.")
    else:
        print("❌ Tests failed. Please check your OpenAI API configuration.")

    print("=" * 80)
    print()

    sys.exit(0 if success else 1)
