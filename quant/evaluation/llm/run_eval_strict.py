#!/usr/bin/env python3
"""STRICT evaluation script with proper system prompts and exact answer matching.

Key improvements:
- Strict system prompts for each question category
- Exact answer matching (no partial credit)
- Support for <think> tags while requiring exact format
- Chat format with proper roles
- Fails if answer includes extra text (e.g., "A. Because..." when only "A" expected)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


# Category-specific system prompts
SYSTEM_PROMPTS = {
    "knowledge": """You are taking a knowledge test. You must answer multiple choice questions with ONLY the letter of the correct answer.

CRITICAL RULES:
- Answer with ONLY the letter (A, B, C, or D)
- NO explanations
- NO additional text
- NO punctuation after the letter
- You may use <think></think> tags to reason, but your final answer must be just the letter

Example:
Question: What is 2+2? A) 3 B) 4 C) 5 D) 6
<think>2 plus 2 equals 4, which is option B</think>
B

WRONG: "B) 4" or "B." or "The answer is B" or "B because..."
RIGHT: "B\"""",

    "common_sense": """You are taking a common sense reasoning test. You must answer with ONLY the letter of the correct answer.

CRITICAL RULES:
- Answer with ONLY the letter (A, B, C, or D)
- NO explanations
- NO additional text
- NO punctuation after the letter
- You may use <think></think> tags to reason, but your final answer must be just the letter

Example:
Question: If it's raining, what will happen to someone without an umbrella? A) Stay dry B) Get wet C) Fly D) Disappear
<think>Someone in rain without protection will get wet</think>
B

WRONG: "B) Get wet" or "B." or "Obviously B" or "B - they will get wet"
RIGHT: "B\"""",

    "math": """You are taking a math test. You must answer with ONLY the exact number - nothing else.

CRITICAL RULES:
- Answer with ONLY the number
- NO units (no "apples", "km", "$", etc.)
- NO explanations
- NO additional text
- You may use <think></think> tags to calculate, but your final answer must be just the number

Example:
Question: What is 15 + 27? Answer with just the number:
<think>15 + 27 = 42</think>
42

WRONG: "42 apples" or "The answer is 42" or "42." or "Forty-two"
RIGHT: "42\"""",

    "reasoning": """You are taking a reasoning test. You must answer with ONLY the letter of the correct answer.

CRITICAL RULES:
- Answer with ONLY the letter (A, B, C, or D)
- NO explanations
- NO additional text
- NO punctuation after the letter
- You may use <think></think> tags to reason through the logic, but your final answer must be just the letter

Example:
Question: Why does ice float? A) Magic B) Less dense than water C) Wind D) Gravity
<think>Ice floats because it is less dense than liquid water due to hydrogen bonding structure</think>
B

WRONG: "B) Less dense than water" or "B." or "The answer is B because..." or "B - density"
RIGHT: "B\"""",

    "truthfulness": """You are taking a test about common myths and facts. You must answer with ONLY the letter of the correct answer.

CRITICAL RULES:
- Answer with ONLY the letter (A, B, C, or D)
- NO explanations
- NO additional text
- NO punctuation after the letter
- Focus on scientific facts, not myths
- You may use <think></think> tags to evaluate the claim, but your final answer must be just the letter

Example:
Question: Does cracking knuckles cause arthritis? A) Yes always B) No, just releases gas C) Only in winter D) Yes if done daily
<think>Scientific studies show no link between knuckle cracking and arthritis. It just releases gas bubbles.</think>
B

WRONG: "B) No, just releases gas" or "B." or "B is correct" or "B - it's a myth"
RIGHT: "B\"""",

    "tool_calling": """You are taking a test about function calling and API usage. You must answer with ONLY the letter of the correct function/tool to call.

CRITICAL RULES:
- Answer with ONLY the letter (A, B, C, or D)
- NO explanations
- NO additional text
- NO punctuation after the letter
- Focus on which function best matches the user's intent
- You may use <think></think> tags to analyze the request, but your final answer must be just the letter

Example:
Question: User says 'What's the weather in Tokyo?' Which function? A) get_time() B) get_weather(location='Tokyo') C) search_web() D) send_email()
<think>User wants weather information for a specific location (Tokyo), so get_weather with location parameter is correct</think>
B

WRONG: "B) get_weather(location='Tokyo')" or "B." or "B because it gets weather" or "The answer is B"
RIGHT: "B\"""",
}


def extract_answer_strict(response: str, expected: str) -> tuple[bool, str]:
    """
    Extract and validate answer with STRICT matching.

    Supports <think> tags but requires exact answer format.
    NO partial matching, NO extraction of embedded answers.

    Args:
        response: Model's full response
        expected: Expected exact answer

    Returns:
        (is_correct, extracted_answer)
    """
    # Remove <think>...</think> tags and everything inside them
    response_no_think = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Get the actual answer (strip whitespace)
    actual_answer = response_no_think.strip()

    # STRICT exact matching
    # For single letter answers (A, B, C, D), must be exactly that
    # For numbers, must be exactly that number
    is_correct = actual_answer == expected

    return is_correct, actual_answer


def load_questions() -> List[Dict[str, Any]]:
    """Load all 140 questions from questions.json."""
    questions_file = Path(__file__).parent / "questions.json"
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    with open(questions_file) as f:
        data = json.load(f)

    print(f"📝 Loaded {data['total_questions']} questions from {questions_file.name}")
    print(f"📊 Categories: {', '.join(f'{k}({v})' for k,v in data['categories'].items())}")
    return data['questions']


def format_chat_prompt(question: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format question as chat messages with appropriate system prompt."""
    category = question.get("category", "knowledge")
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["knowledge"])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question["prompt"]}
    ]


def evaluate_gguf_model(model_path: str, num_questions: int = None) -> Dict[str, Any]:
    """Evaluate GGUF model with STRICT grading."""
    print(f"\n{'='*80}")
    print(f"Testing GGUF Model: {model_path}")
    print(f"Using STRICT evaluation (exact answer matching)")
    print(f"{'='*80}\n")

    from llama_cpp import Llama

    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="chatml"  # Use chat format for proper system prompts
    )
    print("✓ Model loaded\n")

    # Load questions
    try:
        test_prompts = load_questions()
    except:
        print("⚠️  Could not load questions.json, using fallback questions")
        test_prompts = []  # Would use fallback here

    if num_questions:
        test_prompts = test_prompts[:num_questions]

    results = []
    correct = 0
    total = len(test_prompts)

    for i, test in enumerate(test_prompts, 1):
        category = test.get("category", "unknown")
        print(f"[{i}/{total}] {category}: ", end="", flush=True)

        # Format as chat with system prompt
        messages = format_chat_prompt(test)

        start = time.time()
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=200,  # Allow room for <think> tags
            temperature=0.1,
            stop=["</s>", "\n\n", "Question:", "User:"]
        )
        duration = time.time() - start

        full_response = output["choices"][0]["message"]["content"].strip()

        # STRICT matching with <think> support
        is_correct, extracted_answer = extract_answer_strict(full_response, test["expected"])

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            # Show what went wrong
            print(f"✗ ({duration:.1f}s)")
            print(f"    Expected: '{test['expected']}'")
            print(f"    Got: '{extracted_answer[:50]}'")
            if len(full_response) < 100:
                print(f"    Full: {full_response}")

        results.append({
            "category": category,
            "correct": is_correct,
            "duration": duration,
            "expected": test["expected"],
            "extracted_answer": extracted_answer,
            "full_response": full_response[:200]  # Limit stored response length
        })

    accuracy = correct / total if total > 0 else 0
    avg_time = sum(r["duration"] for r in results) / total if total > 0 else 0

    # Calculate category breakdown
    category_stats = {}
    for result in results:
        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if result["correct"]:
            category_stats[cat]["correct"] += 1

    for cat in category_stats:
        category_stats[cat]["accuracy"] = category_stats[cat]["correct"] / category_stats[cat]["total"]

    print(f"\n{'='*80}")
    print(f"GGUF Results: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Average time: {avg_time:.2f}s per question")
    print(f"\nCategory Breakdown:")
    for cat, stats in category_stats.items():
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
    print(f"{'='*80}\n")

    return {
        "format": "gguf",
        "model_path": model_path,
        "evaluation_mode": "strict",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "category_stats": category_stats,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Strict LLM evaluation with proper system prompts and exact matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to test (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: save next to model)"
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_gguf_model(args.model_path, args.num_questions)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        # Save next to the model
        model_dir = Path(args.model_path).parent
        output_path = model_dir / "evaluation_strict.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}")
    print(f"✅ Strict Evaluation Complete!")
    print(f"{'='*80}\n")

    return 0 if results["accuracy"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
