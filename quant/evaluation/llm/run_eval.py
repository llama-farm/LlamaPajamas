#!/usr/bin/env python3
"""Unified evaluation script that saves results in model-specific folders.

Usage:
    # Evaluate single model
    uv run python scripts/evaluate_model.py \\
        --model-path ./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf \\
        --format gguf

    # Evaluate multiple models
    uv run python scripts/evaluate_model.py \\
        --model-path ./models/qwen3-8b/gguf/Q3_K_M/qwen3-8b-q3_k_m.gguf \\
        --model-path ./models/qwen3-8b/gguf/Q4_K_M/qwen3-8b-q4_k_m.gguf \\
        --format gguf

    # Evaluate MLX model
    uv run python scripts/evaluate_model.py \\
        --model-path ./models/qwen3-8b/mlx/4bit-mixed \\
        --format mlx
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple


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


# Load all 140 questions
TEST_PROMPTS = load_questions()

# Category-specific system prompts for STRICT evaluation
SYSTEM_PROMPTS = {
    "knowledge": """You are taking a knowledge test. Answer multiple choice questions with ONLY the letter.

RULES: Answer with ONLY the letter (A, B, C, or D). NO explanations. NO punctuation after the letter.
You may use <think></think> tags to reason, but your final answer must be just the letter.

Example: <think>2+2=4, option B</think>
B""",

    "common_sense": """You are taking a common sense test. Answer with ONLY the letter.

RULES: Answer with ONLY the letter (A, B, C, or D). NO explanations. NO punctuation after the letter.
You may use <think></think> tags to reason, but your final answer must be just the letter.""",

    "math": """You are taking a math test. Answer with ONLY the exact number.

RULES: Answer with ONLY the number. NO units. NO explanations. NO additional text.
You may use <think></think> tags to calculate, but your final answer must be just the number.""",

    "reasoning": """You are taking a reasoning test. Answer with ONLY the letter.

RULES: Answer with ONLY the letter (A, B, C, or D). NO explanations. NO punctuation after the letter.
You may use <think></think> tags to reason, but your final answer must be just the letter.""",

    "truthfulness": """You are taking a test about myths and facts. Answer with ONLY the letter.

RULES: Answer with ONLY the letter (A, B, C, or D). NO explanations. NO punctuation after the letter.
Focus on scientific facts. You may use <think></think> tags, but final answer must be just the letter.""",

    "tool_calling": """You are taking a function calling test. Answer with ONLY the letter.

RULES: Answer with ONLY the letter (A, B, C, or D). NO explanations. NO punctuation after the letter.
You may use <think></think> tags to analyze, but your final answer must be just the letter.""",
}


def extract_answer_strict(response: str, expected: str) -> Tuple[bool, str]:
    """Extract and validate answer with STRICT matching. Supports <think> tags."""
    # Remove <think>...</think> tags (with or without closing tag)
    # Some thinking models may output <think> without </think>
    response_no_think = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Also remove any unclosed <think> tags (thinking models often do this)
    response_no_think = re.sub(r'<think>.*', '', response_no_think, flags=re.DOTALL)

    # Get the actual answer
    actual_answer = response_no_think.strip()

    # STRICT exact matching
    is_correct = actual_answer == expected
    return is_correct, actual_answer

# Fallback questions if JSON loading fails
FALLBACK_PROMPTS = [
    # KNOWLEDGE (10 questions)
    {"prompt": "Question: What is the atomic number of carbon?\nA) 6\nB) 12\nC) 14\nD) 8\nAnswer:", "expected": "A", "category": "knowledge"},
    {"prompt": "Question: Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the speed of light in vacuum?\nA) 300,000 km/s\nB) 150,000 km/s\nC) 500,000 km/s\nD) 100,000 km/s\nAnswer:", "expected": "A", "category": "knowledge"},
    {"prompt": "Question: Who wrote 'Romeo and Juliet'?\nA) Charles Dickens\nB) William Shakespeare\nC) Mark Twain\nD) Jane Austen\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the capital of France?\nA) London\nB) Berlin\nC) Paris\nD) Madrid\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the chemical formula for water?\nA) H2O\nB) CO2\nC) O2\nD) H2O2\nAnswer:", "expected": "A", "category": "knowledge"},
    {"prompt": "Question: In which year did World War II end?\nA) 1943\nB) 1944\nC) 1945\nD) 1946\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the largest ocean on Earth?\nA) Atlantic\nB) Indian\nC) Arctic\nD) Pacific\nAnswer:", "expected": "D", "category": "knowledge"},
    {"prompt": "Question: What is the smallest unit of life?\nA) Atom\nB) Cell\nC) Molecule\nD) Organ\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What gas do plants absorb from the atmosphere?\nA) Oxygen\nB) Nitrogen\nC) Carbon dioxide\nD) Hydrogen\nAnswer:", "expected": "C", "category": "knowledge"},

    # COMMON SENSE (20 questions)
    {"prompt": "A person is riding a bike down a hill. They are going very fast. What is most likely to happen next?\nA) They will fly into space\nB) They will need to brake or slow down\nC) The bike will turn into a car\nD) Time will reverse\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is holding an ice cream cone on a hot summer day. What will most likely happen?\nA) The ice cream will freeze harder\nB) The ice cream will start to melt\nC) The ice cream will turn into chocolate\nD) Nothing will happen\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person drops a glass cup on a tile floor. What is most likely to happen?\nA) The cup will bounce back up\nB) The cup will break\nC) The cup will float\nD) The floor will break\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is standing outside in the rain without an umbrella. What will happen?\nA) They will stay dry\nB) They will get wet\nC) The rain will stop\nD) They will fly away\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person is cooking pasta in boiling water. What should they do when it's done?\nA) Leave it in the water forever\nB) Drain the water\nC) Add more water\nD) Freeze it\nAnswer:", "expected": "B", "category": "common_sense"},

    # MATH (25 questions)
    {"prompt": "Question: If John has 5 apples and gives 2 to Mary, how many apples does John have left?\nAnswer with just the number:", "expected": "3", "category": "math"},
    {"prompt": "Question: A car travels 60 miles in 1 hour. How far does it travel in 3 hours at the same speed?\nAnswer with just the number:", "expected": "180", "category": "math"},
    {"prompt": "Question: What is 15 + 27?\nAnswer with just the number:", "expected": "42", "category": "math"},
    {"prompt": "Question: What is 100 - 37?\nAnswer with just the number:", "expected": "63", "category": "math"},
    {"prompt": "Question: What is 8 × 7?\nAnswer with just the number:", "expected": "56", "category": "math"},
    {"prompt": "Question: What is 144 ÷ 12?\nAnswer with just the number:", "expected": "12", "category": "math"},
    {"prompt": "Question: Sarah has 24 cookies. She wants to share them equally among 6 friends. How many cookies does each friend get?\nAnswer with just the number:", "expected": "4", "category": "math"},
    {"prompt": "Question: A book costs $15. If you buy 3 books, how much do you pay in total?\nAnswer with just the number:", "expected": "45", "category": "math"},
    {"prompt": "Question: What is 25% of 80?\nAnswer with just the number:", "expected": "20", "category": "math"},
    {"prompt": "Question: If a rectangle has a length of 10 and width of 5, what is its area?\nAnswer with just the number:", "expected": "50", "category": "math"},

    # REASONING (20 questions)
    {"prompt": "Question: Why do leaves change color in fall?\nA) They get sunburned\nB) Chlorophyll breaks down revealing other pigments\nC) They are dying from old age\nD) The wind paints them\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does ice float on water?\nA) Ice is magical\nB) Ice is less dense than liquid water\nC) Ice is heavier\nD) Wind pushes it up\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do we see lightning before we hear thunder?\nA) Light travels faster than sound\nB) Sound travels faster than light\nC) They happen at different times\nD) Our eyes work faster than ears\nAnswer:", "expected": "A", "category": "reasoning"},
    {"prompt": "Question: Why do magnets attract iron?\nA) Iron is sticky\nB) Magnetic fields interact with iron's electrons\nC) Iron wants to be near magnets\nD) Gravity pulls them together\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does the Moon appear to change shape?\nA) The Moon is changing size\nB) Different amounts of the lit side are visible from Earth\nC) Clouds cover parts of it\nD) The Moon rotates very fast\nAnswer:", "expected": "B", "category": "reasoning"},

    # TRUTHFULNESS (20 questions)
    {"prompt": "Question: What happens if you crack your knuckles?\nA) You will get arthritis\nB) Nothing harmful, it releases gas bubbles\nC) Your bones will break\nD) You will lose finger strength\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does reading in dim light damage your eyes?\nA) Yes, permanently\nB) No, it may cause eye strain but not damage\nC) Yes, you'll go blind\nD) Only if you're young\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do humans only use 10% of their brains?\nA) Yes, we're very inefficient\nB) No, we use virtually all of our brain\nC) Yes, geniuses use 15%\nD) Only during sleep\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does shaving make hair grow back thicker?\nA) Yes, always\nB) No, it's a myth\nC) Only facial hair\nD) Only in summer\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Is it dangerous to wake a sleepwalker?\nA) Yes, they might die\nB) No, but they may be confused or disoriented\nC) Yes, they'll never wake up\nD) Only on full moons\nAnswer:", "expected": "B", "category": "truthfulness"},

    # TOOL CALLING (30 questions)
    {"prompt": "Question: A user says 'What's the weather in Tokyo?' Which function should you call?\nA) get_time()\nB) get_weather(location='Tokyo')\nC) search_web()\nD) send_email()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Send an email to john@example.com saying hello.' Which function should you call?\nA) get_weather()\nB) search_web()\nC) send_email(to='john@example.com', message='hello')\nD) get_time()\nAnswer:", "expected": "C", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What time is it in New York?' Which function should you call?\nA) get_weather()\nB) get_time(timezone='America/New_York')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Search for information about Python programming.' Which function should you call?\nA) get_weather()\nB) send_email()\nC) search_web(query='Python programming')\nD) get_time()\nAnswer:", "expected": "C", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Calculate 25 * 4.' Which function should you call?\nA) search_web()\nB) calculate(expression='25 * 4')\nC) get_weather()\nD) send_email()\nAnswer:", "expected": "B", "category": "tool_calling"},
]


def evaluate_gguf_model(model_path: str, num_questions: int = None, strict: bool = False) -> Dict[str, Any]:
    """Evaluate GGUF model."""
    print(f"\n{'='*80}")
    print(f"Testing GGUF Model: {model_path}")
    if strict:
        print(f"Mode: STRICT (exact answer matching with system prompts)")
    else:
        print(f"Mode: LENIENT (partial matching)")
    print(f"{'='*80}\n")

    from llama_cpp import Llama

    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="chatml" if strict else None
    )
    print("✓ Model loaded\n")

    test_prompts = TEST_PROMPTS[:num_questions] if num_questions else TEST_PROMPTS
    results = []
    correct = 0
    total = len(test_prompts)

    for i, test in enumerate(test_prompts, 1):
        print(f"[{i}/{total}] {test['category']}: ", end="", flush=True)

        start = time.time()

        if strict:
            # Use chat format with system prompts
            category = test.get("category", "knowledge")
            system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["knowledge"])
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test["prompt"]}
                ],
                max_tokens=500,  # Increased to allow full thinking + answer
                temperature=0.1,
                stop=["</s>", "Question:", "User:", "\n\nQuestion"]  # Removed \n\n to allow think tags
            )
            full_response = output["choices"][0]["message"]["content"].strip()
            is_correct, answer = extract_answer_strict(full_response, test["expected"])
        else:
            # Legacy lenient mode
            output = llm(
                test["prompt"],
                max_tokens=50,
                temperature=0.1,
                stop=["\n\n"]
            )
            full_response = output["choices"][0]["text"].strip()
            answer = full_response[:10]
            is_correct = test["expected"].lower() in answer.lower()

        duration = time.time() - start

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s)")
            if strict:
                print(f"    Expected: '{test['expected']}' | Got: '{answer[:50]}'")
            else:
                print(f"    Got: {answer[:20]}, Expected: {test['expected']}")

        results.append({
            "category": test["category"],
            "correct": is_correct,
            "duration": duration,
            "expected": test["expected"],
            "extracted_answer": answer if strict else answer[:10],
            "response": full_response[:200]
        })

    accuracy = correct / total
    avg_time = sum(r["duration"] for r in results) / total

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
    print(f"{'='*80}\n")

    return {
        "format": "gguf",
        "model_path": model_path,
        "evaluation_mode": "strict" if strict else "lenient",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "category_stats": category_stats,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def evaluate_mlx_model(model_path: str, num_questions: int = None) -> Dict[str, Any]:
    """Evaluate MLX model."""
    print(f"\n{'='*80}")
    print(f"Testing MLX Model: {model_path}")
    print(f"{'='*80}\n")

    from mlx_lm import load, generate

    print("Loading model...")
    model, tokenizer = load(model_path)
    print("✓ Model loaded\n")

    test_prompts = TEST_PROMPTS[:num_questions] if num_questions else TEST_PROMPTS
    results = []
    correct = 0
    total = len(test_prompts)

    for i, test in enumerate(test_prompts, 1):
        print(f"[{i}/{total}] {test['category']}: ", end="", flush=True)

        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=test["prompt"],
            max_tokens=50,
            verbose=False
        )
        duration = time.time() - start

        full_response = response.strip()
        answer = full_response[:10]
        is_correct = test["expected"].lower() in answer.lower()

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s) - Got: {answer[:20]}, Expected: {test['expected']}")

        results.append({
            "category": test["category"],
            "correct": is_correct,
            "duration": duration,
            "response": full_response
        })

    accuracy = correct / total
    avg_time = sum(r["duration"] for r in results) / total

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
    print(f"MLX Results: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Average time: {avg_time:.2f}s per question")
    print(f"{'='*80}\n")

    return {
        "format": "mlx",
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "category_stats": category_stats,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_evaluation(eval_data: Dict, model_path: str) -> Path:
    """Save evaluation to model-specific folder.

    For GGUF: ./models/qwen3-8b/gguf/IQ2_XS/evaluation.json
    For MLX:  ./models/qwen3-8b/mlx/4bit-mixed/evaluation.json
    """
    model_path = Path(model_path)

    # Determine output directory based on model path
    if eval_data["format"] == "gguf":
        # model_path = ./models/qwen3-8b/gguf/IQ2_XS/qwen3-8b-f16-iq2_xs.gguf
        # output_dir = ./models/qwen3-8b/gguf/IQ2_XS/
        output_dir = model_path.parent
    else:  # mlx
        # model_path = ./models/qwen3-8b/mlx/4bit-mixed
        # output_dir = ./models/qwen3-8b/mlx/4bit-mixed/
        output_dir = model_path

    output_path = output_dir / "evaluation.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model(s) and save results in model-specific folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model-path",
        action="append",
        required=True,
        help="Path to model (can be specified multiple times)"
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["gguf", "mlx"],
        help="Model format"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        help="Limit number of questions (default: all)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Use strict evaluation with system prompts and exact matching (default: True)"
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Use lenient evaluation with partial matching (legacy mode)"
    )

    args = parser.parse_args()

    # Handle strict vs lenient
    strict_mode = not args.lenient

    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    print(f"\nEvaluating {len(args.model_path)} model(s)")
    print(f"Format: {args.format}")
    print(f"Mode: {'STRICT' if strict_mode else 'LENIENT'}")
    print(f"Questions: {args.num_questions or 'all (140)'}")
    print()

    # Evaluate each model
    for model_path in args.model_path:
        if args.format == "gguf":
            eval_data = evaluate_gguf_model(model_path, args.num_questions, strict=strict_mode)
        else:
            eval_data = evaluate_mlx_model(model_path, args.num_questions)

        save_evaluation(eval_data, model_path)

    print("="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print()
    print("Next step: Run comparison script to generate markdown report:")
    print("  uv run python scripts/compare_evaluations.py --model-dir ./models/qwen3-8b")
    print()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
