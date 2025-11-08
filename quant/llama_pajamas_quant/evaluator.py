"""LLM-as-Judge evaluation system for quantized models.

Uses OpenAI GPT-5 nano to evaluate generation quality, coherence, and accuracy.
Results are stored alongside quantized models for comparison.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


@dataclass
class EvaluationPrompt:
    """A prompt for evaluation with expected characteristics."""

    prompt: str
    category: str  # e.g., "code", "reasoning", "factual", "creative"
    expected_qualities: List[str]  # e.g., ["accurate", "concise", "well-formatted"]


@dataclass
class EvaluationResult:
    """Result from evaluating a single prompt."""

    prompt: str
    category: str
    model_response: str
    generation_time_seconds: float

    # Scores from LLM judge (0-10 scale)
    accuracy_score: float
    coherence_score: float
    relevance_score: float
    quality_score: float  # overall

    # Judge's reasoning
    judge_reasoning: str

    # Metadata
    timestamp: str


@dataclass
class ModelEvaluation:
    """Complete evaluation results for a quantized model."""

    model_name: str
    model_path: str
    quantization_format: str  # "gguf" or "mlx"
    quantization_config: Dict[str, Any]

    # Individual prompt results
    prompt_results: List[EvaluationResult]

    # Aggregate scores
    avg_accuracy: float
    avg_coherence: float
    avg_relevance: float
    avg_quality: float
    avg_generation_time: float

    # Metadata
    evaluation_timestamp: str
    judge_model: str
    hardware_info: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "quantization_format": self.quantization_format,
            "quantization_config": self.quantization_config,
            "prompt_results": [asdict(r) for r in self.prompt_results],
            "aggregate_scores": {
                "accuracy": self.avg_accuracy,
                "coherence": self.avg_coherence,
                "relevance": self.avg_relevance,
                "quality": self.avg_quality,
                "generation_time": self.avg_generation_time,
            },
            "evaluation_timestamp": self.evaluation_timestamp,
            "judge_model": self.judge_model,
            "hardware_info": self.hardware_info,
        }


# Standard evaluation prompts for consistent testing
STANDARD_PROMPTS = [
    EvaluationPrompt(
        prompt="Write a Python function to calculate the factorial of a number:",
        category="code",
        expected_qualities=["correct", "efficient", "well-commented"],
    ),
    EvaluationPrompt(
        prompt="Explain the concept of recursion in programming:",
        category="technical_explanation",
        expected_qualities=["accurate", "clear", "includes_example"],
    ),
    EvaluationPrompt(
        prompt="What is the difference between a list and a tuple in Python?",
        category="factual",
        expected_qualities=["accurate", "concise", "comprehensive"],
    ),
    EvaluationPrompt(
        prompt="Write a function to reverse a string:",
        category="code",
        expected_qualities=["correct", "simple", "readable"],
    ),
    EvaluationPrompt(
        prompt="Explain what a decorator is in Python:",
        category="technical_explanation",
        expected_qualities=["accurate", "detailed", "includes_example"],
    ),
    EvaluationPrompt(
        prompt="Write a sorting algorithm (your choice) in Python:",
        category="code",
        expected_qualities=["correct", "efficient", "readable"],
    ),
    EvaluationPrompt(
        prompt="What are the key differences between synchronous and asynchronous programming?",
        category="conceptual",
        expected_qualities=["accurate", "clear", "comprehensive"],
    ),
]


class LLMJudge:
    """LLM-as-judge evaluator using OpenAI GPT-5 nano."""

    def __init__(self, model: str = "gpt-5-nano", api_key: Optional[str] = None):
        """Initialize the judge.

        Args:
            model: OpenAI model to use for judging (default: gpt-5-nano)
            api_key: OpenAI API key (defaults to OPEN_AI_KEY env var)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPEN_AI_KEY"))

    def evaluate_response(
        self,
        prompt: str,
        response: str,
        category: str,
        expected_qualities: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a model's response using GPT-5 nano as judge.

        Args:
            prompt: The original prompt
            response: Model's response to evaluate
            category: Category of the prompt (code, reasoning, etc.)
            expected_qualities: List of expected qualities

        Returns:
            Dictionary with scores and reasoning
        """
        # Construct evaluation prompt
        eval_prompt = f"""You are an expert AI evaluator. Your task is to evaluate the quality of an AI model's response.

**Original Prompt:**
{prompt}

**Category:** {category}

**Expected Qualities:** {', '.join(expected_qualities)}

**Model Response:**
{response}

**Evaluation Criteria:**
Please evaluate the response on the following dimensions (0-10 scale):

1. **Accuracy**: Is the information correct and factual?
2. **Coherence**: Is the response well-structured and easy to follow?
3. **Relevance**: Does the response directly address the prompt?
4. **Overall Quality**: Taking everything into account, how good is this response?

**Output Format:**
Provide your evaluation as a JSON object with this exact structure:
{{
    "accuracy_score": <0-10>,
    "coherence_score": <0-10>,
    "relevance_score": <0-10>,
    "quality_score": <0-10>,
    "reasoning": "<your detailed reasoning for these scores>"
}}

Be objective and critical. A score of 10 means perfect, 5 means acceptable, 0 means completely wrong/useless.
"""

        # Call OpenAI API
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format={"type": "json_object"},
                # Note: temperature not set - gpt-5-nano only supports default (1)
            )

            # Parse response
            result = json.loads(completion.choices[0].message.content)
            return result

        except Exception as e:
            print(f"⚠️  Error during evaluation: {e}")
            # Return default scores on error
            return {
                "accuracy_score": 0,
                "coherence_score": 0,
                "relevance_score": 0,
                "quality_score": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }


class ModelEvaluator:
    """Evaluates quantized models using LLM-as-judge."""

    def __init__(
        self,
        judge_model: str = "gpt-5-nano",
        prompts: Optional[List[EvaluationPrompt]] = None,
    ):
        """Initialize evaluator.

        Args:
            judge_model: OpenAI model to use for judging
            prompts: Custom evaluation prompts (defaults to STANDARD_PROMPTS)
        """
        self.judge = LLMJudge(model=judge_model)
        self.prompts = prompts or STANDARD_PROMPTS
        self.judge_model = judge_model

    def evaluate_mlx(
        self,
        model_path: str,
        quant_config: Dict[str, Any],
    ) -> ModelEvaluation:
        """Evaluate MLX quantized model.

        Args:
            model_path: Path to MLX model
            quant_config: Quantization configuration used

        Returns:
            ModelEvaluation with results
        """
        from mlx_lm import load, generate

        print(f"\n🔍 Evaluating MLX model: {model_path}")
        print(f"📝 Running {len(self.prompts)} evaluation prompts...")

        # Load model
        model, tokenizer = load(model_path)

        # Evaluate each prompt
        results = []
        for i, eval_prompt in enumerate(self.prompts, 1):
            print(f"  [{i}/{len(self.prompts)}] {eval_prompt.category}: {eval_prompt.prompt[:50]}...")

            # Generate response
            start_time = time.time()
            response = generate(
                model,
                tokenizer,
                prompt=eval_prompt.prompt,
                max_tokens=150,
                verbose=False,
            )
            gen_time = time.time() - start_time

            # Get judge scores
            scores = self.judge.evaluate_response(
                prompt=eval_prompt.prompt,
                response=response,
                category=eval_prompt.category,
                expected_qualities=eval_prompt.expected_qualities,
            )

            # Store result
            result = EvaluationResult(
                prompt=eval_prompt.prompt,
                category=eval_prompt.category,
                model_response=response,
                generation_time_seconds=gen_time,
                accuracy_score=scores.get("accuracy_score", 0),
                coherence_score=scores.get("coherence_score", 0),
                relevance_score=scores.get("relevance_score", 0),
                quality_score=scores.get("quality_score", 0),
                judge_reasoning=scores.get("reasoning", ""),
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)

            print(f"    ✓ Quality: {result.quality_score:.1f}/10 ({gen_time:.2f}s)")

        # Calculate aggregates
        avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
        avg_coherence = sum(r.coherence_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_quality = sum(r.quality_score for r in results) / len(results)
        avg_gen_time = sum(r.generation_time_seconds for r in results) / len(results)

        # Create evaluation
        evaluation = ModelEvaluation(
            model_name=Path(model_path).name,
            model_path=model_path,
            quantization_format="mlx",
            quantization_config=quant_config,
            prompt_results=results,
            avg_accuracy=avg_accuracy,
            avg_coherence=avg_coherence,
            avg_relevance=avg_relevance,
            avg_quality=avg_quality,
            avg_generation_time=avg_gen_time,
            evaluation_timestamp=datetime.now().isoformat(),
            judge_model=self.judge_model,
            hardware_info=self._get_hardware_info(),
        )

        print(f"\n✅ MLX Evaluation Complete!")
        print(f"   Average Quality: {avg_quality:.1f}/10")
        print(f"   Average Accuracy: {avg_accuracy:.1f}/10")
        print(f"   Average Coherence: {avg_coherence:.1f}/10")

        return evaluation

    def evaluate_gguf(
        self,
        model_path: str,
        quant_config: Dict[str, Any],
    ) -> ModelEvaluation:
        """Evaluate GGUF quantized model.

        Args:
            model_path: Path to GGUF model file
            quant_config: Quantization configuration used

        Returns:
            ModelEvaluation with results
        """
        from llama_cpp import Llama

        print(f"\n🔍 Evaluating GGUF model: {model_path}")
        print(f"📝 Running {len(self.prompts)} evaluation prompts...")

        # Load model
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # Use all GPU layers
            verbose=False,
        )

        # Evaluate each prompt
        results = []
        for i, eval_prompt in enumerate(self.prompts, 1):
            print(f"  [{i}/{len(self.prompts)}] {eval_prompt.category}: {eval_prompt.prompt[:50]}...")

            # Generate response
            start_time = time.time()
            output = llm(
                eval_prompt.prompt,
                max_tokens=150,
                temperature=0.7,
                stop=["\n\n"],
            )
            gen_time = time.time() - start_time
            response = output["choices"][0]["text"]

            # Get judge scores
            scores = self.judge.evaluate_response(
                prompt=eval_prompt.prompt,
                response=response,
                category=eval_prompt.category,
                expected_qualities=eval_prompt.expected_qualities,
            )

            # Store result
            result = EvaluationResult(
                prompt=eval_prompt.prompt,
                category=eval_prompt.category,
                model_response=response,
                generation_time_seconds=gen_time,
                accuracy_score=scores.get("accuracy_score", 0),
                coherence_score=scores.get("coherence_score", 0),
                relevance_score=scores.get("relevance_score", 0),
                quality_score=scores.get("quality_score", 0),
                judge_reasoning=scores.get("reasoning", ""),
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)

            print(f"    ✓ Quality: {result.quality_score:.1f}/10 ({gen_time:.2f}s)")

        # Calculate aggregates
        avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
        avg_coherence = sum(r.coherence_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_quality = sum(r.quality_score for r in results) / len(results)
        avg_gen_time = sum(r.generation_time_seconds for r in results) / len(results)

        # Create evaluation
        evaluation = ModelEvaluation(
            model_name=Path(model_path).name,
            model_path=model_path,
            quantization_format="gguf",
            quantization_config=quant_config,
            prompt_results=results,
            avg_accuracy=avg_accuracy,
            avg_coherence=avg_coherence,
            avg_relevance=avg_relevance,
            avg_quality=avg_quality,
            avg_generation_time=avg_gen_time,
            evaluation_timestamp=datetime.now().isoformat(),
            judge_model=self.judge_model,
            hardware_info=self._get_hardware_info(),
        )

        print(f"\n✅ GGUF Evaluation Complete!")
        print(f"   Average Quality: {avg_quality:.1f}/10")
        print(f"   Average Accuracy: {avg_accuracy:.1f}/10")
        print(f"   Average Coherence: {avg_coherence:.1f}/10")

        return evaluation

    def _get_hardware_info(self) -> Dict[str, str]:
        """Get hardware information for evaluation context."""
        import platform
        import psutil

        return {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "ram_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
        }

    def save_evaluation(self, evaluation: ModelEvaluation, output_path: str):
        """Save evaluation results to JSON file.

        Args:
            evaluation: ModelEvaluation to save
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(evaluation.to_dict(), f, indent=2)

        print(f"\n💾 Evaluation saved to: {output_file}")

    def compare_evaluations(self, eval_paths: List[str]):
        """Compare multiple evaluation results.

        Args:
            eval_paths: List of paths to evaluation JSON files
        """
        evaluations = []
        for path in eval_paths:
            with open(path) as f:
                evaluations.append(json.load(f))

        print("\n" + "=" * 80)
        print("EVALUATION COMPARISON")
        print("=" * 80)

        # Header
        print(f"\n{'Model':<40} {'Format':<10} {'Quality':<10} {'Accuracy':<10} {'Speed':<10}")
        print("-" * 80)

        # Results
        for eval_data in evaluations:
            model_name = eval_data["model_name"][:38]
            format_name = eval_data["quantization_format"].upper()
            quality = eval_data["aggregate_scores"]["quality"]
            accuracy = eval_data["aggregate_scores"]["accuracy"]
            speed = eval_data["aggregate_scores"]["generation_time"]

            print(f"{model_name:<40} {format_name:<10} {quality:<10.1f} {accuracy:<10.1f} {speed:<10.2f}s")

        print("=" * 80)
