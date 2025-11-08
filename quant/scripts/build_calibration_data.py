#!/usr/bin/env python3
"""Build calibration data for importance quantization.

This script generates diverse calibration data by:
1. Using existing curated datasets (tool calling, summarization, RAG)
2. Optionally generating synthetic data via GPT-5-nano for specific use cases
3. Creating train/test split to prevent overfitting
4. Outputting calibration.txt for imatrix generation

Extensible for custom synthetic data generation with configurable:
- Response length (short/medium/long)
- Context length (tokens)
- Domain/topic
- Complexity level
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai

# Import our calibration datasets
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llama_pajamas_quant.calibration import (
    TOOL_CALLING_CALIBRATION,
    SUMMARIZATION_CALIBRATION,
    RAG_CALIBRATION,
)


class CalibrationBuilder:
    """Build calibration data with optional synthetic generation."""

    def __init__(
        self,
        output_dir: Path,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
    ):
        """Initialize calibration builder.

        Args:
            output_dir: Directory to save calibration files
            api_key: OpenAI API key (or loaded from env)
            model: Model to use for synthetic generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load API key
        load_dotenv()
        self.api_key = api_key or os.getenv("OPEN_AI_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.model = model

    def build_from_curated(
        self,
        num_samples: int = 512,
        include_tool_calling: bool = True,
        include_summarization: bool = True,
        include_rag: bool = True,
        seed: int = 42,
    ) -> List[str]:
        """Build calibration data from curated datasets.

        Args:
            num_samples: Target number of calibration samples
            include_tool_calling: Include tool calling examples
            include_summarization: Include summarization examples
            include_rag: Include RAG examples
            seed: Random seed for reproducibility

        Returns:
            List of calibration text samples
        """
        random.seed(seed)
        samples = []

        # Add tool calling examples
        if include_tool_calling:
            samples.extend(TOOL_CALLING_CALIBRATION)
            print(f"Added {len(TOOL_CALLING_CALIBRATION)} tool calling examples")

        # Add summarization examples
        if include_summarization:
            for item in SUMMARIZATION_CALIBRATION:
                samples.append(item)
            print(f"Added {len(SUMMARIZATION_CALIBRATION)} summarization examples")

        # Add RAG examples
        if include_rag:
            for item in RAG_CALIBRATION:
                # Format as "Context: ... Question: ..."
                formatted = f"Context: {item['context']}\n\nQuestion: {item['question']}"
                samples.append(formatted)
            print(f"Added {len(RAG_CALIBRATION)} RAG examples")

        # If we need more samples, we can expand later with synthetic generation
        current_count = len(samples)
        if current_count < num_samples:
            print(f"Warning: Only {current_count} samples available, requested {num_samples}")
            print(f"Consider using --generate-synthetic to create additional samples")

        # Shuffle to mix different types
        random.shuffle(samples)

        return samples[:num_samples]

    def generate_synthetic_samples(
        self,
        num_samples: int,
        response_length: str = "medium",
        context_length: int = 2048,
        domain: str = "general",
        complexity: str = "medium",
        task_type: str = "qa",
    ) -> List[str]:
        """Generate synthetic calibration samples using GPT-5-nano.

        Args:
            num_samples: Number of samples to generate
            response_length: Target response length (short/medium/long)
            context_length: Maximum context length in tokens
            domain: Domain/topic for samples (general, code, math, reasoning, etc.)
            complexity: Complexity level (simple/medium/complex)
            task_type: Type of task (qa, summarization, reasoning, coding, etc.)

        Returns:
            List of generated text samples
        """
        if not self.api_key:
            raise ValueError("OpenAI API key required for synthetic generation")

        # Map response length to approximate word counts
        length_map = {
            "short": "50-100 words",
            "medium": "200-400 words",
            "long": "500-1000 words",
            "very_long": "1000-2000 words",
        }
        target_length = length_map.get(response_length, "200-400 words")

        # Build generation prompt based on parameters
        system_prompt = self._build_generation_prompt(
            domain, complexity, task_type, target_length, context_length
        )

        samples = []
        print(f"\nGenerating {num_samples} synthetic samples...")
        print(f"Domain: {domain}, Complexity: {complexity}, Task: {task_type}")
        print(f"Response length: {response_length}, Context: {context_length} tokens")

        for i in range(num_samples):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate sample {i+1}"}
                    ],
                    max_tokens=context_length,
                    temperature=0.9,  # Higher diversity
                )

                sample_text = response.choices[0].message.content.strip()
                samples.append(sample_text)

                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples...")

            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")
                continue

        print(f"Successfully generated {len(samples)} samples")
        return samples

    def _build_generation_prompt(
        self,
        domain: str,
        complexity: str,
        task_type: str,
        target_length: str,
        context_length: int,
    ) -> str:
        """Build system prompt for synthetic data generation."""

        domain_descriptions = {
            "general": "general knowledge across various topics",
            "code": "programming, algorithms, and software engineering",
            "math": "mathematics, statistics, and quantitative reasoning",
            "reasoning": "logical reasoning, problem-solving, and critical thinking",
            "science": "scientific concepts across physics, chemistry, biology",
            "business": "business, finance, economics, and strategy",
            "creative": "creative writing, storytelling, and literature",
        }

        task_descriptions = {
            "qa": "question-answer pairs with detailed, informative answers",
            "summarization": "long-form text followed by summarization request",
            "reasoning": "complex reasoning problems with step-by-step solutions",
            "coding": "programming challenges with code examples and explanations",
            "dialogue": "multi-turn conversational exchanges",
            "analysis": "analytical tasks with detailed breakdowns",
        }

        return f"""You are a synthetic data generator for model calibration.

Generate diverse, high-quality {task_descriptions.get(task_type, 'text samples')} in the domain of {domain_descriptions.get(domain, domain)}.

Requirements:
- Complexity: {complexity}
- Target length: {target_length}
- Maximum context: {context_length} tokens
- Diversity: Each sample should be unique and cover different aspects
- Quality: Professional, accurate, well-structured

Format: Generate complete, standalone examples that represent realistic use cases.
Each generation should be a single cohesive sample (e.g., a Q&A pair, or a document + task).

Focus on activating diverse model weights and attention patterns."""

    def create_train_test_split(
        self,
        samples: List[str],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[List[str], List[str]]:
        """Split samples into train (calibration) and test (evaluation) sets.

        Args:
            samples: All samples to split
            train_ratio: Ratio for training set (0.8 = 80% train, 20% test)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_samples, test_samples)
        """
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]

        print(f"\nTrain/test split: {len(train)} train, {len(test)} test")
        return train, test

    def save_calibration_file(
        self,
        samples: List[str],
        filename: str = "calibration.txt",
    ) -> Path:
        """Save calibration samples to text file for imatrix generation.

        Args:
            samples: Calibration samples
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Add double newline separator between samples
                f.write(sample + "\n\n")

        print(f"\nSaved {len(samples)} samples to {output_path}")

        # Calculate approximate token count (rough estimate: 1 token ≈ 4 chars)
        total_chars = sum(len(s) for s in samples)
        approx_tokens = total_chars // 4
        print(f"Approximate tokens: {approx_tokens:,}")

        return output_path

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "calibration_metadata.json",
    ) -> Path:
        """Save calibration metadata for reproducibility.

        Args:
            metadata: Metadata dict
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build calibration data for importance quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use curated datasets only (512 samples)
  python build_calibration_data.py --output ./calibration --num-samples 512

  # Generate synthetic samples with specific parameters
  python build_calibration_data.py --output ./calibration \\
      --generate-synthetic 200 \\
      --domain code \\
      --complexity complex \\
      --response-length long

  # Mixed: curated + synthetic
  python build_calibration_data.py --output ./calibration \\
      --num-samples 512 \\
      --generate-synthetic 100 \\
      --domain reasoning

  # Custom split ratio (90% train, 10% test)
  python build_calibration_data.py --output ./calibration \\
      --train-ratio 0.9
        """
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./calibration_data"),
        help="Output directory for calibration files"
    )

    # Curated dataset options
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Target number of calibration samples (default: 512)"
    )
    parser.add_argument(
        "--no-tool-calling",
        action="store_true",
        help="Exclude tool calling examples"
    )
    parser.add_argument(
        "--no-summarization",
        action="store_true",
        help="Exclude summarization examples"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Exclude RAG examples"
    )

    # Synthetic generation options
    parser.add_argument(
        "--generate-synthetic",
        type=int,
        metavar="N",
        help="Generate N synthetic samples using GPT-5-nano"
    )
    parser.add_argument(
        "--domain",
        choices=["general", "code", "math", "reasoning", "science", "business", "creative"],
        default="general",
        help="Domain for synthetic samples (default: general)"
    )
    parser.add_argument(
        "--complexity",
        choices=["simple", "medium", "complex"],
        default="medium",
        help="Complexity level for synthetic samples (default: medium)"
    )
    parser.add_argument(
        "--task-type",
        choices=["qa", "summarization", "reasoning", "coding", "dialogue", "analysis"],
        default="qa",
        help="Task type for synthetic samples (default: qa)"
    )
    parser.add_argument(
        "--response-length",
        choices=["short", "medium", "long", "very_long"],
        default="medium",
        help="Target response length for synthetic samples (default: medium)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Maximum context length in tokens (default: 2048)"
    )

    # Split options
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8 = 80%% train)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # API options
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPEN_AI_KEY in .env)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Model for synthetic generation (default: gpt-5-nano)"
    )

    args = parser.parse_args()

    # Initialize builder
    builder = CalibrationBuilder(
        output_dir=args.output,
        api_key=args.api_key,
        model=args.model,
    )

    print("=" * 60)
    print("CALIBRATION DATA BUILDER")
    print("=" * 60)

    # Build from curated datasets
    samples = builder.build_from_curated(
        num_samples=args.num_samples,
        include_tool_calling=not args.no_tool_calling,
        include_summarization=not args.no_summarization,
        include_rag=not args.no_rag,
        seed=args.seed,
    )

    # Generate synthetic samples if requested
    if args.generate_synthetic:
        synthetic = builder.generate_synthetic_samples(
            num_samples=args.generate_synthetic,
            response_length=args.response_length,
            context_length=args.context_length,
            domain=args.domain,
            complexity=args.complexity,
            task_type=args.task_type,
        )
        samples.extend(synthetic)
        print(f"\nTotal samples: {len(samples)} (curated + synthetic)")

    # Create train/test split
    train_samples, test_samples = builder.create_train_test_split(
        samples=samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Save calibration file (train set for imatrix generation)
    calibration_path = builder.save_calibration_file(
        samples=train_samples,
        filename="calibration.txt",
    )

    # Save evaluation file (test set for held-out benchmarking)
    evaluation_path = builder.save_calibration_file(
        samples=test_samples,
        filename="evaluation.txt",
    )

    # Save metadata
    metadata = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "curated_datasets": {
            "tool_calling": not args.no_tool_calling,
            "summarization": not args.no_summarization,
            "rag": not args.no_rag,
        },
        "synthetic_generation": {
            "enabled": args.generate_synthetic is not None,
            "num_samples": args.generate_synthetic or 0,
            "domain": args.domain,
            "complexity": args.complexity,
            "task_type": args.task_type,
            "response_length": args.response_length,
            "context_length": args.context_length,
            "model": args.model,
        },
        "files": {
            "calibration": str(calibration_path),
            "evaluation": str(evaluation_path),
        }
    }

    builder.save_metadata(metadata)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nCalibration file: {calibration_path}")
    print(f"Evaluation file:  {evaluation_path}")
    print(f"\nNext step: Generate importance matrix with llama-imatrix")
    print(f"Then quantize to IQ2_XS using the imatrix for optimal quality")


if __name__ == "__main__":
    main()
