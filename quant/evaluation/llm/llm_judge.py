#!/usr/bin/env python3
"""Optional LLM-as-Judge evaluation for summarization quality.

This module provides an optional evaluation method for summarization tasks
where simple string matching isn't sufficient. It uses an external LLM API
to judge the quality of generated summaries.

Usage:
    # Enable LLM judge with API key
    export OPENAI_API_KEY=sk-...
    uv run python evaluation/llm/run_eval.py --use-llm-judge

    # Or use a different model
    export LLM_JUDGE_MODEL=gpt-5-nano
    uv run python evaluation/llm/run_eval.py --use-llm-judge
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: float  # 0-10 scale
    reasoning: str
    passed: bool  # score >= threshold


class LLMJudge:
    """LLM-as-judge evaluator for open-ended questions."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        score_threshold: float = 7.0,
    ):
        """Initialize LLM judge.

        Args:
            model: Model to use for judging (gpt-5-nano, gpt-4, etc.)
            api_key: API key (defaults to OPENAI_API_KEY or ANTHROPIC_API_KEY env vars)
            score_threshold: Minimum score to pass (default: 7.0/10)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.score_threshold = score_threshold

        if not self.api_key:
            raise ValueError(
                "API key required for LLM judge. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable"
            )

        # Initialize appropriate client
        if "gpt" in model or "openai" in model:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.provider = "openai"
            except ImportError:
                raise ImportError("OpenAI package required. Install with: uv pip install openai")

        elif "claude" in model or "anthropic" in model:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                self.provider = "anthropic"
            except ImportError:
                raise ImportError("Anthropic package required. Install with: uv pip install anthropic")

        else:
            raise ValueError(f"Unsupported model: {model}. Use gpt-4 or claude-3 models.")

    def evaluate_summary(
        self,
        original_text: str,
        summary: str,
        category: str = "summarization",
    ) -> JudgeResult:
        """Evaluate a summary using LLM judge.

        Args:
            original_text: The original text that was summarized
            summary: The generated summary to evaluate
            category: Category of the question (for context)

        Returns:
            JudgeResult with score, reasoning, and pass/fail
        """
        prompt = f"""You are an expert evaluator assessing the quality of a text summary.

**Original Text:**
{original_text}

**Generated Summary:**
{summary}

**Evaluation Criteria:**
Evaluate the summary on a scale of 0-10 based on:
1. **Accuracy**: Does the summary correctly represent the original text without hallucinations?
2. **Completeness**: Does it capture the key points?
3. **Conciseness**: Is it appropriately brief without unnecessary details?
4. **Coherence**: Is it well-written and easy to understand?

**Output Format:**
Provide your evaluation as JSON:
{{
    "score": <0-10>,
    "reasoning": "<detailed explanation of your score>",
    "accuracy": <0-10>,
    "completeness": <0-10>,
    "conciseness": <0-10>,
    "coherence": <0-10>
}}

Be strict and objective. A score of 10 means perfect, 7+ means good, 5 means acceptable, below 5 means poor."""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert text evaluator. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                result = json.loads(response.choices[0].message.content)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                )
                # Extract JSON from response
                content = response.content[0].text
                result = json.loads(content)

            score = float(result["score"])
            reasoning = result["reasoning"]

            return JudgeResult(
                score=score,
                reasoning=reasoning,
                passed=score >= self.score_threshold,
            )

        except Exception as e:
            print(f"⚠️  Error during LLM judge evaluation: {e}")
            # Return neutral score on error
            return JudgeResult(
                score=5.0,
                reasoning=f"Evaluation failed: {str(e)}",
                passed=False,
            )

    def evaluate_code(
        self,
        prompt: str,
        generated_code: str,
        language: str = "python",
    ) -> JudgeResult:
        """Evaluate generated code using LLM judge.

        Args:
            prompt: The code generation prompt
            generated_code: The generated code to evaluate
            language: Programming language

        Returns:
            JudgeResult with score, reasoning, and pass/fail
        """
        eval_prompt = f"""You are an expert code reviewer evaluating generated code.

**Task:**
{prompt}

**Generated Code:**
```{language}
{generated_code}
```

**Evaluation Criteria:**
Evaluate the code on a scale of 0-10 based on:
1. **Correctness**: Does it solve the problem correctly?
2. **Code Quality**: Is it well-structured, readable, and following best practices?
3. **Completeness**: Does it include necessary components (error handling, edge cases, etc.)?
4. **Efficiency**: Is the algorithm/approach reasonable?

**Output Format:**
Provide your evaluation as JSON:
{{
    "score": <0-10>,
    "reasoning": "<detailed explanation>",
    "correctness": <0-10>,
    "code_quality": <0-10>,
    "completeness": <0-10>,
    "efficiency": <0-10>
}}

Be objective. A score of 10 means production-ready code, 7+ means good, 5 means works but needs improvement, below 5 means significant issues."""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert code reviewer. Respond only with valid JSON."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                result = json.loads(response.choices[0].message.content)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": eval_prompt}
                    ],
                    temperature=0.1,
                )
                content = response.content[0].text
                result = json.loads(content)

            score = float(result["score"])
            reasoning = result["reasoning"]

            return JudgeResult(
                score=score,
                reasoning=reasoning,
                passed=score >= self.score_threshold,
            )

        except Exception as e:
            print(f"⚠️  Error during code evaluation: {e}")
            return JudgeResult(
                score=5.0,
                reasoning=f"Evaluation failed: {str(e)}",
                passed=False,
            )


# Convenience function
def create_judge(
    use_judge: bool = False,
    model: Optional[str] = None,
    threshold: float = 7.0,
) -> Optional[LLMJudge]:
    """Create LLM judge if enabled.

    Args:
        use_judge: Whether to enable LLM judge
        model: Model to use (default: from env or gpt-4-turbo-preview)
        threshold: Score threshold for passing

    Returns:
        LLMJudge instance if enabled, None otherwise
    """
    if not use_judge:
        return None

    model = model or os.getenv("LLM_JUDGE_MODEL", "gpt-4-turbo-preview")

    try:
        return LLMJudge(model=model, score_threshold=threshold)
    except Exception as e:
        print(f"⚠️  Failed to initialize LLM judge: {e}")
        print("   Continuing without LLM judge evaluation...")
        return None
