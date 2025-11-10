"""Evaluation commands."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_command(subparsers):
    """Register evaluate command."""
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_subparsers = eval_parser.add_subparsers(dest='eval_type')

    # LLM evaluation
    llm_parser = eval_subparsers.add_parser('llm', help='Evaluate LLM')
    llm_parser.add_argument('--model-dir', required=True, type=Path, help='Model directory')
    llm_parser.add_argument('--num-questions', type=int, default=140, help='Number of questions')
    llm_parser.add_argument('--use-llm-judge', action='store_true', help='Use LLM judge for summarization')
    llm_parser.set_defaults(func=evaluate_llm)

    # Vision evaluation
    vision_parser = eval_subparsers.add_parser('vision', help='Evaluate vision model')
    vision_parser.add_argument('--model', required=True, help='Model name')
    vision_parser.add_argument('--models-dir', required=True, type=Path, help='Models directory')
    vision_parser.add_argument('--images', required=True, type=Path, help='Image directory')
    vision_parser.set_defaults(func=evaluate_vision)

    # Compare models
    compare_parser = eval_subparsers.add_parser('compare', help='Compare model evaluations')
    compare_parser.add_argument('--model-dir', required=True, type=Path, help='Model directory')
    compare_parser.set_defaults(func=compare_evaluations)


def evaluate_llm(args):
    """Evaluate LLM."""
    logger.info(f"Evaluating LLM in {args.model_dir}")
    logger.info(f"Questions: {args.num_questions}")
    logger.info("Running evaluation scripts...")
    logger.error("LLM evaluation command not yet fully implemented")
    logger.info("Please use: cd quant && uv run python evaluation/llm/run_eval.py")
    return 1


def evaluate_vision(args):
    """Evaluate vision model."""
    logger.info(f"Evaluating vision model: {args.model}")
    logger.error("Vision evaluation command not yet fully implemented")
    logger.info("Please use: cd run-coreml && uv run python ../quant/evaluation/vision/run_eval.py")
    return 1


def compare_evaluations(args):
    """Compare model evaluations."""
    logger.info(f"Comparing evaluations in {args.model_dir}")
    logger.error("Comparison command not yet fully implemented")
    logger.info("Please use: cd quant && uv run python evaluation/llm/compare_evaluations.py")
    return 1
