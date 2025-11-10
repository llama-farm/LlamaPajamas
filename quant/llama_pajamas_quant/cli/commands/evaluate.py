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
    import subprocess
    import sys
    from pathlib import Path

    print(f"Evaluating LLM in {args.model_dir}")
    print(f"Questions: {args.num_questions}")

    # Find evaluation script
    eval_script = Path(__file__).parent.parent.parent.parent / "evaluation" / "llm" / "run_eval.py"

    if not eval_script.exists():
        logger.error(f"Evaluation script not found: {eval_script}")
        return 1

    # Build command
    cmd = [
        sys.executable,
        str(eval_script),
        "--model-dir", str(args.model_dir),
        "--num-questions", str(args.num_questions),
    ]

    if args.use_llm_judge:
        cmd.append("--use-llm-judge")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def evaluate_vision(args):
    """Evaluate vision model."""
    import subprocess
    import sys
    from pathlib import Path

    print(f"Evaluating vision model: {args.model}")

    # Find evaluation script
    eval_script = Path(__file__).parent.parent.parent.parent / "evaluation" / "vision" / "run_eval.py"

    if not eval_script.exists():
        logger.error(f"Evaluation script not found: {eval_script}")
        logger.info("Please ensure evaluation scripts are in quant/evaluation/")
        return 1

    # Build command
    cmd = [
        sys.executable,
        str(eval_script),
        "--model", args.model,
        "--models-dir", str(args.models_dir),
        "--images", str(args.images),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def compare_evaluations(args):
    """Compare model evaluations."""
    import subprocess
    import sys
    from pathlib import Path

    print(f"Comparing evaluations in {args.model_dir}")

    # Find comparison script
    compare_script = Path(__file__).parent.parent.parent.parent / "evaluation" / "llm" / "compare_evaluations.py"

    if not compare_script.exists():
        logger.error(f"Comparison script not found: {compare_script}")
        return 1

    # Build command
    cmd = [
        sys.executable,
        str(compare_script),
        "--model-dir", str(args.model_dir),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Comparison failed: {e}")
        return 1
