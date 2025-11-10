"""Batch processing commands."""

import argparse
import json
import logging
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def register_command(subparsers):
    """Register batch command."""
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch process models',
        description='Process multiple models from configuration file'
    )
    batch_parser.add_argument('--config', required=True, type=Path, help='Batch config file (YAML/JSON)')
    batch_parser.add_argument('--parallel', type=int, default=1, help='Parallel workers')
    batch_parser.add_argument('--dry-run', action='store_true', help='Dry run (no execution)')
    batch_parser.set_defaults(func=batch_process)


def batch_process(args):
    """Batch process models from config."""
    config_path = args.config

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    # Load config
    try:
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    models = config.get('models', [])
    if not models:
        logger.error("No models found in config")
        return 1

    parallel = config.get('parallel', args.parallel)

    logger.info(f"Loaded {len(models)} models from config")
    logger.info(f"Parallel workers: {parallel}")

    if args.dry_run:
        print("\nDry run mode - would process:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model.get('model', model.get('name'))}")
        return 0

    # Process models
    logger.info("\nStarting batch processing...")

    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(process_model, model): model for model in models}

            for future in as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    logger.info(f"✅ Completed: {model.get('model')}")
                except Exception as e:
                    logger.error(f"❌ Failed: {model.get('model')} - {e}")
    else:
        for model in models:
            try:
                process_model(model)
                logger.info(f"✅ Completed: {model.get('model')}")
            except Exception as e:
                logger.error(f"❌ Failed: {model.get('model')} - {e}")

    logger.info("\nBatch processing complete!")
    return 0


def process_model(model_config):
    """Process single model from config."""
    model_type = model_config.get('type', 'llm')

    if model_type == 'llm':
        from ...core import Quantizer
        quantizer = Quantizer()

        return quantizer.convert(
            model_path=model_config['model'],
            output_dir=model_config['output'],
            formats=model_config.get('formats', ['gguf']),
            gguf_precision=model_config.get('gguf_precision', 'Q4_K_M'),
            mlx_bits=model_config.get('mlx_bits', 4),
        )
    else:
        logger.warning(f"Unsupported model type: {model_type}")
        return None
