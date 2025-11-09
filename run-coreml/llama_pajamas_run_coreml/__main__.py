"""CLI for llama-pajamas-coreml."""

import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Llama-Pajamas CoreML Runtime (Apple Silicon Multi-Modal)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Vision commands
    vision_parser = subparsers.add_parser("detect", help="Object detection")
    vision_parser.add_argument("--model", required=True, help="Path to CoreML model")
    vision_parser.add_argument("--image", required=True, help="Path to input image")
    vision_parser.add_argument("--confidence", type=float, default=0.5)

    classify_parser = subparsers.add_parser("classify", help="Image classification")
    classify_parser.add_argument("--model", required=True, help="Path to CoreML model")
    classify_parser.add_argument("--image", required=True, help="Path to input image")
    classify_parser.add_argument("--top-k", type=int, default=5)

    embed_parser = subparsers.add_parser("embed", help="Image embeddings")
    embed_parser.add_argument("--model", required=True, help="Path to CoreML model")
    embed_parser.add_argument("--image", required=True, help="Path to input image")

    # Speech commands
    stt_parser = subparsers.add_parser("transcribe", help="Speech-to-text")
    stt_parser.add_argument("--model", required=True, help="Path to CoreML model")
    stt_parser.add_argument("--audio", required=True, help="Path to input audio")
    stt_parser.add_argument("--language", help="Language code (e.g., 'en', 'zh')")

    tts_parser = subparsers.add_parser("synthesize", help="Text-to-speech")
    tts_parser.add_argument("--model", required=True, help="Path to CoreML model")
    tts_parser.add_argument("--text", required=True, help="Text to synthesize")
    tts_parser.add_argument("--output", required=True, help="Output audio file")
    tts_parser.add_argument("--speaker", type=int, default=0)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # TODO: Implement command handlers
    logger.error(f"Command '{args.command}' not yet implemented")
    logger.info("Week 1-2: Setting up infrastructure")
    logger.info("Week 3-6: Implementing vision and speech pipelines")
    return 1


if __name__ == "__main__":
    sys.exit(main())
