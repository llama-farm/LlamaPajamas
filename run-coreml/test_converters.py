"""Test script for CoreML converters.

Tests YOLO-v8, CLIP, and ViT conversion to CoreML.
"""

import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_yolo_conversion(output_dir: Path = Path("./models")):
    """Test YOLO-v8 conversion."""
    logger.info("=" * 70)
    logger.info("TEST 1: YOLO-v8 Conversion")
    logger.info("=" * 70)

    try:
        from llama_pajamas_run_coreml.converters import convert_yolo, CoreMLConverterConfig

        # Use smallest YOLO model for testing
        config = CoreMLConverterConfig(
            precision="float16",
            compute_units="ALL",
            optimize_for_ane=True,
        )

        output_path = output_dir / "yolo-v8n-coreml"
        logger.info(f"Converting YOLO-v8n to: {output_path}")

        model_path = convert_yolo(
            model_name="yolov8n.pt",
            output_dir=str(output_path),
            config=config,
        )

        logger.info(f"✅ YOLO-v8 conversion successful: {model_path}")
        logger.info(f"Model size: {(model_path.stat().st_size / 1024 / 1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ YOLO-v8 conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_conversion(output_dir: Path = Path("./models")):
    """Test CLIP conversion."""
    logger.info("=" * 70)
    logger.info("TEST 2: CLIP Vision Encoder Conversion")
    logger.info("=" * 70)

    try:
        from llama_pajamas_run_coreml.converters import convert_clip, CoreMLConverterConfig

        config = CoreMLConverterConfig(
            precision="float16",
            compute_units="ALL",
            optimize_for_ane=True,
        )

        output_path = output_dir / "clip-vit-base-coreml"
        logger.info(f"Converting CLIP to: {output_path}")

        model_path = convert_clip(
            model_name="openai/clip-vit-base-patch32",
            output_dir=str(output_path),
            config=config,
        )

        logger.info(f"✅ CLIP conversion successful: {model_path}")
        logger.info(f"Model size: {(model_path.stat().st_size / 1024 / 1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ CLIP conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vit_conversion(output_dir: Path = Path("./models")):
    """Test ViT conversion."""
    logger.info("=" * 70)
    logger.info("TEST 3: ViT Classification Conversion")
    logger.info("=" * 70)

    try:
        from llama_pajamas_run_coreml.converters import convert_vit, CoreMLConverterConfig

        config = CoreMLConverterConfig(
            precision="float16",
            compute_units="ALL",
            optimize_for_ane=True,
        )

        output_path = output_dir / "vit-base-coreml"
        logger.info(f"Converting ViT to: {output_path}")

        model_path = convert_vit(
            model_name="google/vit-base-patch16-224",
            output_dir=str(output_path),
            config=config,
        )

        logger.info(f"✅ ViT conversion successful: {model_path}")
        logger.info(f"Model size: {(model_path.stat().st_size / 1024 / 1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ ViT conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all converter tests."""
    parser = argparse.ArgumentParser(description="Test CoreML converters")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for converted models"
    )
    parser.add_argument(
        "--test",
        choices=["yolo", "clip", "vit", "all"],
        default="all",
        help="Which converter to test"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("")

    results = {}

    if args.test in ["yolo", "all"]:
        results["yolo"] = test_yolo_conversion(output_dir)
        logger.info("")

    if args.test in ["clip", "all"]:
        results["clip"] = test_clip_conversion(output_dir)
        logger.info("")

    if args.test in ["vit", "all"]:
        results["vit"] = test_vit_conversion(output_dir)
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name.upper()}: {status}")

    all_passed = all(results.values())
    logger.info("")
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.info("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
