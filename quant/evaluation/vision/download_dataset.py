#!/usr/bin/env python3
"""Download diverse images for comprehensive vision model evaluation.

This script downloads 200-300 images from Open Images V7, Google's large-scale dataset.
See: https://storage.googleapis.com/openimages/web/download_v7.html

The script downloads:
- Image URLs from validation set (190k images available)
- Selects diverse categories matching COCO classes
- Downloads images in parallel for speed

Usage:
    # Download 200 images from Open Images V7
    uv run python download_dataset.py --num-images 200

    # Download specific number
    uv run python download_dataset.py --num-images 300

    # Download to custom directory
    uv run python download_dataset.py --output ./custom_images --num-images 200
"""

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import io


# Open Images URLs (using publicly accessible validation set)
OPEN_IMAGES_BASE = "https://storage.googleapis.com/openimages"
VALIDATION_IMAGES_URL = f"{OPEN_IMAGES_BASE}/2018_04/validation/validation-images-with-rotation.csv"

# Map COCO-like categories to Open Images class names
# Open Images uses different naming - these are common object categories
TARGET_CATEGORIES = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck",
    "Boat", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cattle", "Elephant", "Bear",
    "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Suitcase",
    "Bottle", "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl",
    "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Pizza",
    "Tree", "Building", "Furniture", "Food", "Animal", "Vehicle", "Plant"
]


def download_image(url: str, output_path: Path, timeout: int = 10) -> bool:
    """Download a single image.

    Args:
        url: Image URL
        output_path: Where to save the image
        timeout: Download timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; llamafarm/1.0)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(data)
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def download_open_images_metadata() -> List[Dict[str, str]]:
    """Download Open Images V7 validation metadata CSV.

    Returns:
        List of image metadata dicts with ImageID, Subset, OriginalURL, etc.
    """
    print(f"\n📥 Downloading Open Images V7 metadata...")
    print(f"📂 URL: {VALIDATION_IMAGES_URL}")

    try:
        req = urllib.request.Request(
            VALIDATION_IMAGES_URL,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; llamafarm-vision-eval/1.0)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            csv_data = response.read().decode('utf-8')

        # Parse CSV
        reader = csv.DictReader(io.StringIO(csv_data))
        metadata = list(reader)

        print(f"✅ Downloaded metadata for {len(metadata)} images")
        return metadata

    except Exception as e:
        print(f"❌ Failed to download metadata: {e}")
        print(f"\n💡 Alternative: Download manually from:")
        print(f"   {VALIDATION_IMAGES_URL}")
        print(f"   Save to: validation_metadata.csv")
        return []


def select_diverse_images(
    metadata: List[Dict[str, str]],
    num_images: int = 200,
    target_categories: Set[str] = None,
) -> List[Dict[str, str]]:
    """Select diverse images from metadata.

    Args:
        metadata: List of image metadata dicts
        num_images: Number of images to select
        target_categories: Set of categories to focus on (optional)

    Returns:
        Selected subset of metadata
    """
    print(f"\n🔍 Selecting {num_images} diverse images...")

    # Shuffle for randomness
    random.shuffle(metadata)

    # Take first N images (already shuffled for diversity)
    selected = metadata[:num_images]

    print(f"✅ Selected {len(selected)} images")
    return selected


def download_open_images_dataset(
    output_dir: Path,
    num_images: int = 200,
) -> Dict[str, Any]:
    """Download images from Open Images V7 validation set.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to download

    Returns:
        Dataset metadata
    """
    print(f"\n📥 Downloading {num_images} images from Open Images V7...")
    print(f"📂 Output directory: {output_dir}")

    # Step 1: Download metadata
    metadata = download_open_images_metadata()
    if not metadata:
        print("❌ Failed to download metadata. Cannot proceed.")
        return {}

    # Step 2: Select diverse images
    selected = select_diverse_images(metadata, num_images)

    # Step 3: Create subdirectories
    detection_dir = output_dir / "detection"
    classification_dir = output_dir / "classification"
    embedding_dir = output_dir / "embedding"

    for dir in [detection_dir, classification_dir, embedding_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # Distribute images across tasks
    num_detection = int(num_images * 0.5)  # 50% for detection
    num_classification = int(num_images * 0.35)  # 35% for classification
    num_embedding = num_images - num_detection - num_classification  # Rest for embedding

    downloaded = []
    failed = []

    def download_with_metadata(idx: int, image_meta: Dict[str, str], task_dir: Path) -> tuple:
        """Download single image from Open Images."""
        image_id = image_meta.get('ImageID', f'image_{idx}')
        url = image_meta.get('OriginalURL', '')

        if not url:
            return (False, f"No URL for {image_id}")

        output_path = task_dir / f"{image_id}.jpg"

        if download_image(url, output_path):
            return (True, str(output_path.relative_to(output_dir)))
        return (False, image_id)

    # Step 4: Download images
    print(f"\n🔍 Downloading detection images ({num_detection})...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(num_detection):
            if i < len(selected):
                futures.append(executor.submit(download_with_metadata, i, selected[i], detection_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Detection"):
            success, result = future.result()
            if success:
                downloaded.append(result)
            else:
                failed.append(result)

    print(f"\n🖼️  Downloading classification images ({num_classification})...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(num_detection, num_detection + num_classification):
            if i < len(selected):
                futures.append(executor.submit(download_with_metadata, i, selected[i], classification_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Classification"):
            success, result = future.result()
            if success:
                downloaded.append(result)
            else:
                failed.append(result)

    print(f"\n🧬 Downloading embedding images ({num_embedding})...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(num_detection + num_classification, num_images):
            if i < len(selected):
                futures.append(executor.submit(download_with_metadata, i, selected[i], embedding_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            success, result = future.result()
            if success:
                downloaded.append(result)
            else:
                failed.append(result)

    # Create dataset metadata
    result_metadata = {
        "version": "1.0",
        "source": "open_images_v7_validation",
        "source_url": VALIDATION_IMAGES_URL,
        "total_images": len(downloaded),
        "failed": len(failed),
        "distribution": {
            "detection": len([d for d in downloaded if "detection" in str(d)]),
            "classification": len([d for d in downloaded if "classification" in str(d)]),
            "embedding": len([d for d in downloaded if "embedding" in str(d)]),
        },
        "notes": "Images downloaded from Open Images V7 validation set. Licensed under CC BY 4.0."
    }

    # Save metadata
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(result_metadata, f, indent=2)

    print(f"\n✅ Download complete!")
    print(f"   Downloaded: {len(downloaded)} images")
    print(f"   Failed: {len(failed)} images")
    print(f"   Metadata saved to: {metadata_path}")

    return result_metadata


def download_sample_images(output_dir: Path) -> None:
    """Download a small set of sample images for quick testing.

    This downloads ~20-30 diverse images suitable for testing the evaluation pipeline.
    """
    print("\n📥 Downloading sample images for testing...")

    sample_urls = [
        # Sample URLs for diverse testing (these are example placeholders)
        ("https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=800", "detection/cat_01.jpg"),
        ("https://images.unsplash.com/photo-1537151625747-768eb6cf92b2?w=800", "detection/dog_01.jpg"),
        ("https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=800", "detection/car_01.jpg"),
        ("https://images.unsplash.com/photo-1551782450-a2132b4ba21d?w=800", "classification/food_01.jpg"),
        ("https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=800", "embedding/landscape_01.jpg"),
    ]

    for url, rel_path in tqdm(sample_urls, desc="Downloading samples"):
        output_path = output_dir / rel_path
        download_image(url, output_path)

    print(f"✅ Sample images downloaded to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download diverse images from Open Images V7 for vision model evaluation"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="Number of images to download (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="images",
        help="Output directory (default: images/)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Vision Evaluation Dataset Downloader - Open Images V7")
    print("=" * 70)

    # Download from Open Images V7
    download_open_images_dataset(
        output_dir,
        num_images=args.num_images,
    )

    print(f"\n{'=' * 70}")
    print("✅ Dataset download complete!")
    print(f"📂 Images saved to: {output_dir.absolute()}")
    print(f"\nNext step: Run vision evaluation:")
    print(f"  uv run python evaluation/vision/run_eval.py \\")
    print(f"    --models-dir ../../models \\")
    print(f"    --images {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
