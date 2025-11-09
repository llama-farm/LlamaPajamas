"""Download Open Images v7 evaluation dataset.

Downloads a diverse subset of ~250 images for vision model evaluation.
Categories: animals, vehicles, objects, people, scenes.
"""

import logging
import urllib.request
import csv
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Open Images V7 URLs
VALIDATION_ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-validation-annotations-bbox.csv"
CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"
IMAGE_BASE_URL = "https://storage.googleapis.com/openimages/data"

# Target categories for diverse evaluation (COCO-like classes)
TARGET_CATEGORIES = {
    # Animals
    "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe",
    "Bird", "Chicken", "Duck", "Goose", "Penguin",
    # Vehicles
    "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat",
    "Bicycle", "Van", "Taxi",
    # Objects
    "Traffic sign", "Traffic light", "Stop sign", "Parking meter",
    "Bench", "Chair", "Table", "Couch", "Bed",
    "Bottle", "Cup", "Fork", "Knife", "Spoon", "Bowl",
    "Backpack", "Umbrella", "Handbag", "Suitcase",
    # People & activities
    "Person", "Man", "Woman", "Boy", "Girl",
    # Food
    "Apple", "Banana", "Orange", "Pizza", "Cake", "Hot dog",
    # Electronics
    "Mobile phone", "Computer keyboard", "Computer monitor", "Laptop",
    "Television", "Remote control",
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download file from URL."""
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def load_class_descriptions(output_dir: Path) -> Dict[str, str]:
    """Download and parse class descriptions."""
    logger.info("Downloading class descriptions...")

    csv_path = output_dir / "class-descriptions.csv"
    if not csv_path.exists():
        download_file(CLASS_DESCRIPTIONS_URL, csv_path)

    # Parse CSV: label_name -> class_id
    class_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                class_id, class_name = row[0], row[1]
                class_map[class_name] = class_id

    logger.info(f"Loaded {len(class_map)} class descriptions")
    return class_map


def download_validation_annotations(output_dir: Path) -> Path:
    """Download validation annotations CSV."""
    logger.info("Downloading validation annotations...")

    csv_path = output_dir / "validation-annotations.csv"
    if csv_path.exists():
        logger.info(f"Annotations already downloaded: {csv_path}")
        return csv_path

    download_file(VALIDATION_ANNOTATIONS_URL, csv_path)
    logger.info(f"Saved to: {csv_path}")
    return csv_path


def select_diverse_images(
    annotations_csv: Path,
    class_map: Dict[str, str],
    target_count: int = 250,
) -> List[Dict[str, str]]:
    """Select diverse images from validation set."""
    logger.info(f"Selecting {target_count} diverse images...")

    # Map class names to IDs
    target_class_ids = {class_map.get(name) for name in TARGET_CATEGORIES if class_map.get(name)}
    logger.info(f"Targeting {len(target_class_ids)} relevant classes")

    # Parse annotations and group by image
    image_data = {}  # image_id -> {classes: Set[str], annotations: int}

    with open(annotations_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageID']
            label_name = row['LabelName']

            # Only include images with target classes
            if label_name not in target_class_ids:
                continue

            if image_id not in image_data:
                image_data[image_id] = {'classes': set(), 'annotations': 0}

            image_data[image_id]['classes'].add(label_name)
            image_data[image_id]['annotations'] += 1

    logger.info(f"Found {len(image_data)} images with target classes")

    # Sort by number of target classes (prefer images with multiple objects)
    ranked_images = sorted(
        image_data.items(),
        key=lambda x: (len(x[1]['classes']), x[1]['annotations']),
        reverse=True
    )

    # Select diverse subset
    selected = []
    seen_classes = set()

    # First pass: prioritize images with new classes
    for image_id, data in ranked_images:
        if len(selected) >= target_count:
            break

        # Check if this image introduces new classes
        new_classes = data['classes'] - seen_classes
        if new_classes or len(selected) < target_count // 2:
            selected.append({
                'image_id': image_id,
                'classes': data['classes'],
                'annotations': data['annotations'],
            })
            seen_classes.update(data['classes'])

    # Second pass: fill remaining slots with random images
    if len(selected) < target_count:
        remaining = [img for img_id, _ in ranked_images
                    if img_id not in {s['image_id'] for s in selected}]
        random.shuffle(remaining)
        for image_id in remaining[:target_count - len(selected)]:
            data = image_data[image_id]
            selected.append({
                'image_id': image_id,
                'classes': data['classes'],
                'annotations': data['annotations'],
            })

    logger.info(f"Selected {len(selected)} images covering {len(seen_classes)} classes")
    return selected


def download_image(image_id: str, output_dir: Path) -> bool:
    """Download single image."""
    # Open Images uses the image ID to construct the URL
    # Format: https://storage.googleapis.com/openimages/data/{image_id}.jpg
    url = f"{IMAGE_BASE_URL}/{image_id}.jpg"
    dest_path = output_dir / f"{image_id}.jpg"

    if dest_path.exists():
        return True

    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        logger.debug(f"Failed to download {image_id}: {e}")
        return False


def download_images_parallel(
    image_list: List[Dict[str, str]],
    output_dir: Path,
    max_workers: int = 10,
) -> int:
    """Download images in parallel."""
    logger.info(f"Downloading {len(image_list)} images with {max_workers} workers...")

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_image = {
            executor.submit(download_image, img['image_id'], images_dir): img
            for img in image_list
        }

        # Process completed downloads
        for i, future in enumerate(as_completed(future_to_image), 1):
            img = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1

                if i % 25 == 0:
                    logger.info(f"Progress: {i}/{len(image_list)} ({successful} success, {failed} failed)")
            except Exception as e:
                logger.error(f"Error downloading {img['image_id']}: {e}")
                failed += 1

    logger.info(f"✅ Downloaded {successful}/{len(image_list)} images")
    logger.info(f"   Failed: {failed}")
    return successful


def main():
    """Download Open Images v7 evaluation dataset."""
    # Setup
    output_dir = Path("../run/llama_pajamas_run/evaluation/open_images_v7")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Open Images V7 Evaluation Dataset Download")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("")

    # Step 1: Download class descriptions
    class_map = load_class_descriptions(output_dir)

    # Step 2: Download validation annotations
    annotations_csv = download_validation_annotations(output_dir)

    # Step 3: Select diverse images
    selected_images = select_diverse_images(
        annotations_csv=annotations_csv,
        class_map=class_map,
        target_count=250,
    )

    # Step 4: Download images
    successful = download_images_parallel(
        image_list=selected_images,
        output_dir=output_dir,
        max_workers=10,
    )

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Successfully downloaded {successful} images")
    logger.info(f"📁 Saved to: {output_dir / 'images'}")
    logger.info("")

    if successful >= 200:
        logger.info("🎉 Dataset download complete!")
        return 0
    else:
        logger.warning("⚠️  Some images failed to download")
        return 1


if __name__ == "__main__":
    exit(main())
