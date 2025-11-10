#!/usr/bin/env python3
"""
Download LibriSpeech test-clean audio samples for STT evaluation.

LibriSpeech is a corpus of read English speech (1000 hours), derived from
audiobooks. The test-clean subset has clean speech with known transcriptions.

Usage:
    uv run python evaluation/stt/download_audio.py --num-samples 20
"""

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Any


LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_TEST_CLEAN_SIZE_MB = 346  # Approximate size


def download_librispeech(output_dir: Path, num_samples: int = 20) -> Dict[str, Any]:
    """
    Download LibriSpeech test-clean subset.

    Args:
        output_dir: Directory to save audio files
        num_samples: Number of audio samples to extract (default: 20)

    Returns:
        Metadata about downloaded samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading LibriSpeech test-clean subset...")
    print(f"   Size: ~{LIBRISPEECH_TEST_CLEAN_SIZE_MB} MB")
    print(f"   URL: {LIBRISPEECH_URL}")
    print(f"   This may take a few minutes...")

    # Download
    tar_path = output_dir / "test-clean.tar.gz"

    if not tar_path.exists():
        print(f"\n⬇️  Downloading...")
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)
        print(f"   ✅ Downloaded to {tar_path}")
    else:
        print(f"   ℹ️  Using cached download: {tar_path}")

    # Extract samples
    print(f"\n📦 Extracting {num_samples} audio samples...")

    samples = []
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        # Find .flac files and their corresponding .txt transcriptions
        flac_files = [m for m in members if m.name.endswith(".flac")][:num_samples]

        for idx, flac_member in enumerate(flac_files, 1):
            # Extract audio file
            flac_path = Path(flac_member.name)
            sample_id = flac_path.stem

            # Save with simple name
            output_audio = audio_dir / f"sample_{idx:03d}.flac"

            # Extract
            tar.extract(flac_member, path=output_dir / "temp")
            temp_flac = output_dir / "temp" / flac_member.name

            # Move to audio dir
            temp_flac.rename(output_audio)

            # Find corresponding transcription
            # LibriSpeech format: speaker-book-utterance.flac
            # Transcription in: speaker-book.trans.txt
            parts = sample_id.split("-")
            if len(parts) >= 2:
                trans_file = f"{parts[0]}-{parts[1]}.trans.txt"
                trans_member = next((m for m in members if m.name.endswith(trans_file)), None)

                if trans_member:
                    tar.extract(trans_member, path=output_dir / "temp")
                    trans_path = output_dir / "temp" / trans_member.name

                    # Read transcription
                    with open(trans_path) as f:
                        for line in f:
                            if line.startswith(sample_id):
                                text = line.split(" ", 1)[1].strip()

                                samples.append({
                                    "id": f"sample_{idx:03d}",
                                    "audio_file": str(output_audio.name),
                                    "text": text,
                                    "duration_sec": None,  # Will be calculated later
                                    "source": "LibriSpeech test-clean",
                                    "original_id": sample_id
                                })
                                break

            print(f"   [{idx}/{num_samples}] Extracted: {output_audio.name}")

    # Cleanup temp directory
    import shutil
    temp_dir = output_dir / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Calculate durations using ffprobe if available
    print(f"\n⏱️  Calculating audio durations...")
    try:
        import subprocess
        for sample in samples:
            audio_path = audio_dir / sample["audio_file"]
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                sample["duration_sec"] = round(float(result.stdout.strip()), 2)
            else:
                sample["duration_sec"] = None
    except FileNotFoundError:
        print("   ⚠️  ffprobe not found - durations will be calculated during evaluation")

    # Save metadata
    metadata = {
        "version": "1.0",
        "source": "LibriSpeech test-clean",
        "source_url": LIBRISPEECH_URL,
        "total_samples": len(samples),
        "samples": samples,
        "notes": "LibriSpeech test-clean subset - clean read English speech with transcriptions"
    }

    metadata_path = output_dir / "dataset.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Downloaded {len(samples)} audio samples")
    print(f"📁 Audio files: {audio_dir}")
    print(f"📄 Metadata: {metadata_path}")
    print(f"\n💡 To clean up the large .tar.gz file:")
    print(f"   rm {tar_path}")

    return metadata


def download_common_voice(output_dir: Path, num_samples: int = 20) -> Dict[str, Any]:
    """
    Alternative: Download Mozilla Common Voice samples.

    Note: Requires manual download and extraction.
    """
    print("⚠️  Common Voice requires manual download from:")
    print("   https://commonvoice.mozilla.org/en/datasets")
    print("\n💡 Use LibriSpeech for automated downloads:")
    print("   uv run python evaluation/stt/download_audio.py --source librispeech")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Download audio samples for STT evaluation")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of audio samples to download (default: 20)"
    )
    parser.add_argument(
        "--source",
        default="librispeech",
        choices=["librispeech", "commonvoice"],
        help="Audio source (default: librispeech)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: evaluation/stt/)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent

    # Download
    if args.source == "librispeech":
        download_librispeech(output_dir, args.num_samples)
    elif args.source == "commonvoice":
        download_common_voice(output_dir, args.num_samples)


if __name__ == "__main__":
    main()
