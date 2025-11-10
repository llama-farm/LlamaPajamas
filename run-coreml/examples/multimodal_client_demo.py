#!/usr/bin/env python3
"""
Multi-modal client demonstration.

Tests the multi-modal server endpoints for Vision and STT.

Usage:
    # Start the server first (in another terminal):
    uv run python examples/multimodal_server_demo.py

    # Then run this client:
    uv run python examples/multimodal_client_demo.py
"""

import base64
import requests
from pathlib import Path


def test_health():
    """Test health check endpoint."""
    print("\n" + "="*70)
    print("Testing: Health Check")
    print("="*70)

    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


def test_list_models():
    """Test model listing endpoint."""
    print("\n" + "="*70)
    print("Testing: List Models")
    print("="*70)

    response = requests.get("http://localhost:8000/v1/models")
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Models available: {len(data['data'])}")
    for model in data['data']:
        print(f"  - {model['id']}: {model['capabilities']}")


def test_vision_detection():
    """Test vision detection endpoint."""
    print("\n" + "="*70)
    print("Testing: Vision Detection")
    print("="*70)

    # Load a test image (create a simple solid color image)
    from PIL import Image
    import io

    # Create a simple test image (blue square)
    img = Image.new("RGB", (640, 480), color=(0, 0, 255))

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Make request
    response = requests.post(
        "http://localhost:8000/v1/images/detect",
        json={
            "image": f"data:image/jpeg;base64,{img_base64}",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Detections: {len(data['detections'])}")
        for det in data['detections'][:3]:  # Show first 3
            print(f"  - {det['label']}: {det['confidence']:.3f} @ {det['box']}")
    else:
        print(f"Error: {response.text}")


def test_stt_transcription():
    """Test STT transcription endpoint."""
    print("\n" + "="*70)
    print("Testing: STT Transcription")
    print("="*70)

    # Check if test audio exists
    audio_dir = Path(__file__).parent.parent.parent / "quant/evaluation/stt/audio"
    test_audio = audio_dir / "sample_001.flac"

    if not test_audio.exists():
        print(f"⚠️  Test audio not found: {test_audio}")
        print("   Download with: cd quant/evaluation/stt && uv run python download_audio.py")
        return

    # Make request
    with open(test_audio, "rb") as f:
        response = requests.post(
            "http://localhost:8000/v1/audio/transcriptions",
            files={"file": ("audio.flac", f, "audio/flac")},
            data={
                "model": "whisper-tiny",
                "response_format": "json",
            }
        )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Transcription: {data['text']}")
    else:
        print(f"Error: {response.text}")


def test_tts():
    """Test TTS endpoint."""
    print("\n" + "="*70)
    print("Testing: TTS (Text-to-Speech)")
    print("="*70)

    response = requests.post(
        "http://localhost:8000/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello! This is a test of the text to speech system.",
            "voice": "alloy",  # Maps to Samantha
            "response_format": "wav",
        }
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        # Save audio to file
        output_path = Path("/tmp/tts_test.wav")
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to: {output_path}")
        print(f"Audio size: {len(response.content)} bytes")
    else:
        print(f"Error: {response.text}")


def test_stt_verbose():
    """Test STT with verbose output (includes segments)."""
    print("\n" + "="*70)
    print("Testing: STT Transcription (Verbose)")
    print("="*70)

    audio_dir = Path(__file__).parent.parent.parent / "quant/evaluation/stt/audio"
    test_audio = audio_dir / "sample_001.flac"

    if not test_audio.exists():
        print(f"⚠️  Test audio not found: {test_audio}")
        return

    with open(test_audio, "rb") as f:
        response = requests.post(
            "http://localhost:8000/v1/audio/transcriptions",
            files={"file": ("audio.flac", f, "audio/flac")},
            data={
                "model": "whisper-tiny",
                "response_format": "verbose_json",
                "language": "en",
            }
        )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Text: {data['text']}")
        print(f"Language: {data.get('language', 'unknown')}")
        print(f"Segments: {len(data.get('segments', []))}")
    else:
        print(f"Error: {response.text}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 Multi-Modal Server Client Demo")
    print("="*70)
    print("\nMake sure the server is running:")
    print("  uv run python examples/multimodal_server_demo.py")
    print("="*70)

    try:
        # Test in order
        test_health()
        test_list_models()
        test_vision_detection()
        test_stt_transcription()
        test_stt_verbose()
        test_tts()

        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server!")
        print("   Make sure the server is running:")
        print("   uv run python examples/multimodal_server_demo.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
