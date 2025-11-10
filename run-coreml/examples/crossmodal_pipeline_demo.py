#!/usr/bin/env python3
"""
Cross-Modal Pipeline Demo: Audio → STT → LLM → TTS → Audio

Demonstrates complete multi-modal workflow:
1. Load audio file
2. Transcribe with Whisper STT (CoreML + ANE)
3. Process with LLM (MLX on Apple Silicon)
4. Synthesize response with TTS (System TTS)
5. Save audio output

This shows the full "voice assistant" pipeline on-device.

Usage:
    uv run python examples/crossmodal_pipeline_demo.py
"""

import sys
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run-core"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "run"))

from llama_pajamas_run_coreml.backends import CoreMLSTTBackend, SystemTTSBackend
from llama_pajamas_run.backends import MLXBackend
from llama_pajamas_run.config import RuntimeConfig
from llama_pajamas_run_core.utils.audio_utils import load_audio
import numpy as np


def demo_crossmodal_pipeline():
    """Run complete cross-modal pipeline demo."""

    print("\n" + "="*70)
    print("🎙️  Cross-Modal Pipeline Demo")
    print("="*70)
    print("\nPipeline: Audio → STT → LLM → TTS → Audio")
    print("          (Voice Assistant Simulation)")
    print("="*70 + "\n")

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    stt_model = base_dir / "quant/models/whisper-tiny/coreml/int8/encoder.mlpackage"
    audio_file = base_dir / "quant/evaluation/stt/audio/sample_001.flac"

    # Check if models exist
    if not stt_model.exists():
        print(f"⚠️  STT model not found: {stt_model}")
        print("   Run: cd quant && uv run python scripts/quantize_whisper_coreml.py --model whisper-tiny")
        stt_model = base_dir / "quant/models/whisper-tiny/coreml/float16/encoder.mlpackage"
        if not stt_model.exists():
            print("❌ No STT model found! Exiting.")
            return

    if not audio_file.exists():
        print(f"⚠️  Audio file not found: {audio_file}")
        print("   Run: cd quant/evaluation/stt && uv run python download_audio.py")
        return

    # ========================================================================
    # Step 1: Speech-to-Text (STT)
    # ========================================================================
    print("Step 1: Speech-to-Text (Whisper INT8 + ANE)")
    print("-" * 70)

    stt_backend = CoreMLSTTBackend()
    stt_backend.load_model(str(stt_model), model_name="tiny")

    print(f"Loading audio: {audio_file.name}")
    audio = load_audio(str(audio_file), sample_rate=16000)
    audio_duration = len(audio) / 16000

    print(f"Transcribing... ({audio_duration:.1f}s audio)")
    import time
    start = time.time()
    transcription = stt_backend.transcribe(audio, sample_rate=16000)
    stt_time = (time.time() - start) * 1000

    print(f"✅ Transcription: \"{transcription.text}\"")
    print(f"   Latency: {stt_time:.1f}ms")
    print(f"   RTF: {(stt_time/1000)/audio_duration:.3f}")

    # ========================================================================
    # Step 2: LLM Processing (Simulated)
    # ========================================================================
    print("\nStep 2: LLM Processing (Response Generation)")
    print("-" * 70)

    # Simulate LLM processing
    # In production, this would use MLXBackend with a quantized model
    user_text = transcription.text
    print(f"User said: \"{user_text}\"")

    # Generate a simple response (simulated)
    # In production: response = llm.generate(f"User: {user_text}\nAssistant: ")
    llm_response = (
        f"I heard you say: {user_text.lower()}. "
        f"That's a recording from the LibriSpeech dataset. "
        f"This demonstrates the complete voice assistant pipeline running on-device."
    )

    print(f"✅ LLM Response: \"{llm_response}\"")
    print(f"   (Simulated - in production, use MLX backend)")

    # ========================================================================
    # Step 3: Text-to-Speech (TTS)
    # ========================================================================
    print("\nStep 3: Text-to-Speech (System TTS)")
    print("-" * 70)

    tts_backend = SystemTTSBackend()
    tts_backend.load_model(voice="Albert")

    print(f"Synthesizing response...")
    start = time.time()
    audio_output = tts_backend.synthesize(llm_response, sample_rate=22050)
    tts_time = (time.time() - start) * 1000

    print(f"✅ Synthesized {len(audio_output)/22050:.1f}s of audio")
    print(f"   Latency: {tts_time:.1f}ms")
    print(f"   Voice: Samantha (System TTS)")

    # ========================================================================
    # Step 4: Save Output
    # ========================================================================
    print("\nStep 4: Save Output Audio")
    print("-" * 70)

    output_path = Path("/tmp/crossmodal_output.wav")
    import soundfile as sf
    sf.write(output_path, audio_output, 22050)

    print(f"✅ Audio saved: {output_path}")
    print(f"   Play with: afplay {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("📊 Pipeline Summary")
    print("="*70)
    print(f"Input:  {audio_file.name} ({audio_duration:.1f}s)")
    print(f"STT:    {stt_time:.1f}ms (Whisper INT8 + ANE)")
    print(f"LLM:    Simulated (use MLX backend in production)")
    print(f"TTS:    {tts_time:.1f}ms (System TTS)")
    print(f"Output: {output_path} ({len(audio_output)/22050:.1f}s)")
    print(f"\nTotal:  {stt_time + tts_time:.1f}ms end-to-end")
    print(f"        ({(stt_time + tts_time)/1000:.2f}s latency)")

    print("\n" + "="*70)
    print("✅ Cross-Modal Pipeline Complete!")
    print("="*70)

    print("\n💡 Next Steps:")
    print("   1. Add MLX backend for true LLM processing")
    print("   2. Integrate Vision (image → description → TTS)")
    print("   3. Deploy as iOS app with whisper-tiny INT8 (7.9 MB)")
    print("   4. Add streaming for real-time voice assistant")
    print()


if __name__ == "__main__":
    demo_crossmodal_pipeline()
