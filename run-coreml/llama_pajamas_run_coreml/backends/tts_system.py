"""System TTS backend using Apple AVFoundation (AVSpeechSynthesizer).

Provides text-to-speech using macOS/iOS built-in voices.
No model files needed - uses system TTS engine.

This is a practical implementation while CoreML TTS models are being developed.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Iterator
import numpy as np

from llama_pajamas_run_core.backends.speech_base import TTSBackend

logger = logging.getLogger(__name__)


class SystemTTSBackend(TTSBackend):
    """System TTS backend using Apple AVFoundation.

    Uses macOS/iOS built-in TTS voices via 'say' command.
    No model loading required.
    """

    def __init__(self):
        """Initialize System TTS backend."""
        self._sample_rate: int = 22050
        self._voice: str = "Albert"  # Default macOS voice
        self._loaded: bool = False

        # Available voices (macOS defaults)
        self._available_voices = [
            "Albert",    # Male (US)
            "Alice",     # Female (IT)
            "Bad News",  # Male (US, novelty)
            "Bahh",      # Male (US, novelty)
            "Bells",     # Female (US, novelty)
            "Boing",     # Male (US, novelty)
        ]

    def load_model(self, model_path: str = None, **kwargs) -> None:
        """'Load' system TTS (no actual model loading needed).

        Args:
            model_path: Ignored (system TTS doesn't use files)
            **kwargs:
                - voice: str (default: "Albert")
                - sample_rate: int (default: 22050)
        """
        self._voice = kwargs.get("voice", "Albert")
        self._sample_rate = kwargs.get("sample_rate", 22050)

        # Verify 'say' command is available (macOS)
        try:
            subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                check=True,
                timeout=5
            )
            self._loaded = True
            logger.info(f"✅ System TTS loaded successfully")
            logger.info(f"   Voice: {self._voice}")
            logger.info(f"   Sample rate: {self._sample_rate} Hz")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"System TTS not available. Is this macOS? Error: {e}"
            )

    def synthesize(
        self,
        text: str,
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize speech from text using system TTS.

        Args:
            text: Text to synthesize
            sample_rate: Target sample rate (will resample if needed)
            speaker_id: Voice index (0-5, maps to available voices)
            **kwargs:
                - voice: str (override voice name)
                - rate: int (words per minute, default: 200)

        Returns:
            Audio waveform as numpy array (float32, mono)
        """
        if not self.is_loaded:
            raise RuntimeError("System TTS not loaded. Call load_model() first.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Select voice
        voice = kwargs.get("voice")
        if voice is None:
            # Map speaker_id to voice
            if 0 <= speaker_id < len(self._available_voices):
                voice = self._available_voices[speaker_id]
            else:
                voice = self._voice

        rate = kwargs.get("rate", 200)  # words per minute

        # Synthesize to temporary file
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Use 'say' command to synthesize
            cmd = [
                "say",
                "-v", voice,
                "-r", str(rate),
                "-o", tmp_path,
                text
            ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=60)

            # Load audio file
            import soundfile as sf
            audio, sr = sf.read(tmp_path)

            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            # Convert to float32 and normalize
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            return audio

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def synthesize_streaming(
        self,
        text: str,
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs,
    ) -> Iterator[np.ndarray]:
        """Streaming TTS synthesis.

        Note: System TTS doesn't support streaming, so this synthesizes
        the full text and yields it in chunks.
        """
        if not self.is_loaded:
            raise RuntimeError("System TTS not loaded. Call load_model() first.")

        # Synthesize full text
        audio = self.synthesize(text, sample_rate, speaker_id, **kwargs)

        # Yield in chunks (e.g., 1 second chunks)
        chunk_size = sample_rate  # 1 second
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]

    def batch_synthesize(
        self,
        texts: List[str],
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs,
    ) -> List[np.ndarray]:
        """Batch TTS synthesis."""
        return [
            self.synthesize(text, sample_rate, speaker_id, **kwargs)
            for text in texts
        ]

    @property
    def num_speakers(self) -> int:
        """Get number of available voices."""
        return len(self._available_voices)

    @property
    def speaker_names(self) -> List[str]:
        """Get list of available voice names."""
        return self._available_voices.copy()

    def unload(self) -> None:
        """Unload TTS (no-op for system TTS)."""
        self._loaded = False
        logger.info("System TTS unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if TTS is loaded."""
        return self._loaded
