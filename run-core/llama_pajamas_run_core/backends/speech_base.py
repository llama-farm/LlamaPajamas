"""Base backend interface for speech model inference (STT and TTS)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Optional
import numpy as np


@dataclass
class TranscriptionSegment:
    """Speech-to-text transcription segment with timestamps."""

    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str  # Full transcription
    segments: List[TranscriptionSegment]  # Timestamped segments
    language: Optional[str] = None  # Detected or specified language code

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "segments": [seg.to_dict() for seg in self.segments],
            "language": self.language,
        }


class SpeechBackend(ABC):
    """Abstract base class for speech model inference.

    Base class for both STT and TTS backends.
    """

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load speech model.

        Args:
            model_path: Path to model file (.mlpackage, .engine, .onnx, etc.)
            **kwargs: Backend-specific parameters
                - sample_rate: int (default: 16000 for STT, 22050 for TTS)
                - language: str for STT (e.g., 'en', 'zh')
                - speaker_id: int for multi-speaker TTS
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class STTBackend(SpeechBackend):
    """Abstract base class for Speech-to-Text (STT) inference.

    Supports:
    - Batch transcription (audio file → text)
    - Streaming transcription (audio chunks → text)
    - Multilingual models (Whisper, etc.)
    """

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio samples (float32, mono, range: -1.0 to 1.0)
                   Shape: (num_samples,)
            sample_rate: Sample rate in Hz (default: 16000)
            language: Language code (e.g., 'en', 'zh'). None = auto-detect
            **kwargs: Backend-specific parameters
                - task: 'transcribe' or 'translate' (for Whisper)
                - beam_size: int for beam search
                - best_of: int for sampling

        Returns:
            TranscriptionResult with full text and timestamped segments

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If audio format is invalid
        """
        pass

    @abstractmethod
    def transcribe_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int = 16000,
        chunk_duration: float = 10.0,
        **kwargs,
    ) -> Iterator[str]:
        """Streaming transcription (real-time).

        Args:
            audio_stream: Iterator yielding audio chunks (float32 mono)
            sample_rate: Sample rate in Hz
            chunk_duration: Duration of each chunk in seconds (default: 10s)
            **kwargs: Backend-specific parameters

        Yields:
            Partial transcriptions as strings

        Raises:
            RuntimeError: If model is not loaded
        """
        pass

    @abstractmethod
    def batch_transcribe(
        self,
        audio_list: List[np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> List[TranscriptionResult]:
        """Batch transcription.

        Args:
            audio_list: List of audio samples
            sample_rate: Sample rate in Hz
            language: Language code (shared across batch)

        Returns:
            List of transcription results
        """
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        pass


class TTSBackend(SpeechBackend):
    """Abstract base class for Text-to-Speech (TTS) inference.

    Supports:
    - Batch synthesis (text → audio file)
    - Streaming synthesis (text → audio chunks)
    - Multi-speaker models
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            sample_rate: Output sample rate in Hz (default: 22050)
            speaker_id: Speaker/voice ID for multi-speaker models
            **kwargs: Backend-specific parameters
                - speed: float (speaking rate, default: 1.0)
                - pitch: float (pitch shift in semitones)
                - energy: float (volume/energy level)

        Returns:
            Audio samples (float32, mono, range: -1.0 to 1.0)
            Shape: (num_samples,)

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If text is empty or too long
        """
        pass

    @abstractmethod
    def synthesize_streaming(
        self,
        text: str,
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs,
    ) -> Iterator[np.ndarray]:
        """Streaming TTS synthesis.

        Args:
            text: Input text to synthesize
            sample_rate: Output sample rate in Hz
            speaker_id: Speaker/voice ID

        Yields:
            Audio chunks (float32 mono, typically 50-100ms each)

        Raises:
            RuntimeError: If model is not loaded
        """
        pass

    @abstractmethod
    def batch_synthesize(
        self,
        texts: List[str],
        sample_rate: int = 22050,
        speaker_id: int = 0,
        **kwargs,
    ) -> List[np.ndarray]:
        """Batch TTS synthesis.

        Args:
            texts: List of input texts
            sample_rate: Output sample rate in Hz
            speaker_id: Speaker/voice ID (shared across batch)

        Returns:
            List of audio samples (one per input text)
        """
        pass

    @property
    @abstractmethod
    def num_speakers(self) -> int:
        """Get number of available speakers/voices."""
        pass

    @property
    @abstractmethod
    def speaker_names(self) -> List[str]:
        """Get list of speaker names (if available)."""
        pass
