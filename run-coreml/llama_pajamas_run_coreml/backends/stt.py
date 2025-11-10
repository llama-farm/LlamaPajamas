"""CoreML STT backend for Apple Silicon.

Supports:
- Whisper encoder (mel-spectrogram → audio embeddings)
- Full transcription pipeline (encoder[CoreML] + decoder[Python])

Optimized for Apple Neural Engine (ANE).
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Any, Iterator
import numpy as np

from llama_pajamas_run_core.backends.speech_base import (
    STTBackend,
    TranscriptionResult,
    TranscriptionSegment,
)

# Import shared audio utilities from core
from llama_pajamas_run_core.utils.audio_utils import (
    load_audio,
    compute_mel_spectrogram,
    pad_or_trim,
)

logger = logging.getLogger(__name__)


class CoreMLSTTBackend(STTBackend):
    """CoreML implementation of STTBackend for Whisper models.

    Uses coremltools for encoder inference on Apple Silicon (ANE-optimized),
    and openai-whisper for decoder (Python).

    Architecture:
    - Encoder (CoreML + ANE): mel-spectrogram → audio embeddings
    - Decoder (Python): audio embeddings → text tokens
    """

    def __init__(self):
        """Initialize CoreML STT backend."""
        self.encoder_model = None
        self.whisper_model = None  # Full Whisper model for decoder
        self._model_path: Optional[Path] = None
        self._model_name: Optional[str] = None  # 'tiny', 'base', 'small', etc.
        self._sample_rate: int = 16000
        self._n_mels: int = 80

        # Check CoreML availability
        try:
            import coremltools as ct  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "coremltools not installed. Install with: "
                "uv pip install coremltools"
            )

        # Check Whisper availability
        try:
            import whisper  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "openai-whisper not installed. Install with: "
                "uv pip install openai-whisper"
            )

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load CoreML Whisper encoder model.

        Args:
            model_path: Path to encoder .mlpackage file
            **kwargs:
                - model_name: str ('tiny', 'base', 'small') for loading decoder
                - sample_rate: int (default: 16000)
                - language: str for STT (e.g., 'en', 'zh')
        """
        import coremltools as ct
        import whisper

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Extract model name from path (e.g., whisper-tiny/coreml/float16/encoder.mlpackage)
        model_name = kwargs.get("model_name")
        if not model_name:
            # Try to infer from path
            parts = model_path.parts
            for part in parts:
                if part.startswith("whisper-"):
                    model_name = part.replace("whisper-", "")
                    break

        if not model_name:
            raise ValueError("Could not determine model name. Pass model_name='tiny'/'base'/'small'")

        logger.info(f"Loading CoreML encoder: {model_path}")
        logger.info(f"Model type: {model_name}")

        # Load CoreML encoder with ANE support
        self.encoder_model = ct.models.MLModel(
            str(model_path),
            compute_units=ct.ComputeUnit.ALL,  # Use ANE when possible
        )

        logger.info("✅ CoreML model loaded successfully")
        logger.info("   Compute units: ALL (ANE + GPU + CPU)")

        # Load full Whisper model for decoder
        logger.info(f"Loading Whisper decoder ({model_name})...")
        self.whisper_model = whisper.load_model(model_name)

        logger.info("✅ Whisper decoder loaded successfully")

        self._model_path = model_path
        self._model_name = model_name
        self._sample_rate = kwargs.get("sample_rate", 16000)

    def unload(self) -> None:
        """Unload model and free resources."""
        self.encoder_model = None
        self.whisper_model = None
        self._model_path = None
        self._model_name = None
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.encoder_model is not None and self.whisper_model is not None

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text using CoreML encoder + Python decoder.

        Args:
            audio: Audio samples (float32, mono, range: -1.0 to 1.0)
            sample_rate: Sample rate in Hz (default: 16000)
            language: Language code (e.g., 'en', 'zh'). None = auto-detect
            **kwargs:
                - task: 'transcribe' or 'translate' (default: 'transcribe')
                - fp16: bool (default: True)

        Returns:
            TranscriptionResult with full text and timestamped segments
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Resample if needed
        if sample_rate != self._sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self._sample_rate)

        # Pad or trim to 30 seconds (Whisper's expected input length)
        audio_30s = pad_or_trim(audio, 30 * self._sample_rate)

        # Compute mel-spectrogram
        mel_features = compute_mel_spectrogram(
            audio_30s,
            sample_rate=self._sample_rate,
            n_mels=self._n_mels
        )

        # Pad to expected length (3000 frames for 30 seconds)
        mel_features = self._pad_mel_features(mel_features, target_frames=3000)

        # Run CoreML encoder
        logger.debug(f"Running CoreML encoder on mel features: {mel_features.shape}")
        encoder_output = self._run_encoder(mel_features)

        # Run Whisper decoder (Python)
        logger.debug("Running Whisper decoder (Python)...")
        result = self._run_decoder(
            encoder_output,
            language=language,
            task=kwargs.get("task", "transcribe")
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        # Create TranscriptionResult
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                text=seg["text"],
                start_time=seg["start"],
                end_time=seg["end"],
                confidence=1.0,  # Whisper doesn't provide confidence per segment
            ))

        return TranscriptionResult(
            text=result["text"],
            segments=segments,
            language=result.get("language")
        )

    def _pad_mel_features(self, mel_features: np.ndarray, target_frames: int = 3000) -> np.ndarray:
        """Pad mel features to target number of frames."""
        n_mels, n_frames = mel_features.shape

        if n_frames < target_frames:
            # Pad with zeros
            padding = target_frames - n_frames
            mel_features = np.pad(mel_features, ((0, 0), (0, padding)), mode='constant')
        elif n_frames > target_frames:
            # Trim
            mel_features = mel_features[:, :target_frames]

        return mel_features

    def _run_encoder(self, mel_features: np.ndarray) -> np.ndarray:
        """Run CoreML encoder on mel features.

        Args:
            mel_features: Mel-spectrogram (shape: [80, 3000])

        Returns:
            Audio embeddings from encoder
        """
        # Prepare input (add batch dimension)
        mel_input = mel_features[np.newaxis, :, :]  # Shape: [1, 80, 3000]

        # Run CoreML model
        output = self.encoder_model.predict({"mel_features": mel_input})

        # Extract audio features (output name may vary)
        if "audio_features" in output:
            audio_features = output["audio_features"]
        elif "var_1345" in output:  # CoreML sometimes uses generic names
            audio_features = output["var_1345"]
        else:
            # Try first output
            audio_features = list(output.values())[0]

        return audio_features

    def _run_decoder(
        self,
        audio_features: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> dict:
        """Run Whisper decoder on audio features.

        Args:
            audio_features: Audio embeddings from encoder
            language: Language code
            task: 'transcribe' or 'translate'

        Returns:
            Transcription result dict
        """
        import whisper
        import torch

        # Convert audio_features to torch tensor
        audio_features_tensor = torch.from_numpy(audio_features).to(self.whisper_model.device)

        # Prepare decoder options
        decode_options = {
            "task": task,
            "fp16": False,  # Already using CoreML encoder
        }

        if language:
            decode_options["language"] = language

        # Create a DecodingOptions object
        options = whisper.DecodingOptions(**decode_options)

        # Decode (returns a list of results, one per batch element)
        results = whisper.decode(self.whisper_model, audio_features_tensor, options)

        # Extract first result (we only have one batch element)
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # Format result similar to transcribe output
        return {
            "text": result.text,
            "segments": [],  # Simple decode doesn't provide segments
            "language": result.language if hasattr(result, 'language') else language
        }

    def transcribe_streaming(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int = 16000,
        chunk_duration: float = 10.0,
        **kwargs,
    ) -> Iterator[str]:
        """Streaming transcription (real-time).

        Note: This is a simplified implementation. True streaming requires
        more sophisticated buffering and state management.

        Args:
            audio_stream: Iterator yielding audio chunks
            sample_rate: Sample rate in Hz
            chunk_duration: Duration of each chunk in seconds

        Yields:
            Partial transcriptions as strings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        buffer = []
        buffer_duration = 0.0
        target_samples = int(chunk_duration * sample_rate)

        for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            buffer_duration += len(audio_chunk) / sample_rate

            # Process when we have enough audio
            if buffer_duration >= chunk_duration:
                # Concatenate buffer
                audio = np.concatenate(buffer)

                # Transcribe
                result = self.transcribe(audio, sample_rate=sample_rate, **kwargs)

                yield result.text

                # Clear buffer
                buffer = []
                buffer_duration = 0.0

        # Process remaining audio
        if buffer:
            audio = np.concatenate(buffer)
            result = self.transcribe(audio, sample_rate=sample_rate, **kwargs)
            yield result.text

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
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        for audio in audio_list:
            result = self.transcribe(
                audio,
                sample_rate=sample_rate,
                language=language,
                **kwargs
            )
            results.append(result)

        return results

    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes.

        Returns Whisper's 99 supported languages.
        """
        import whisper

        return list(whisper.tokenizer.LANGUAGES.keys())
