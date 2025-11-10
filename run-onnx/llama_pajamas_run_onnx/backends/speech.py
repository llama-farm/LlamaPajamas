"""ONNX Runtime backend for speech models.

Supports:
- Speech-to-text (Whisper)
- Speech embeddings (Wav2Vec)
- TTS (future)

Optimized for CPU, AMD GPU, ARM processors, NVIDIA Jetson.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXSpeechBackend:
    """ONNX Runtime backend for speech models."""

    def __init__(self):
        self.encoder_session: Optional[ort.InferenceSession] = None
        self.decoder_session: Optional[ort.InferenceSession] = None
        self.model_name: Optional[str] = None

    def load_model(
        self,
        encoder_path: str,
        decoder_path: Optional[str] = None,
        model_name: str = "whisper-tiny",
        providers: Optional[List[str]] = None,
        **kwargs,
    ):
        """Load ONNX speech model.

        Args:
            encoder_path: Path to encoder .onnx file
            decoder_path: Path to decoder .onnx file (if separate)
            model_name: Model identifier
            providers: Execution providers (default: ["CPUExecutionProvider"])
            **kwargs: Additional session options
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.info(f"Loading ONNX speech model: {model_name}")
        logger.info(f"Using execution providers: {providers}")

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Apply custom options
        if "num_threads" in kwargs:
            sess_options.intra_op_num_threads = kwargs["num_threads"]

        # Load encoder
        self.encoder_session = ort.InferenceSession(
            str(encoder_path), sess_options=sess_options, providers=providers
        )

        logger.info(f"Encoder loaded: {encoder_path}")

        # Load decoder if provided
        if decoder_path:
            self.decoder_session = ort.InferenceSession(
                str(decoder_path), sess_options=sess_options, providers=providers
            )
            logger.info(f"Decoder loaded: {decoder_path}")

        self.model_name = model_name

        # Get available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str = "en"
    ) -> Dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform (float32, 1D array)
            sample_rate: Sample rate (must be 16000 for Whisper)
            language: Language code (e.g., "en", "es", "fr")

        Returns:
            {"text": str, "language": str, "segments": List[...]}
        """
        if self.encoder_session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Resample if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # Preprocess audio
        mel_spectrogram = self._compute_mel_spectrogram(audio)

        # Run encoder
        encoder_input_name = self.encoder_session.get_inputs()[0].name
        encoder_outputs = self.encoder_session.run(
            None, {encoder_input_name: mel_spectrogram}
        )

        # Run decoder (if available)
        if self.decoder_session:
            # Decode audio features to text
            text = self._decode(encoder_outputs[0])
        else:
            # Use external decoder (OpenAI Whisper)
            text = self._decode_external(encoder_outputs[0])

        return {
            "text": text,
            "language": language,
            "segments": [],  # TODO: Add segment-level transcription
        }

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram for Whisper.

        Args:
            audio: Audio waveform (float32, 1D)

        Returns:
            Mel spectrogram (float32, [1, 80, 3000])
        """
        import librosa

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            fmin=0,
            fmax=8000,
        )

        # Convert to log scale
        log_mel = np.log10(np.maximum(mel, 1e-10))

        # Normalize
        log_mel = (log_mel + 4.0) / 4.0

        # Pad or truncate to 3000 frames (30 seconds)
        if log_mel.shape[1] < 3000:
            log_mel = np.pad(
                log_mel, ((0, 0), (0, 3000 - log_mel.shape[1])), mode="constant"
            )
        else:
            log_mel = log_mel[:, :3000]

        # Add batch dimension
        log_mel = np.expand_dims(log_mel, axis=0)

        return log_mel.astype(np.float32)

    def _decode(self, audio_features: np.ndarray) -> str:
        """Decode audio features using ONNX decoder.

        Args:
            audio_features: Encoder outputs

        Returns:
            Transcribed text
        """
        # TODO: Implement ONNX-based decoder
        # For now, raise error
        raise NotImplementedError(
            "ONNX decoder not implemented. Use external decoder or load decoder model."
        )

    def _decode_external(self, audio_features: np.ndarray) -> str:
        """Decode using external Whisper decoder.

        Args:
            audio_features: Encoder outputs

        Returns:
            Transcribed text
        """
        import whisper

        # Load Whisper model (decoder only)
        model_size = self.model_name.split("-")[-1] if self.model_name else "tiny"
        model = whisper.load_model(model_size)

        # Create decoding options
        options = whisper.DecodingOptions(language="en", without_timestamps=True)

        # Decode
        result = whisper.decode(model, audio_features, options)

        if isinstance(result, list):
            result = result[0]

        return result.text

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio."""
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def unload_model(self):
        """Unload model from memory."""
        self.encoder_session = None
        self.decoder_session = None
        logger.info("Model unloaded")
