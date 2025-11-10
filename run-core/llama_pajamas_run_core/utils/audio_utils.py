"""
Audio processing utilities for STT/TTS.

Shared utilities for audio loading, preprocessing, and mel-spectrogram computation.
"""

from pathlib import Path
from typing import Union
import numpy as np


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file (supports .wav, .flac, .mp3)
        sample_rate: Target sample rate (default: 16000 Hz for Whisper)

    Returns:
        Audio waveform as numpy array (shape: [samples])

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Try soundfile first (fast, native)
        import soundfile as sf
        audio, sr = sf.read(str(file_path))

        # Resample if needed
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    except ImportError:
        # Fallback to librosa (slower but more compatible)
        import librosa
        audio, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize to [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    return audio.astype(np.float32)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160
) -> np.ndarray:
    """
    Compute mel-spectrogram features for Whisper-style STT.

    Args:
        audio: Audio waveform (shape: [samples])
        sample_rate: Audio sample rate (default: 16000 Hz)
        n_mels: Number of mel filterbanks (default: 80 for Whisper)
        n_fft: FFT window size (default: 400 for Whisper)
        hop_length: Hop length in samples (default: 160 for Whisper, 10ms at 16kHz)

    Returns:
        Mel-spectrogram features (shape: [n_mels, n_frames])

    Note:
        These parameters are optimized for Whisper models:
        - 16kHz sample rate
        - 80 mel filterbanks
        - 25ms FFT window (400 samples)
        - 10ms hop length (160 samples)
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required. Install with: uv pip install librosa")

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1] range (Whisper expects this)
    mel_spec_normalized = (mel_spec_db + 80) / 80

    return mel_spec_normalized.astype(np.float32)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

    Args:
        reference: Ground truth transcription
        hypothesis: Model-generated transcription

    Returns:
        WER as float (0.0 = perfect, higher = worse)

    Example:
        >>> calculate_wer("the quick brown fox", "the quick brown box")
        0.25  # 1 substitution out of 4 words
    """
    # Normalize text
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    # Dynamic programming for edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + 1  # Substitution
                )

    # WER = edit distance / reference length
    if n == 0:
        return 0.0 if m == 0 else float('inf')

    return dp[n][m] / n


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent WER calculation.

    - Convert to uppercase
    - Remove punctuation
    - Remove extra whitespace

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    import re

    # Convert to uppercase
    text = text.upper()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or trim audio to target length.

    Args:
        audio: Input audio (shape: [samples])
        target_length: Target length in samples

    Returns:
        Padded or trimmed audio (shape: [target_length])
    """
    current_length = len(audio)

    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        return np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Trim
        return audio[:target_length]
    else:
        return audio


def calculate_rtf(processing_time_sec: float, audio_duration_sec: float) -> float:
    """
    Calculate Real-Time Factor (RTF).

    RTF = Processing Time / Audio Duration
    - RTF < 1.0: Faster than real-time
    - RTF = 1.0: Real-time
    - RTF > 1.0: Slower than real-time

    Args:
        processing_time_sec: Time taken to process audio
        audio_duration_sec: Duration of input audio

    Returns:
        RTF as float
    """
    if audio_duration_sec == 0:
        return float('inf')
    return processing_time_sec / audio_duration_sec
