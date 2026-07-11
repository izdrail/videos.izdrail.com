"""
Audio Processing Utilities
Functions for audio normalization, filtering, and enhancement
"""

import logging
import torch
import torchaudio
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter

logger = logging.getLogger(__name__)


def trim_silence(
    audio: AudioSegment, silence_thresh_db: int = -40, min_silence_ms: int = 300
) -> AudioSegment:
    """Remove leading/trailing silence and long internal pauses.

    Args:
        audio: Input audio segment.
        silence_thresh_db: Threshold in dBFS below which audio is considered silent.
        min_silence_ms: Minimum silence duration (ms) to trim from edges or shorten internally.

    Returns:
        Trimmed AudioSegment.
    """
    try:
        if len(audio) == 0:
            return audio

        # Strip leading silence
        start_trim = 0
        for i in range(len(audio)):
            if audio[i : i + 10].dBFS > silence_thresh_db:
                start_trim = max(0, i - min_silence_ms // 2)
                break

        # Strip trailing silence
        end_trim = len(audio)
        for i in range(len(audio) - 1, 0, -10):
            if audio[i : i + 10].dBFS > silence_thresh_db:
                end_trim = min(len(audio), i + min_silence_ms // 2)
                break

        if start_trim >= end_trim:
            return audio

        trimmed = audio[start_trim:end_trim]
        return trimmed
    except Exception as e:
        logger.debug("[Audio] Silence trimming skipped: %s", e)
        return audio


def improve_audio_quality(audio_path: Path, sample_rate: int = 24000) -> Path:
    """
    Standardize and improve audio quality

    Args:
        audio_path: Path to input audio
        sample_rate: Target sample rate (default 24kHz for XTTS/Kokoro)

    Returns:
        Path to improved audio file
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))

        # Strip leading/trailing silence
        audio = trim_silence(audio)

        # Standardize sample rate
        audio = audio.set_frame_rate(sample_rate)

        # Peak normalization
        audio = normalize(audio, headroom=0.1)

        # Subtle fades to prevent pops
        audio = audio.fade_in(20).fade_out(20)

        # High pass filter to remove low-end rumble/DC offset
        audio = audio.high_pass_filter(80)

        # Gentle compression to even out dynamics
        audio = audio.compress_dynamic_range(
            threshold=-20.0,
            ratio=3.0,
            attack=5.0,
            release=50.0,
        )

        improved_path = audio_path.parent / f"improved_{audio_path.name}"
        audio.export(str(improved_path), format="wav")
        return improved_path
    except Exception as e:
        logger.warning("[Audio] Enhancement warning for %s: %s", audio_path.name, e)
        return audio_path


def remove_metallic_artifacts(
    waveforms: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    """
    Peak normalization for raw torch waveforms
    """
    try:
        max_val = waveforms.abs().max()
        if max_val > 0.01:
            waveforms = waveforms / max_val * 0.95
        return waveforms
    except Exception as e:
        logger.warning("[Audio] Waveform normalization warning: %s", e)
        return waveforms


def mix_audio_with_music(
    voice_path: Path, music_path: Path, output_path: Path, music_volume_db: int = -20
) -> bool:
    """
    Mix voice with background music
    """
    try:
        voice = AudioSegment.from_file(str(voice_path))
        music = AudioSegment.from_file(str(music_path))

        # Loop music if it's shorter than voice
        if len(music) < len(voice):
            music = music * (len(voice) // len(music) + 1)

        music = music[: len(voice)]
        music = music + music_volume_db  # Apply volume

        mixed = voice.overlay(music)
        mixed.export(str(output_path), format="wav")
        return True
    except Exception as e:
        logger.error("[Audio] Mixing error: %s", e)
        return False
