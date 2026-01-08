"""
Audio Processing Utilities
Functions for audio normalization, filtering, and enhancement
"""
import torch
import torchaudio
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter

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
        # Standardize sample rate
        audio = audio.set_frame_rate(sample_rate)
        
        # Peak normalization
        audio = normalize(audio, headroom=0.1)
        
        # Subtle fades to prevent pops
        audio = audio.fade_in(20).fade_out(20)
        
        # High pass filter to remove low-end rumble/DC offset
        audio = audio.high_pass_filter(80)
        
        improved_path = audio_path.parent / f"improved_{audio_path.name}"
        audio.export(str(improved_path), format="wav")
        return improved_path
    except Exception as e:
        print(f"[Audio] Enhancement warning for {audio_path.name}: {e}")
        return audio_path

def remove_metallic_artifacts(waveforms: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Peak normalization for raw torch waveforms
    """
    try:
        max_val = waveforms.abs().max()
        if max_val > 0.01:
            waveforms = waveforms / max_val * 0.95
        return waveforms
    except Exception as e:
        print(f"[Audio] Waveform normalization warning: {e}")
        return waveforms

def mix_audio_with_music(voice_path: Path, music_path: Path, output_path: Path, 
                        music_volume_db: int = -20) -> bool:
    """
    Mix voice with background music
    """
    try:
        voice = AudioSegment.from_file(str(voice_path))
        music = AudioSegment.from_file(str(music_path))
        
        # Loop music if it's shorter than voice
        if len(music) < len(voice):
            music = music * (len(voice) // len(music) + 1)
        
        music = music[:len(voice)]
        music = music + music_volume_db # Apply volume
        
        mixed = voice.overlay(music)
        mixed.export(str(output_path), format="wav")
        return True
    except Exception as e:
        print(f"[Audio] Mixing error: {e}")
        return False
