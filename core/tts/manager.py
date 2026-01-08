"""
TTS Manager
Unified interface for multiple TTS engines (Kokoro, XTTS2, SpeechBrain)
"""
import os
import uuid
import torch
from pathlib import Path
from typing import List, Dict, Optional
from ..database import GenerationDB
from ..utils.audio import improve_audio_quality

# Engine-specific imports handled lazily
KOKORO_AVAILABLE = False
XTTS_AVAILABLE = False
SPEECHBRAIN_AVAILABLE = False

class TTSManager:
    """Unified manager for various TTS engines"""
    
    def __init__(self, config, engine: str = "auto"):
        self.config = config
        self.engine = engine
        self.model = None
        self.db = GenerationDB()
        self.loaded_engine = None
        
    def _load_engine(self, target_engine: str):
        """Lazy load the requested engine"""
        if self.loaded_engine == target_engine:
            return
            
        if target_engine == "kokoro":
            self._load_kokoro()
        elif target_engine == "xtts":
            self._load_xtts()
            
    def _load_kokoro(self):
        global KOKORO_AVAILABLE
        try:
            from kokoro import KPipeline
            self.model = KPipeline(lang_code="a") # Default to English
            self.loaded_engine = "kokoro"
            KOKORO_AVAILABLE = True
            print("[TTS] Kokoro-82M engine loaded")
        except ImportError:
            print("[TTS] Kokoro library not found")
            
    def _load_xtts(self):
        global XTTS_AVAILABLE
        try:
            from TTS.api import TTS
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.config.DEVICE)
            self.loaded_engine = "xtts"
            XTTS_AVAILABLE = True
            print("[TTS] Coqui XTTSv2 engine loaded")
        except Exception as e:
            print(f"[TTS] Coqui XTTS error: {e}")

    def generate_speech(self, text: str, voice_id: str, language: str = "en", 
                       engine: Optional[str] = None) -> Path:
        """
        Generate speech from text
        
        Args:
            text: Input text
            voice_id: Voice identifier or speaker name
            language: ISO language code
            engine: Override default engine
            
        Returns:
            Path to generated audio
        """
        # Check cache
        cached = self.db.get_cached_tts(text, voice_id, language)
        if cached:
            return cached
            
        # Select engine
        engine = engine or self.engine
        if engine == "auto":
            # Heuristic: Kokoro for fast English, XTTS for others or cloning
            if language.startswith("en") and not voice_id.endswith(".wav"):
                engine = "kokoro"
            else:
                engine = "xtts"
                
        self._load_engine(engine)
        
        output_path = self.config.TEMP_DIR / f"tts_{uuid.uuid4().hex}.wav"
        
        try:
            if self.loaded_engine == "kokoro":
                self._generate_kokoro(text, voice_id, output_path)
            elif self.loaded_engine == "xtts":
                self._generate_xtts(text, voice_id, language, output_path)
            else:
                raise ValueError(f"No functional engine for {engine}")
                
            # Post-process
            improved_path = improve_audio_quality(output_path)
            if improved_path != output_path:
                output_path.unlink(missing_ok=True)
                output_path = improved_path
                
            # Save to cache
            self.db.save_tts(text, voice_id, language, output_path)
            return output_path
            
        except Exception as e:
            print(f"[TTS] Generation failed with {engine}: {e}")
            if output_path.exists():
                output_path.unlink()
            return None

    def _generate_kokoro(self, text: str, voice_id: str, output_path: Path):
        import scipy.io.wavfile as wavfile
        import numpy as np
        # Process chunks
        all_audio = []
        for gs, ps, audio in self.model(text, voice=voice_id, speed=1.0):
            all_audio.append(audio)
        full_audio = np.concatenate(all_audio)
        full_audio = (full_audio * 32767).astype(np.int16)
        wavfile.write(str(output_path), 24000, full_audio)

    def _generate_xtts(self, text: str, voice_id: str, language: str, output_path: Path):
        # Handle cloning if voice_id is a path or name of sample
        speaker_wav = None
        if voice_id.endswith(".wav"):
            speaker_wav = voice_id
        elif (self.config.VOICE_SAMPLES_DIR / voice_id).exists():
            sample_dir = self.config.VOICE_SAMPLES_DIR / voice_id
            for f in sample_dir.glob("*.wav"):
                speaker_wav = str(f)
                break
        
        if speaker_wav:
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language=language[:2]
            )
        else:
            # Fallback to standard speaker
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker="Conditioning Latent", # Default standard
                language=language[:2]
            )

    def get_available_voices(self, engine: str = "kokoro") -> List[str]:
        """Get voices supported by the engine"""
        if engine == "kokoro":
            return ["af_heart", "af_bella", "af_nicole", "am_michael", "am_liam", "am_puck"]
        return ["Standard"]
