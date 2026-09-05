"""
TTS Manager
Unified interface for multiple TTS engines (Kokoro, XTTS2, SpeechBrain)
"""

import os
import uuid
import torch
from pathlib import Path
from typing import List, Dict, Optional
from ..database import DB
from ..utils.audio import improve_audio_quality
from pydub import AudioSegment

# Engine-specific imports handled lazily
KOKORO_AVAILABLE = False
XTTS_AVAILABLE = False
SPEECHBRAIN_AVAILABLE = False
GTTS_AVAILABLE = False
MMS_AVAILABLE = False
CHATTERBOX_AVAILABLE = False

import threading


class TTSManager:
    """Unified manager for various TTS engines"""

    def __init__(self, config, engine: str = "auto"):
        self.config = config
        self.engine = engine
        self.model = None
        self.db = DB
        self.loaded_engine = None
        self.current_kokoro_lang = None
        self.last_status_message = "Idle. No engine loaded."
        self._lock = threading.Lock()  # Lock for _load_engine
        self.lock = threading.Lock()  # Lock for generate_speech
        self.health_status = self.check_engines()

    def _load_engine(self, target_engine: str, lang_code: str = "a"):
        """Lazy load the requested engine"""
        with self._lock:
            if self.loaded_engine == target_engine:
                if target_engine == "kokoro" and self.current_kokoro_lang != lang_code:
                    # Reload Kokoro for new language
                    pass
                else:
                    return

            if target_engine == "kokoro":
                self._load_kokoro(lang_code)
            elif target_engine == "xtts":
                self._load_xtts()
            elif target_engine == "gtts":
                self._load_gtts()
            elif target_engine == "mms":
                self._load_mms()
            elif target_engine == "chatterbox":
                self._load_chatterbox()

    def _load_kokoro(self, lang_code: str = "a"):
        global KOKORO_AVAILABLE
        try:
            from kokoro import KPipeline

            # lang_code ensures we load the correct phonemizer/vocabulary
            try:
                self.model = KPipeline(lang_code=lang_code)
                self.loaded_engine = "kokoro"
                self.current_kokoro_lang = lang_code
                KOKORO_AVAILABLE = True
                self.last_status_message = f"✅ Kokoro-82M loaded (Lang: {lang_code})"
                print(f"[TTS] {self.last_status_message}")
            except Exception as e:
                self.last_status_message = f"❌ Kokoro Load Error: {e}"
                print(f"[TTS] {self.last_status_message}")
                # Fallback or re-raise if critical
                raise
        except ImportError:
            self.last_status_message = "❌ Kokoro library not found"
            print(f"[TTS] {self.last_status_message}")

    def _load_xtts(self):
        global XTTS_AVAILABLE
        try:
            from TTS.api import TTS

            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                self.config.DEVICE
            )
            self.loaded_engine = "xtts"
            XTTS_AVAILABLE = True
            self.last_status_message = "✅ Coqui XTTSv2 loaded"
            print(f"[TTS] {self.last_status_message}")
        except Exception as e:
            self.last_status_message = f"❌ XTTS Error: {e}"
            print(f"[TTS] {self.last_status_message}")

    def _load_gtts(self):
        global GTTS_AVAILABLE
        try:
            from gtts import gTTS

            self.loaded_engine = "gtts"
            GTTS_AVAILABLE = True
            self.last_status_message = "✅ gTTS (Google) loaded"
            print(f"[TTS] {self.last_status_message}")
        except ImportError:
            self.last_status_message = "❌ gTTS library not found"
            print(f"[TTS] {self.last_status_message}")

    def _load_mms(self, repo_id: str = "facebook/mms-tts-ron"):
        global MMS_AVAILABLE
        try:
            from transformers import VitsModel, AutoTokenizer
            import torch

            print(f"[TTS] Loading MMS model: {repo_id}...")
            # Attempt to load from cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id, local_files_only=True
            )
            self.model = VitsModel.from_pretrained(repo_id, local_files_only=True)
            self.loaded_engine = "mms"
            MMS_AVAILABLE = True
            self.last_status_message = f"✅ MMS Loaded: {repo_id}"
            print(f"[TTS] {self.last_status_message}")
        except Exception as e:
            # Fallback to online if local fails (though in Docker it shouldn't)
            try:
                from transformers import VitsModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
                self.model = VitsModel.from_pretrained(repo_id)
                self.loaded_engine = "mms"
                MMS_AVAILABLE = True
                self.last_status_message = f"✅ MMS Loaded (Online): {repo_id}"
                print(f"[TTS] {self.last_status_message}")
            except Exception as e2:
                self.last_status_message = f"❌ MMS Error: {e2}"
                print(f"[TTS] {self.last_status_message}")

    def _load_chatterbox(self):
        global CHATTERBOX_AVAILABLE
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            import torchaudio

            self.last_status_message = "⏳ Loading Chatterbox..."
            print(f"[TTS] {self.last_status_message}")

            # Using Multilingual model by default for broader support
            self.model = ChatterboxMultilingualTTS.from_pretrained(
                device=self.config.DEVICE
            )
            self.loaded_engine = "chatterbox"
            CHATTERBOX_AVAILABLE = True

            self.last_status_message = "✅ Chatterbox Multilingual loaded"
            print(f"[TTS] {self.last_status_message}")
        except Exception as e:
            self.last_status_message = f"❌ Chatterbox Error: {e}"
            print(f"[TTS] {self.last_status_message}")

    def _clean_text(self, text: str) -> str:
        """Clean text of custom metadata tags and problematic punctuation for TTS."""
        import re

        # Remove [N level] or [N levels] optionally followed by (...)
        text = re.sub(r"\[\d+\s+levels?\](?:\([^)]+\))?", "", text)

        # Standardize ellipsis
        text = re.sub(r"\.{2,}", "...", text)

        # Remove characters that often cause phonemizer warnings or "words count mismatch"
        # We keep standard punctuation . , ! ? : ; ' " -
        # But we remove or replace others.
        text = text.replace("*", " ").replace("#", " ").replace("@", " ")
        text = text.replace("_", " ").replace("~", " ")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def check_engines(self) -> Dict[str, bool]:
        """Check status of all engines without loading models fully"""
        status = {
            "kokoro": False,
            "xtts": False,
            "gtts": False,
            "mms": False,
            "chatterbox": False,
        }

        # Check Kokoro
        try:
            import kokoro

            status["kokoro"] = True
        except ImportError:
            pass

        # Check XTTS
        try:
            from TTS.api import TTS

            status["xtts"] = True
        except ImportError:
            pass

        # Check gTTS
        try:
            import gtts

            status["gtts"] = True
        except ImportError:
            pass

        # Check MMS
        try:
            import transformers

            status["mms"] = True
        except ImportError:
            pass

        # Check Chatterbox
        try:
            import chatterbox

            status["chatterbox"] = True
        except ImportError:
            pass

        return status

    def generate_speech(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        engine: Optional[str] = None,
        speed: float = 1.0,
    ) -> Path:
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
        # Clean text first
        text = self._clean_text(text)

        # 1. Immediate Cache Check
        # We check the database for an existing file that matches the text, voice, language, and speed.
        cached = self.db.get_cached_tts(text, voice_id, language, speed=speed)
        if cached and Path(cached).exists():
            print(f"🔁 [TTS] Reusing cached audio for: '{text[:30]}...' ({voice_id})")
            return Path(cached)

        # Determine capabilities
        lang_config = self.config.SUPPORTED_LANGUAGES.get(language, {})
        kokoro_code = lang_config.get("kokoro_code")

        # Select engine
        engine = engine or self.engine

        # Check if voice_id is a clone (folder exists)
        is_clone = False
        try:
            if (self.config.VOICE_SAMPLES_DIR / voice_id).exists() and (
                self.config.VOICE_SAMPLES_DIR / voice_id
            ).is_dir():
                is_clone = True
        except Exception:
            pass

        if engine == "auto":
            # Heuristic:
            # 1. If it's a clone voice -> XTTS (XTTS is best for cloning)
            # 2. If it's Romanian -> MMS (XTTS doesn't support RO, Kokoro doesn't support RO)
            # 3. If valid Kokoro lang -> Kokoro (Kokoro is fastest)
            # 4. Fallback to gTTS (Most compatible but online)
            if is_clone:
                engine = "xtts"
            elif language == "ro":
                engine = "mms"
            elif kokoro_code:
                engine = "kokoro"
            else:
                engine = "gtts"

        # Force MMS if the voice is explicitly "MMS-TTS Romanian"
        if voice_id == "MMS-TTS Romanian":
            engine = "mms"
        # Only force Kokoro for these names if it's NOT a clone
        elif not is_clone and (
            voice_id
            in [
                "sexy",
                "af_heart",
                "af_bella",
                "af_nicole",
                "af_sarah",
                "af_sky",
                "am_michael",
                "am_adam",
                "am_liam",
                "am_puck",
            ]
            or voice_id.startswith("af_")
            or voice_id.startswith("am_")
            or voice_id.startswith("bf_")
            or voice_id.startswith("bm_")
        ):
            engine = "kokoro"

        with self.lock:
            output_path = self.config.TEMP_AUDIO_DIR / f"tts_{uuid.uuid4().hex}.wav"
            print(f"▶ [TTS] Starting generation for: '{text[:50]}...' using {engine}")
            last_err = None
            for attempt in range(3):
                try:
                    self._load_engine(engine, lang_code=kokoro_code or "a")
                    if self.loaded_engine == "kokoro":
                        self._generate_kokoro(text, voice_id, output_path, speed)
                    elif self.loaded_engine == "xtts":
                        self._generate_xtts(
                            text, voice_id, language, output_path, speed
                        )
                    elif self.loaded_engine == "gtts":
                        self._generate_gtts(text, language, output_path)
                    elif self.loaded_engine == "mms":
                        self._generate_mms(text, output_path)
                    elif self.loaded_engine == "chatterbox":
                        self._generate_chatterbox(text, output_path, language)
                    else:
                        raise ValueError(f"No functional engine for {engine}")
                    if not output_path.exists():
                        raise FileNotFoundError(
                            f"Engine {self.loaded_engine} claims success but {output_path} is missing"
                        )
                    break
                except Exception as e:
                    last_err = e
                    print(f"⚠️ [TTS] Attempt {attempt + 1}/3 failed with {engine}: {e}")
                    import time

                    time.sleep(0.5 * (2**attempt))
                    if attempt == 2:
                        print(f"❌ [TTS] All retries failed: {last_err}")
                        import traceback

                        traceback.print_exc()
                        try:
                            silent = AudioSegment.silent(duration=1000)
                            silent.export(str(output_path), format="wav")
                            print(
                                f"📁 [TTS] Created emergency silent fallback at: {output_path}"
                            )
                            return output_path
                        except Exception as e2:
                            print(f"💀 [TTS] Even emergency fallback failed: {e2}")
                            return None
            try:
                if not output_path.exists():
                    raise FileNotFoundError(
                        f"Engine {self.loaded_engine} claims success but {output_path} is missing"
                    )
                print(f"🪄 [TTS] Post-processing audio quality...")
                improved_path = improve_audio_quality(output_path)
                if improved_path != output_path:
                    output_path.unlink(missing_ok=True)
                    output_path = improved_path
                self.db.save_tts(text, voice_id, language, output_path, speed=speed)
                print(f"✅ [TTS] Successfully generated: {output_path}")
                return output_path
            except Exception as e:
                print(f"❌ [TTS] Post-process failed: {e}")
                if output_path.exists():
                    return output_path
                try:
                    silent = AudioSegment.silent(duration=1000)
                    silent.export(str(output_path), format="wav")
                    return output_path
                except Exception as e2:
                    print(f"💀 [TTS] Fallback failed: {e2}")
                    return None

    def _generate_kokoro(
        self, text: str, voice_id: str, output_path: Path, speed: float = 1.0
    ):
        import scipy.io.wavfile as wavfile
        import numpy as np

        # Voice mapping: translates descriptive names to Kokoro IDs
        # Standard Kokoro v1.0 voices
        VOICE_MAPPING = {
            # American English
            self.config.STANDARD_VOICE_NAME: "af_heart",
            "american-woman": "af_heart",
            "american-woman-2": "af_bella",
            "american-woman-3": "af_nicole",
            "american-woman-4": "af_sarah",
            "american-woman-5": "af_sky",
            "american-man": "am_michael",
            "american-man-2": "am_adam",
            "american-man-3": "am_liam",
            "american-man-4": "am_puck",
            # British English
            "british-woman": "bf_emma",
            "british-woman-2": "bf_isabella",
            "british-man": "bm_george",
            "british-man-2": "bm_lewis",
            # Other languages
            "spanish-woman": "es_karen",
            "french-woman": "ff_siwis",
            "italian-woman": "if_sara",
            "italian-man": "im_nicola",
            "portuguese-woman": "pf_dora",
            "portuguese-man": "pm_alex",
            "japanese-voice": "jf_choing",
            "japanese-man": "jm_kumo",
            "chinese-voice": "zf_xiaobei",
            "chinese-man": "zm_yunjian",
            "hindi-voice": "hf_alpha",
            "hindi-man": "hm_omega",
            # Legacy mappings
            "asrm": "af_bella",
            "austrian": "af_bella",
            "bianca": "af_nicole",
            "churchil": "am_michael",
            "gabriel": "am_liam",
            "iliescu": "am_michael",
            "man": "am_michael",
            "megan": "af_nicole",
            "sexy": "af_heart",
        }

        # Fallback to af_heart if voice_id not found or is None
        kk_voice = VOICE_MAPPING.get(voice_id, "af_heart")

        # Process chunks
        all_audio = []
        for gs, ps, audio in self.model(text, voice=kk_voice, speed=speed):
            if audio is not None and len(audio) > 0:
                all_audio.append(audio)

        if not all_audio:
            raise ValueError(
                f"Kokoro failed to generate any audio chunks for text: {text[:50]}..."
            )

        full_audio = np.concatenate(all_audio)
        full_audio = (full_audio * 32767).astype(np.int16)
        wavfile.write(str(output_path), 24000, full_audio)

    def _generate_xtts(
        self,
        text: str,
        voice_id: str,
        language: str,
        output_path: Path,
        speed: float = 1.0,
    ):
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
                language=language[:2],
                speed=speed,
            )
        else:
            # Fallback to standard speaker
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker="Conditioning Latent",  # Default standard
                language=language[:2],
                speed=speed,
            )

    def _generate_gtts(self, text: str, language: str, output_path: Path):
        from gtts import gTTS

        tts = gTTS(text=text, lang=language)
        tts.save(str(output_path))

    def _generate_mms(self, text: str, output_path: Path):
        import scipy.io.wavfile as wavfile
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform

        # Squeeze and convert to numpy
        audio_data = output.cpu().numpy().squeeze()
        # MMS outputs float32, usually normalized. Scale to int16.
        # But VITS might be at a specific sample rate.
        # Check model config for sampling rate.
        sampling_rate = self.model.config.sampling_rate

        wavfile.write(str(output_path), sampling_rate, audio_data)

    def _generate_chatterbox(self, text: str, output_path: Path, language: str):
        import torchaudio

        # Clean text
        text = self._clean_text(text)

        # Chatterbox expects language codes like 'en', 'fr', etc.
        # Ensure we pass the correct language ID if supported
        # Defaulting to 'en' if not specified or fallback
        lang_id = language if language else "en"

        # Generate
        # Note: The model might be multilingual or single language depending on how we refine _load_chatterbox
        # Currently assuming Multilingual
        wav = self.model.generate(text, language_id=lang_id)

        # Save
        torchaudio.save(str(output_path), wav, self.model.sr)

    def get_available_voices(self, engine: str = "kokoro") -> List[str]:
        """Get voices supported by the engine"""
        if engine == "kokoro":
            return [
                "american-woman",
                "american-woman-2",
                "american-woman-3",
                "american-woman-4",
                "american-woman-5",
                "american-man",
                "american-man-2",
                "american-man-3",
                "american-man-4",
                "british-woman",
                "british-woman-2",
                "british-man",
                "british-man-2",
                "spanish-woman",
                "french-woman",
                "italian-woman",
                "italian-man",
                "portuguese-woman",
                "portuguese-man",
                "japanese-voice",
                "japanese-man",
                "chinese-voice",
                "chinese-man",
                "hindi-voice",
                "hindi-man",
                "MMS-TTS Romanian",
                "Chatterbox Multilingual",
            ]
        return ["Standard"]
