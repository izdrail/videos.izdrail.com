"""
Application Configuration
Centralized configuration for video generation applications
"""

import os
import uuid
from pathlib import Path
from datetime import datetime


class Config:
    """Base configuration for video generation applications"""

    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent.parent.resolve()
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.VIDEOS_DIR = self.ROOT_DIR / "background_videos"
        self.MUSIC_DIR = self.ROOT_DIR / "background_music"
        self.IMAGES_DIR = self.ROOT_DIR / "background_images"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        self.BACKUP_OUTPUT_DIR = self.ROOT_DIR / "backup_output"
        self.BACKGROUND_VIDEOS_DIR = self.VIDEOS_DIR  # Alias for clearer usage
        self.IMAGE_GENERATION_CACHE_DIR = (
            self.ROOT_DIR / os.getenv("IMAGE_GENERATION_CACHE_DIR", "cache/images")
        )

        # Create all directories
        for d in [
            self.VOICE_SAMPLES_DIR,
            self.VIDEOS_DIR,
            self.MUSIC_DIR,
            self.IMAGES_DIR,
            self.TEMP_DIR,
            self.OUTPUT_DIR,
            self.BACKUP_OUTPUT_DIR,
            self.IMAGE_GENERATION_CACHE_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Device configuration
        self.DEVICE = "cpu"

        # Standard voice for default selection
        self.STANDARD_VOICE_NAME = "sexy"

        # Keyword selection / entity search configuration
        self.KEYWORD_DEBUG_MODE = (
            os.getenv("KEYWORD_DEBUG_MODE", "False").lower() == "true"
        )
        self.ENTITY_ENABLED = True
        self.ENTITY_WEIGHT_BOOST = 1.5
        self.KEYWORD_HISTORY_LIMIT = int(os.getenv("KEYWORD_HISTORY_LIMIT", "200"))

        # Image Generation Settings (SD-Turbo)
        self.IMAGE_GENERATION_ENABLED = (
            os.getenv("IMAGE_GENERATION_ENABLED", "true").lower() == "true"
        )
        self.IMAGE_GENERATION_MODEL = os.getenv(
            "IMAGE_GENERATION_MODEL", "stabilityai/sd-turbo"
        )
        self.IMAGE_GENERATION_DEVICE = os.getenv("IMAGE_GENERATION_DEVICE", "auto")
        self.IMAGE_GENERATION_STEPS = int(os.getenv("IMAGE_GENERATION_STEPS", "1"))
        self.IMAGE_GENERATION_GUIDANCE_SCALE = float(
            os.getenv("IMAGE_GENERATION_GUIDANCE_SCALE", "0.0")
        )
        self.IMAGE_GENERATION_WIDTH = int(os.getenv("IMAGE_GENERATION_WIDTH", "512"))
        self.IMAGE_GENERATION_HEIGHT = int(os.getenv("IMAGE_GENERATION_HEIGHT", "512"))

        # Visual Source Settings
        self.VISUAL_SOURCE = os.getenv("VISUAL_SOURCE", "stock")  # stock, ai, mixed
        self.MIXED_MODE_IMAGE_RATIO = float(
            os.getenv("MIXED_MODE_IMAGE_RATIO", "0.5")
        )

        # Additional directory paths
        self.VIDEO_OVERLAYS_DIR = self.ROOT_DIR / "video-overlays"
        self.CIRCLE_OVERLAYS_DIR = self.ROOT_DIR / "circle_overlays"
        self.SD_MODEL_DIR = self.ROOT_DIR / "models/sd-turbo"

        # Ensure additional directories exist
        for d in [self.VIDEO_OVERLAYS_DIR, self.CIRCLE_OVERLAYS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Video configuration
        self.VIDEO_WIDTH = 1080
        self.VIDEO_HEIGHT = 1920
        self.VIDEO_SIZE = (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        self.VIDEO_PRESET = "ultrafast"
        self.VIDEO_CRF = 28
        self.FPS = 30
        self.VIDEO_CODEC = "libx264"
        self.AUDIO_CODEC = "aac"
        self.MIXED_MODE_SD_RATIO = 0.2

        # Detailed component configurations
        self.CIRCLE_OVERLAY_CONFIG = {
            "diameter": 300,
            "position": "top-right",
            "border_width": 5,
            "border_color": (255, 255, 255),
            "margin": 20,
        }

        self.LOGO_CONFIG = {
            "max_width": 200,
            "max_height": 100,
            "opacity": 0.7,
            "position": "top-right",
            "margin": 20,
        }

        self.MUSIC_CONFIG = {
            "music_volume_db": -15,
            "crossfade_duration": 2000,  # ms
            "fade_in_duration": 3000,  # ms
            "fade_out_duration": 3000,  # ms
        }

        self.TRANSITION_CONFIG = {
            "duration": 1.0,
            "fade_in_duration": 0.5,
            "fade_out_duration": 0.5,
        }

        # Text configuration
        self.TEXT_SIZE_CONFIG = {
            "font_size": 50,
            "line_spacing": 1.2,
            "max_width": 900,
            "bottom_margin": 150,
        }

        self.SENTENCE_MERGE_ENABLED = False
        self.SENTENCE_MERGE_MIN_WORDS = 3
        self.SENTENCE_MERGE_MIN_CHARS = 15
        self.ENABLE_SD_FALLBACK = (
            os.getenv("ENABLE_SD_FALLBACK", "True").lower() == "true"
        )
        self.MAX_PARALLEL_SLIDES = 4
        self.MIN_IMAGE_DURATION = 10.0

        # Parallel Worker Pools
        self.WORKER_POOL_NLP = 20  # IO-bound (API requests to Ollama)
        self.WORKER_POOL_TTS = 4  # CPU/GPU-bound (XTTS is heavy)
        self.WORKER_POOL_MEDIA = 10  # IO-bound (Media downloads)
        self.WORKER_POOL_RENDERING = 4  # CPU-bound (FFmpeg/MoviePy)

        # Audio configuration
        self.TEMP_AUDIO_DIR = self.TEMP_DIR / "audio_cache"
        self.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        # Default messages
        self.INTRO_MESSAGES = {
            "en": "Welcome to our channel!",
            "zh": "欢迎来到我们的频道！",
            "es": "¡Bienvenido a nuestro canal!",
            "hi": "हमारे चैनल में आपका स्वागत है!",
            "ar": "مرحبا بكم في قناتنا!",
            "ro": "Bine ați venit pe canalul nostru!",
        }
        self.CTA_MESSAGES = {
            "en": "Like, share, and subscribe!",
            "zh": "点赞、分享和订阅！",
            "es": "¡Dale me gusta, comparte y suscríbete!",
            "hi": "लाइक करें, शेयर करें और सब्सक्राइब करें!",
            "ar": "أعجبني، شارك، واشترك!",
            "ro": "Apreciază, distribuie și abonează-te!",
        }
        self.INTRO_MESSAGE = self.INTRO_MESSAGES["en"]
        self.CTA_MESSAGE = self.CTA_MESSAGES["en"]
        self.OUTRO_MESSAGE = "Thanks for watching! Don't forget to like and subscribe!"

        # Language configuration
        self.SUPPORTED_LANGUAGES = {
            "en": {
                "name": "English",
                "code": "en",
                "tts_code": "en",
                "kokoro_code": "a",
            },
            "zh": {
                "name": "Chinese (Mandarin)",
                "code": "zh-cn",
                "tts_code": "zh-cn",
                "kokoro_code": "z",
            },
            "es": {
                "name": "Spanish",
                "code": "es",
                "tts_code": "es",
                "kokoro_code": "e",
            },
            "fr": {
                "name": "French",
                "code": "fr",
                "tts_code": "fr",
                "kokoro_code": "f",
            },
            "it": {
                "name": "Italian",
                "code": "it",
                "tts_code": "it",
                "kokoro_code": "i",
            },
            "pt": {
                "name": "Brazilian Portuguese",
                "code": "pt",
                "tts_code": "pt",
                "kokoro_code": "p",
            },
            "hi": {"name": "Hindi", "code": "hi", "tts_code": "hi", "kokoro_code": "h"},
            "ja": {
                "name": "Japanese",
                "code": "ja",
                "tts_code": "ja",
                "kokoro_code": "j",
            },
            # Arabic and Romanian are NOT supported by Kokoro v0.19/v1.0.
            "ar": {
                "name": "Arabic",
                "code": "ar",
                "tts_code": "ar",
                "kokoro_code": None,
            },  # Use gTTS/XTTS
            "ro": {
                "name": "Romanian",
                "code": "ro",
                "tts_code": "ro",
                "kokoro_code": None,
            },  # Use gTTS/MMS-TTS
            "auto": {
                "name": "✨ Auto Detect",
                "code": "auto",
                "tts_code": "en",
                "kokoro_code": "a",
            },
        }

        # Unsplash Configuration
        self.UNSPLASH_APP_ID = os.getenv("UNSPLASH_APP_ID")
        self.UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
        self.UNSPLASH_SECRET_KEY = os.getenv("UNSPLASH_SECRET_KEY")

        # AI/NLP Configuration
        self.AI_MODEL = os.getenv("AI_MODEL", "gemma4:e2b")
        self.OLLAMA_API_URL = os.getenv(
            "OLLAMA_API_URL", "https://ai.izdrail.com/api/generate"
        )

        # Ollama Resilience Configuration
        self.OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.OLLAMA_RETRY_BASE_DELAY = float(
            os.getenv("OLLAMA_RETRY_BASE_DELAY", "1.0")
        )
        self.OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
        self.OLLAMA_CACHE_MAX_SIZE = int(os.getenv("OLLAMA_CACHE_MAX_SIZE", "512"))
        self.OLLAMA_FALLBACK_KEYWORDS = os.getenv(
            "OLLAMA_FALLBACK_KEYWORDS",
            "abstract,motion,light,texture,landscape,cityscape",
        ).split(",")
        self.OLLAMA_FALLBACK_MOOD = os.getenv("OLLAMA_FALLBACK_MOOD", "Cinematic")
        self.OLLAMA_FALLBACK_SCRIPT = os.getenv(
            "OLLAMA_FALLBACK_SCRIPT",
            "This video was generated without AI assistance due to a temporary service issue.",
        )

        # YouTube Free Audio Library API
        self.YOUTUBE_AUDIO_API_URL = (
            "https://thibaultjanbeyer.github.io/YouTube-Free-Audio-Library-API/api.json"
        )

        # Aspect ratio profiles
        self.ASPECT_RATIOS = {
            "9:16 Portrait (TikTok/Shorts)": {
                "width": 1080,
                "height": 1920,
                "label": "9:16",
            },
            "16:9 Landscape (YouTube)": {
                "width": 1920,
                "height": 1080,
                "label": "16:9",
            },
            "1:1 Square (Instagram)": {"width": 1080, "height": 1080, "label": "1:1"},
            "4:5 Vertical (Instagram)": {"width": 1080, "height": 1350, "label": "4:5"},
        }

        # Quality presets
        self.QUALITY_PRESETS = {
            "Low (Fastest)": {
                "preset": "ultrafast",
                "crf": 35,
                "fps": 24,
                "label": "Low",
            },
            "Medium (Balanced)": {
                "preset": "veryfast",
                "crf": 28,
                "fps": 30,
                "label": "Medium",
            },
            "High (Quality)": {
                "preset": "medium",
                "crf": 22,
                "fps": 30,
                "label": "High",
            },
            "Ultra (Best)": {"preset": "slow", "crf": 18, "fps": 60, "label": "Ultra"},
        }

    def validate(self) -> list:
        """Validate system configuration and return list of warnings/errors."""
        import shutil

        warnings = []

        # Check FFmpeg
        if not shutil.which("ffmpeg"):
            warnings.append("FFmpeg not found in PATH — video generation will fail")
        if not shutil.which("ffprobe"):
            warnings.append("FFprobe not found in PATH — duration detection will fail")

        # Check critical directories
        for name, d in [
            ("TEMP_DIR", self.TEMP_DIR),
            ("OUTPUT_DIR", self.OUTPUT_DIR),
            ("MUSIC_DIR", self.MUSIC_DIR),
            ("VOICE_SAMPLES_DIR", self.VOICE_SAMPLES_DIR),
        ]:
            if not d.exists():
                try:
                    d.mkdir(parents=True, exist_ok=True)
                except Exception:
                    warnings.append(f"Cannot create {name}: {d}")

        # Check API keys
        if not os.getenv("PEXELS_API_KEY"):
            warnings.append("PEXELS_API_KEY not set — Pexels video search disabled")
        if not os.getenv("UNSPLASH_ACCESS_KEY"):
            warnings.append(
                "UNSPLASH_ACCESS_KEY not set — Unsplash image search disabled"
            )

        return warnings

    def get_aspect_ratio(self, name: str = "9:16 Portrait (TikTok/Shorts)") -> dict:
        """Get aspect ratio dimensions by preset name."""
        return self.ASPECT_RATIOS.get(
            name, self.ASPECT_RATIOS["9:16 Portrait (TikTok/Shorts)"]
        )

    def get_quality(self, name: str = "Medium (Balanced)") -> dict:
        """Get quality preset by name."""
        return self.QUALITY_PRESETS.get(name, self.QUALITY_PRESETS["Medium (Balanced)"])

    def get_temp_audio_file(self, prefix: str = "audio") -> Path:
        """Generate a temporary audio file path"""
        return self.TEMP_AUDIO_DIR / f"{prefix}_{uuid.uuid4().hex}.wav"

    def get_backup_path(self, original_path: Path) -> Path:
        """Get a backup path for a file to ensure it's preserved"""
        backup_name = (
            f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_path.name}"
        )
        return self.BACKUP_OUTPUT_DIR / backup_name
