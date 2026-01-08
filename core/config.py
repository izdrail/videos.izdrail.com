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
        
        # Create all directories
        for d in [self.VOICE_SAMPLES_DIR, self.VIDEOS_DIR, self.MUSIC_DIR,
                  self.IMAGES_DIR, self.TEMP_DIR, self.OUTPUT_DIR, self.BACKUP_OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.DEVICE = "cpu"
        
        # Video configuration
        self.VIDEO_WIDTH = 1080
        self.VIDEO_HEIGHT = 1920
        self.VIDEO_SIZE = (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        self.VIDEO_PRESET = 'ultrafast'
        self.VIDEO_CRF = 28
        
        # Text configuration
        self.TEXT_SIZE_CONFIG = {
            'font_size': 50,
            'line_spacing': 1.2,
            'max_width': 900,
            'bottom_margin': 150
        }
        
        # Processing configuration
        self.MAX_PARALLEL_SLIDES = 3
        self.MIN_IMAGE_DURATION = 10.0
        
        # Audio configuration
        self.TEMP_AUDIO_DIR = self.TEMP_DIR / "audio_cache"
        self.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        
        # Default messages
        self.INTRO_MESSAGES = {
            'en': "Welcome to our channel!",
            'zh': "欢迎来到我们的频道！",
            'es': "¡Bienvenido a nuestro canal!",
            'hi': "हमारे चैनल में आपका स्वागत है!",
            'ar': "مرحبا بكم في قناتنا!",
            'ro': "Bine ați venit pe canalul nostru!"
        }
        self.CTA_MESSAGES = {
            'en': "Like, share, and subscribe!",
            'zh': "点赞、分享和订阅！",
            'es': "¡Dale me gusta, comparte y suscríbete!",
            'hi': "लाइक करें, शेयर करें और सब्सक्राइब करें!",
            'ar': "أعجبني، شارك، واشترك!",
            'ro': "Apreciază, distribuie și abonează-te!"
        }
        self.INTRO_MESSAGE = self.INTRO_MESSAGES['en']
        self.CTA_MESSAGE = self.CTA_MESSAGES['en']
        self.OUTRO_MESSAGE = "Thanks for watching! Don't forget to like and subscribe!"
        
        # Language configuration
        self.SUPPORTED_LANGUAGES = {
            'en': {'name': 'English', 'code': 'en', 'tts_code': 'en', 'kokoro_code': 'a'},
            'zh': {'name': 'Chinese (Mandarin)', 'code': 'zh-cn', 'tts_code': 'zh-cn', 'kokoro_code': 'z'},
            'es': {'name': 'Spanish', 'code': 'es', 'tts_code': 'es', 'kokoro_code': 'e'},
            'hi': {'name': 'Hindi', 'code': 'hi', 'tts_code': 'hi', 'kokoro_code': 'h'},
            'ar': {'name': 'Arabic', 'code': 'ar', 'tts_code': 'ar', 'kokoro_code': 'a'},
            'ro': {'name': 'Romanian', 'code': 'ro', 'tts_code': 'ro', 'kokoro_code': 'a'}
        }
    
    def get_temp_audio_file(self, prefix: str = "audio") -> Path:
        """Generate a temporary audio file path"""
        return self.TEMP_AUDIO_DIR / f"{prefix}_{uuid.uuid4().hex}.wav"
    
    def get_backup_path(self, original_path: Path) -> Path:
        """Get a backup path for a file to ensure it's preserved"""
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_path.name}"
        return self.BACKUP_OUTPUT_DIR / backup_name
