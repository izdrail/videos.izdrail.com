#!/usr/bin/env python3
import os
import re
import glob
import random
import shutil
import uuid
import platform
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchaudio
import gradio as gr
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
import textwrap

try:
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    from TTS.api import TTS
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"TTS libraries not found: {e}")
    MODELS_AVAILABLE = False

class Config:
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.IMAGES_DIR = self.ROOT_DIR / "background_images"
        self.MUSIC_DIR = self.ROOT_DIR / "background_music"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        for dir_path in [self.VOICE_SAMPLES_DIR, self.IMAGES_DIR, self.MUSIC_DIR,
                         self.TEMP_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(exist_ok=True)
        self.STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["COQUI_TOS_AGREED"] = "1"

        # CTA MESSAGE
        self.CTA_MESSAGE = "Like, share, and subscribe to our channel!"

        # PORTRAIT VIDEO SETTINGS (9:16 for mobile)
        self.VIDEO_WIDTH = 1080
        self.VIDEO_HEIGHT = 1920
        self.VIDEO_SIZE = (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)

        self.TEXT_SIZE_CONFIG = {
            'target_coverage': 0.4,
            'initial_font_size': 120,
            'min_font_size': 50,
            'max_font_size': 180,
            'wrap_width_range': (12, 25),
            'margin_percentage': 0.08,
        }
        self.MUSIC_CONFIG = {
            'voice_volume_db': 0,
            'music_volume_db': -15,
            'fade_in_duration': 2000,
            'fade_out_duration': 2000,
            'crossfade_duration': 500,
        }
        self.AUDIO_QUALITY_CONFIG = {
            'sample_rate': 22050,
            'low_pass_cutoff': 6000,
            'normalize_audio': True,
            'remove_silence_threshold': -40,
            'apply_compression': True,
            'high_pass_cutoff': 80,
            'apply_warmth': True,
            'reduce_sibilance': True,
        }

        # PERFORMANCE SETTINGS
        self.MAX_PARALLEL_SLIDES = 4
        self.ENABLE_CACHING = True
        self.IMAGE_CACHE_SIZE = 10

        # VIBRANT COLOR PALETTE for text
        self.TEXT_COLORS = [
            (255, 255, 255),   # White (classic)
            (255, 215, 0),     # Gold
            (255, 105, 180),   # Hot Pink
            (0, 255, 255),     # Cyan
            (255, 165, 0),     # Orange
            (50, 205, 50),     # Lime Green
            (255, 20, 147),    # Deep Pink
            (255, 255, 0),     # Yellow
            (138, 43, 226),    # Blue Violet
            (255, 69, 0),      # Red Orange
            (0, 255, 127),     # Spring Green
            (255, 192, 203),   # Pink
            (173, 216, 230),   # Light Blue
            (255, 218, 185),   # Peach
            (144, 238, 144),   # Light Green
        ]

class UnsplashAPI:
    def __init__(self):
        self.base_url = "https://api.unsplash.com"
        self.client_id = None
        self.cache = {}
        self.cache_limit = 50
    def set_client_id(self, client_id: str):
        self.client_id = client_id
    def _manage_cache(self):
        if len(self.cache) > self.cache_limit:
            items_to_remove = len(self.cache) - self.cache_limit
            for key in list(self.cache.keys())[:items_to_remove]:
                del self.cache[key]
    def search_photos(self, query: str, per_page: int = 10) -> List[Dict]:
        if not self.client_id:
            raise ValueError("Unsplash client ID not set")
        url = f"{self.base_url}/search/photos"
        params = {
            "query": query,
            "per_page": min(per_page, 30),
            "client_id": self.client_id,
            "orientation": "portrait"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"[Unsplash] Error fetching images: {e}")
            return []
    def download_image(self, photo_url: str) -> Optional[Image.Image]:
        if photo_url in self.cache:
            return self.cache[photo_url].copy()
        try:
            response = requests.get(photo_url, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            self._manage_cache()
            self.cache[photo_url] = img.copy()
            return img
        except Exception as e:
            print(f"[Unsplash] Error downloading image: {e}")
            return None
    def get_random_image(self, query: str, size: Tuple[int, int] = (1080, 1920)) -> Optional[Image.Image]:
        photos = self.search_photos(query, per_page=10)
        if not photos:
            print(f"[Unsplash] No images found for query: '{query}'")
            return None
        photo = random.choice(photos)
        photo_url = photo.get("urls", {}).get("regular")
        if not photo_url:
            return None
        img = self.download_image(photo_url)
        if img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(size, Image.Resampling.LANCZOS)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.6)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            download_url = photo.get("links", {}).get("download_location")
            if download_url and self.client_id:
                try:
                    requests.get(download_url, params={"client_id": self.client_id}, timeout=5)
                except:
                    pass
        return img
    def clear_cache(self):
        self.cache.clear()

class TTSManager:
    def __init__(self, config: Config):
        self.config = config
        self.voice_model = None
        self.standard_models: Dict[str, any] = {}
        self._load_models()
    def _load_models(self):
        if not MODELS_AVAILABLE:
            print("[TTS] Required libraries missing.")
            return
        print(f"[TTS] Loading models on device: {self.config.DEVICE}")
        try:
            print("[TTS] Loading Coqui XTTS for voice cloning...")
            self.voice_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.config.DEVICE)
            print("[TTS] Coqui XTTS loaded successfully.")
        except Exception as e:
            print(f"[TTS] ERROR loading Coqui: {e}")
            self.voice_model = None
        try:
            print("[TTS] Loading SpeechBrain models...")
            tmp_tts = self.config.ROOT_DIR / "tmpdir_tts"
            tmp_vocoder = self.config.ROOT_DIR / "tmpdir_vocoder"
            self.standard_models['tacotron2'] = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech", savedir=tmp_tts
            )
            self.standard_models['hifi_gan'] = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech", savedir=tmp_vocoder
            )
            print("[TTS] SpeechBrain models loaded successfully.")
        except Exception as e:
            print(f"[TTS] ERROR loading SpeechBrain: {e}")
    @staticmethod
    def preprocess_text(text: str) -> str:
        return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    def improve_audio_quality(self, audio_path: Path) -> Path:
        try:
            print(f"[TTS] Improving audio quality for smoother, warmer sound...")
            audio = AudioSegment.from_file(str(audio_path))
            cfg = self.config.AUDIO_QUALITY_CONFIG

            if audio.frame_rate != cfg['sample_rate']:
                audio = audio.set_frame_rate(cfg['sample_rate'])

            print(f"[TTS]   - Removing low-frequency rumble ({cfg['high_pass_cutoff']}Hz)")
            audio = audio.high_pass_filter(cfg['high_pass_cutoff'])

            print(f"[TTS]   - Removing metallic frequencies (low-pass at {cfg['low_pass_cutoff']}Hz)")
            audio = low_pass_filter(audio, cfg['low_pass_cutoff'])

            if cfg['reduce_sibilance']:
                print(f"[TTS]   - Reducing sibilance (harsh 's' sounds)")
                audio = audio.low_pass_filter(7000)

            if cfg['apply_warmth']:
                print(f"[TTS]   - Adding warmth to voice")
                warm_audio = audio.low_pass_filter(500) + 2
                audio = audio.overlay(warm_audio - 15)

            if cfg['apply_compression']:
                print(f"[TTS]   - Applying gentle compression for smooth dynamics")
                audio = audio.compress_dynamic_range(
                    threshold=-25.0,
                    ratio=2.5,
                    attack=10.0,
                    release=100.0
                )

            if cfg['normalize_audio']:
                print(f"[TTS]   - Normalizing audio levels")
                audio = normalize(audio, headroom=0.1)

            audio = audio.strip_silence(
                silence_len=150,
                silence_thresh=cfg['remove_silence_threshold'],
                padding=150
            )

            audio = audio.fade_in(50).fade_out(50)

            improved_path = self.config.TEMP_DIR / f"improved_{audio_path.name}"
            audio.export(str(improved_path), format="wav", parameters=["-q:a", "0"])
            print(f"[TTS] ✓ Audio quality improved - smoother, warmer, less metallic")
            return improved_path
        except Exception as e:
            print(f"[TTS] Warning: Could not improve audio quality: {e}")
            return audio_path
    def generate_speech(self, text: str, speaker_id: str) -> Path:
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        processed_text = self.preprocess_text(text)
        temp_wav_path = self.config.TEMP_DIR / f"tts_{uuid.uuid4()}.wav"
        if speaker_id == self.config.STANDARD_VOICE_NAME:
            if not self.standard_models:
                raise ValueError("Standard TTS models unavailable.")
            tacotron2 = self.standard_models['tacotron2']
            hifi_gan = self.standard_models['hifi_gan']
            mel_outputs, _, _ = tacotron2.encode_text(processed_text)
            waveforms = hifi_gan.decode_batch(mel_outputs)
            torchaudio.save(str(temp_wav_path), waveforms.squeeze(1), 22050)
        else:
            if not self.voice_model:
                raise ValueError("Voice cloning model unavailable.")
            reference_audio = self.config.VOICE_SAMPLES_DIR / speaker_id / "reference.wav"
            if not reference_audio.exists():
                raise ValueError(f"Reference audio not found for '{speaker_id}'.")
            self.voice_model.tts_to_file(
                text=processed_text,
                file_path=str(temp_wav_path),
                speaker_wav=str(reference_audio),
                language="en",
                split_sentences=False,
            )
        if not temp_wav_path.exists():
            raise ValueError("TTS generation failed.")
        improved_path = self.improve_audio_quality(temp_wav_path)
        if improved_path != temp_wav_path:
            try:
                temp_wav_path.unlink(missing_ok=True)
            except:
                pass
        return improved_path

class VideoGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.available_fonts = self._discover_fonts()
        self.unsplash = UnsplashAPI()
        self.background_cache = []
        if not self.available_fonts:
            print("[Video] WARNING: No system fonts found. Using default.")
    def _discover_fonts(self) -> List[Path]:
        font_paths = []
        system = platform.system()
        if system == "Windows":
            font_paths.append(Path("C:/Windows/Fonts"))
        elif system == "Darwin":
            font_paths.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts")])
        elif system == "Linux":
            font_paths.extend([Path("/usr/share/fonts/truetype"), Path.home() / ".fonts"])
        discovered = []
        common_fonts = [
            "arialbd.ttf", "Arial Bold.ttf",
            "calibrib.ttf", "Calibri Bold.ttf",
            "arial.ttf", "Arial.ttf",
            "calibri.ttf", "Calibri.ttf",
            "times.ttf", "Times.ttf",
            "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
            "LiberationSans-Bold.ttf", "LiberationSans-Regular.ttf"
        ]
        for path in font_paths:
            if path.is_dir():
                for font_name in common_fonts:
                    font_file = None
                    if (path / font_name).exists():
                        font_file = path / font_name
                    else:
                        for found in path.rglob(font_name):
                            font_file = found
                            break
                    if font_file and font_file not in discovered:
                        discovered.append(font_file)
        return discovered

    def preload_backgrounds(self, count: int, unsplash_keyword: Optional[str] = None):
        """Preload backgrounds for better performance"""
        if not self.config.ENABLE_CACHING:
            return

        print(f"[Video] Preloading {count} backgrounds...")
        for _ in range(min(count, self.config.IMAGE_CACHE_SIZE)):
            bg = self.get_background_image(self.config.VIDEO_SIZE, unsplash_keyword)
            if bg:
                self.background_cache.append(bg.copy())

    def get_background_image(self, size: Tuple[int, int], unsplash_keyword: Optional[str] = None) -> Optional[Image.Image]:
        if self.background_cache and self.config.ENABLE_CACHING:
            return self.background_cache.pop(0).copy()

        if unsplash_keyword and self.unsplash.client_id:
            print(f"[Video] Fetching portrait image from Unsplash: '{unsplash_keyword}'")
            img = self.unsplash.get_random_image(unsplash_keyword, size)
            if img:
                return img
            print("[Video] Falling back to local images...")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        if self.config.IMAGES_DIR.exists():
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(self.config.IMAGES_DIR, ext)))
                image_files.extend(glob.glob(os.path.join(self.config.IMAGES_DIR, ext.upper())))
        if image_files:
            selected_image = random.choice(image_files)
            try:
                img = Image.open(selected_image)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(size, Image.Resampling.LANCZOS)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.6)
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
                return img
            except Exception as e:
                print(f"[Video] Error loading background: {e}")
        return None

    def get_random_text_color(self) -> Tuple[int, int, int]:
        """Get a random vibrant color for text"""
        return random.choice(self.config.TEXT_COLORS)

    def get_random_background_music(self) -> Optional[Path]:
        music_extensions = ['*.mp3', '*.MP3', '*.wav', '*.WAV']
        music_files = []
        if self.config.MUSIC_DIR.exists():
            for ext in music_extensions:
                music_files.extend(glob.glob(os.path.join(self.config.MUSIC_DIR, ext)))
        if music_files:
            selected = random.choice(music_files)
            print(f"[Music] Selected: {os.path.basename(selected)}")
            return Path(selected)
        print("[Music] No background music found in background_music/ folder")
        return None

    def mix_audio_with_music(self, voice_audio_path: Path, total_duration_ms: int,
                            music_path: Optional[Path] = None) -> Path:
        if music_path is None:
            music_path = self.get_random_background_music()
        if music_path is None or not music_path.exists():
            print("[Music] No background music - using voice only")
            return voice_audio_path
        try:
            print(f"[Music] Mixing audio with background music...")
            voice = AudioSegment.from_file(str(voice_audio_path))
            music = AudioSegment.from_file(str(music_path))
            cfg = self.config.MUSIC_CONFIG
            voice = voice + cfg['voice_volume_db']
            music = music + cfg['music_volume_db']
            print(f"[Music] Voice duration: {len(voice)/1000:.1f}s, Music duration: {len(music)/1000:.1f}s")
            if len(music) < len(voice):
                loops_needed = (len(voice) // len(music)) + 2
                print(f"[Music] Looping music {loops_needed} times with crossfade")
                looped_music = music
                for _ in range(loops_needed - 1):
                    looped_music = looped_music.append(music, crossfade=cfg['crossfade_duration'])
                music = looped_music
            music = music[:len(voice)]
            music = music.fade_in(cfg['fade_in_duration']).fade_out(cfg['fade_out_duration'])
            mixed = voice.overlay(music)
            output_path = self.config.TEMP_DIR / f"final_mixed_{uuid.uuid4()}.wav"
            mixed.export(str(output_path), format="wav", parameters=["-q:a", "0"])
            print(f"[Music] ✓ Successfully mixed {len(mixed)/1000:.1f}s of audio with background music")
            return output_path
        except Exception as e:
            print(f"[Music] ✗ Error mixing audio: {e}")
            return voice_audio_path

    def create_cta_slide(self, audio_path: Path, bg_color: Tuple[int, int, int] = (74, 144, 226),
                         unsplash_keyword: Optional[str] = None) -> Tuple[ImageClip, Path]:
        """Create a Call-To-Action slide with Like, Share, Subscribe message"""
        print("\n🎬 Creating Call-To-Action slide with audio...")

        # Get audio duration
        audio_segment = AudioSegment.from_file(str(audio_path))
        duration_sec = len(audio_segment) / 1000.0

        size = self.config.VIDEO_SIZE

        # Get background
        background = self.get_background_image(size, unsplash_keyword)
        img = background.copy() if background else Image.new('RGB', size, bg_color)

        # Create semi-transparent overlay
        overlay = Image.new('RGBA', size, (0, 0, 0, 180))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Font setup
        font_path = self.available_fonts[0] if self.available_fonts else None

        # Main CTA text - large and bold
        main_text = "LIKE\n SHARE\n SUBSCRIBE"
        main_font_size = 140
        main_font = ImageFont.truetype(str(font_path), main_font_size) if font_path else ImageFont.load_default()

        # Secondary text
        secondary_text = "TO OUR CHANNEL"
        secondary_font_size = 80
        secondary_font = ImageFont.truetype(str(font_path), secondary_font_size) if font_path else ImageFont.load_default()

        # Get text dimensions
        main_bbox = draw.textbbox((0, 0), main_text, font=main_font)
        main_w, main_h = main_bbox[2] - main_bbox[0], main_bbox[3] - main_bbox[1]

        secondary_bbox = draw.textbbox((0, 0), secondary_text, font=secondary_font)
        secondary_w, secondary_h = secondary_bbox[2] - secondary_bbox[0], secondary_bbox[3] - secondary_bbox[1]

        # Position main text (center-upper)
        main_y = (size[1] - main_h - secondary_h - 80) // 2
        main_x = (size[0] - main_w) // 2

        # Position secondary text (below main)
        secondary_y = main_y + main_h + 60
        secondary_x = (size[0] - secondary_w) // 2

        # Draw glowing effect (multiple shadows)
        glow_color = (255, 215, 0)  # Gold glow
        for offset in range(8, 0, -2):
            alpha = int(100 - (offset * 10))
            glow_overlay = Image.new('RGBA', size, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_overlay)

            for dx in [-offset, 0, offset]:
                for dy in [-offset, 0, offset]:
                    if dx == 0 and dy == 0:
                        continue
                    glow_draw.text(
                        (main_x + dx, main_y + dy),
                        main_text,
                        font=main_font,
                        fill=glow_color + (alpha,),
                        align="center"
                    )

            img = Image.alpha_composite(img.convert('RGBA'), glow_overlay).convert('RGB')
            draw = ImageDraw.Draw(img)

        # Draw main text with shadow
        shadow_offset = 4
        for offset in [(-shadow_offset, -shadow_offset), (shadow_offset, -shadow_offset),
                       (-shadow_offset, shadow_offset), (shadow_offset, shadow_offset)]:
            draw.text(
                (main_x + offset[0], main_y + offset[1]),
                main_text,
                font=main_font,
                fill=(0, 0, 0),
                align="center"
            )

        # Draw main text in bright yellow
        draw.text((main_x, main_y), main_text, font=main_font, fill=(255, 255, 0), align="center")

        # Draw secondary text with shadow
        for offset in [(-shadow_offset, -shadow_offset), (shadow_offset, -shadow_offset),
                       (-shadow_offset, shadow_offset), (shadow_offset, shadow_offset)]:
            draw.text(
                (secondary_x + offset[0], secondary_y + offset[1]),
                secondary_text,
                font=secondary_font,
                fill=(0, 0, 0),
                align="center"
            )

        # Draw secondary text in white
        draw.text((secondary_x, secondary_y), secondary_text, font=secondary_font, fill=(255, 255, 255), align="center")

        # Add decorative elements (stars/sparkles)
        star_positions = [
            (size[0] // 4, size[1] // 3),
            (size[0] * 3 // 4, size[1] // 3),
            (size[0] // 4, size[1] * 2 // 3),
            (size[0] * 3 // 4, size[1] * 2 // 3),
        ]

        for x, y in star_positions:
            star_size = 40
            star_font = ImageFont.truetype(str(font_path), star_size) if font_path else ImageFont.load_default()
            draw.text((x, y), "⭐", font=star_font, fill=(255, 215, 0), align="center")

        # Save image
        image_path = self.config.TEMP_DIR / f"cta_slide_{uuid.uuid4()}.png"
        img.save(image_path, quality=95)

        # Create video clip with audio
        audio_clip = AudioFileClip(str(audio_path))
        video_clip = ImageClip(str(image_path), duration=duration_sec).set_audio(audio_clip)

        print(f"✓ CTA slide created: {duration_sec:.2f}s duration with audio")
        return (video_clip, image_path)

    def _create_text_image(self, text: str, size: Tuple[int, int] = None,
                           bg_color: Tuple[int, int, int] = (74, 144, 226),
                           unsplash_keyword: Optional[str] = None,
                           text_color: Optional[Tuple[int, int, int]] = None) -> Path:
        if size is None:
            size = self.config.VIDEO_SIZE

        if text_color is None:
            text_color = self.get_random_text_color()

        background = self.get_background_image(size, unsplash_keyword)
        img = background.copy() if background else Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(img)
        cfg = self.config.TEXT_SIZE_CONFIG
        margin = int(size[0] * cfg['margin_percentage'])
        text_width = size[0] - (margin * 2)
        target_text_area = (size[0] * size[1]) * cfg['target_coverage']
        font_path = self.available_fonts[0] if self.available_fonts else None
        font_size = cfg['initial_font_size']
        min_font_size = cfg['min_font_size']
        max_font_size = cfg['max_font_size']
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
        wrap_width = 15
        wrapped_text = textwrap.fill(text, width=wrap_width)
        best_font_size = min_font_size
        best_wrapped = wrapped_text
        best_area = 0

        for test_font_size in range(max_font_size, min_font_size - 1, -10):
            test_font = ImageFont.truetype(str(font_path), test_font_size) if font_path else ImageFont.load_default()
            for w in range(cfg['wrap_width_range'][0], cfg['wrap_width_range'][1], 2):
                test_wrapped = textwrap.fill(text, width=w)
                bbox = draw.textbbox((0, 0), test_wrapped, font=test_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w <= text_width and text_h <= size[1] * 0.85:
                    text_area = text_w * text_h
                    if text_area >= target_text_area:
                        best_font_size = test_font_size
                        best_wrapped = test_wrapped
                        best_area = text_area
                        break
                    elif text_area > best_area:
                        best_font_size = test_font_size
                        best_wrapped = test_wrapped
                        best_area = text_area
            if best_area >= target_text_area:
                break

        font_size = best_font_size
        wrapped_text = best_wrapped
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2)
        coverage_percentage = (text_w * text_h / (size[0] * size[1]) * 100)

        color_name = f"RGB{text_color}"
        print(f"[Video] Font: {font_size}px, Color: {color_name}, Coverage: {coverage_percentage:.1f}%")

        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 30
        rect_coords = [
            text_pos[0] - padding, text_pos[1] - padding,
            text_pos[0] + text_w + padding, text_pos[1] + text_h + padding
        ]
        overlay_draw.rounded_rectangle(rect_coords, radius=20, fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        shadow_offset = 3
        for offset in [(-shadow_offset, -shadow_offset), (shadow_offset, -shadow_offset),
                       (-shadow_offset, shadow_offset), (shadow_offset, shadow_offset)]:
            draw.text((text_pos[0] + offset[0], text_pos[1] + offset[1]),
                      wrapped_text, font=font, fill="black", align="center")

        draw.text(text_pos, wrapped_text, font=font, fill=text_color, align="center")

        image_path = self.config.TEMP_DIR / f"slide_{uuid.uuid4()}.png"
        img.save(image_path, quality=95)
        return image_path

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r'\bDr\.', 'Dr<dot>', text)
        text = re.sub(r'\bMr\.', 'Mr<dot>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<dot>', text)
        text = re.sub(r'\bMs\.', 'Ms<dot>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<dot>', '.').strip() for s in sentences if s.strip()]
        for i, sentence in enumerate(sentences):
            if not sentence.endswith(('.', '!', '?')):
                sentences[i] = sentence + '.'
        return sentences

    def _create_single_slide(self, sentence: str, audio_path: Path, bg_color: Tuple[int, int, int],
                            unsplash_keyword: Optional[str], slide_num: int) -> Tuple[ImageClip, Path]:
        """Create a single slide - optimized for parallel processing"""
        try:
            audio_segment = AudioSegment.from_file(str(audio_path))
            duration_sec = len(audio_segment) / 1000.0

            text_color = self.get_random_text_color()

            image_path = self._create_text_image(sentence, self.config.VIDEO_SIZE, bg_color,
                                                 unsplash_keyword, text_color)

            audio_clip = AudioFileClip(str(audio_path))
            video_clip = ImageClip(str(image_path), duration=duration_sec).set_audio(audio_clip)

            return (video_clip, image_path)
        except Exception as e:
            print(f"[Video] Error creating slide {slide_num}: {e}")
            return (None, None)

    def create_video_per_sentence(self, sentences: List[str], audio_paths: List[Path],
                                  cta_audio_path: Optional[Path] = None,
                                  bg_color: Tuple[int, int, int] = (74, 144, 226),
                                  unsplash_keyword: Optional[str] = None,
                                  add_cta_slide: bool = True,
                                  progress_callback=None) -> Path:
        size = self.config.VIDEO_SIZE
        clips = []
        temp_image_paths = []

        print(f"\n🎨 Creating {len(sentences)} slides with RANDOM COLORS (9:16)...")
        print(f"⚡ Using parallel processing (max {self.config.MAX_PARALLEL_SLIDES} workers)")

        if unsplash_keyword:
            self.preload_backgrounds(len(sentences), unsplash_keyword)

        with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_SLIDES) as executor:
            futures = {}
            for i, (sentence, audio_path) in enumerate(zip(sentences, audio_paths)):
                future = executor.submit(
                    self._create_single_slide,
                    sentence, audio_path, bg_color, unsplash_keyword, i + 1
                )
                futures[future] = i

            slide_results = [None] * len(sentences)
            for future in as_completed(futures):
                i = futures[future]
                try:
                    video_clip, image_path = future.result()
                    if video_clip:
                        slide_results[i] = (video_clip, image_path)
                        audio_segment = AudioSegment.from_file(str(audio_paths[i]))
                        duration_sec = len(audio_segment) / 1000.0
                        print(f"  ✓ Slide {i + 1}/{len(sentences)}: {duration_sec:.2f}s - '{sentences[i][:50]}...'")
                        if progress_callback:
                            progress_callback(i + 1, len(sentences), f"Created slide {i + 1}")
                except Exception as e:
                    print(f"  ✗ Error processing slide {i + 1}: {e}")

        for result in slide_results:
            if result:
                video_clip, image_path = result
                clips.append(video_clip)
                temp_image_paths.append(image_path)

        if not clips:
            raise ValueError("No clips were created - all slides failed")

        if add_cta_slide and cta_audio_path:
            try:
                cta_clip, cta_image_path = self.create_cta_slide(
                    cta_audio_path,
                    bg_color=bg_color,
                    unsplash_keyword=unsplash_keyword
                )
                clips.append(cta_clip)
                temp_image_paths.append(cta_image_path)
                print("✓ Added Call-To-Action slide at the end with audio")
            except Exception as e:
                print(f"✗ Warning: Could not create CTA slide: {e}")

        print("\n🎬 Assembling final portrait video with colorful text...")
        if progress_callback:
            progress_callback(len(sentences), len(sentences), "Assembling final video...")
        final_clip = concatenate_videoclips(clips, method="compose")
        output_path = self.config.TEMP_DIR / f"video_{uuid.uuid4()}.mp4"
        final_clip.write_videofile(
            str(output_path),
            fps=30,
            codec='libx264',
            audio_codec='aac',
            logger=None,
            preset='medium',
            threads=4
        )
        for path in temp_image_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[Video] Warning: Could not delete temp file {path}: {e}")
        for clip in clips:
            try:
                clip.close()
            except Exception as e:
                print(f"[Video] Warning: Error closing clip: {e}")
        try:
            final_clip.close()
        except Exception as e:
            print(f"[Video] Warning: Error closing final clip: {e}")
        return output_path

class TextToVideoGenerator:
    def __init__(self):
        self.config = Config()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config)
        self.available_voices = self._get_available_voices()
    def _get_available_voices(self) -> List[str]:
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                       bg_color: Tuple[int, int, int] = (74, 144, 226),
                       unsplash_keyword: Optional[str] = None,
                       unsplash_client_id: Optional[str] = None,
                       enable_background_music: bool = True,
                       music_volume_db: int = -15,
                       add_call_to_action: bool = True,
                       progress_callback=None) -> Dict:
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}
        if len(text) > 10000:
            return {"error": "Text is too long (max 10,000 characters)", "success": False}
        if music_volume_db != self.config.MUSIC_CONFIG['music_volume_db']:
            self.config.MUSIC_CONFIG['music_volume_db'] = music_volume_db
        if unsplash_client_id and unsplash_client_id.strip():
            self.video_generator.unsplash.set_client_id(unsplash_client_id.strip())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        audio_paths = []
        cta_audio_path = None
        music_path = None
        try:
            print("📝 Splitting text into sentences...")
            sentences = self.video_generator.split_into_sentences(text)
            print(f"✓ Found {len(sentences)} sentences")
            if len(sentences) > 100:
                return {"error": "Too many sentences (max 100)", "success": False}
            if enable_background_music:
                music_path = self.video_generator.get_random_background_music()
                if music_path:
                    print(f"[Music] ✓ Using background music: {music_path.name}")
                else:
                    print("[Music] ✗ No background music found - continuing without music")

            print("\n🎙️ Generating audio for each sentence...")

            for i, sentence in enumerate(sentences):
                print(f"  Sentence {i + 1}/{len(sentences)}: '{sentence[:50]}...'")
                if progress_callback:
                    progress_callback(i + 1, len(sentences) * 2, f"Generating audio {i + 1}/{len(sentences)}")
                audio_path = self.tts_manager.generate_speech(sentence, speaker_id)
                audio_paths.append(audio_path)

            if add_call_to_action:
                print(f"\n🎙️ Generating CTA audio: '{self.config.CTA_MESSAGE}'")
                cta_audio_path = self.tts_manager.generate_speech(self.config.CTA_MESSAGE, speaker_id)
                print(f"✓ CTA audio generated")

            print("\n🔊 Combining all voice audio...")
            combined_voice = AudioSegment.empty()
            for audio_path in audio_paths:
                segment = AudioSegment.from_file(str(audio_path))
                combined_voice += segment

            if add_call_to_action and cta_audio_path:
                cta_segment = AudioSegment.from_file(str(cta_audio_path))
                combined_voice += cta_segment
                print(f"✓ Added CTA audio to combined voice ({len(cta_segment)/1000:.2f}s)")

            total_duration_ms = len(combined_voice)
            print(f"[Audio] Total voice duration: {total_duration_ms/1000:.2f}s")
            temp_combined_voice_path = self.config.TEMP_DIR / f"combined_voice_{uuid.uuid4()}.wav"
            combined_voice.export(str(temp_combined_voice_path), format="wav")
            final_audio_path = temp_combined_voice_path
            if enable_background_music and music_path:
                print("\n🎵 Mixing entire audio with background music...")
                final_audio_path = self.video_generator.mix_audio_with_music(
                    temp_combined_voice_path,
                    total_duration_ms,
                    music_path
                )
            else:
                print("\n🎵 No background music - using voice only")
            audio_mp3_path = session_dir / f"audio_{timestamp}.mp3"
            final_audio_segment = AudioSegment.from_file(str(final_audio_path))
            final_audio_segment.export(str(audio_mp3_path), format="mp3", bitrate="192k")
            print(f"[Audio] ✓ Final audio saved: {audio_mp3_path}")

            print("\n🎨 Creating portrait video with RANDOM COLORED TEXT (9:16)...")
            def video_progress(current, total, message):
                if progress_callback:
                    progress_callback(len(sentences) + current, len(sentences) * 2, message)
            video_temp_path = self.video_generator.create_video_per_sentence(
                sentences, audio_paths,
                cta_audio_path=cta_audio_path,
                bg_color=bg_color,
                unsplash_keyword=unsplash_keyword,
                add_cta_slide=add_call_to_action,
                progress_callback=video_progress
            )
            print("\n🎬 Replacing video audio with mixed audio...")
            from moviepy.editor import VideoFileClip
            video_clip = VideoFileClip(str(video_temp_path))
            final_audio_clip = AudioFileClip(str(final_audio_path))
            final_video = video_clip.set_audio(final_audio_clip)
            video_final_path = session_dir / f"video_portrait_{timestamp}.mp4"
            final_video.write_videofile(
                str(video_final_path),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                logger=None,
                preset='medium',
                threads=4
            )
            video_clip.close()
            final_audio_clip.close()
            final_video.close()
            try:
                video_temp_path.unlink(missing_ok=True)
            except:
                pass
            print(f"\n✅ Portrait video created successfully: {video_final_path}")
            print(f"🎨 Text colors: RANDOMIZED for each slide!")
            print(f"🎬 CTA slide: {'✓ Included' if add_call_to_action else '✗ Disabled'}")
            print(f"📱 Perfect for: YouTube Shorts, TikTok, Instagram Reels")
            print(f"⚡ Performance optimizations: Parallel processing enabled")
            return {
                "success": True,
                "audio_path": str(audio_mp3_path),
                "video_path": str(video_final_path),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "background_music": enable_background_music and music_path is not None,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "text_colors": "Random vibrant colors per slide"
            }
        except Exception as e:
            print(f"\n❌ Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "success": False}
        finally:
            for audio_path in audio_paths:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"[Cleanup] Warning: Could not delete {audio_path}: {e}")
            if cta_audio_path:
                try:
                    cta_audio_path.unlink(missing_ok=True)
                except:
                    pass
            try:
                if 'temp_combined_voice_path' in locals():
                    temp_combined_voice_path.unlink(missing_ok=True)
            except:
                pass

def setup_ui(generator: TextToVideoGenerator):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   title="Portrait Video Generator (9:16) - Random Colors + CTA",
                   css="""
                   .large-text textarea { font-size: 16px !important; }
                   .progress-bar { margin: 10px 0; }
                   .gradio-container .gr-tabs { padding: 8px 12px; }
                   .gradio-container .gr-tabitem { padding: 10px; }
                   """) as demo:
        gr.Markdown("# 🎨 Portrait Video Generator - Random Colors + CTA Edition")
        with gr.Tabs():
            with gr.TabItem("Controls"):
                gr.Markdown(
                    "✅ **NEW: AUTO CTA SLIDE!** 🎬\n"
                    "- Every video ends with: **'Like, share, and subscribe to our channel!'**\n"
                    "- Includes **spoken audio** + **animated visual CTA**\n"
                    "- Professional gold glow effects with emojis (👍💬🔔)\n\n"
                    "✅ **RANDOM VIBRANT TEXT COLORS!** 🌈\n"
                    "- Each slide gets a **random vibrant color** (Gold, Pink, Cyan, Orange, Lime, etc.)\n"
                    "- Makes your videos **more engaging and eye-catching**\n\n"
                    "✅ **PERFORMANCE OPTIMIZATIONS:** ⚡\n"
                    "- **Parallel slide creation** - up to 4x faster video assembly\n"
                    "- **Background caching** - smoother generation\n"
                    "- **Optimized font sizing** - faster text rendering\n\n"
                    "✅ **PORTRAIT MODE (9:16):**\n"
                    "- Perfect for **YouTube Shorts**, **TikTok**, **Instagram Reels**\n"
                    "- Resolution: **1080x1920** (mobile optimized)\n"
                    "- **ENHANCED AUDIO:** Smooth, warm, natural voice quality\n\n"
                    "Convert your text into a mobile-friendly portrait video with **colorful text**, "
                    "perfectly synchronized audio, background music, and **automatic CTA**."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Enter Your Text",
                            placeholder="Type or paste your text here. Each sentence will become a separate slide with a random color...",
                            lines=8,
                            max_lines=15,
                            elem_classes=["large-text"]
                        )
                        with gr.Row():
                            speaker_dropdown = gr.Dropdown(
                                label="Select Voice",
                                choices=generator.available_voices,
                                value=generator.config.STANDARD_VOICE_NAME
                            )
                            bg_color_picker = gr.ColorPicker(
                                label="Background Color (fallback)",
                                value="#4A90E2"
                            )
                        gr.Markdown("### 🎬 Call-To-Action Settings")
                        enable_cta = gr.Checkbox(
                            label="Add CTA Slide (Like, Share, Subscribe)",
                            value=True,
                            info="Automatically adds CTA at the end with audio"
                        )
                        gr.Markdown("### 🎵 Background Music Settings")
                        with gr.Row():
                            enable_music = gr.Checkbox(
                                label="Enable Background Music",
                                value=True,
                                info="Add background music from background_music/ folder"
                            )
                            music_volume = gr.Slider(
                                minimum=-40,
                                maximum=-5,
                                value=-15,
                                step=1,
                                label="Music Volume (dB)",
                                info="Lower = quieter music. -15dB default"
                            )
                        gr.Markdown("### 🖼️ Unsplash Background Settings")
                        with gr.Row():
                            unsplash_keyword = gr.Textbox(
                                label="Image Search Keyword",
                                placeholder="e.g., nature, business, technology, abstract",
                                info="Leave empty to use local images or plain background"
                            )
                            unsplash_client_id = gr.Textbox(
                                label="Unsplash API Client ID",
                                placeholder="Your Unsplash Access Key",
                                type="password",
                                info="Get your free API key from unsplash.com/developers"
                            )
                        progress_bar = gr.Textbox(
                            label="Progress",
                            value="Ready to generate portrait video with random colors + CTA...",
                            interactive=False,
                            elem_classes=["progress-bar"]
                        )
                        generate_button = gr.Button("🎨 Generate Colorful Video + CTA (9:16)", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎬 CTA Feature")
                        gr.Markdown(
                            "**Auto Call-To-Action:**\n"
                            "- 🎙️ **Spoken:** 'Like, share, and subscribe to our channel!'\n"
                            "- 👍 **Visual:** Animated emoji text\n"
                            "- ✨ **Gold glow** effects\n"
                            "- ⏱️ **Duration:** ~3 seconds\n"
                            "- 🎯 **Boosts engagement**\n\n"
                            "**15 Vibrant Colors:**\n"
                            "- 🤍 White, 🟡 Gold\n"
                            "- 💗 Hot Pink, 🔵 Cyan\n"
                            "- 🟠 Orange, 🟢 Lime\n"
                            "- 💕 Deep Pink, 🟨 Yellow\n"
                            "- And 7 more!\n\n"
                            "**Performance:**\n"
                            "- ⚡ **Faster** generation\n"
                            "- 🔄 Parallel processing\n"
                            "- 💾 Smart caching\n"
                        )
            with gr.TabItem("Preview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_output = gr.Audio(label="Generated Audio (with CTA)")
                        video_output = gr.Video(label="Generated Portrait Video (9:16) 🎨")
                        status_output = gr.Markdown()
                    with gr.Column(scale=1):
                        gr.Markdown("### Output / Status")
                        status_output_2 = gr.Markdown("Ready to generate...")
        gr.Examples(
            examples=[
                ["Welcome to our channel. Today we explore amazing technology. Artificial intelligence is changing everything.",
                 "Standard Voice (Non-Cloned)", "#4A90E2", "technology", "", True, -15, True],
                ["Follow for daily motivation. Success starts with believing in yourself. Every day is a new opportunity. Let's grow together.",
                 "Standard Voice (Non-Cloned)", "#8B5CF6", "motivation", "", True, -15, True],
                ["Check out this incredible fact. Did you know the ocean is deeper than Mount Everest is tall. Nature is truly amazing.",
                 "Standard Voice (Non-Cloned)", "#1E3A8A", "ocean", "", True, -15, True],
            ],
            inputs=[text_input, speaker_dropdown, bg_color_picker, unsplash_keyword,
                   unsplash_client_id, enable_music, music_volume, enable_cta],
            label="📝 Example Texts (Auto CTA at end!)"
        )
        def generate_video_wrapper(text, speaker, bg_color_hex, keyword, client_id,
                                   enable_music, music_vol, enable_cta, progress=gr.Progress()):
            if not text or not text.strip():
                return None, None, "❌ Error: Please enter some text", "Ready to generate..."
            bg_color_hex = bg_color_hex.lstrip('#')
            bg_color = tuple(int(bg_color_hex[i:i + 2], 16) for i in (0, 2, 4))
            keyword = keyword.strip() if keyword else None
            client_id = client_id.strip() if client_id else None
            def update_progress(current, total, message):
                progress_text = f"Progress: {current}/{total} - {message}"
                progress((current, total), desc=message)
                return progress_text
            progress(0, desc="Starting portrait video generation with CTA...")
            result = generator.generate_video(
                text, speaker, bg_color, keyword, client_id,
                enable_background_music=enable_music,
                music_volume_db=music_vol,
                add_call_to_action=enable_cta,
                progress_callback=update_progress
            )
            if result.get("success"):
                music_status = "✓ With background music" if result.get("background_music") else "○ Voice only"
                cta_status = "✓ CTA included" if result.get("cta_included") else "○ No CTA"
                final_status = (
                    f"✅ Portrait video created successfully!\n\n"
                    f"**Format:** {result.get('video_format', '9:16 Portrait')}\n\n"
                    f"🎨 **Text Colors:** {result.get('text_colors', 'Random per slide')}!\n\n"
                    f"🎬 **CTA:** {cta_status}\n\n"
                    f"**Perfect for:** YouTube Shorts, TikTok, Instagram Reels\n\n"
                    f"**Sentences processed:** {result['sentence_count']}\n\n"
                    f"**Audio:** {music_status} (Music: {music_vol}dB)\n\n"
                    f"**Performance:** ⚡ Optimized with parallel processing\n\n"
                    f"**Audio enhancements:** Smooth, warm, natural sound\n\n"
                    f"**Output Directory:** `{result['output_directory']}`"
                )
                return (
                    result["audio_path"],
                    result["video_path"],
                    final_status,
                    "✅ Colorful portrait video with CTA complete!"
                )
            error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
            return None, None, error_msg, "❌ Generation failed"
        generate_button.click(
            fn=generate_video_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color_picker, unsplash_keyword,
                   unsplash_client_id, enable_music, music_volume, enable_cta],
            outputs=[audio_output, video_output, status_output, progress_bar]
        )
    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)

if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\n❌ Missing required libraries. Please install:")
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests")
    else:
        print("\n🎨 Starting Portrait Video Generator - RANDOM COLORS + CTA EDITION")
        print("=" * 80)
        print("✨ NEW FEATURES:")
        print("  • 🎬 AUTO CTA SLIDE - 'Like, share, subscribe' with audio!")
        print("  • 🌈 RANDOM VIBRANT TEXT COLORS - Each slide gets a different color!")
        print("  • ⚡ PARALLEL SLIDE CREATION - Faster video generation")
        print("  • 💾 SMART CACHING - Smoother, more efficient workflow")
        print("\n✅ PORTRAIT MODE FEATURES:")
        print("  • Resolution: 1080×1920 (9:16 aspect ratio)")
        print("  • Perfect for YouTube Shorts, TikTok, Instagram Reels")
        print("  • FPS: 30 for smooth mobile playback")
        print("  • CTA Message: 'Like, share, and subscribe to our channel!'")
        print("\n🎨 15 VIBRANT COLORS AVAILABLE:")
        print("  • White, Gold, Hot Pink, Cyan, Orange, Lime Green")
        print("  • Deep Pink, Yellow, Blue Violet, Red Orange")
        print("  • Spring Green, Pink, Light Blue, Peach, Light Green")
        print("\n🎙️ ENHANCED AUDIO PROCESSING:")
        print("  • Removes metallic/harsh frequencies")
        print("  • Reduces sibilance")
        print("  • Adds warmth and depth")
        print("  • Gentle compression for smooth dynamics")
        print("  • CTA audio automatically generated and synced")
        print("\n⚠️ PERFORMANCE NOTE:")
        print("  • Audio generation: Sequential (TTS models aren't thread-safe)")
        print("  • Slide creation: Parallel (up to 4x faster)")
        print("=" * 80)
        print("\n📁 Required Folders:")
        print("  • background_music/ - Add MP3/WAV files")
        print("  • background_images/ - Add portrait images (optional)")
        print("  • voice_samples/ - Add voice samples (optional)")
        print("=" * 80)
        print()
        generator = TextToVideoGenerator()
        setup_ui(generator)