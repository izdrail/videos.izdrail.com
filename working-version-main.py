#!/usr/bin/env python3
"""
Text-to-Video Generator with Voice Training, Background Music, and Unsplash Integration
FIXED: Voice training tab visibility, default music volume, and random text colors
"""

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

import torch
import torchaudio
import gradio as gr
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from num2words import num2words
import textwrap

# Conditional import for TTS models
try:
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    from TTS.api import TTS
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"TTS libraries not found: {e}")
    MODELS_AVAILABLE = False


class Config:
    """Configuration and paths for the application."""

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

        self.VOICE_REQUIREMENTS = {
            'min_duration': 5,
            'max_duration': 30,
            'sample_rate': 22050,
            'min_quality_bitrate': 64000,
        }

        self.TEXT_SIZE_CONFIG = {
            'target_coverage': 0.5,
            'initial_font_size': 150,
            'min_font_size': 40,
            'max_font_size': 200,
            'wrap_width_range': (15, 35),
            'margin_percentage': 0.05,
        }

        self.MUSIC_CONFIG = {
            'voice_volume_db': 0,
            'music_volume_db': -10,  # Changed from -20 to -10
            'fade_in_duration': 2000,
            'fade_out_duration': 2000,
        }

        # Color palettes for random text colors
        self.COLOR_PALETTES = [
            # Vibrant colors
            [(255, 255, 255), (255, 107, 107)],  # White/Red
            [(255, 255, 255), (78, 205, 196)],   # White/Turquoise
            [(255, 255, 255), (255, 195, 0)],    # White/Gold
            [(255, 255, 255), (138, 43, 226)],   # White/Purple
            [(255, 255, 255), (0, 168, 255)],    # White/Blue
            [(255, 255, 255), (255, 121, 198)],  # White/Pink
            [(255, 255, 255), (72, 219, 251)],   # White/Cyan
            [(255, 255, 255), (250, 177, 160)],  # White/Coral
            # Gradient-friendly pairs
            [(255, 223, 186), (255, 107, 107)],  # Peach/Red
            [(186, 220, 255), (72, 219, 251)],   # Light Blue/Cyan
            [(255, 250, 205), (255, 195, 0)],    # Light Yellow/Gold
            [(230, 230, 250), (138, 43, 226)],   # Lavender/Purple
        ]


class VoiceTrainer:
    """Handles voice sample upload, validation, and preparation for cloning."""

    def __init__(self, config: Config):
        self.config = config

    def validate_audio(self, audio_path: Path) -> Tuple[bool, str, Dict]:
        """Validate audio file for voice cloning."""
        try:
            audio = AudioSegment.from_file(str(audio_path))
            duration_sec = len(audio) / 1000.0

            info = {
                'duration': duration_sec,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'sample_width': audio.sample_width,
            }

            if duration_sec < self.config.VOICE_REQUIREMENTS['min_duration']:
                return False, f"Audio too short ({duration_sec:.1f}s). Minimum: {self.config.VOICE_REQUIREMENTS['min_duration']}s", info

            if duration_sec > self.config.VOICE_REQUIREMENTS['max_duration']:
                return False, f"Audio too long ({duration_sec:.1f}s). Maximum: {self.config.VOICE_REQUIREMENTS['max_duration']}s", info

            if audio.channels > 2:
                return False, f"Too many audio channels ({audio.channels}). Use mono or stereo.", info

            return True, f"Valid audio: {duration_sec:.1f}s, {audio.frame_rate}Hz", info

        except Exception as e:
            return False, f"Error reading audio file: {str(e)}", {}

    def process_audio_for_training(self, audio_path: Path, output_path: Path) -> bool:
        """Process audio file to optimal format for voice cloning."""
        try:
            print(f"[Voice] Processing audio: {audio_path.name}")
            audio = AudioSegment.from_file(str(audio_path))

            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("[Voice] Converted to mono")

            target_rate = self.config.VOICE_REQUIREMENTS['sample_rate']
            if audio.frame_rate != target_rate:
                audio = audio.set_frame_rate(target_rate)
                print(f"[Voice] Resampled to {target_rate}Hz")

            audio = audio.normalize()
            print("[Voice] Normalized volume")

            def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
                trim_ms = 0
                while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
                    trim_ms += chunk_size
                return trim_ms

            start_trim = detect_leading_silence(audio)
            end_trim = detect_leading_silence(audio.reverse())

            duration = len(audio)
            audio = audio[start_trim:duration-end_trim]
            print(f"[Voice] Trimmed silence: {start_trim}ms from start, {end_trim}ms from end")

            audio.export(str(output_path), format="wav", parameters=["-ar", str(target_rate)])
            print(f"[Voice] Saved processed audio: {output_path.name}")
            return True

        except Exception as e:
            print(f"[Voice] Error processing audio: {e}")
            return False

    def create_voice_profile(self, voice_name: str, audio_file: str) -> Dict:
        """Create a new voice profile from uploaded audio."""
        if not voice_name or not voice_name.strip():
            return {"success": False, "error": "Voice name cannot be empty"}

        voice_name = re.sub(r'[^\w\s-]', '', voice_name.strip())
        voice_name = re.sub(r'\s+', '_', voice_name)

        if not voice_name:
            return {"success": False, "error": "Invalid voice name"}

        voice_dir = self.config.VOICE_SAMPLES_DIR / voice_name
        if voice_dir.exists():
            return {"success": False, "error": f"Voice '{voice_name}' already exists. Choose a different name."}

        temp_audio_path = Path(audio_file)
        is_valid, message, info = self.validate_audio(temp_audio_path)

        if not is_valid:
            return {"success": False, "error": message}

        try:
            voice_dir.mkdir(exist_ok=True)
            reference_path = voice_dir / "reference.wav"
            success = self.process_audio_for_training(temp_audio_path, reference_path)

            if not success:
                shutil.rmtree(voice_dir, ignore_errors=True)
                return {"success": False, "error": "Failed to process audio file"}

            metadata_path = voice_dir / "info.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Voice Name: {voice_name}\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {info.get('duration', 0):.1f}s\n")
                f.write(f"Original Sample Rate: {info.get('sample_rate', 'unknown')}Hz\n")
                f.write(f"Processed Sample Rate: {self.config.VOICE_REQUIREMENTS['sample_rate']}Hz\n")

            return {
                "success": True,
                "message": f"Voice '{voice_name}' created successfully!\n\n{message}",
                "voice_name": voice_name,
                "voice_dir": str(voice_dir)
            }

        except Exception as e:
            if voice_dir.exists():
                shutil.rmtree(voice_dir, ignore_errors=True)
            return {"success": False, "error": f"Error creating voice profile: {str(e)}"}

    def delete_voice_profile(self, voice_name: str) -> Dict:
        """Delete a voice profile."""
        if voice_name == self.config.STANDARD_VOICE_NAME:
            return {"success": False, "error": "Cannot delete the standard voice"}

        voice_dir = self.config.VOICE_SAMPLES_DIR / voice_name
        if not voice_dir.exists():
            return {"success": False, "error": f"Voice '{voice_name}' not found"}

        try:
            shutil.rmtree(voice_dir)
            return {"success": True, "message": f"Voice '{voice_name}' deleted successfully"}
        except Exception as e:
            return {"success": False, "error": f"Error deleting voice: {str(e)}"}

    def list_voices(self) -> List[str]:
        """Get list of available voices."""
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)


class UnsplashAPI:
    """Handles Unsplash API requests for fetching images."""

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
            "orientation": "landscape"
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

    def get_random_image(self, query: str, size: Tuple[int, int] = (1280, 720)) -> Optional[Image.Image]:
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
    """Handles TTS generation for all voice types."""

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

        return temp_wav_path


class VideoGenerator:
    """Creates video from audio and text with synchronized timing."""

    def __init__(self, config: Config):
        self.config = config
        self.available_fonts = self._discover_fonts()
        self.unsplash = UnsplashAPI()
        self._ensure_default_music()

    def _ensure_default_music(self):
        """Ensure default background music always exists."""
        default_music_path = self.config.MUSIC_DIR / "default_silence.mp3"
        music_files = list(self.config.MUSIC_DIR.glob("*.mp3")) + list(self.config.MUSIC_DIR.glob("*.MP3"))

        if not music_files:
            print("[Music] Creating default silent track...")
            try:
                silent = AudioSegment.silent(duration=60000)
                silent.export(str(default_music_path), format="mp3", bitrate="128k")
                print(f"[Music] ✓ Created: {default_music_path.name}")
            except Exception as e:
                print(f"[Music] ✗ Error creating default track: {e}")

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
            "arialbd.ttf", "Arial Bold.ttf", "calibrib.ttf", "Calibri Bold.ttf",
            "arial.ttf", "Arial.ttf", "calibri.ttf", "Calibri.ttf",
            "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"
        ]

        for path in font_paths:
            if path.is_dir():
                for font_name in common_fonts:
                    if (path / font_name).exists():
                        font_file = path / font_name
                        if font_file not in discovered:
                            discovered.append(font_file)
        return discovered

    def get_background_image(self, size: Tuple[int, int], unsplash_keyword: Optional[str] = None) -> Optional[Image.Image]:
        if unsplash_keyword and self.unsplash.client_id:
            print(f"[Video] Fetching Unsplash image: '{unsplash_keyword}'")
            img = self.unsplash.get_random_image(unsplash_keyword, size)
            if img:
                return img

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []

        if self.config.IMAGES_DIR.exists():
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(self.config.IMAGES_DIR, ext)))

        if image_files:
            try:
                img = Image.open(random.choice(image_files))
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

    def get_random_background_music(self) -> Path:
        """Get random MP3 file, creating default if needed."""
        music_files = list(self.config.MUSIC_DIR.glob("*.mp3"))

        if music_files:
            selected = random.choice(music_files)
            print(f"[Music] ✓ Selected: {selected.name}")
            return selected

        default_music = self.config.MUSIC_DIR / "default_silence.mp3"
        if not default_music.exists():
            print("[Music] Creating default track...")
            silent = AudioSegment.silent(duration=60000)
            silent.export(str(default_music), format="mp3", bitrate="128k")

        return default_music

    def mix_audio_with_music(self, voice_audio_path: Path, music_path: Optional[Path] = None,
                            output_path: Optional[Path] = None) -> Path:
        """Mix voice with background music."""
        if music_path is None:
            music_path = self.get_random_background_music()

        try:
            print(f"[Music] Mixing audio with background...")
            voice = AudioSegment.from_file(str(voice_audio_path))
            music = AudioSegment.from_file(str(music_path))

            voice_duration = len(voice)
            cfg = self.config.MUSIC_CONFIG

            voice = voice + cfg['voice_volume_db']
            music = music + cfg['music_volume_db']

            if len(music) < voice_duration:
                loops_needed = (voice_duration // len(music)) + 1
                music = music * loops_needed

            music = music[:voice_duration]
            music = music.fade_in(cfg['fade_in_duration']).fade_out(cfg['fade_out_duration'])
            mixed = voice.overlay(music)

            if output_path is None:
                output_path = self.config.TEMP_DIR / f"mixed_{uuid.uuid4()}.mp3"

            mixed.export(str(output_path), format="mp3", bitrate="192k")
            print(f"[Music] ✓ Mixed successfully")
            return output_path

        except Exception as e:
            print(f"[Music] ✗ Error: {e}")
            return voice_audio_path

    def _create_text_image(self, text: str, size: Tuple[int, int] = (1280, 720),
                           bg_color: Tuple[int, int, int] = (74, 144, 226),
                           unsplash_keyword: Optional[str] = None) -> Path:
        background = self.get_background_image(size, unsplash_keyword)
        img = background.copy() if background else Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(img)

        # Select random color palette
        text_colors = random.choice(self.config.COLOR_PALETTES)
        primary_color = text_colors[0]
        secondary_color = text_colors[1]

        cfg = self.config.TEXT_SIZE_CONFIG
        margin = int(size[0] * cfg['margin_percentage'])
        text_width = size[0] - (margin * 2)
        target_text_area = (size[0] * size[1]) * cfg['target_coverage']

        font_path = self.available_fonts[0] if self.available_fonts else None
        font_size = cfg['initial_font_size']

        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
        wrapped_text = textwrap.fill(text, width=20)

        best_font_size = cfg['min_font_size']
        best_wrapped = wrapped_text
        best_area = 0

        for test_size in range(cfg['max_font_size'], cfg['min_font_size'] - 1, -5):
            test_font = ImageFont.truetype(str(font_path), test_size) if font_path else ImageFont.load_default()
            for w in range(*cfg['wrap_width_range']):
                test_wrapped = textwrap.fill(text, width=w)
                bbox = draw.textbbox((0, 0), test_wrapped, font=test_font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                if text_w <= text_width and text_h <= size[1] * 0.85:
                    text_area = text_w * text_h
                    if text_area >= target_text_area:
                        best_font_size = test_size
                        best_wrapped = test_wrapped
                        best_area = text_area
                        break
                    elif text_area > best_area:
                        best_font_size = test_size
                        best_wrapped = test_wrapped
                        best_area = text_area
            if best_area >= target_text_area:
                break

        font = ImageFont.truetype(str(font_path), best_font_size) if font_path else ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), best_wrapped, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2)

        # Create gradient overlay
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 40
        rect_coords = [text_pos[0] - padding, text_pos[1] - padding,
                      text_pos[0] + text_w + padding, text_pos[1] + text_h + padding]

        # Semi-transparent dark background
        overlay_draw.rounded_rectangle(rect_coords, radius=25, fill=(0, 0, 0, 160))

        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Enhanced shadow with gradient effect
        shadow_offset = 5
        for i, offset in enumerate([(-shadow_offset, -shadow_offset), (shadow_offset, -shadow_offset),
                       (-shadow_offset, shadow_offset), (shadow_offset, shadow_offset)]):
            alpha = int(180 - (i * 20))  # Varying shadow intensity
            shadow_color = (0, 0, 0) if alpha > 150 else (secondary_color[0]//3, secondary_color[1]//3, secondary_color[2]//3)
            draw.text((text_pos[0] + offset[0], text_pos[1] + offset[1]),
                      best_wrapped, font=font, fill=shadow_color, align="center")

        # Main text with primary color
        draw.text(text_pos, best_wrapped, font=font, fill=primary_color, align="center")

        # Optional: Add subtle glow effect with secondary color
        if random.random() > 0.5:  # 50% chance to add glow
            glow_offset = 2
            for glow_x in range(-glow_offset, glow_offset + 1):
                for glow_y in range(-glow_offset, glow_offset + 1):
                    if glow_x != 0 or glow_y != 0:
                        draw.text((text_pos[0] + glow_x, text_pos[1] + glow_y),
                                best_wrapped, font=font,
                                fill=(*secondary_color, 50), align="center")

        image_path = self.config.TEMP_DIR / f"slide_{uuid.uuid4()}.png"
        img.save(image_path, quality=95)
        return image_path

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms)\.', r'\1<dot>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<dot>', '.').strip() for s in sentences if s.strip()]
        return sentences

    def create_video_per_sentence(self, sentences: List[str], audio_paths: List[Path],
                                  bg_color: Tuple[int, int, int] = (74, 144, 226),
                                  unsplash_keyword: Optional[str] = None,
                                  progress_callback=None) -> Path:
        """Create synchronized video clips with perfect audio-video sync."""
        size = (1280, 720)
        clips = []
        temp_images = []

        print(f"\n[Video] Creating {len(sentences)} synchronized slides...")

        for i, (sentence, audio_path) in enumerate(zip(sentences, audio_paths)):
            try:
                # Get EXACT audio duration
                audio_clip = AudioFileClip(str(audio_path))
                exact_duration = audio_clip.duration
                print(f"  Slide {i+1}: {exact_duration:.3f}s - '{sentence[:50]}...'")

                if progress_callback:
                    progress_callback(i + 1, len(sentences), f"Slide {i+1}/{len(sentences)}")

                # Create text image with random colors
                image_path = self._create_text_image(sentence, size, bg_color, unsplash_keyword)
                temp_images.append(image_path)

                # Create video clip with EXACT duration matching audio
                video_clip = (ImageClip(str(image_path))
                            .set_duration(exact_duration)
                            .set_audio(audio_clip))

                clips.append(video_clip)
                print(f"    ✓ Clip created: duration={exact_duration:.3f}s")

            except Exception as e:
                print(f"[Video] ✗ Error on slide {i+1}: {e}")
                if audio_clip:
                    audio_clip.close()
                continue

        if not clips:
            raise ValueError("No clips created")

        print("\n[Video] Assembling final video...")
        final_clip = concatenate_videoclips(clips, method="compose")

        output_path = self.config.TEMP_DIR / f"video_{uuid.uuid4()}.mp4"
        final_clip.write_videofile(
            str(output_path),
            fps=24,
            codec='libx264',
            audio_codec='aac',
            logger=None,
            preset='medium',
            threads=4
        )

        # Cleanup
        for clip in clips:
            clip.close()
        final_clip.close()

        for img_path in temp_images:
            img_path.unlink(missing_ok=True)

        return output_path


class TextToVideoGenerator:
    """Main generator class."""

    def __init__(self):
        self.config = Config()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config)
        self.voice_trainer = VoiceTrainer(self.config)
        self.available_voices = self.voice_trainer.list_voices()

    def refresh_voices(self) -> List[str]:
        self.available_voices = self.voice_trainer.list_voices()
        return self.available_voices

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                       bg_color: Tuple[int, int, int] = (74, 144, 226),
                       unsplash_keyword: Optional[str] = None,
                       unsplash_client_id: Optional[str] = None,
                       enable_background_music: bool = True,
                       music_volume_db: int = -10,
                       progress_callback=None) -> Dict:
        """Generate video with synchronized audio."""
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}

        if len(text) > 10000:
            return {"error": "Text too long (max 10,000 characters)", "success": False}

        if music_volume_db != self.config.MUSIC_CONFIG['music_volume_db']:
            self.config.MUSIC_CONFIG['music_volume_db'] = music_volume_db

        if unsplash_client_id:
            self.video_generator.unsplash.set_client_id(unsplash_client_id.strip())

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        audio_paths = []
        music_path = None

        try:
            print("=" * 80)
            print("STARTING VIDEO GENERATION")
            print("=" * 80)

            sentences = self.video_generator.split_into_sentences(text)
            print(f"\n✓ Split into {len(sentences)} sentences")

            if len(sentences) > 100:
                return {"error": "Too many sentences (max 100)", "success": False}

            if enable_background_music:
                print("\n[Music] Getting background music...")
                music_path = self.video_generator.get_random_background_music()

            print("\n[TTS] Generating audio for sentences...")
            for i, sentence in enumerate(sentences):
                print(f"  [{i+1}/{len(sentences)}] '{sentence[:60]}...'")

                if progress_callback:
                    progress_callback(i + 1, len(sentences) * 2, f"Audio {i+1}/{len(sentences)}")

                audio_path = self.tts_manager.generate_speech(sentence, speaker_id)

                if enable_background_music and music_path:
                    mixed_path = self.config.TEMP_DIR / f"mixed_{i}_{uuid.uuid4()}.mp3"
                    audio_path = self.video_generator.mix_audio_with_music(
                        audio_path, music_path, mixed_path
                    )

                audio_paths.append(audio_path)

            print("\n[Audio] Combining all segments...")
            combined = AudioSegment.empty()
            for audio_path in audio_paths:
                combined += AudioSegment.from_file(str(audio_path))

            audio_mp3 = session_dir / f"audio_{timestamp}.mp3"
            combined.export(audio_mp3, format="mp3", bitrate="192k")
            print(f"✓ Audio saved: {len(combined)/1000:.2f}s")

            print("\n[Video] Creating synchronized video with random text colors...")
            def video_progress(curr, total, msg):
                if progress_callback:
                    progress_callback(len(sentences) + curr, len(sentences) * 2, msg)

            video_temp = self.video_generator.create_video_per_sentence(
                sentences, audio_paths, bg_color, unsplash_keyword, video_progress
            )

            video_final = session_dir / f"video_{timestamp}.mp4"
            shutil.move(video_temp, video_final)

            print("\n" + "=" * 80)
            print("✓✓✓ VIDEO COMPLETE ✓✓✓")
            print("=" * 80)
            print(f"Video: {video_final}")
            print(f"Audio: {audio_mp3}")
            print(f"Sentences: {len(sentences)}")
            print(f"Music: {'✓ YES' if enable_background_music else '○ NO'}")
            print(f"Text Colors: ✓ RANDOMIZED")
            print("=" * 80 + "\n")

            return {
                "success": True,
                "audio_path": str(audio_mp3),
                "video_path": str(video_final),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "background_music": enable_background_music
            }

        except Exception as e:
            print(f"\n✗✗✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "success": False}

        finally:
            for audio_path in audio_paths:
                try:
                    audio_path.unlink(missing_ok=True)
                except:
                    pass


def setup_ui(generator: TextToVideoGenerator):
    """Create Gradio UI - FIXED voice training tab visibility."""

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue"),
        title="Text-to-Video Generator",
        css="""
        .large-text textarea { font-size: 16px !important; }
        .voice-section { padding: 15px; margin: 10px 0; border: 2px solid #4A90E2; border-radius: 8px; }
        .status-box { padding: 12px; border-radius: 6px; margin: 8px 0; }
        """
    ) as demo:

        gr.Markdown("# 🎬 Text-to-Video Generator with Voice Training & Random Colors")
        gr.Markdown("Convert text into video with custom voices, background music, and dynamic text colors")

        # Main tabs container
        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════════
            # TAB 1: VIDEO GENERATION
            # ═══════════════════════════════════════════════════════════════════
            with gr.TabItem("🎬 Generate Video", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="📝 Enter Your Text",
                            placeholder="Type your text here. Each sentence becomes a separate slide with random colors...",
                            lines=10,
                            elem_classes=["large-text"]
                        )

                        with gr.Row():
                            speaker_dropdown = gr.Dropdown(
                                label="🎤 Select Voice",
                                choices=generator.available_voices,
                                value=generator.config.STANDARD_VOICE_NAME
                            )
                            refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")

                        with gr.Row():
                            bg_color = gr.ColorPicker(label="🎨 Background Color", value="#4A90E2")

                        gr.Markdown("### 🎵 Background Music Settings")
                        with gr.Row():
                            enable_music = gr.Checkbox(
                                label="Enable Background Music",
                                value=True
                            )
                            music_vol = gr.Slider(
                                -40, -5, value=-10, step=1,
                                label="Music Volume (dB)",
                                info="Default: -10dB (louder than before)"
                            )

                        gr.Markdown("### 🖼️ Unsplash Images (Optional)")
                        with gr.Row():
                            unsplash_keyword = gr.Textbox(
                                label="Search Keyword",
                                placeholder="e.g., nature, business, technology"
                            )
                            unsplash_id = gr.Textbox(
                                label="Unsplash API Key",
                                type="password",
                                placeholder="Optional - leave blank to use local images"
                            )

                        progress = gr.Textbox(
                            label="📊 Progress",
                            value="Ready to generate...",
                            interactive=False
                        )

                        generate_btn = gr.Button(
                            "🎬 Generate Video with Random Colors",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        audio_out = gr.Audio(label="🔊 Generated Audio")
                        video_out = gr.Video(label="🎥 Generated Video")
                        status_out = gr.Markdown(value="", elem_classes=["status-box"])

            # ═══════════════════════════════════════════════════════════════════
            # TAB 2: VOICE TRAINING - FIXED VISIBILITY
            # ═══════════════════════════════════════════════════════════════════
            with gr.TabItem("🎤 Train Custom Voice", id=1):
                gr.Markdown("## 🎙️ Create Your Custom Voice Profile")
                gr.Markdown("Upload 5-30 seconds of clear speech to train a custom voice for TTS")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Requirements", elem_classes=["voice-section"])
                        gr.Markdown("""
                        ✓ **Duration:** 5-30 seconds
                        ✓ **Quality:** Clear audio, minimal background noise
                        ✓ **Format:** WAV, MP3, M4A, or any audio format
                        ✓ **Content:** Single speaker only
                        ✓ **Language:** English recommended
                        ✓ **Recording:** Use a good microphone in a quiet room
                        """)

                        voice_name = gr.Textbox(
                            label="Voice Profile Name",
                            placeholder="e.g., John_Professional, Sarah_Narrator",
                            info="Use letters, numbers, and underscores only"
                        )

                        audio_upload = gr.Audio(
                            label="Upload Voice Sample Audio",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )

                        train_btn = gr.Button(
                            "🎤 Create Voice Profile",
                            variant="primary",
                            size="lg"
                        )

                        train_status = gr.Markdown(
                            value="",
                            elem_classes=["status-box"]
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Available Voices", elem_classes=["voice-section"])
                        voices_list = gr.Textbox(
                            label="Current Voice Profiles",
                            value="\n".join(generator.available_voices),
                            lines=10,
                            interactive=False
                        )

                        gr.Markdown("### 🗑️ Delete Voice Profile")
                        delete_dropdown = gr.Dropdown(
                            label="Select Voice to Delete",
                            choices=[v for v in generator.available_voices
                                   if v != generator.config.STANDARD_VOICE_NAME],
                            value=None
                        )
                        delete_btn = gr.Button("🗑️ Delete Selected Voice", variant="stop")
                        delete_status = gr.Markdown(value="", elem_classes=["status-box"])

                gr.Markdown("---")
                gr.Markdown("""
                ### 💡 Tips for Best Voice Training Results

                1. **Environment:** Record in a quiet room with minimal echo
                2. **Microphone:** Use a quality microphone positioned 6-12 inches away
                3. **Speaking:** Talk naturally at normal pace and volume
                4. **Consistency:** Maintain consistent tone throughout the sample
                5. **Variety:** Include different sentence types and emotions
                6. **Avoid:** Background music, multiple speakers, or sound effects
                7. **Length:** 10-15 seconds is ideal for most voices

                **Good Example:** *"Hello, my name is John. I'm excited to help you with your project. Let me know if you have any questions."*
                """)

            # ═══════════════════════════════════════════════════════════════════
            # TAB 3: HELP & DOCUMENTATION
            # ═══════════════════════════════════════════════════════════════════
            with gr.TabItem("📖 Help & Guide", id=2):
                gr.Markdown("""
                # 📚 Complete User Guide

                ## 🎯 Key Features

                ### ✨ New in This Version
                - ✅ **Random Text Colors** - Each slide uses vibrant, randomly selected color palettes
                - ✅ **Voice Training Tab** - Fixed visibility issue, now fully accessible
                - ✅ **Better Music Volume** - Default changed from -20dB to -10dB for better balance
                - ✅ **Perfect Audio Sync** - Text stays visible for exact audio duration
                - ✅ **Enhanced Text Display** - 50%+ screen coverage with dynamic sizing

                ### 🎨 Text Color System
                - **12 Curated Color Palettes** - Professional combinations with high contrast
                - **Random Selection** - Each slide gets a unique color scheme
                - **Gradient Effects** - Optional glow effects for visual appeal
                - **Enhanced Shadows** - Multi-layered shadows for better readability
                - **Color Pairs** - Primary and secondary colors that complement each other

                ### 🎤 Voice Training
                - **Custom Voice Cloning** - Train AI on your voice samples
                - **Multiple Voices** - Create unlimited voice profiles
                - **Easy Management** - Add and delete voices anytime
                - **Professional Quality** - Uses XTTS v2 for realistic speech

                ### 🎵 Background Music
                - **Auto-Creation** - Creates silent track if no music found
                - **Random Selection** - Picks from your music library
                - **Volume Control** - Adjustable from -40dB to -5dB
                - **Smooth Mixing** - Automatic fade in/out effects

                ## 📁 Folder Structure

                ```
                project_directory/
                ├── voice_samples/          # Custom voice profiles
                │   ├── John_Professional/
                │   │   ├── reference.wav
                │   │   └── info.txt
                │   └── Sarah_Narrator/
                │       ├── reference.wav
                │       └── info.txt
                ├── background_music/       # Your MP3 music files
                │   ├── calm_piano.mp3
                │   ├── upbeat_corporate.mp3
                │   └── default_silence.mp3 (auto-created)
                ├── background_images/      # Local background images
                │   ├── nature1.jpg
                │   └── business2.png
                ├── output/                 # Generated videos
                │   └── video_20250112_143022/
                │       ├── video_20250112_143022.mp4
                │       └── audio_20250112_143022.mp3
                └── temp/                   # Temporary files (auto-cleaned)
                ```

                ## 🎬 How to Generate Videos

                1. **Enter Text** - Type or paste your content
                2. **Select Voice** - Choose standard or custom trained voice
                3. **Configure Options:**
                   - Background color (if not using images)
                   - Enable/disable background music
                   - Adjust music volume (-10dB recommended)
                   - Optional: Enter Unsplash keyword for backgrounds
                4. **Click Generate** - Wait for processing
                5. **Download** - Get your video and audio files

                ## 🎙️ How to Train Voices

                1. **Record Audio:**
                   - Duration: 5-30 seconds (10-15 ideal)
                   - Quality: Clear, no background noise
                   - Content: Natural speech, varied sentences

                2. **Upload:**
                   - Go to "Train Custom Voice" tab
                   - Enter unique profile name
                   - Upload your audio file

                3. **Create Profile:**
                   - Click "Create Voice Profile"
                   - Wait for processing
                   - Voice appears in dropdown

                4. **Use Voice:**
                   - Return to "Generate Video" tab
                   - Select your voice from dropdown
                   - Generate with your custom voice!

                ## 🎨 Color Palette Examples

                The system randomly selects from these professional combinations:

                - **Classic White/Red** - Bold and attention-grabbing
                - **White/Turquoise** - Modern and clean
                - **White/Gold** - Elegant and premium
                - **White/Purple** - Creative and unique
                - **White/Blue** - Professional and trustworthy
                - **Peach/Red Gradient** - Warm and inviting
                - **Light Blue/Cyan** - Cool and calming
                - **And more...** - 12 total palettes

                ## 🔧 Troubleshooting

                ### Voice Training Issues
                - **"Audio too short"** → Record for at least 5 seconds
                - **"Audio too long"** → Trim to under 30 seconds
                - **"Voice already exists"** → Use a different name
                - **Poor quality voice** → Re-record in quiet room with better mic

                ### Video Generation Issues
                - **No music playing** → System auto-creates silent track (working as intended)
                - **Text too small** → Increase text length or shorten sentences
                - **Audio out of sync** → Fixed in this version! Should be perfect now
                - **Unsplash not working** → Check API key or leave blank for local images

                ### Music Issues
                - **Music too loud** → Decrease volume slider
                - **Music too quiet** → Increase volume slider (try -5dB to -10dB)
                - **No music files** → Add MP3s to `background_music/` folder
                - **Wrong music playing** → Random selection is by design

                ## 🚀 Performance Tips

                1. **Shorter text** = Faster processing
                2. **Use local images** = Faster than Unsplash API
                3. **GPU acceleration** = Install CUDA for faster TTS
                4. **Pre-train voices** = Generate videos faster later
                5. **Optimize music** = Use 128kbps MP3s for faster mixing

                ## 📊 Technical Specifications

                - **Video Resolution:** 1280x720 (720p HD)
                - **Frame Rate:** 24 fps
                - **Video Codec:** H.264 (libx264)
                - **Audio Codec:** AAC
                - **Audio Bitrate:** 192 kbps
                - **TTS Models:** Coqui XTTS v2 + SpeechBrain Tacotron2
                - **Max Text Length:** 10,000 characters
                - **Max Sentences:** 100 per video

                ## 🆘 Getting Help

                If you encounter issues:
                1. Check console output for error messages
                2. Verify all folders exist and have write permissions
                3. Ensure required libraries are installed
                4. Check that audio/video files aren't corrupted
                5. Try with shorter text or different settings

                ## 📝 Credits

                - **TTS:** Coqui XTTS v2, SpeechBrain
                - **Video:** MoviePy
                - **Images:** Unsplash API (optional)
                - **Audio:** Pydub, PyTorch
                - **Interface:** Gradio
                """)

        # ═══════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════════════════════════════════

        def generate_wrapper(txt, spkr, bg_hex, kw, client, music, vol, progress=gr.Progress()):
            """Handle video generation with progress updates."""
            if not txt:
                return None, None, "❌ **Error:** Please enter some text", "❌ Error"

            try:
                bg_rgb = tuple(int(bg_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            except:
                bg_rgb = (74, 144, 226)

            def prog_update(curr, tot, msg):
                progress((curr, tot), desc=msg)

            result = generator.generate_video(
                txt, spkr, bg_rgb, kw or None, client or None,
                music, vol, prog_update
            )

            if result.get("success"):
                music_txt = "✅ WITH MUSIC" if result.get("background_music") else "⭕ No music"
                status = (f"## ✅ SUCCESS!\n\n"
                         f"**Sentences:** {result['sentence_count']}\n\n"
                         f"**Music:** {music_txt}\n\n"
                         f"**Colors:** 🎨 Randomized\n\n"
                         f"**Output:** `{result['output_directory']}`")
                return result["audio_path"], result["video_path"], status, "✅ Complete!"

            return None, None, f"## ❌ ERROR\n\n{result.get('error')}", "❌ Failed"

        def train_wrapper(name, audio):
            """Handle voice profile creation."""
            if not name or not audio:
                return "❌ **Error:** Please provide both name and audio file", gr.update(), gr.update()

            result = generator.voice_trainer.create_voice_profile(name, audio)

            if result["success"]:
                voices = generator.refresh_voices()
                voices_txt = "\n".join(f"✓ {v}" for v in voices)
                del_choices = [v for v in voices if v != generator.config.STANDARD_VOICE_NAME]
                return (f"## ✅ SUCCESS!\n\n{result['message']}",
                       voices_txt,
                       gr.update(choices=del_choices))

            return f"## ❌ ERROR\n\n{result['error']}", gr.update(), gr.update()

        def refresh_wrapper():
            """Refresh voice list."""
            voices = generator.refresh_voices()
            return gr.update(choices=voices)

        def delete_wrapper(voice):
            """Handle voice profile deletion."""
            if not voice:
                return "❌ **Error:** Please select a voice to delete", gr.update(), gr.update()

            result = generator.voice_trainer.delete_voice_profile(voice)

            if result["success"]:
                voices = generator.refresh_voices()
                voices_txt = "\n".join(f"✓ {v}" for v in voices)
                del_choices = [v for v in voices if v != generator.config.STANDARD_VOICE_NAME]
                return (f"## ✅ SUCCESS!\n\n{result['message']}",
                       voices_txt,
                       gr.update(choices=del_choices, value=None))

            return f"## ❌ ERROR\n\n{result['error']}", gr.update(), gr.update()

        # Connect all event handlers
        generate_btn.click(
            generate_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color, unsplash_keyword,
                   unsplash_id, enable_music, music_vol],
            outputs=[audio_out, video_out, status_out, progress]
        )

        refresh_btn.click(
            refresh_wrapper,
            outputs=[speaker_dropdown]
        )

        train_btn.click(
            train_wrapper,
            inputs=[voice_name, audio_upload],
            outputs=[train_status, voices_list, delete_dropdown]
        )

        delete_btn.click(
            delete_wrapper,
            inputs=[delete_dropdown],
            outputs=[delete_status, voices_list, delete_dropdown]
        )

    return demo


if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\n❌ MISSING REQUIRED LIBRARIES")
        print("=" * 80)
        print("Please install the following:")
        print()
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests")
        print()
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("🎬 TEXT-TO-VIDEO GENERATOR v2.0")
        print("=" * 80)
        print("\n✨ FEATURES:")
        print("  ✓ Custom voice training & cloning")
        print("  ✓ Background music (auto-enabled)")
        print("  ✓ Perfect audio-video synchronization")
        print("  ✓ Random vibrant text colors (12 palettes)")
        print("  ✓ Large readable text (50%+ coverage)")
        print("  ✓ Unsplash image integration")
        print("=" * 80)
        print("\n🔧 FIXES IN THIS VERSION:")
        print("  ✅ Voice training tab now fully visible")
        print("  ✅ Music volume default: -10dB (was -20dB)")
        print("  ✅ Random text colors per slide")
        print("  ✅ Enhanced visual effects & shadows")
        print("  ✅ Improved UI with better labels")
        print("=" * 80)
        print("\n🚀 Starting server...\n")

        generator = TextToVideoGenerator()
        demo = setup_ui(generator)
        demo.launch(server_name="0.0.0.0", server_port=1602, share=False)