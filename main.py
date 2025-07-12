#!/usr/bin/env python3
"""
Enhanced Zodiac Signs Horoscope Video Generator with Thumbnail Display
Creates videos with background images, better typography, synchronized audio, and displays image thumbnails
"""

import requests
import json
import os
import re
import glob
import random
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import uuid
import platform
import torch
import torchaudio
import numpy as np
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.silence import split_on_silence
from num2words import num2words
import textwrap
import gradio as gr

# Conditional import for TTS models
try:
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    from TTS.api import TTS
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Crucial libraries not found: {e}. Please install speechbrain and TTS.")
    MODELS_AVAILABLE = False


# --- Configuration Class ---
class Config:
    """Holds all static configuration and paths for the application."""
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.GENERATED_AUDIO_DIR = self.ROOT_DIR / "generated_audio"
        self.IMAGES_DIR = self.ROOT_DIR / "images"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        self.ZODIAC_IMAGES_DIR = self.IMAGES_DIR / ""

        # Create necessary directories
        for dir_path in [self.VOICE_SAMPLES_DIR, self.GENERATED_AUDIO_DIR, self.IMAGES_DIR,
                         self.TEMP_DIR, self.OUTPUT_DIR, self.ZODIAC_IMAGES_DIR]:
            dir_path.mkdir(exist_ok=True)

        # Constants
        self.STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["COQUI_TOS_AGREED"] = "1"


# --- Text-to-Speech Management Class ---
class TTSManager:
    """Handles model loading and TTS generation for all voice types."""
    def __init__(self, config: Config):
        self.config = config
        self.voice_model = None
        self.standard_models: Dict[str, Tacotron2 | HIFIGAN] = {}
        self._load_models()

    def _load_models(self):
        """Loads all required TTS models at startup."""
        if not MODELS_AVAILABLE:
            print("[TTS Manager] Cannot load models as required libraries are missing.")
            return

        print(f"[TTS Manager] Initializing models on device: {self.config.DEVICE}")
        try:
            print("[TTS Manager] Loading Coqui XTTS model for voice cloning...")
            self.voice_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.config.DEVICE)
            print("[TTS Manager] Coqui XTTS model loaded successfully.")
        except Exception as e:
            print(f"[TTS Manager] ERROR: Could not load Coqui TTS model: {e}")
            self.voice_model = None

        try:
            print("[TTS Manager] Loading SpeechBrain models for standard TTS...")
            tmp_tts = self.config.ROOT_DIR / "tmpdir_tts"
            tmp_vocoder = self.config.ROOT_DIR / "tmpdir_vocoder"
            self.standard_models['tacotron2'] = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech", savedir=tmp_tts
            )
            self.standard_models['hifi_gan'] = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech", savedir=tmp_vocoder
            )
            print("[TTS Manager] SpeechBrain models loaded successfully.")
        except Exception as e:
            print(f"[TTS Manager] ERROR: Could not load standard SpeechBrain models: {e}")
            self.standard_models = {}

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Converts numbers to words to improve TTS pronunciation."""
        return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

    def generate_speech(self, text: str, speaker_id: str) -> Path:
        """Generates speech from text using the selected voice and returns the path to the audio file."""
        if not text.strip():
            raise ValueError("Text input cannot be empty.")

        processed_text = self.preprocess_text(text)
        temp_wav_path = self.config.TEMP_DIR / f"tts_{uuid.uuid4()}.wav"

        if speaker_id == self.config.STANDARD_VOICE_NAME:
            if not self.standard_models:
                raise ValueError("Standard TTS models are not available. Please check the logs.")
            tacotron2 = self.standard_models['tacotron2']
            hifi_gan = self.standard_models['hifi_gan']

            mel_outputs, _, _ = tacotron2.encode_text(processed_text)
            waveforms = hifi_gan.decode_batch(mel_outputs)
            torchaudio.save(str(temp_wav_path), waveforms.squeeze(1), 22050)
        else:
            if not self.voice_model:
                raise ValueError("Voice cloning model (Coqui TTS) is not available. Please check logs.")

            reference_audio = self.config.VOICE_SAMPLES_DIR / speaker_id / "reference.wav"
            if not reference_audio.exists():
                raise ValueError(f"Speaker reference audio not found for ID '{speaker_id}'.")

            self.voice_model.tts_to_file(
                text=processed_text,
                file_path=str(temp_wav_path),
                speaker_wav=str(reference_audio),
                language="en",
                split_sentences=True,
            )

        if not temp_wav_path.exists():
            raise ValueError("TTS generation failed to produce an audio file.")

        return temp_wav_path


# --- Video Generation Class ---
class VideoGenerator:
    """Handles the creation of video from audio and text with zodiac-specific styling."""
    def __init__(self, config: Config, zodiac_colors: Dict[str, Tuple[int, int, int]]):
        self.config = config
        self.zodiac_colors = zodiac_colors
        self.available_fonts = self._discover_fonts()
        if not self.available_fonts:
            print("[Video Generator] WARNING: No system fonts found. Using default.")

    def _discover_fonts(self) -> List[Path]:
        """Finds common .ttf font files on the system."""
        font_paths = []
        system = platform.system()
        if system == "Windows":
            font_paths.append(Path("C:/Windows/Fonts"))
        elif system == "Darwin":  # macOS
            font_paths.append(Path("/System/Library/Fonts"))
            font_paths.append(Path("/Library/Fonts"))
        elif system == "Linux":
            font_paths.append(Path("/usr/share/fonts/truetype"))
            font_paths.append(Path.home() / ".fonts")

        discovered = []
        common_fonts = ["arial.ttf", "arialbd.ttf", "calibri.ttf", "times.ttf", "DejaVuSans.ttf"]
        for path in font_paths:
            if path.is_dir():
                for font_name in common_fonts:
                    if (path / font_name).exists():
                        discovered.append(path / font_name)
        return discovered

    def get_background_image(self, sign: str, size: Tuple[int, int]) -> Optional[Image.Image]:
        """Get zodiac-specific background image."""
        image_folder = self.config.ZODIAC_IMAGES_DIR / sign.lower()
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []

        if image_folder.exists():
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(image_folder, ext)))
                image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))

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
                print(f"Error loading background image {selected_image}: {e}")
        return None

    def _create_text_image(self, text: str, sign: str, size: Tuple[int, int] = (1280, 720)) -> Path:
        """Creates an image with text overlay, zodiac-specific background, and title."""
        background = self.get_background_image(sign, size)
        if background:
            img = background.copy()
        else:
            bg_color = self.zodiac_colors.get(sign, (74, 144, 226))
            img = Image.new('RGB', size, bg_color)

        draw = ImageDraw.Draw(img)
        margin = 60
        text_width = size[0] - (margin * 2)
        text_height = size[1] - (margin * 3)

        font_path = self.available_fonts[0] if self.available_fonts else None
        font_size = 80
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

        wrapped_text = textwrap.fill(text, width=30)
        while font_size > 20:
            bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= text_width * 0.9:
                break
            font_size -= 4
            font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2 + 30)

        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 20
        rect_coords = [
            text_pos[0] - padding,
            text_pos[1] - padding,
            text_pos[0] + text_w + padding,
            text_pos[1] + text_h + padding
        ]
        overlay_draw.rounded_rectangle(rect_coords, radius=15, fill=(0, 0, 0, 120))

        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        shadow_color, text_color = "black", "white"
        for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((text_pos[0] + offset[0], text_pos[1] + offset[1]), wrapped_text, font=font, fill=shadow_color, align="center")
        draw.text(text_pos, wrapped_text, font=font, fill=text_color, align="center")

        title_text = f"{sign.upper()} HOROSCOPE"
        title_size = min(60, font_size + 20)
        title_font = None
        for fp in self.available_fonts:
            if 'bold' in fp.lower() or 'Bold' in fp:
                try:
                    title_font = ImageFont.truetype(fp, title_size)
                    break
                except:
                    continue
        if not title_font:
            title_font = font

        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (size[0] - title_width) // 2
        title_y = 40

        title_bg_coords = [
            title_x - 20,
            title_y - 10,
            title_x + title_width + 20,
            title_y + title_size + 10
        ]
        draw.rounded_rectangle(title_bg_coords, radius=10, fill=(0, 0, 0, 150))

        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                if adj_x != 0 or adj_y != 0:
                    draw.text((title_x + adj_x, title_y + adj_y), title_text, font=title_font, fill="black")
        draw.text((title_x, title_y), title_text, font=title_font, fill=(255, 215, 0))

        image_path = self.config.TEMP_DIR / f"slide_{uuid.uuid4()}.png"
        img.save(image_path, quality=95)
        return image_path

    def create_video(self, audio_path: Path, full_text: str, sign: str) -> Path:
        """Orchestrates video creation from audio and text slides for a zodiac sign."""
        size = (1280, 720)
        print("Analyzing audio for sentence timing...")
        audio = AudioSegment.from_file(str(audio_path))

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
        if not sentences:
            sentences = [full_text]

        audio_chunks = split_on_silence(
            audio, min_silence_len=400, silence_thresh=audio.dBFS - 16, keep_silence=200
        )

        clips = []
        temp_image_paths = []
        num_segments = min(len(sentences), len(audio_chunks))

        for i in range(num_segments):
            sentence = sentences[i]
            chunk = audio_chunks[i]
            duration_sec = len(chunk) / 1000.0

            print(f"Creating slide {i + 1}/{num_segments}...")
            image_path = self._create_text_image(sentence, sign, size)
            temp_image_paths.append(image_path)

            chunk_path = self.config.TEMP_DIR / f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            audio_clip = AudioFileClip(str(chunk_path))

            video_clip = ImageClip(str(image_path), duration=duration_sec).set_audio(audio_clip)
            clips.append(video_clip)

        print("Assembling final video file...")
        if not clips:
            image_path = self._create_text_image(full_text, sign, size)
            temp_image_paths.append(image_path)
            final_clip = ImageClip(str(image_path), duration=audio.duration_seconds).set_audio(
                AudioFileClip(str(audio_path)))
        else:
            final_clip = concatenate_videoclips(clips)

        output_path = self.config.TEMP_DIR / f"video_{uuid.uuid4()}.mp4"
        final_clip.write_videofile(
            str(output_path), fps=24, codec='libx264', audio_codec='aac', logger=None
        )

        for path in temp_image_paths:
            path.unlink(missing_ok=True)
        for clip in clips:
            clip.close()
        final_clip.close()
        return output_path


# --- Main Horoscope Generator Class ---
class EnhancedHoroscopeGenerator:
    """Enhanced class for horoscope video generation with background images and thumbnail display."""
    def __init__(self):
        self.config = Config()
        self.zodiac_signs = [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]
        self.zodiac_colors = {
            "Aries": (255, 107, 107),
            "Taurus": (78, 205, 196),
            "Gemini": (69, 183, 209),
            "Cancer": (150, 206, 180),
            "Leo": (255, 234, 167),
            "Virgo": (221, 160, 221),
            "Libra": (152, 216, 200),
            "Scorpio": (247, 220, 111),
            "Sagittarius": (187, 143, 206),
            "Capricorn": (133, 193, 233),
            "Aquarius": (248, 196, 113),
            "Pisces": (130, 224, 170)
        }
        self.base_url = "https://horoscope-app-api.vercel.app/api/v1"
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config, self.zodiac_colors)
        self.available_voices = self._get_available_voices()

    def _get_available_voices(self) -> List[str]:
        """Get list of available voices from voice_samples directory."""
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)

    def _extract_horoscope_text(self, horoscope_data: Dict) -> str:
        """Extract horoscope text from API response."""
        horoscope_text = ""
        if 'data' in horoscope_data:
            data = horoscope_data['data']
            if 'horoscope_data' in data:
                horoscope_text = data['horoscope_data']
            elif 'content' in data:
                horoscope_text = data['content']
            elif isinstance(data, str):
                horoscope_text = data
        elif 'horoscope' in horoscope_data:
            horoscope_text = horoscope_data['horoscope']
        elif isinstance(horoscope_data, str):
            horoscope_text = horoscope_data
        return horoscope_text

    def get_daily_horoscope(self, sign: str, day: str = "TODAY") -> Optional[Dict]:
        """Get daily horoscope for a specific zodiac sign."""
        url = f"{self.base_url}/get-horoscope/daily"
        params = {"sign": sign, "day": day}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching daily horoscope for {sign}: {e}")
            return None

    def get_weekly_horoscope(self, sign: str) -> Optional[Dict]:
        """Get weekly horoscope for a specific zodiac sign."""
        url = f"{self.base_url}/get-horoscope/weekly"
        params = {"sign": sign}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weekly horoscope for {sign}: {e}")
            return None

    def get_monthly_horoscope(self, sign: str) -> Optional[Dict]:
        """Get monthly horoscope for a specific zodiac sign."""
        url = f"{self.base_url}/get-horoscope/monthly"
        params = {"sign": sign}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching monthly horoscope for {sign}: {e}")
            return None

    def generate_horoscope_audio(self, sign: str, horoscope_data: Dict, horoscope_type: str = "daily",
                                speaker_id: str = "Standard Voice (Non-Cloned)", output_dir: str = None) -> Optional[Tuple[str, str]]:
        """Generate audio for a horoscope and return audio path and full text."""
        if not horoscope_data:
            print(f"No horoscope data provided for {sign}")
            return None

        horoscope_text = self._extract_horoscope_text(horoscope_data)
        if not horoscope_text:
            print(f"Could not extract horoscope text for {sign}")
            return None

        intro_text = f"Here is your {horoscope_type} horoscope for {sign}. "
        full_text = intro_text + horoscope_text

        print(f"Generating audio for {sign} {horoscope_type} horoscope...")
        try:
            audio_wav_path = self.tts_manager.generate_speech(full_text, speaker_id)
            audio = AudioSegment.from_wav(audio_wav_path)
            temp_mp3_path = self.config.TEMP_DIR / f"output_{uuid.uuid4()}.mp3"
            audio.export(temp_mp3_path, format="mp3")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_text = re.sub(r'[^\w-]', '', full_text[:20]).strip()
                perm_filename = f"{sign}_{horoscope_type}_audio_{timestamp}.mp3"
                perm_path = Path(output_dir) / perm_filename
                shutil.copy(temp_mp3_path, perm_path)
                audio_wav_path.unlink(missing_ok=True)
                return str(perm_path), full_text
            else:
                audio_wav_path.unlink(missing_ok=True)
                return str(temp_mp3_path), full_text
        except Exception as e:
            print(f"Error generating audio for {sign}: {e}")
            return None

    def create_enhanced_video(self, sign: str, audio_path: str, full_text: str,
                             horoscope_type: str = "daily", output_dir: str = None) -> Optional[str]:
        """Create an enhanced video with text images synchronized to audio."""
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None

        try:
            if not output_dir:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = os.path.join(self.config.OUTPUT_DIR, f"horoscope_videos_{horoscope_type}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            video_path = self.video_generator.create_video(Path(audio_path), full_text, sign)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{sign}_{horoscope_type}_enhanced_{timestamp}.mp4"
            final_path = os.path.join(output_dir, output_filename)
            shutil.move(video_path, final_path)
            print(f"Enhanced video created successfully: {final_path}")
            return final_path
        except Exception as e:
            print(f"Error creating enhanced video for {sign}: {e}")
            return None

    def process_single_horoscope(self, sign: str, horoscope_type: str = "daily",
                                day: str = "TODAY", speaker_id: str = "Standard Voice (Non-Cloned)") -> Dict:
        """Process a single horoscope and generate enhanced video."""
        print(f"Processing {sign} {horoscope_type} horoscope...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(self.config.OUTPUT_DIR, f"single_horoscope_{horoscope_type}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        if horoscope_type == "daily":
            horoscope = self.get_daily_horoscope(sign, day)
        elif horoscope_type == "weekly":
            horoscope = self.get_weekly_horoscope(sign)
        elif horoscope_type == "monthly":
            horoscope = self.get_monthly_horoscope(sign)
        else:
            return {"error": "Invalid horoscope type"}

        if not horoscope:
            return {"error": "Could not fetch horoscope data"}

        audio_result = self.generate_horoscope_audio(sign, horoscope, horoscope_type, speaker_id, session_dir)
        if not audio_result:
            return {"error": "Failed to generate audio"}

        audio_path, full_text = audio_result
        final_video_path = self.create_enhanced_video(sign, audio_path, full_text, horoscope_type, session_dir)

        return {
            "horoscope_data": horoscope,
            "audio_path": audio_path,
            "video_path": final_video_path,
            "full_text": full_text,
            "output_directory": session_dir,
            "success": final_video_path is not None
        }

    def process_all_horoscopes(self, horoscope_type: str = "daily", day: str = "TODAY",
                              speaker_id: str = "Standard Voice (Non-Cloned)") -> Dict[str, Dict]:
        """Process all zodiac signs and generate enhanced videos."""
        all_data = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(self.config.OUTPUT_DIR, f"batch_horoscope_videos_{horoscope_type}_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        print(f"Processing all {horoscope_type} horoscopes...")
        print(f"Output directory: {batch_dir}")

        for sign in self.zodiac_signs:
            print(f"\n{'=' * 50}")
            print(f"Processing {sign}...")
            print(f"{'=' * 50}")

            sign_dir = os.path.join(batch_dir, sign.lower())
            os.makedirs(sign_dir, exist_ok=True)

            if horoscope_type == "daily":
                horoscope = self.get_daily_horoscope(sign, day)
            elif horoscope_type == "weekly":
                horoscope = self.get_weekly_horoscope(sign)
            elif horoscope_type == "monthly":
                horoscope = self.get_monthly_horoscope(sign)
            else:
                all_data[sign] = {"error": "Invalid horoscope type"}
                continue

            if not horoscope:
                all_data[sign] = {"error": "Could not fetch horoscope data"}
                continue

            audio_result = self.generate_horoscope_audio(sign, horoscope, horoscope_type, speaker_id, sign_dir)
            if not audio_result:
                all_data[sign] = {"error": "Failed to generate audio"}
                continue

            audio_path, full_text = audio_result
            final_video_path = self.create_enhanced_video(sign, audio_path, full_text, horoscope_type, sign_dir)

            all_data[sign] = {
                "horoscope_data": horoscope,
                "audio_path": audio_path,
                "video_path": final_video_path,
                "full_text": full_text,
                "output_directory": sign_dir,
                "success": final_video_path is not None
            }

        return all_data

    def get_images_info(self) -> Tuple[str, List[str]]:
        """Get information about background images and return thumbnails."""
        if not self.config.ZODIAC_IMAGES_DIR.exists():
            return "❌ Zodiac images directory not found. Please create 'images/' folder.", []

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        image_files = {}
        total_images = 0

        # Collect images for each zodiac sign
        for sign in self.zodiac_signs:
            image_folder = self.config.ZODIAC_IMAGES_DIR / sign.lower()
            if image_folder.exists():
                image_files[sign] = []
                for ext in image_extensions:
                    image_files[sign].extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
                    image_files[sign].extend(glob.glob(os.path.join(image_folder, f"*{ext.upper()}")))
                total_images += len(image_files[sign])

        if not total_images:
            return f"📁 Zodiac images directory exists but no images found.\nSupported formats: {', '.join(image_extensions)}", []

        # Generate thumbnails
        thumbnails = []
        thumbnail_size = (100, 100)  # Thumbnail dimensions
        for sign in image_files:
            for image_path in image_files[sign][:3]:  # Limit to 3 thumbnails per sign for performance
                try:
                    img = Image.open(image_path)
                    img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    thumbnail_path = self.config.TEMP_DIR / f"thumbnail_{sign}_{os.path.basename(image_path)}"
                    img.save(thumbnail_path, quality=85)
                    thumbnails.append(str(thumbnail_path))
                except Exception as e:
                    print(f"Error generating thumbnail for {image_path}: {e}")

        info_text = f"✅ Found {total_images} background images across {len(image_files)} zodiac signs:\n\n"
        for sign in image_files:
            count = len(image_files[sign])
            info_text += f"**{sign}**: {count} image{'s' if count != 1 else ''}\n"

        return info_text, thumbnails


# --- Gradio UI Setup ---
def setup_ui(generator: EnhancedHoroscopeGenerator):
    """Creates and launches the Gradio user interface."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
                   title="Zodiac Horoscope Video Generator") as demo:
        gr.Markdown("# 🗣️ Zodiac Horoscope Video Generator")
        gr.Markdown("Generate horoscope videos with synchronized audio and text slides, and view available background images.")

        with gr.Tabs():
            with gr.TabItem("🚀 Generate Horoscope"):
                with gr.Row():
                    with gr.Column(scale=2):
                        zodiac_sign = gr.Dropdown(label="Select Zodiac Sign", choices=generator.zodiac_signs)
                        horoscope_type = gr.Dropdown(label="Horoscope Type", choices=["daily", "weekly", "monthly"], value="daily")
                        speaker_dropdown = gr.Dropdown(label="Select Speaker", choices=generator.available_voices,
                                                      value=generator.config.STANDARD_VOICE_NAME)
                        generate_button = gr.Button("Generate Horoscope Video", variant="primary")

                    with gr.Column(scale=1):
                        audio_output = gr.Audio(label="Generated Audio")
                        video_output = gr.Video(label="Generated Video")
                        status_output = gr.Markdown()

            with gr.TabItem("🖼️ Background Images"):
                gr.Markdown("## Manage Zodiac Background Images")
                gr.Markdown(
                    "View available background images for each zodiac sign in the 'images/zodiac_chosen/<sign>' folders. "
                    "Supported formats: JPG, PNG, BMP, GIF, TIFF, WebP"
                )
                refresh_images_button = gr.Button("🔄 Check Images", variant="secondary")
                images_info_display = gr.Markdown(value="Click 'Check Images' to see available background images.")
                thumbnails_display = gr.Gallery(label="Image Thumbnails", columns=4, height="auto")

        def generate_horoscope(sign, h_type, speaker):
            result = generator.process_single_horoscope(sign, h_type, speaker_id=speaker)
            if result.get("success"):
                return result["audio_path"], result["video_path"], f"✅ Video created: {result['video_path']}"
            return None, None, f"❌ Error: {result.get('error', 'Unknown error')}"

        generate_button.click(
            fn=generate_horoscope,
            inputs=[zodiac_sign, horoscope_type, speaker_dropdown],
            outputs=[audio_output, video_output, status_output]
        )

        refresh_images_button.click(
            fn=generator.get_images_info,
            outputs=[images_info_display, thumbnails_display]
        )

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\nApplication cannot start because essential libraries (TTS, speechbrain) are missing.")
        print("Please run: pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio")
    else:
        generator = EnhancedHoroscopeGenerator()
        setup_ui(generator)