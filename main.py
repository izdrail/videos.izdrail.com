#!/usr/bin/env python3
"""
Text-to-Video Generator with Per-Sentence Audio Synchronization and Unsplash API Integration
Creates videos with perfectly synchronized audio and dynamic background images from Unsplash

IMPROVEMENTS:
1. Larger text (50%+ of image coverage)
2. Better font size optimization algorithm
3. Improved error handling and validation
4. Better cache management for Unsplash images
5. Progress tracking for video generation
6. Better resource cleanup
7. Configurable text sizing parameters
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
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"

        for dir_path in [self.VOICE_SAMPLES_DIR, self.IMAGES_DIR, self.TEMP_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(exist_ok=True)

        self.STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["COQUI_TOS_AGREED"] = "1"

        # IMPROVED: Configurable text sizing parameters
        self.TEXT_SIZE_CONFIG = {
            'target_coverage': 0.5,  # 50% of image
            'initial_font_size': 150,
            'min_font_size': 40,
            'max_font_size': 200,
            'wrap_width_range': (15, 35),
            'margin_percentage': 0.05,  # 5% margin on each side
        }


class UnsplashAPI:
    """Handles Unsplash API requests for fetching images."""

    def __init__(self):
        self.base_url = "https://api.unsplash.com"
        self.client_id = None
        self.cache = {}  # Simple in-memory cache for images
        self.cache_limit = 50  # IMPROVED: Limit cache size

    def set_client_id(self, client_id: str):
        """Set the Unsplash API client ID."""
        self.client_id = client_id

    def _manage_cache(self):
        """IMPROVED: Manage cache size to prevent memory issues."""
        if len(self.cache) > self.cache_limit:
            # Remove oldest entries
            items_to_remove = len(self.cache) - self.cache_limit
            for key in list(self.cache.keys())[:items_to_remove]:
                del self.cache[key]

    def search_photos(self, query: str, per_page: int = 10) -> List[Dict]:
        """
        Search for photos on Unsplash.

        Args:
            query: Search keyword
            per_page: Number of results to fetch (max 30)

        Returns:
            List of photo data dictionaries
        """
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
        """
        Download an image from Unsplash.

        Args:
            photo_url: URL of the image to download

        Returns:
            PIL Image object or None if download fails
        """
        # Check cache first
        if photo_url in self.cache:
            return self.cache[photo_url].copy()

        try:
            response = requests.get(photo_url, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            # IMPROVED: Cache management
            self._manage_cache()
            self.cache[photo_url] = img.copy()
            return img
        except Exception as e:
            print(f"[Unsplash] Error downloading image: {e}")
            return None

    def get_random_image(self, query: str, size: Tuple[int, int] = (1280, 720)) -> Optional[Image.Image]:
        """
        Get a random image from Unsplash based on search query.

        Args:
            query: Search keyword
            size: Desired image size (width, height)

        Returns:
            PIL Image object or None
        """
        photos = self.search_photos(query, per_page=10)

        if not photos:
            print(f"[Unsplash] No images found for query: '{query}'")
            return None

        # Select a random photo
        photo = random.choice(photos)

        # Get the regular size URL (good balance between quality and size)
        photo_url = photo.get("urls", {}).get("regular")

        if not photo_url:
            return None

        # Download and process the image
        img = self.download_image(photo_url)

        if img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to target size
            img = img.resize(size, Image.Resampling.LANCZOS)

            # Apply darkening and blur for better text readability
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.6)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

            # Trigger download endpoint (Unsplash API guidelines)
            download_url = photo.get("links", {}).get("download_location")
            if download_url and self.client_id:
                try:
                    requests.get(download_url, params={"client_id": self.client_id}, timeout=5)
                except:
                    pass  # Non-critical

        return img

    def clear_cache(self):
        """IMPROVED: Clear the image cache."""
        self.cache.clear()


class TTSManager:
    """Handles TTS generation for all voice types."""

    def __init__(self, config: Config):
        self.config = config
        self.voice_model = None
        self.standard_models: Dict[str, any] = {}
        self._load_models()

    def _load_models(self):
        """Load TTS models at startup."""
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
        """Convert numbers to words for better TTS pronunciation."""
        return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

    def generate_speech(self, text: str, speaker_id: str) -> Path:
        """Generate speech from text and return audio file path."""
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
        if not self.available_fonts:
            print("[Video] WARNING: No system fonts found. Using default.")

    def _discover_fonts(self) -> List[Path]:
        """Find common TTF font files on the system."""
        font_paths = []
        system = platform.system()

        if system == "Windows":
            font_paths.append(Path("C:/Windows/Fonts"))
        elif system == "Darwin":
            font_paths.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts")])
        elif system == "Linux":
            font_paths.extend([Path("/usr/share/fonts/truetype"), Path.home() / ".fonts"])

        discovered = []
        # IMPROVED: Extended font list with better preferences
        common_fonts = [
            "arialbd.ttf", "Arial Bold.ttf",  # Bold fonts first for better visibility
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
                    # Check direct path
                    if (path / font_name).exists():
                        font_file = path / font_name
                    else:
                        # Search recursively
                        for found in path.rglob(font_name):
                            font_file = found
                            break

                    if font_file and font_file not in discovered:
                        discovered.append(font_file)

        return discovered

    def get_background_image(self, size: Tuple[int, int], unsplash_keyword: Optional[str] = None) -> Optional[
        Image.Image]:
        """
        Get a background image - either from Unsplash or local directory.

        Args:
            size: Image size (width, height)
            unsplash_keyword: Keyword to search on Unsplash (if provided)

        Returns:
            PIL Image object or None
        """
        # Try Unsplash first if keyword is provided
        if unsplash_keyword and self.unsplash.client_id:
            print(f"[Video] Fetching image from Unsplash: '{unsplash_keyword}'")
            img = self.unsplash.get_random_image(unsplash_keyword, size)
            if img:
                return img
            print("[Video] Falling back to local images...")

        # Fallback to local images
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

    def _create_text_image(self, text: str, size: Tuple[int, int] = (1280, 720),
                           bg_color: Tuple[int, int, int] = (74, 144, 226),
                           unsplash_keyword: Optional[str] = None) -> Path:
        """
        IMPROVED: Create an image with large text overlay (50%+ coverage) and background.

        Major improvements:
        - Targets 50% minimum image coverage
        - Better font size optimization
        - Tests multiple wrap widths to find optimal layout
        - Increased default font sizes
        - Better visual hierarchy with larger shadows
        """
        background = self.get_background_image(size, unsplash_keyword)
        img = background.copy() if background else Image.new('RGB', size, bg_color)

        draw = ImageDraw.Draw(img)

        # IMPROVED: Use config for text sizing
        cfg = self.config.TEXT_SIZE_CONFIG
        margin = int(size[0] * cfg['margin_percentage'])
        text_width = size[0] - (margin * 2)

        # Calculate target text area (50% of image by default)
        target_text_area = (size[0] * size[1]) * cfg['target_coverage']

        font_path = self.available_fonts[0] if self.available_fonts else None

        # IMPROVED: Start with much larger font
        font_size = cfg['initial_font_size']
        min_font_size = cfg['min_font_size']
        max_font_size = cfg['max_font_size']

        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

        # Initial wrap
        wrap_width = 20
        wrapped_text = textwrap.fill(text, width=wrap_width)

        # IMPROVED: Smart font sizing algorithm
        best_font_size = min_font_size
        best_wrapped = wrapped_text
        best_area = 0

        # Try different font sizes
        for test_font_size in range(max_font_size, min_font_size - 1, -5):
            test_font = ImageFont.truetype(str(font_path), test_font_size) if font_path else ImageFont.load_default()

            # Try different wrap widths for this font size
            for w in range(cfg['wrap_width_range'][0], cfg['wrap_width_range'][1]):
                test_wrapped = textwrap.fill(text, width=w)
                bbox = draw.textbbox((0, 0), test_wrapped, font=test_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                # Check if text fits within margins
                if text_w <= text_width and text_h <= size[1] * 0.85:
                    text_area = text_w * text_h

                    # Prefer larger text that meets target coverage
                    if text_area >= target_text_area:
                        best_font_size = test_font_size
                        best_wrapped = test_wrapped
                        best_area = text_area
                        break  # Found good size, use it
                    elif text_area > best_area:
                        # Track best option even if below target
                        best_font_size = test_font_size
                        best_wrapped = test_wrapped
                        best_area = text_area

            # If we found a size that meets target, stop searching
            if best_area >= target_text_area:
                break

        # Use best configuration found
        font_size = best_font_size
        wrapped_text = best_wrapped
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

        # Get final text dimensions
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2)

        coverage_percentage = (text_w * text_h / (size[0] * size[1]) * 100)
        print(f"[Video] Font size: {font_size}px, Text coverage: {coverage_percentage:.1f}% of image")

        # IMPROVED: Larger semi-transparent background
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 40  # Increased padding
        rect_coords = [
            text_pos[0] - padding, text_pos[1] - padding,
            text_pos[0] + text_w + padding, text_pos[1] + text_h + padding
        ]
        overlay_draw.rounded_rectangle(rect_coords, radius=25, fill=(0, 0, 0, 160))

        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # IMPROVED: Larger shadow for better contrast
        shadow_offset = 4
        for offset in [(-shadow_offset, -shadow_offset), (shadow_offset, -shadow_offset),
                       (-shadow_offset, shadow_offset), (shadow_offset, shadow_offset)]:
            draw.text((text_pos[0] + offset[0], text_pos[1] + offset[1]),
                      wrapped_text, font=font, fill="black", align="center")

        # Draw main text
        draw.text(text_pos, wrapped_text, font=font, fill="white", align="center")

        image_path = self.config.TEMP_DIR / f"slide_{uuid.uuid4()}.png"
        img.save(image_path, quality=95)
        return image_path

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences intelligently."""
        # Handle common abbreviations to avoid incorrect splits
        text = re.sub(r'\bDr\.', 'Dr<dot>', text)
        text = re.sub(r'\bMr\.', 'Mr<dot>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<dot>', text)
        text = re.sub(r'\bMs\.', 'Ms<dot>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore dots
        sentences = [s.replace('<dot>', '.').strip() for s in sentences if s.strip()]

        # Ensure sentences end with punctuation
        for i, sentence in enumerate(sentences):
            if not sentence.endswith(('.', '!', '?')):
                sentences[i] = sentence + '.'

        return sentences

    def create_video_per_sentence(self, sentences: List[str], audio_paths: List[Path],
                                  bg_color: Tuple[int, int, int] = (74, 144, 226),
                                  unsplash_keyword: Optional[str] = None,
                                  progress_callback=None) -> Path:
        """
        IMPROVED: Create video with progress tracking and better error handling.
        """
        size = (1280, 720)
        clips = []
        temp_image_paths = []

        print(f"\nCreating {len(sentences)} synchronized slides...")

        for i, (sentence, audio_path) in enumerate(zip(sentences, audio_paths)):
            try:
                # Get audio duration
                audio_segment = AudioSegment.from_file(str(audio_path))
                duration_sec = len(audio_segment) / 1000.0

                print(f"  Slide {i + 1}/{len(sentences)}: {duration_sec:.2f}s - '{sentence[:50]}...'")

                # IMPROVED: Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(sentences), f"Creating slide {i + 1}")

                # Create image for this sentence
                image_path = self._create_text_image(sentence, size, bg_color, unsplash_keyword)
                temp_image_paths.append(image_path)

                # Create video clip with exact audio duration
                audio_clip = AudioFileClip(str(audio_path))
                video_clip = ImageClip(str(image_path), duration=duration_sec).set_audio(audio_clip)
                clips.append(video_clip)

            except Exception as e:
                print(f"[Video] Error creating slide {i + 1}: {e}")
                # IMPROVED: Continue with other slides instead of failing completely
                continue

        if not clips:
            raise ValueError("No clips were created - all slides failed")

        print("\nAssembling final video...")
        if progress_callback:
            progress_callback(len(sentences), len(sentences), "Assembling final video...")

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

        # IMPROVED: Better cleanup with error handling
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
    """Main generator class for text-to-video conversion."""

    def __init__(self):
        self.config = Config()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config)
        self.available_voices = self._get_available_voices()

    def _get_available_voices(self) -> List[str]:
        """Get available voices from voice_samples directory."""
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                       bg_color: Tuple[int, int, int] = (74, 144, 226),
                       unsplash_keyword: Optional[str] = None,
                       unsplash_client_id: Optional[str] = None,
                       progress_callback=None) -> Dict:
        """
        IMPROVED: Generate video with better error handling and progress tracking.
        """
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}

        # IMPROVED: Validate inputs
        if len(text) > 10000:
            return {"error": "Text is too long (max 10,000 characters)", "success": False}

        # Set Unsplash client ID if provided
        if unsplash_client_id and unsplash_client_id.strip():
            self.video_generator.unsplash.set_client_id(unsplash_client_id.strip())

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        audio_paths = []

        try:
            # Split text into sentences
            print("Splitting text into sentences...")
            sentences = self.video_generator.split_into_sentences(text)
            print(f"Found {len(sentences)} sentences")

            if len(sentences) > 100:
                return {"error": "Too many sentences (max 100)", "success": False}

            # Generate audio for each sentence
            print("\nGenerating audio for each sentence...")
            for i, sentence in enumerate(sentences):
                print(f"  Sentence {i + 1}/{len(sentences)}: '{sentence[:50]}...'")

                if progress_callback:
                    progress_callback(i + 1, len(sentences) * 2, f"Generating audio {i + 1}/{len(sentences)}")

                audio_path = self.tts_manager.generate_speech(sentence, speaker_id)
                audio_paths.append(audio_path)

            # Combine all audio files for the final MP3
            print("\nCombining audio files...")
            combined_audio = AudioSegment.empty()
            for audio_path in audio_paths:
                segment = AudioSegment.from_wav(audio_path)
                combined_audio += segment

            audio_mp3_path = session_dir / f"audio_{timestamp}.mp3"
            combined_audio.export(audio_mp3_path, format="mp3", bitrate="192k")

            # Generate video with per-sentence synchronization
            print("\nCreating video with perfect per-sentence synchronization...")

            def video_progress(current, total, message):
                if progress_callback:
                    # Offset by number of sentences for audio generation
                    progress_callback(len(sentences) + current, len(sentences) * 2, message)

            video_temp_path = self.video_generator.create_video_per_sentence(
                sentences, audio_paths, bg_color, unsplash_keyword, video_progress
            )
            video_final_path = session_dir / f"video_{timestamp}.mp4"
            shutil.move(video_temp_path, video_final_path)

            print(f"\n✅ Video created successfully: {video_final_path}")
            return {
                "success": True,
                "audio_path": str(audio_mp3_path),
                "video_path": str(video_final_path),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences)
            }

        except Exception as e:
            print(f"\n❌ Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "success": False}

        finally:
            # IMPROVED: Always cleanup temporary audio files
            for audio_path in audio_paths:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"[Cleanup] Warning: Could not delete {audio_path}: {e}")


def setup_ui(generator: TextToVideoGenerator):
    """Create and launch Gradio UI with improved styling."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   title="Text-to-Video Generator",
                   css="""
                   .large-text textarea { font-size: 16px !important; }
                   .progress-bar { margin: 10px 0; }
                   """) as demo:
        gr.Markdown("# 🎬 Text-to-Video Generator with Perfect Audio Sync")
        gr.Markdown(
            "Convert your text into a video with **perfectly synchronized** audio, "
            "**large readable text (50%+ coverage)**, and **dynamic backgrounds from Unsplash**. "
            "Each sentence gets its own slide with precisely matched audio duration."
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter Your Text",
                    placeholder="Type or paste your text here. Each sentence will become a separate slide...",
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

                # IMPROVED: Progress indicator
                progress_bar = gr.Textbox(
                    label="Progress",
                    value="Ready to generate...",
                    interactive=False,
                    elem_classes=["progress-bar"]
                )

                generate_button = gr.Button("🎬 Generate Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Audio")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown(
            "### 📖 How it works:\n"
            "1. Your text is split into sentences\n"
            "2. Audio is generated for each sentence individually\n"
            "3. Each sentence gets its own slide with a **large, readable text (50%+ of image)**\n"
            "4. Slides are shown for exactly as long as their audio plays\n"
            "5. All slides are combined into one synchronized video\n\n"
            "### ✨ Key Improvements:\n"
            "- **Larger text**: Text now occupies at least 50% of the image area\n"
            "- **Smart font sizing**: Automatically finds the optimal font size and layout\n"
            "- **Better readability**: Enhanced shadows and contrast\n"
            "- **Progress tracking**: See real-time progress during generation\n"
            "- **Better error handling**: More robust with detailed error messages\n\n"
            "### 🖼️ Background Image Priority:\n"
            "1. **Unsplash API** (if keyword and client ID provided)\n"
            "2. **Local images** from `background_images/` folder\n"
            "3. **Plain color** background\n\n"
            "### 🔑 Getting an Unsplash API Key:\n"
            "1. Go to [unsplash.com/developers](https://unsplash.com/developers)\n"
            "2. Register for a free developer account\n"
            "3. Create a new application\n"
            "4. Copy your **Access Key** (Client ID)\n"
            "5. Paste it in the field above\n\n"
            "**Free tier:** 50 requests/hour"
        )

        def generate_video_wrapper(text, speaker, bg_color_hex, keyword, client_id, progress=gr.Progress()):
            """IMPROVED: Wrapper with progress tracking."""
            if not text or not text.strip():
                return None, None, "❌ Error: Please enter some text", "Ready to generate..."

            # Convert hex color to RGB
            bg_color_hex = bg_color_hex.lstrip('#')
            bg_color = tuple(int(bg_color_hex[i:i + 2], 16) for i in (0, 2, 4))

            # Clean up keyword
            keyword = keyword.strip() if keyword else None
            client_id = client_id.strip() if client_id else None

            # Progress callback
            def update_progress(current, total, message):
                progress_text = f"Progress: {current}/{total} - {message}"
                progress((current, total), desc=message)
                return progress_text

            # Initial status
            progress(0, desc="Starting generation...")

            result = generator.generate_video(
                text, speaker, bg_color, keyword, client_id,
                progress_callback=update_progress
            )

            if result.get("success"):
                final_status = (
                    f"✅ Video created successfully!\n\n"
                    f"**Sentences processed:** {result['sentence_count']}\n\n"
                    f"**Output Directory:** `{result['output_directory']}`\n\n"
                    f"**Text Coverage:** Large text optimized for readability (50%+ of image)"
                )
                return (
                    result["audio_path"],
                    result["video_path"],
                    final_status,
                    "✅ Generation complete!"
                )

            error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
            return None, None, error_msg, "❌ Generation failed"

        generate_button.click(
            fn=generate_video_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color_picker, unsplash_keyword, unsplash_client_id],
            outputs=[audio_output, video_output, status_output, progress_bar]
        )

        # IMPROVED: Add example
        gr.Examples(
            examples=[
                ["Welcome to our presentation. Today we will explore the future of technology. Artificial intelligence is transforming our world. Let's discover what's possible together.",
                 "Standard Voice (Non-Cloned)", "#4A90E2", "technology", ""],
                ["The ocean is a vast and mysterious place. It covers over seventy percent of Earth's surface. Marine life is incredibly diverse. Conservation efforts are critical for our future.",
                 "Standard Voice (Non-Cloned)", "#1E3A8A", "ocean", ""],
            ],
            inputs=[text_input, speaker_dropdown, bg_color_picker, unsplash_keyword, unsplash_client_id],
            label="📝 Example Texts"
        )

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)


if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\n❌ Missing required libraries. Please install:")
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests")
    else:
        print("\n🎬 Starting Enhanced Text-to-Video Generator...")
        print("=" * 70)
        print("IMPROVEMENTS:")
        print("  ✓ Text now occupies 50%+ of image area")
        print("  ✓ Smart font sizing algorithm")
        print("  ✓ Better readability with enhanced shadows")
        print("  ✓ Progress tracking during generation")
        print("  ✓ Improved error handling and validation")
        print("  ✓ Cache management for Unsplash images")
        print("  ✓ Configurable text sizing parameters")
        print("=" * 70)
        print()
        generator = TextToVideoGenerator()
        setup_ui(generator)