#!/usr/bin/env python3
"""
Text-to-Video Generator with Per-Sentence Audio Synchronization
Creates videos with perfectly synchronized audio by generating each sentence separately
"""

import os
import re
import glob
import random
import shutil
import uuid
import platform
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

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
                split_sentences=False,  # We handle sentence splitting ourselves
            )

        if not temp_wav_path.exists():
            raise ValueError("TTS generation failed.")

        return temp_wav_path


class VideoGenerator:
    """Creates video from audio and text with synchronized timing."""

    def __init__(self, config: Config):
        self.config = config
        self.available_fonts = self._discover_fonts()
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
        common_fonts = ["arial.ttf", "arialbd.ttf", "calibri.ttf", "times.ttf", "DejaVuSans.ttf"]
        for path in font_paths:
            if path.is_dir():
                for font_name in common_fonts:
                    if (path / font_name).exists():
                        discovered.append(path / font_name)
        return discovered

    def get_background_image(self, size: Tuple[int, int]) -> Optional[Image.Image]:
        """Get a random background image from the images directory."""
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
                print(f"Error loading background: {e}")
        return None

    def _create_text_image(self, text: str, size: Tuple[int, int] = (1280, 720),
                           bg_color: Tuple[int, int, int] = (74, 144, 226)) -> Path:
        """Create an image with text overlay and background."""
        background = self.get_background_image(size)
        img = background.copy() if background else Image.new('RGB', size, bg_color)

        draw = ImageDraw.Draw(img)
        margin = 60
        text_width = size[0] - (margin * 2)

        font_path = self.available_fonts[0] if self.available_fonts else None
        font_size = 80
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

        wrapped_text = textwrap.fill(text, width=30)

        # Auto-size font to fit
        while font_size > 20:
            bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= text_width * 0.9:
                break
            font_size -= 4
            font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2)

        # Create semi-transparent background for text
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 20
        rect_coords = [
            text_pos[0] - padding, text_pos[1] - padding,
            text_pos[0] + text_w + padding, text_pos[1] + text_h + padding
        ]
        overlay_draw.rounded_rectangle(rect_coords, radius=15, fill=(0, 0, 0, 140))

        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Draw text with shadow
        for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((text_pos[0] + offset[0], text_pos[1] + offset[1]),
                      wrapped_text, font=font, fill="black", align="center")
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
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)  # Handle initials

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
                                  bg_color: Tuple[int, int, int] = (74, 144, 226)) -> Path:
        """Create video with one slide per sentence, synchronized with its audio."""
        size = (1280, 720)
        clips = []
        temp_image_paths = []

        print(f"\nCreating {len(sentences)} synchronized slides...")

        for i, (sentence, audio_path) in enumerate(zip(sentences, audio_paths)):
            # Get audio duration
            audio_segment = AudioSegment.from_file(str(audio_path))
            duration_sec = len(audio_segment) / 1000.0

            print(f"  Slide {i + 1}/{len(sentences)}: {duration_sec:.2f}s - '{sentence[:50]}...'")

            # Create image for this sentence
            image_path = self._create_text_image(sentence, size, bg_color)
            temp_image_paths.append(image_path)

            # Create video clip with exact audio duration
            audio_clip = AudioFileClip(str(audio_path))
            video_clip = ImageClip(str(image_path), duration=duration_sec).set_audio(audio_clip)
            clips.append(video_clip)

        if not clips:
            raise ValueError("No clips were created")

        print("\nAssembling final video...")
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
        for path in temp_image_paths:
            path.unlink(missing_ok=True)
        for clip in clips:
            clip.close()
        final_clip.close()

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
                       bg_color: Tuple[int, int, int] = (74, 144, 226)) -> Dict:
        """Generate video from text with per-sentence audio synchronization."""
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        audio_paths = []

        try:
            # Split text into sentences
            print("Splitting text into sentences...")
            sentences = self.video_generator.split_into_sentences(text)
            print(f"Found {len(sentences)} sentences")

            # Generate audio for each sentence
            print("\nGenerating audio for each sentence...")
            for i, sentence in enumerate(sentences):
                print(f"  Sentence {i + 1}/{len(sentences)}: '{sentence[:50]}...'")
                audio_path = self.tts_manager.generate_speech(sentence, speaker_id)
                audio_paths.append(audio_path)

            # Combine all audio files for the final MP3
            print("\nCombining audio files...")
            combined_audio = AudioSegment.empty()
            for audio_path in audio_paths:
                segment = AudioSegment.from_wav(audio_path)
                combined_audio += segment

            audio_mp3_path = session_dir / f"audio_{timestamp}.mp3"
            combined_audio.export(audio_mp3_path, format="mp3")

            # Generate video with per-sentence synchronization
            print("\nCreating video with perfect per-sentence synchronization...")
            video_temp_path = self.video_generator.create_video_per_sentence(
                sentences, audio_paths, bg_color
            )
            video_final_path = session_dir / f"video_{timestamp}.mp4"
            shutil.move(video_temp_path, video_final_path)

            # Cleanup temporary audio files
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)

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
            # Cleanup on error
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)
            return {"error": str(e), "success": False}


def setup_ui(generator: TextToVideoGenerator):
    """Create and launch Gradio UI."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   title="Text-to-Video Generator") as demo:
        gr.Markdown("# 🎬 Text-to-Video Generator with Perfect Audio Sync")
        gr.Markdown(
            "Convert your text into a video with **perfectly synchronized** audio. "
            "Each sentence gets its own slide with precisely matched audio duration."
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter Your Text",
                    placeholder="Type or paste your text here. Each sentence will become a separate slide...",
                    lines=8,
                    max_lines=15
                )
                speaker_dropdown = gr.Dropdown(
                    label="Select Voice",
                    choices=generator.available_voices,
                    value=generator.config.STANDARD_VOICE_NAME
                )
                bg_color_picker = gr.ColorPicker(
                    label="Background Color (if no image)",
                    value="#4A90E2"
                )
                generate_button = gr.Button("🎬 Generate Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Audio")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown(
            "**How it works:**\n"
            "1. Your text is split into sentences\n"
            "2. Audio is generated for each sentence individually\n"
            "3. Each sentence gets its own slide\n"
            "4. Slides are shown for exactly as long as their audio plays\n"
            "5. All slides are combined into one synchronized video\n\n"
            "**Tip:** Place background images in `background_images/` folder for custom backgrounds. "
            "Supported formats: JPG, PNG, BMP, GIF."
        )

        def generate_video_wrapper(text, speaker, bg_color_hex):
            if not text or not text.strip():
                return None, None, "❌ Error: Please enter some text"

            # Convert hex color to RGB
            bg_color_hex = bg_color_hex.lstrip('#')
            bg_color = tuple(int(bg_color_hex[i:i + 2], 16) for i in (0, 2, 4))

            result = generator.generate_video(text, speaker, bg_color)

            if result.get("success"):
                return (
                    result["audio_path"],
                    result["video_path"],
                    f"✅ Video created successfully!\n\n"
                    f"**Sentences processed:** {result['sentence_count']}\n\n"
                    f"**Output Directory:** `{result['output_directory']}`"
                )
            return None, None, f"❌ Error: {result.get('error', 'Unknown error')}"

        generate_button.click(
            fn=generate_video_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color_picker],
            outputs=[audio_output, video_output, status_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)


if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\n❌ Missing required libraries. Please install:")
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio")
    else:
        print("\n🎬 Starting Text-to-Video Generator...")
        generator = TextToVideoGenerator()
        setup_ui(generator)