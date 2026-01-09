import os
import re
import glob
import random
import shutil
import uuid
import platform
import requests
import sqlite3
import hashlib
import numpy as np
import traceback
import threading
import textwrap
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import torch
import torchaudio
import gradio as gr
from moviepy.editor import (
    AudioFileClip, ImageSequenceClip, ImageClip, VideoFileClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip, vfx
)
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
from dotenv import load_dotenv

# Import core modules
from core.utils.pytorch_compat import setup_pytorch_allowlist
from core.config import Config
from core.database import DB
from core.media.manager import MediaManager
from core.nlp.keyword_extractor import KeywordExtractor
from core.ai.stable_diffusion import StableDiffusionManager, SD_AVAILABLE
from core.tts.manager import TTSManager
from core.utils.audio import improve_audio_quality, remove_metallic_artifacts

# Availability flags
MODELS_AVAILABLE = True # Assumed true since imports above succeeded
SPACY_AVAILABLE = True  # Used in main block

# Global setups
setup_pytorch_allowlist()
load_dotenv()

# Enforce CPU globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)

# Fix PIL.ANTIALIAS deprecation
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# Initialize global DB derived from core.database
# (DB already initialized in core.database)

# DATABASE Setup handled by core.database

# =============== MEDIA SOURCE CACHE ===============
_media_source_cache = {}

# Stable Diffusion handled by core.ai

# Keyword Extraction handled by core.nlp

# Config and APIs handled by core.config and core.media

# TTS Management handled by core.tts

# =============== VIDEO EFFECTS, CIRCLE OVERLAY ===============
class VideoEffectsManager:
    @staticmethod
    def apply_ken_burns(clip: ImageClip, duration: float, direction: str = "zoom_in") -> ImageClip:
        def zoom_effect(t):
            if direction == "zoom_in":
                return 1 + 0.1 * (t / duration)
            elif direction == "zoom_out":
                return 1.1 - 0.1 * (t / duration)
            else:
                if t < duration / 2:
                    return 1 + 0.1 * (t / (duration / 2))
                else:
                    return 1.1 - 0.1 * ((t - duration / 2) / (duration / 2))
        return clip.resize(zoom_effect)

    @staticmethod
    def apply_pan(clip: ImageClip, duration: float, direction: str = "left") -> ImageClip:
        w, h = clip.size
        pan_distance = 100
        def pan_position(t):
            progress = t / duration
            if direction == "left":
                return (-pan_distance * progress, 0)
            elif direction == "right":
                return (pan_distance * progress, 0)
            elif direction == "up":
                return (0, -pan_distance * progress)
            elif direction == "down":
                return (0, pan_distance * progress)
            else:
                angle = progress * 2 * np.pi
                return (np.cos(angle) * pan_distance / 2, np.sin(angle) * pan_distance / 2)
        return clip.set_position(pan_position)

    @staticmethod
    def apply_parallax(clip: ImageClip, duration: float, intensity: float = 0.5) -> ImageClip:
        def parallax_pos(t):
            progress = np.sin(t / duration * np.pi)
            return (intensity * 50 * progress, intensity * 30 * progress)
        return clip.set_position(parallax_pos)

    @staticmethod
    def apply_rotation(clip: ImageClip, duration: float, degrees: float = 5) -> ImageClip:
        def rotate_angle(t):
            progress = t / duration
            return degrees * np.sin(progress * 2 * np.pi)
        return clip.rotate(rotate_angle, unit='deg')

    @staticmethod
    def get_random_effect_sequence(clip: ImageClip, duration: float) -> ImageClip:
        effects = [
            ("ken_burns_in", lambda c, d: VideoEffectsManager.apply_ken_burns(c, d, "zoom_in")),
            ("ken_burns_out", lambda c, d: VideoEffectsManager.apply_ken_burns(c, d, "zoom_out")),
            ("pan_left", lambda c, d: VideoEffectsManager.apply_pan(c, d, "left")),
            ("pan_right", lambda c, d: VideoEffectsManager.apply_pan(c, d, "right")),
            ("parallax", lambda c, d: VideoEffectsManager.apply_parallax(c, d, 0.5)),
            ("rotation", lambda c, d: VideoEffectsManager.apply_rotation(c, d, 3)),
        ]
        num_effects = random.randint(1, 2)
        selected_effects = random.sample(effects, num_effects)
        result_clip = clip.set_duration(duration)
        for effect_name, effect_func in selected_effects:
            result_clip = effect_func(result_clip, duration)
            print(f"[Effects] Applied: {effect_name}")
        return result_clip

class CircleOverlayManager:
    def __init__(self, config: Config):
        self.config = config
        self.overlays_dir = config.VIDEO_OVERLAYS_DIR

    def get_available_overlay_videos(self) -> List[Path]:
        video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']
        video_files = []
        if self.overlays_dir.exists():
            for ext in video_extensions:
                video_files.extend(self.overlays_dir.glob(ext))
        return sorted(video_files)

    def get_random_overlay_video(self) -> Optional[Path]:
        videos = self.get_available_overlay_videos()
        if videos:
            return random.choice(videos)
        return None

    def create_circular_mask(self, size: int) -> Image.Image:
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        return mask

    def create_circle_overlay_clip(
        self,
        video_path: Path,
        duration: float,
        diameter: Optional[int] = None,
        position: Optional[str] = None,
        border_width: Optional[int] = None,
        border_color: Optional[Tuple[int, int, int]] = None
    ) -> Optional[VideoFileClip]:
        try:
            cfg = self.config.CIRCLE_OVERLAY_CONFIG
            diameter = diameter or cfg['diameter']
            position = position or cfg['position']
            border_width = border_width or cfg['border_width']
            border_color = border_color or cfg['border_color']
            print(f"[Circle Overlay] Loading video: {video_path.name}")
            overlay_video = VideoFileClip(str(video_path), audio=False)
            if overlay_video.duration < duration:
                n_loops = int(duration / overlay_video.duration) + 1
                overlay_video = concatenate_videoclips([overlay_video] * n_loops)
            overlay_video = overlay_video.subclip(0, min(duration, overlay_video.duration))
            overlay_video = overlay_video.resize((diameter, diameter))
            circular_mask = self.create_circular_mask(diameter)
            mask_array = np.array(circular_mask) / 255.0
            mask_clip = ImageClip(mask_array, ismask=True, duration=duration)
            overlay_video = overlay_video.set_mask(mask_clip)
            if border_width > 0:
                border_img = Image.new('RGB', (diameter, diameter), (0, 0, 0))
                border_draw = ImageDraw.Draw(border_img)
                border_draw.ellipse(
                    [(0, 0), (diameter-1, diameter-1)],
                    outline=border_color,
                    width=border_width
                )
                border_array = np.array(border_img)
                border_clip = ImageClip(border_array, duration=duration)
                border_mask_img = Image.new('L', (diameter, diameter), 0)
                border_mask_draw = ImageDraw.Draw(border_mask_img)
                border_mask_draw.ellipse([(0, 0), (diameter-1, diameter-1)], fill=255)
                if border_width > 0:
                    inner_size = diameter - 2 * border_width
                    border_mask_draw.ellipse(
                        [(border_width, border_width), (border_width + inner_size, border_width + inner_size)],
                        fill=0
                    )
                border_mask_array = np.array(border_mask_img) / 255.0
                border_mask_clip = ImageClip(border_mask_array, ismask=True, duration=duration)
                border_clip = border_clip.set_mask(border_mask_clip)
                overlay_video = CompositeVideoClip([overlay_video, border_clip], size=(diameter, diameter))
            margin = cfg['margin']
            w, h = self.config.VIDEO_WIDTH, self.config.VIDEO_HEIGHT
            pos_map = {
                'top-left': (margin, margin),
                'top-right': (w - diameter - margin, margin),
                'bottom-left': (margin, h - diameter - margin),
                'bottom-right': (w - diameter - margin, h - diameter - margin),
                'center': ((w - diameter) // 2, (h - diameter) // 2)
            }
            overlay_position = pos_map.get(position, pos_map['top-right'])
            overlay_video = overlay_video.set_position(overlay_position)
            print(f"[Circle Overlay] Created at {position} position ({diameter}px diameter)")
            return overlay_video
        except Exception as e:
            print(f"[Circle Overlay] Error creating overlay: {e}")
            import traceback
            traceback.print_exc()
            return None

# =============== LARAVEL COMPANY THEME ===============
LARAVEL_BG_GRADIENT = ("#0f172a", "#2a1030") # Dark Blue to Dark Purple
LARAVEL_ACCENT_GRADIENT = ("#7c3aed", "#ec4899") # Purple to Pink

def create_gradient_image(size: Tuple[int, int], colors: Tuple[str, str], direction: str = "135deg") -> Image.Image:
    """Creates a linear gradient image using Pillow."""
    base = Image.new('RGB', size, colors[0])
    top = Image.new('RGB', size, colors[1])
    mask = Image.new('L', size)
    mask_data = []
    
    w, h = size
    if direction == "135deg":
        for y in range(h):
            for x in range(w):
                # Simple diagonal gradient mask
                mask_data.append(int(255 * (x / w + y / h) / 2))
    elif direction == "to_right":
        for y in range(h):
            for x in range(w):
                mask_data.append(int(255 * (x / w)))
    else: # to_bottom
        for y in range(h):
            for x in range(w):
                mask_data.append(int(255 * (y / h)))
                
    mask.putdata(mask_data)
    return Image.composite(top, base, mask)

# =============== VIDEO GENERATOR — with DB video caching ===============
class VideoGenerator:
    def __init__(self, config: Config, keyword_extractor: Optional[KeywordExtractor] = None):
        self.config = config
        self.font_path = self._discover_fonts()
        self.media_manager = MediaManager()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.logo_clip = self._load_logo()
        self.effects_manager = VideoEffectsManager()
        self.circle_overlay_manager = CircleOverlayManager(config)
        self.sd_manager = None
        if SD_AVAILABLE:
            try:
                sd_model_path = str(config.SD_MODEL_DIR) if config.SD_MODEL_DIR.exists() else "/models/stable-diffusion-v1-5"
                self.sd_manager = StableDiffusionManager(model_path=sd_model_path)
                print("[SD] Stable Diffusion enabled for background generation")
            except Exception as e:
                print(f"[SD] Could not initialize Stable Diffusion: {e}")
                self.sd_manager = None

    def _load_logo(self) -> Optional[ImageClip]:
        logo_extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
        logo_files = []
        if self.config.IMAGES_DIR.exists():
            for ext in logo_extensions:
                logo_files.extend(glob.glob(os.path.join(self.config.IMAGES_DIR, ext)))
        if not logo_files:
            return None
        logo_path = logo_files[0]
        try:
            logo_img = Image.open(logo_path).convert('RGBA')
            cfg = self.config.LOGO_CONFIG
            logo_img.thumbnail((cfg['max_width'], cfg['max_height']), Image.LANCZOS)
            if cfg['opacity'] < 1.0:
                alpha = logo_img.split()[3]
                alpha = ImageEnhance.Brightness(alpha).enhance(cfg['opacity'])
                logo_img.putalpha(alpha)
            logo_array = np.array(logo_img)
            logo_clip = ImageClip(logo_array, transparent=True)
            margin = cfg['margin']
            position = cfg['position']
            w, h = logo_img.size
            pos_map = {
                'top-left': (margin, margin),
                'top-right': (self.config.VIDEO_WIDTH - w - margin, margin),
                'bottom-left': (margin, self.config.VIDEO_HEIGHT - h - margin),
                'bottom-right': (self.config.VIDEO_WIDTH - w - margin, self.config.VIDEO_HEIGHT - h - margin),
                'center': 'center'
            }
            logo_clip = logo_clip.set_position(pos_map[position])
            return logo_clip
        except Exception as e:
            print(f"[Logo] Error loading logo: {e}")
            return None

    def _discover_fonts(self) -> str:
        font_paths = []
        system = platform.system()
        if system == "Windows":
            font_paths.append(Path("C:/Windows/Fonts"))
        elif system == "Darwin":
            font_paths.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts")])
        elif system == "Linux":
            font_paths.extend([
                Path("/usr/share/fonts/truetype"),
                Path("/usr/share/fonts/truetype/dejavu"),
                Path("/usr/share/fonts/truetype/liberation"),
                Path("/usr/share/fonts/TTF"),
                Path.home() / ".fonts"
            ])
        common_fonts = [
            "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
            "LiberationSans-Bold.ttf", "LiberationSans-Regular.ttf",
            "arialbd.ttf", "Arial-Bold.ttf",
            "calibrib.ttf", "Calibri-Bold.ttf",
            "arial.ttf", "Arial.ttf",
            "FreeSans.ttf", "FreeSansBold.ttf",
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
                    if font_file and isinstance(font_file, Path) and font_file.exists():
                        return str(font_file.resolve())
        return "DejaVuSans"

    def _create_audio_visualizer_clip(self, audio_path: Path, duration: float, height: int = 80) -> ImageClip:
        from pydub import AudioSegment
        import numpy as np
        audio_seg = AudioSegment.from_file(str(audio_path))
        if audio_seg.frame_rate != 22050:
            audio_seg = audio_seg.set_frame_rate(22050)
        samples = np.array(audio_seg.get_array_of_samples())
        if audio_seg.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1).astype(np.int32)
        fps = 30
        frame_samples = int(audio_seg.frame_rate / fps)
        amplitudes = []
        for i in range(0, len(samples), frame_samples):
            chunk = samples[i:i + frame_samples]
            if len(chunk) == 0:
                amp = 0
            else:
                amp = np.abs(chunk).mean()
            amplitudes.append(amp)
        max_amp = max(amplitudes) if amplitudes else 1
        if max_amp == 0:
            max_amp = 1
        normalized = [min(a / max_amp, 1.0) for a in amplitudes]
        frames = []
        bar_width = self.config.VIDEO_WIDTH // 30
        for amp in normalized:
            img = Image.new('RGBA', (self.config.VIDEO_WIDTH, self.config.VIDEO_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            for i in range(30):
                bar_height = int(amp * height * (0.5 + 0.5 * np.sin(i * 0.2)))
                x = i * bar_width
                y_top = self.config.VIDEO_HEIGHT - height - bar_height
                y_bottom = self.config.VIDEO_HEIGHT - height
                color = (0, int(200 + 55 * amp), int(255 * amp), int(200 * amp))
                draw.rectangle([x, y_top, x + bar_width - 2, y_bottom], fill=color)
            frames.append(np.array(img))
        target_frames = int(duration * fps)
        if len(frames) < target_frames:
            last_frame = frames[-1] if frames else np.zeros((self.config.VIDEO_HEIGHT, self.config.VIDEO_WIDTH, 4), dtype=np.uint8)
            frames.extend([last_frame] * (target_frames - len(frames)))
        elif len(frames) > target_frames:
            frames = frames[:target_frames]
        vis_clip = ImageSequenceClip(frames, fps=fps)
        vis_clip = vis_clip.set_duration(duration)
        return vis_clip

    def get_available_music_files(self) -> List[Dict[str, str]]:
        music_extensions = ['*.mp3', '*.MP3', '*.wav', '*.WAV']
        music_files = []
        if self.config.MUSIC_DIR.exists():
            for ext in music_extensions:
                found_files = glob.glob(os.path.join(self.config.MUSIC_DIR, ext))
                for file_path in found_files:
                    music_files.append({
                        'name': os.path.basename(file_path),
                        'path': file_path
                    })
        return sorted(music_files, key=lambda x: x['name'])

    def get_music_by_name(self, music_name: str) -> Optional[Path]:
        if not music_name or music_name == "Random":
            return self.get_random_background_music()
        music_files = self.get_available_music_files()
        for music_file in music_files:
            if music_file['name'] == music_name:
                return Path(music_file['path'])
        return self.get_random_background_music()

    def get_random_background_music(self) -> Optional[Path]:
        music_extensions = ['*.mp3', '*.MP3', '*.wav', '*.WAV']
        music_files = []
        if self.config.MUSIC_DIR.exists():
            for ext in music_extensions:
                music_files.extend(glob.glob(os.path.join(self.config.MUSIC_DIR, ext)))
        if music_files:
            return Path(random.choice(music_files))
        return None

    def get_single_ai_background_image(self, text: str, pexels_keyword: Optional[str] = None) -> Optional[Path]:
        if not self.sd_manager:
            print("[SD] Stable Diffusion not available for single image mode")
            return None
        keyword = pexels_keyword or self.keyword_extractor.get_best_unique_keyword(text)
        print(f"[Single Image Mode] Generating AI image for: {keyword or 'full text'}")
        image_path = self.sd_manager.generate_image(text, keyword=keyword)
        if image_path:
            _media_source_cache["single_image_mode"] = 'sd'
            print(f"[Single Image Mode] Image generated: {image_path}")
        return image_path

    def create_single_image_slide_with_effects(
        self,
        sentence: str,
        audio_path: Path,
        single_image_path: Path,
        slide_num: int
    ) -> Optional[VideoFileClip]:
        try:
            audio_clip = AudioFileClip(str(audio_path))
            duration_sec = audio_clip.duration
            img = Image.open(single_image_path)
            img_array = np.array(img)
            image_clip = ImageClip(img_array).set_duration(duration_sec)
            image_clip = self.effects_manager.get_random_effect_sequence(image_clip, duration_sec)
            dimming_clip = ColorClip(size=self.config.VIDEO_SIZE, color=(0,0,0), duration=duration_sec).set_opacity(0.4)
            video_clip = CompositeVideoClip([image_clip, dimming_clip])
            text_clip = self._create_subtitle_overlay_pil(sentence, duration_sec)
            layers = [video_clip, text_clip]
            if self.logo_clip:
                logo = self.logo_clip.set_duration(duration_sec)
                layers.append(logo)
            final_clip = CompositeVideoClip(layers)
            final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
            print(f"[Effects] Slide {slide_num} created with random effects")
            return final_clip
        except Exception as e:
            print(f"[Video] Slide {slide_num} (single image with effects) error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_background_media(self, pexels_keyword: Optional[str] = None,
                            sentence: Optional[str] = None,
                            use_sd: bool = False,
                            media_type: str = "mixed",
                            force_video: bool = False) -> Optional[Path]:
        """
        Prefer real videos. Only use Stable Diffusion as fallback.
        """
        # If explicitly in "sd_only" mode, skip videos
        if media_type == "sd_only":
            if self.sd_manager:
                keyword = pexels_keyword or (self.keyword_extractor.get_best_unique_keyword(sentence) if sentence else None)
                image_path = self.sd_manager.generate_image(sentence or "abstract art", keyword=keyword)
                if image_path:
                    _media_source_cache[keyword or "default"] = 'sd'
                    return image_path
            return None

        # Step 1: Try searching for videos via MediaManager
        if pexels_keyword or sentence:
            # Collect all candidate keywords
            search_keywords = []
            if pexels_keyword:
                search_keywords.append(self.keyword_extractor.sanitize_keyword(pexels_keyword))
            if sentence:
                # Get multiple keywords but try the unused ones first
                extracted = self.keyword_extractor.extract_keywords(sentence, top_n=10)
                # Sort: unused first
                extracted.sort(key=lambda kw: kw in self.keyword_extractor.used_keywords)
                search_keywords.extend(extracted)
            
            for kw in search_keywords:
                if not kw: continue
                # Skip if already used in this session (unless it's the only choice)
                if kw in self.keyword_extractor.used_keywords and len(search_keywords) > 1:
                    continue
                    
                # MediaManager handles Pexels, Giphy, YouTube
                video_path = self.media_manager.get_random_media(kw, self.config.VIDEO_SIZE)
                if video_path:
                    self.keyword_extractor.used_keywords.add(kw)
                    _media_source_cache[kw] = 'media_manager'
                    return video_path

        # Step 2: Try local background videos
        video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV']
        video_files = []
        if self.config.VIDEOS_DIR.exists():
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.config.VIDEOS_DIR, ext)))
        if video_files:
            return Path(random.choice(video_files))

        # Step 3: ONLY FALL BACK TO STABLE DIFFUSION IF ALL VIDEO SOURCES FAIL
        if self.sd_manager and (use_sd or media_type == "mixed"):
            keyword = pexels_keyword or (self.keyword_extractor.get_best_unique_keyword(sentence) if sentence else None)
            image_path = self.sd_manager.generate_image(sentence or "abstract background", keyword=keyword)
            if image_path:
                _media_source_cache[keyword or "default"] = 'sd'
                return image_path

        return None  # No media found

    def _create_subtitle_overlay_pil(self, text: str, duration: float) -> ImageClip:
        # 1. Setup dimensions and fonts
        img_size = self.config.VIDEO_SIZE
        base_font_size = self.config.TEXT_SIZE_CONFIG['font_size']
        if len(text) > 60:
            font_size = max(30, int(base_font_size * (1.0 - (len(text) - 60) / 200)))
        else:
            font_size = base_font_size
            
        try:
            font = ImageFont.truetype(self.font_path, font_size) if self.font_path and os.path.exists(self.font_path) else ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # 2. Wrap text and calculate bounding box
        wrapped_text = textwrap.fill(text, width=35)
        # Create a temp image to calculate bbox
        temp_draw = ImageDraw.Draw(Image.new('L', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 3. Create the text mask (white text on black background)
        # Add padding for stroke
        padding = 10
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new('L', mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        
        # Position in mask
        text_pos = (padding, padding)
        
        # Draw stroke in mask if needed (actually for mask we just want the text area)
        mask_draw.text(text_pos, wrapped_text, font=font, fill=255)
        
        # 4. Create gradient image for the text
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")
        
        # 5. Composite text onto final frame
        final_frame = Image.new('RGBA', img_size, (0, 0, 0, 0))
        
        # First draw the shadow/outline on final_frame for better legibility
        shadow_draw = ImageDraw.Draw(final_frame)
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = self.config.VIDEO_HEIGHT - text_height - self.config.TEXT_SIZE_CONFIG['bottom_margin']
        
        stroke_width = 3
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                if adj_x != 0 or adj_y != 0:
                    shadow_draw.text((x + adj_x, y + adj_y), wrapped_text, font=font, fill=(0, 0, 0, 200))

        # Composite the gradient text
        text_layer = Image.new('RGBA', mask_size, (0, 0, 0, 0))
        text_layer.paste(grad_img, (0, 0), mask_img)
        
        final_frame.paste(text_layer, (x - padding, y - padding), text_layer)
        
        img_clip = ImageClip(np.array(final_frame)).set_duration(duration)
        return img_clip

    def create_intro_slide(self, audio_path: Path, bg_color: Tuple[int, int, int] = (74, 144, 226),
                          pexels_keyword: Optional[str] = None, single_image_path: Optional[Path] = None) -> VideoFileClip:
        audio_clip = AudioFileClip(str(audio_path))
        duration_sec = audio_clip.duration
        if single_image_path and single_image_path.exists():
            img = Image.open(single_image_path)
            img_array = np.array(img)
            video_clip = ImageClip(img_array).set_duration(duration_sec)
            video_clip = self.effects_manager.apply_ken_burns(video_clip, duration_sec, "zoom_in")
        else:
            background_video = self.get_background_media(pexels_keyword=pexels_keyword, media_type="video_only")
            if background_video and background_video.exists():
                try:
                    video_clip = VideoFileClip(str(background_video), audio=False)
                    target_ratio = self.config.VIDEO_WIDTH / self.config.VIDEO_HEIGHT
                    current_ratio = video_clip.size[0] / video_clip.size[1]
                    if current_ratio > target_ratio:
                        new_width = int(video_clip.size[1] * target_ratio)
                        x_center = video_clip.size[0] / 2
                        x1 = int(x_center - new_width / 2)
                        video_clip = video_clip.crop(x1=x1, width=new_width)
                    else:
                        new_height = int(video_clip.size[0] / target_ratio)
                        y_center = video_clip.size[1] / 2
                        y1 = int(y_center - new_height / 2)
                        video_clip = video_clip.crop(y1=y1, height=new_height)
                    video_clip = video_clip.resize(self.config.VIDEO_SIZE)
                    if video_clip.duration < duration_sec:
                        n_loops = int(duration_sec / video_clip.duration) + 1
                        video_clip = video_clip.loop(n=n_loops)
                    video_clip = video_clip.subclip(0, min(duration_sec, video_clip.duration))
                    dimming_clip = ColorClip(size=self.config.VIDEO_SIZE, color=(0,0,0), duration=duration_sec).set_opacity(0.4)
                    video_clip = CompositeVideoClip([video_clip, dimming_clip])
                except Exception as e:
                    print(f"[Video] Intro error: {e}")
                    video_clip = None
            else:
                video_clip = None
        if video_clip is None:
            video_clip = ColorClip(size=self.config.VIDEO_SIZE, color=list(bg_color), duration=duration_sec)
        layers = [video_clip]
        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            if self.font_path and os.path.exists(self.font_path):
                font_large = ImageFont.truetype(self.font_path, 120)
                font_small = ImageFont.truetype(self.font_path, 70)
            else:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        main_text_str = "WELCOME"
        bbox = draw.textbbox((0, 0), main_text_str, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2 - 80
        for adj in range(-3, 4):
            draw.text((x + adj, y), main_text_str, font=font_large, fill='black')
            draw.text((x, y + adj), main_text_str, font=font_large, fill='black')
        draw.text((x, y), main_text_str, font=font_large, fill='cyan')
        sec_text_str = "TO OUR CHANNEL"
        bbox2 = draw.textbbox((0, 0), sec_text_str, font=font_small)
        text_width2 = bbox2[2] - bbox2[0]
        x2 = (self.config.VIDEO_WIDTH - text_width2) // 2
        y2 = int(self.config.VIDEO_HEIGHT * 0.58)
        for adj in range(-2, 3):
            draw.text((x2 + adj, y2), sec_text_str, font=font_small, fill='black')
            draw.text((x2, y2 + adj), sec_text_str, font=font_small, fill='black')
        draw.text((x2, y2), sec_text_str, font=font_small, fill='white')
        text_clip = ImageClip(np.array(img)).set_duration(duration_sec)
        layers.append(text_clip)
        if self.logo_clip:
            logo = self.logo_clip.set_duration(duration_sec)
            layers.append(logo)
        final_clip = CompositeVideoClip(layers)
        final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
        return final_clip

    def create_cta_slide(self, audio_path: Path, bg_color: Tuple[int, int, int] = (74, 144, 226),
                         pexels_keyword: Optional[str] = None, single_image_path: Optional[Path] = None) -> VideoFileClip:
        audio_clip = AudioFileClip(str(audio_path))
        duration_sec = audio_clip.duration
        if single_image_path and single_image_path.exists():
            img = Image.open(single_image_path)
            img_array = np.array(img)
            video_clip = ImageClip(img_array).set_duration(duration_sec)
            video_clip = self.effects_manager.apply_ken_burns(video_clip, duration_sec, "zoom_out")
        else:
            background_video = self.get_background_media(pexels_keyword=pexels_keyword, media_type="video_only")
            if background_video and background_video.exists():
                try:
                    video_clip = VideoFileClip(str(background_video), audio=False)
                    target_ratio = self.config.VIDEO_WIDTH / self.config.VIDEO_HEIGHT
                    current_ratio = video_clip.size[0] / video_clip.size[1]
                    if current_ratio > target_ratio:
                        new_width = int(video_clip.size[1] * target_ratio)
                        x_center = video_clip.size[0] / 2
                        x1 = int(x_center - new_width / 2)
                        video_clip = video_clip.crop(x1=x1, width=new_width)
                    else:
                        new_height = int(video_clip.size[0] / target_ratio)
                        y_center = video_clip.size[1] / 2
                        y1 = int(y_center - new_height / 2)
                        video_clip = video_clip.crop(y1=y1, height=new_height)
                    video_clip = video_clip.resize(self.config.VIDEO_SIZE)
                    if video_clip.duration < duration_sec:
                        n_loops = int(duration_sec / video_clip.duration) + 1
                        video_clip = video_clip.loop(n=n_loops)
                    video_clip = video_clip.subclip(0, min(duration_sec, video_clip.duration))
                    dimming_clip = ColorClip(size=self.config.VIDEO_SIZE, color=(0,0,0), duration=duration_sec).set_opacity(0.4)
                    video_clip = CompositeVideoClip([video_clip, dimming_clip])
                except Exception as e:
                    print(f"[Video] CTA error: {e}")
                    video_clip = None
            else:
                video_clip = None
        if video_clip is None:
            video_clip = ColorClip(size=self.config.VIDEO_SIZE, color=list(bg_color), duration=duration_sec)
        layers = [video_clip]
        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            if self.font_path and os.path.exists(self.font_path):
                font_large = ImageFont.truetype(self.font_path, 140)
                font_small = ImageFont.truetype(self.font_path, 80)
            else:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        main_text_str = "LIKE\nSHARE\nSUBSCRIBE"
        bbox = draw.textbbox((0, 0), main_text_str, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2 - 100
        for adj in range(-3, 4):
            draw.text((x + adj, y), main_text_str, font=font_large, fill='black')
            draw.text((x, y + adj), main_text_str, font=font_large, fill='black')
        draw.text((x, y), main_text_str, font=font_large, fill='yellow')
        sec_text_str = "TO OUR CHANNEL"
        bbox2 = draw.textbbox((0, 0), sec_text_str, font=font_small)
        text_width2 = bbox2[2] - bbox2[0]
        x2 = (self.config.VIDEO_WIDTH - text_width2) // 2
        y2 = int(self.config.VIDEO_HEIGHT * 0.65)
        for adj in range(-2, 3):
            draw.text((x2 + adj, y2), sec_text_str, font=font_small, fill='black')
            draw.text((x2, y2 + adj), sec_text_str, font=font_small, fill='black')
        draw.text((x2, y2), sec_text_str, font=font_small, fill='white')
        text_clip = ImageClip(np.array(img)).set_duration(duration_sec)
        layers.append(text_clip)
        if self.logo_clip:
            logo = self.logo_clip.set_duration(duration_sec)
            layers.append(logo)
        final_clip = CompositeVideoClip(layers)
        final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
        return final_clip

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r'\bDr\.', 'Dr<dot>', text)
        text = re.sub(r'\bMr\.', 'Mr<dot>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<dot>', text)
        text = re.sub(r'\bMs\.', 'Ms<dot>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<dot>', '.').strip() for s in raw_sentences if s.strip()]
        for i, sentence in enumerate(sentences):
            if not sentence.endswith(('.', '!', '?')):
                sentences[i] = sentence + '.'
        filtered = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            if len(current.split()) < 3 and len(current) < 15 and i + 1 < len(sentences):
                merged = current + " " + sentences[i + 1]
                filtered.append(merged)
                i += 2
            else:
                filtered.append(current)
                i += 1
        return filtered

    def _create_single_slide_with_fixed_image(self, sentence: str, audio_path: Path, bg_color: Tuple[int, int, int],
                                              single_image_path: Path, slide_num: int) -> Optional[VideoFileClip]:
        try:
            audio_clip = AudioFileClip(str(audio_path))
            duration_sec = audio_clip.duration
            img = Image.open(single_image_path)
            img_array = np.array(img)
            image_clip = ImageClip(img_array).set_duration(duration_sec)
            image_clip = self.effects_manager.get_random_effect_sequence(image_clip, duration_sec)
            dimming_clip = ColorClip(size=self.config.VIDEO_SIZE, color=(0,0,0), duration=duration_sec).set_opacity(0.4)
            video_clip = CompositeVideoClip([image_clip, dimming_clip])
            layers = [video_clip]
            text_clip = self._create_subtitle_overlay_pil(sentence, duration_sec)
            layers.append(text_clip)
            if self.logo_clip:
                logo = self.logo_clip.set_duration(duration_sec)
                layers.append(logo)
            final_clip = CompositeVideoClip(layers)
            final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
            return final_clip
        except Exception as e:
            print(f"[Video] Slide {slide_num} (fixed image) error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_single_slide(self, sentence: str, audio_path: Path, bg_color: Tuple[int, int, int],
                             pexels_keyword: Optional[str], slide_num: int,
                             use_sd: bool = False, media_type: str = "mixed",
                             force_video: bool = False) -> Optional[VideoFileClip]:
        try:
            if audio_path is None or not Path(audio_path).exists():
                print(f"[Video] Slide {slide_num} error: audio_path is None or missing")
                return None
                
            audio_clip = AudioFileClip(str(audio_path))
            duration_sec = audio_clip.duration
            media_path = self.get_background_media(
                pexels_keyword=pexels_keyword,
                sentence=sentence,
                use_sd=use_sd,
                media_type=media_type,
                force_video=force_video
            )
            video_clip = None
            if media_path and media_path.exists():
                try:
                    if media_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        img = Image.open(media_path)
                        img_array = np.array(img)
                        video_clip = ImageClip(img_array).set_duration(duration_sec)
                        print(f"[Video] Slide {slide_num}: Using SD-generated image")
                    else:
                        video_clip = VideoFileClip(str(media_path), audio=False)
                        target_ratio = self.config.VIDEO_WIDTH / self.config.VIDEO_HEIGHT
                        current_ratio = video_clip.size[0] / video_clip.size[1]
                        if current_ratio > target_ratio:
                            new_width = int(video_clip.size[1] * target_ratio)
                            x_center = video_clip.size[0] / 2
                            x1 = int(x_center - new_width / 2)
                            video_clip = video_clip.crop(x1=x1, width=new_width)
                        else:
                            new_height = int(video_clip.size[0] / target_ratio)
                            y_center = video_clip.size[1] / 2
                            y1 = int(y_center - new_height / 2)
                            video_clip = video_clip.crop(y1=y1, height=new_height)
                        video_clip = video_clip.resize(self.config.VIDEO_SIZE)
                        if video_clip.duration < duration_sec:
                            n_loops = int(duration_sec / video_clip.duration) + 1
                            video_clip = video_clip.loop(n=n_loops)
                        video_clip = video_clip.subclip(0, min(duration_sec, video_clip.duration))
                except Exception as e:
                    print(f"[Video] Slide {slide_num} media error: {e}")
                    video_clip = None
            if video_clip is not None:
                print(f"[Video] Slide {slide_num}: Composition started. Layers: background={media_path.name if media_path else 'None'}, duration={duration_sec:.2f}s")
                dimming_clip = ColorClip(size=self.config.VIDEO_SIZE, color=(0,0,0), duration=duration_sec).set_opacity(0.4)
                video_clip = CompositeVideoClip([video_clip, dimming_clip])
            else:
                print(f"[Video] Slide {slide_num}: Using Laravel Company Branded Gradient Background.")
                grad_img = create_gradient_image(self.config.VIDEO_SIZE, LARAVEL_BG_GRADIENT, "135deg")
                video_clip = ImageClip(np.array(grad_img)).set_duration(duration_sec)
            
            text_clip = self._create_subtitle_overlay_pil(sentence, duration_sec)
            layers = [video_clip, text_clip]
            if self.logo_clip:
                logo = self.logo_clip.set_duration(duration_sec)
                layers.append(logo)
            final_clip = CompositeVideoClip(layers)
            final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
            print(f"[Video] Slide {slide_num}: Successfully composed.")
            return final_clip
        except Exception as e:
            print(f"[Video] Slide {slide_num} error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extend_clip_with_silence(self, clip: VideoFileClip, additional_duration: float) -> VideoFileClip:
        from pydub import AudioSegment
        temp_audio_path = self.config.TEMP_DIR / f"temp_extend_audio_{uuid.uuid4()}.wav"
        clip.audio.write_audiofile(str(temp_audio_path), logger=None)
        audio_seg = AudioSegment.from_file(str(temp_audio_path))
        silence = AudioSegment.silent(duration=int(additional_duration * 1000))
        extended_audio = audio_seg + silence
        extended_audio_path = self.config.TEMP_DIR / f"extended_audio_{uuid.uuid4()}.wav"
        extended_audio.export(str(extended_audio_path), format="wav")
        extended_audio_clip = AudioFileClip(str(extended_audio_path))
        video_without_audio = clip.without_audio()
        extended_video = video_without_audio.set_duration(clip.duration + additional_duration)
        extended_clip = extended_video.set_audio(extended_audio_clip)
        try:
            temp_audio_path.unlink(missing_ok=True)
        except:
            pass
        return extended_clip

    def _apply_transition_between(self, clip1: VideoFileClip, clip2: VideoFileClip, duration: float = 1.0) -> VideoFileClip:
        if duration <= 0 or duration >= min(clip1.duration, clip2.duration):
            return concatenate_videoclips([clip1, clip2], method="compose")
        
        transition_type = random.choice(['crossfade', 'slide_left', 'slide_right', 'fade_black'])
        w, h = self.config.VIDEO_WIDTH, self.config.VIDEO_HEIGHT
        
        # Audio is usually concatenated simply to keep timing exact
        full_audio = concatenate_videoclips([clip1, clip2], method="compose").audio
        
        if transition_type == 'crossfade':
            # Create a transition where clip2 fades in over clip1
            clip2_with_fade = clip2.set_start(clip1.duration - duration).crossfadein(duration)
            result = CompositeVideoClip([clip1, clip2_with_fade], size=(w, h))
        elif transition_type == 'slide_left':
            # Slide left transition: clip2 slides in from the right
            clip2_with_slide = clip2.set_start(clip1.duration - duration).set_position(lambda t: (w * (1 - t/duration) if t < duration else 0, 0))
            result = CompositeVideoClip([clip1, clip2_with_slide], size=(w, h))
        elif transition_type == 'slide_right':
            # Slide right transition: clip2 slides in from the left
            clip2_with_slide = clip2.set_start(clip1.duration - duration).set_position(lambda t: (-w * (1 - t/duration) if t < duration else 0, 0))
            result = CompositeVideoClip([clip1, clip2_with_slide], size=(w, h))
        elif transition_type == 'fade_black':
            black = ColorClip((w, h), color=(0, 0, 0), duration=duration).set_start(clip1.duration - duration/2).crossfadein(duration/2).crossfadeout(duration/2)
            clip2_delayed = clip2.set_start(clip1.duration) # No overlap for fade_black to keep it simple
            # Actually for a nice fade black we want both to fade to black
            # But let's keep it simple: clip2 starts after clip1 + gap
            # Or just use concatenate with black clip
            black = ColorClip((w, h), color=(0, 0, 0), duration=duration)
            result = concatenate_videoclips([clip1, black, clip2], method="compose")
            return result.set_audio(full_audio)
        else:
            return concatenate_videoclips([clip1, clip2], method="compose")

        return result.set_duration(clip1.duration + clip2.duration - duration).set_audio(full_audio)

    def create_video_per_sentence(self, sentences: List[str], audio_paths: List[Path],
                                  sentence_keywords: List[Optional[str]],
                                  intro_audio_path: Optional[Path] = None,
                                  cta_audio_path: Optional[Path] = None,
                                  bg_color: Tuple[int, int, int] = (74, 144, 226),
                                  add_intro_slide: bool = True,
                                  add_cta_slide: bool = True,
                                  use_stable_diffusion: bool = False,
                                  media_type: str = "mixed",
                                  single_image_path: Optional[Path] = None,
                                  enable_circle_overlay: bool = False,
                                  circle_overlay_config: Optional[Dict] = None,
                                  overlay_selection: str = "Random",
                                  progress_callback=None) -> Path:
        clips = []
        use_single_image = single_image_path is not None
        transition_duration = self.config.TRANSITION_CONFIG['duration']
        sd_slide_indices = set()
        if media_type == "mixed" and self.sd_manager and not use_single_image:
            num_sd_slides = max(1, int(len(sentences) * self.config.MIXED_MODE_SD_RATIO))
            sd_slide_indices = set(random.sample(range(len(sentences)), min(num_sd_slides, len(sentences))))
            print(f"[OPTIMIZATION] Mixed mode: {num_sd_slides} AI images, {len(sentences) - num_sd_slides} videos")
        full_audio_path = None
        if use_single_image:
            from pydub import AudioSegment
            combined = AudioSegment.silent(duration=0)
            if add_intro_slide and intro_audio_path:
                combined += AudioSegment.from_file(str(intro_audio_path))
            for ap in audio_paths:
                combined += AudioSegment.from_file(str(ap))
            if add_cta_slide and cta_audio_path:
                combined += AudioSegment.from_file(str(cta_audio_path))
            full_audio_path = self.config.TEMP_DIR / f"full_audio_{uuid.uuid4()}.wav"
            combined.export(str(full_audio_path), format="wav")
        if add_intro_slide and intro_audio_path:
            try:
                intro_kw = sentence_keywords[0] if sentence_keywords else None
                intro_clip = self.create_intro_slide(
                    intro_audio_path, bg_color=bg_color,
                    pexels_keyword=intro_kw, single_image_path=single_image_path
                )
                if not use_single_image:
                    intro_clip = self._extend_clip_with_silence(intro_clip, transition_duration)
                clips.append(intro_clip)
                print("[Video] Intro slide created")
            except Exception as e:
                print(f"[Video] Intro warning: {e}")
        with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_SLIDES) as executor:
            futures = {}
            for i, (sentence, audio_path, keyword) in enumerate(zip(sentences, audio_paths, sentence_keywords)):
                if single_image_path:
                    future = executor.submit(
                        self._create_single_slide_with_fixed_image,
                        sentence, audio_path, bg_color, single_image_path, i + 1
                    )
                else:
                    force_video = (media_type == "mixed" and i not in sd_slide_indices)
                    future = executor.submit(
                        self._create_single_slide,
                        sentence, audio_path, bg_color, keyword, i + 1,
                        use_stable_diffusion, media_type, force_video
                    )
                futures[future] = i
            slide_results = [None] * len(sentences)
            for future in as_completed(futures):
                i = futures[future]
                try:
                    video_clip = future.result()
                    if video_clip:
                        if not use_single_image and i < len(slide_results) - 1:
                            video_clip = self._extend_clip_with_silence(video_clip, transition_duration)
                        slide_results[i] = video_clip
                        if progress_callback:
                            progress_callback(i + 1, len(sentences), f"Created slide {i + 1}")
                except Exception as e:
                    print(f"[Video] Slide {i + 1} processing error: {e}")
        for result in slide_results:
            if result:
                clips.append(result)
        if not clips:
            raise ValueError("No clips were created successfully")
        if add_cta_slide and cta_audio_path:
            try:
                cta_kw = sentence_keywords[-1] if sentence_keywords else None
                cta_clip = self.create_cta_slide(
                    cta_audio_path, bg_color=bg_color,
                    pexels_keyword=cta_kw, single_image_path=single_image_path
                )
                clips.append(cta_clip)
            except Exception as e:
                print(f"[Video] CTA warning: {e}")
        if use_single_image:
            if progress_callback:
                progress_callback(len(sentences), len(sentences), "Assembling final video with audio visualizer...")
            final_clip = concatenate_videoclips(clips, method="compose")
            vis_clip = self._create_audio_visualizer_clip(full_audio_path, final_clip.duration)
            final_clip = CompositeVideoClip([final_clip, vis_clip])
        else:
            if progress_callback:
                progress_callback(len(sentences), len(sentences), "Applying random transitions...")
            if len(clips) > 1:
                current = clips[0]
                for i in range(1, len(clips)):
                    current = self._apply_transition_between(current, clips[i], duration=transition_duration)
                clips = [current]
            final_clip = clips[0]
        if enable_circle_overlay:
            if progress_callback:
                progress_callback(len(sentences), len(sentences), "Adding circle overlay...")
            
            if overlay_selection and overlay_selection != "Random":
                overlay_video_path = self.config.VIDEO_OVERLAYS_DIR / overlay_selection
                if not overlay_video_path.exists():
                    print(f"[Circle Overlay] Selected overlay {overlay_selection} not found, falling back to random")
                    overlay_video_path = self.circle_overlay_manager.get_random_overlay_video()
            else:
                overlay_video_path = self.circle_overlay_manager.get_random_overlay_video()
            if overlay_video_path:
                circle_overlay = self.circle_overlay_manager.create_circle_overlay_clip(
                    overlay_video_path,
                    final_clip.duration,
                    diameter=circle_overlay_config.get('diameter') if circle_overlay_config else None,
                    position=circle_overlay_config.get('position') if circle_overlay_config else None,
                    border_width=circle_overlay_config.get('border_width') if circle_overlay_config else None,
                    border_color=circle_overlay_config.get('border_color') if circle_overlay_config else None
                )
                if circle_overlay:
                    final_clip = CompositeVideoClip([final_clip, circle_overlay])
                    print("[Circle Overlay] Successfully added to video")
                else:
                    print("[Circle Overlay] Failed to create overlay, continuing without it")
            else:
                print("[Circle Overlay] No overlay videos found in video-overlays folder")
        if progress_callback:
            progress_callback(len(sentences), len(sentences), "Exporting final video...")
        output_path = self.config.TEMP_DIR / f"video_{uuid.uuid4()}.mp4"
        print(f"[MoviePy] Final export phase started. Output: {output_path}")
        try:
            import time
            start_time = time.time()
            final_clip.write_videofile(
                str(output_path),
                fps=self.config.FPS,
                codec=self.config.VIDEO_CODEC,
                audio_codec=self.config.AUDIO_CODEC,
                audio_bitrate='192k',
                logger='bar',
                preset=self.config.VIDEO_PRESET,
                threads=4,
                ffmpeg_params=["-crf", str(self.config.VIDEO_CRF)]
            )
            elapsed = time.time() - start_time
            print(f"[MoviePy] Export finished in {elapsed:.2f}s")
        except Exception as e:
            print(f"[MoviePy] EXPORT ERROR: {e}")
            raise
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        try:
            final_clip.close()
        except:
            pass
        if full_audio_path and full_audio_path.exists():
            try:
                full_audio_path.unlink()
            except:
                pass
        return output_path

# =============== TEXT TO VIDEO GENERATOR — WITH DB CACHING ===============
class TextToVideoGenerator:
    def __init__(self):
        self.config = Config()
        self.keyword_extractor = KeywordExtractor()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config, keyword_extractor=self.keyword_extractor)
        self.available_voices = self._get_available_voices()
        self.available_music = self._get_available_music()
        self.available_overlays = self._get_available_overlays()

    def _get_available_voices(self) -> List[str]:
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)

    def _get_available_music(self) -> List[str]:
        music_files = self.video_generator.get_available_music_files()
        music_names = ["Random"] + [m['name'] for m in music_files]
        return music_names

    def _get_available_overlays(self) -> List[str]:
        overlay_videos = self.video_generator.circle_overlay_manager.get_available_overlay_videos()
        return [v.name for v in overlay_videos]

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                       bg_color: Tuple[int, int, int] = (74, 144, 226),
                       pexels_keyword: Optional[str] = None,
                       enable_background_music: bool = True,
                       music_selection: str = "Random",
                       music_volume_db: int = -27,
                       add_intro_slide: bool = True,
                       add_call_to_action: bool = True,
                       use_random_voices: bool = False,
                       use_stable_diffusion: bool = True,
                       media_type: str = "mixed",
                       use_single_image_mode: bool = False,
                       enable_circle_overlay: bool = False,
                       circle_diameter: int = 300,
                       circle_position: str = "top-right",
                       circle_border_width: int = 5,
                       overlay_selection: str = "Random",
                       progress_callback=None) -> Dict:
        # Prepare input parameters for DB caching
        input_params = {
            'text': text,
            'speaker_id': speaker_id,
            'bg_color': bg_color,
            'pexels_keyword': pexels_keyword,
            'enable_background_music': enable_background_music,
            'music_selection': music_selection,
            'music_volume_db': music_volume_db,
            'add_intro_slide': add_intro_slide,
            'add_call_to_action': add_call_to_action,
            'use_random_voices': use_random_voices,
            'use_stable_diffusion': use_stable_diffusion,
            'media_type': media_type,
            'use_single_image_mode': use_single_image_mode,
            'enable_circle_overlay': enable_circle_overlay,
            'circle_diameter': circle_diameter,
            'circle_position': circle_position,
            'circle_border_width': circle_border_width,
            'overlay_selection': overlay_selection,
        }
        # Check if this exact video was generated before
        cached_result = DB.get_cached_video(input_params)
        if cached_result:
            print("[DB] Reusing previously generated video from cache")
            return cached_result
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}
        if len(text) > 10000:
            return {"error": "Text is too long (max 10,000 characters)", "success": False}
        if music_volume_db != self.config.MUSIC_CONFIG['music_volume_db']:
            self.config.MUSIC_CONFIG['music_volume_db'] = music_volume_db
        sentences = self.video_generator.split_into_sentences(text)
        if len(sentences) > 100:
            return {"error": "Too many sentences (max 100)", "success": False}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        single_image_path = None
        if use_single_image_mode:
            single_image_path = self.video_generator.get_single_ai_background_image(text, pexels_keyword)
            
        # Reset and prepare keywords for this session
        self.keyword_extractor.clear_used()
        if pexels_keyword and pexels_keyword.strip():
            sentence_keywords = [pexels_keyword.strip()] * len(sentences)
        else:
            sentence_keywords = []
            for sent in sentences:
                kw = self.keyword_extractor.get_best_unique_keyword(sent)
                sentence_keywords.append(kw)
        audio_paths = []
        intro_audio_path = None
        cta_audio_path = None
        music_path = None
        try:
            if enable_background_music:
                music_path = self.video_generator.get_music_by_name(music_selection)
            if use_random_voices:
                voices_for_sentences = [random.choice(self.available_voices) for _ in sentences]
            else:
                voices_for_sentences = [speaker_id] * len(sentences)
            if add_intro_slide:
                intro_voice = speaker_id if not use_random_voices else random.choice(self.available_voices)
                intro_audio_path = self.tts_manager.generate_speech(self.config.INTRO_MESSAGE, intro_voice)
            for i, (sentence, voice) in enumerate(zip(sentences, voices_for_sentences)):
                if progress_callback:
                    voice_name = voice if len(voice) < 30 else voice[:27] + "..."
                    progress_callback(i + 1, len(sentences) * 2, f"Audio {i + 1}/{len(sentences)} ({voice_name})")
                audio_path = self.tts_manager.generate_speech(sentence, voice)
                audio_paths.append(audio_path)
            if add_call_to_action:
                cta_voice = speaker_id if not use_random_voices else voices_for_sentences[-1]
                cta_audio_path = self.tts_manager.generate_speech(self.config.CTA_MESSAGE, cta_voice)
            def video_progress(current, total, message):
                if progress_callback:
                    progress_callback(len(sentences) + current, len(sentences) * 2, message)
            circle_overlay_config = None
            if enable_circle_overlay:
                circle_overlay_config = {
                    'diameter': circle_diameter,
                    'position': circle_position,
                    'border_width': circle_border_width,
                    'border_color': (255, 255, 255)
                }
            video_temp_path = self.video_generator.create_video_per_sentence(
                sentences=sentences,
                audio_paths=audio_paths,
                sentence_keywords=sentence_keywords,
                intro_audio_path=intro_audio_path,
                cta_audio_path=cta_audio_path,
                bg_color=bg_color,
                add_intro_slide=add_intro_slide,
                add_cta_slide=add_call_to_action,
                use_stable_diffusion=use_stable_diffusion,
                media_type=media_type,
                single_image_path=single_image_path,
                enable_circle_overlay=enable_circle_overlay,
                circle_overlay_config=circle_overlay_config,
                overlay_selection=overlay_selection,
                progress_callback=video_progress
            )
            final_video_clip = VideoFileClip(str(video_temp_path))
            if enable_background_music and music_path and music_path.exists():
                voice_segment = AudioSegment.from_file(str(video_temp_path), format="mp4")
                music = AudioSegment.from_file(str(music_path))
                cfg = self.config.MUSIC_CONFIG
                music = music + cfg['music_volume_db']
                if len(music) < len(voice_segment):
                    loops_needed = (len(voice_segment) // len(music)) + 2
                    looped_music = music
                    for _ in range(loops_needed - 1):
                        looped_music = looped_music.append(music, crossfade=cfg['crossfade_duration'])
                    music = looped_music
                music = music[:len(voice_segment)]
                music = music.fade_in(cfg['fade_in_duration']).fade_out(cfg['fade_out_duration'])
                mixed = voice_segment.overlay(music)
                mixed_audio_path = self.config.TEMP_DIR / f"final_mixed_audio_{uuid.uuid4()}.wav"
                mixed.export(str(mixed_audio_path), format="wav")
                mixed_audio_clip = AudioFileClip(str(mixed_audio_path))
                final_video_clip = final_video_clip.set_audio(mixed_audio_clip)
            else:
                mixed_audio_path = None
            video_final_path = session_dir / f"video_portrait_{timestamp}.mp4"
            final_video_clip.write_videofile(
                str(video_final_path),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                logger=None,
                preset=self.config.VIDEO_PRESET,
                threads=4,
                ffmpeg_params=["-crf", str(self.config.VIDEO_CRF)]
            )
            audio_only_path = session_dir / f"audio_{timestamp}.mp3"
            if mixed_audio_path:
                mixed_audio_segment = AudioSegment.from_file(str(mixed_audio_path))
            else:
                original_audio = AudioSegment.from_file(str(video_temp_path), format="mp4")
                mixed_audio_segment = original_audio
            mixed_audio_segment.export(str(audio_only_path), format="mp3", bitrate="192k")
            try:
                video_temp_path.unlink(missing_ok=True)
                if mixed_audio_path:
                    mixed_audio_path.unlink(missing_ok=True)
            except:
                pass
            sd_images_count = 0
            if use_single_image_mode and single_image_path:
                sd_images_count = 1
            else:
                for kw in sentence_keywords:
                    if kw and kw in _media_source_cache and _media_source_cache[kw] == 'sd':
                        sd_images_count += 1
            result = {
                "success": True,
                "audio_path": str(audio_only_path),
                "video_path": str(video_final_path),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "background_music": enable_background_music and music_path is not None,
                "music_used": os.path.basename(music_path) if music_path else None,
                "intro_included": add_intro_slide,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "text_style": "White subtitle-style at bottom, no background, dynamic font size",
                "logo_included": self.video_generator.logo_clip is not None,
                "transitions": "Random: crossfade, slide, zoom, fade-to-black (SYNCED)" if not use_single_image_mode else "None (Visualizer + Effects)",
                "video_backgrounds": f"Pexels/Giphy/SD ({media_type})",
                "random_voices": use_random_voices,
                "voices_used": voices_for_sentences if use_random_voices else None,
                "media_type": media_type,
                "sd_images_used": sd_images_count,
                "single_image_mode": use_single_image_mode,
                "circle_overlay_enabled": enable_circle_overlay,
                "circle_overlay_position": circle_position if enable_circle_overlay else None,
                "circle_overlay_selection": overlay_selection if enable_circle_overlay else None,
            }
            # Save to DB
            DB.save_video(input_params, result)
            return result
        except Exception as e:
            print(f"[Error] Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "success": False}
        finally:
            for audio_path in audio_paths:
                try:
                    audio_path.unlink(missing_ok=True)
                except:
                    pass
            if intro_audio_path:
                try:
                    intro_audio_path.unlink(missing_ok=True)
                except:
                    pass
            if cta_audio_path:
                try:
                    cta_audio_path.unlink(missing_ok=True)
                except:
                    pass

# =============== UI ===============
def setup_ui(generator: TextToVideoGenerator):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   title="Optimized Portrait Video Generator") as demo:
        gr.Markdown("# 🎥 Optimized Portrait Video Generator")
        gr.Markdown("**CPU-ONLY** + **TTS & Video Caching via SQLite**!")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter Your Text",
                    placeholder="Each sentence will get backgrounds!",
                    lines=8
                )
                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        label="Voice",
                        choices=generator.available_voices,
                        value=generator.config.STANDARD_VOICE_NAME
                    )
                    bg_color_picker = gr.ColorPicker(
                        label="Background Color (Fallback)",
                        value="#4A90E2"
                    )
                media_type_radio = gr.Radio(
                    label="🖼️ Background Media Type",
                    choices=[
                        ("✅ Mixed (20% AI + 80% Videos) - FAST", "mixed"),
                        ("🎬 Videos Only (Pexels/Giphy) - FASTEST", "video_only"),
                        ("🎨 AI Images Only (Stable Diffusion) - SLOW", "sd_only")
                    ],
                    value="mixed",
                    info="Mixed mode is now OPTIMIZED for speed!"
                )
                use_sd = gr.Checkbox(
                    label="🖼️ Prefer Stable Diffusion (Mixed Mode Only)",
                    value=False,
                    info="Increase AI image ratio in mixed mode (slower but more creative)"
                )
                use_single_image_mode = gr.Checkbox(
                    label="📷 Single AI Image Mode (Podcast Style)",
                    value=False,
                    info="One AI image for entire video with audio visualizer and Ken Burns effects"
                )
                gr.Markdown("---")
                gr.Markdown("### ⭕ Circle Video Overlay (NEW!)")
                enable_circle_overlay = gr.Checkbox(
                    label="Enable Circle Video Overlay",
                    value=False,
                    info="Add a circular video overlay from video-overlays folder."
                )
                overlay_selection = gr.Dropdown(
                    label="Circle Overlay Video",
                    choices=["Random"] + generator.available_overlays,
                    value="Random",
                    info="Choose a specific video for the circle overlay"
                )
                with gr.Row():
                    circle_diameter = gr.Slider(
                        minimum=150,
                        maximum=500,
                        value=300,
                        step=50,
                        label="Circle Diameter (px)",
                        info="Size of the circular overlay"
                    )
                    circle_position = gr.Dropdown(
                        label="Circle Position",
                        choices=[
                            "top-left",
                            "top-right",
                            "bottom-left",
                            "bottom-right",
                            "center"
                        ],
                        value="top-right",
                        info="Where to place the circle overlay"
                    )
                circle_border_width = gr.Slider(
                    minimum=0,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Border Width (px)",
                    info="Thickness of the white border around circle"
                )
                gr.Markdown("---")
                enable_intro = gr.Checkbox(label="Add Welcome Intro Slide", value=True)
                enable_cta = gr.Checkbox(label="Add Call-to-Action Outro Slide", value=True)
                enable_music = gr.Checkbox(label="Enable Background Music", value=True)
                music_dropdown = gr.Dropdown(
                    label="🎵 Select Background Music",
                    choices=generator.available_music,
                    value="Random"
                )
                use_random_voices = gr.Checkbox(
                    label="🗣️ Use Random Voice Per Sentence",
                    value=False
                )
                music_volume = gr.Slider(-40, -5, value=-15, step=1, label="Music Volume (dB)")
                pexels_keyword = gr.Textbox(
                    label="Manual Keyword Override (Optional)",
                    placeholder="Leave empty for per-sentence AI extraction"
                )
                progress_bar = gr.Textbox(label="Progress", value="Ready...", interactive=False)
                generate_button = gr.Button("🎥 Generate Video", variant="primary", size="lg")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown()
        gr.Markdown("""
        ### ✅ Performance Optimization
        - **SQLite Caching**: TTS and full videos are reused if inputs unchanged
        - **CPU-Only**: Safe for systems without GPU
        - **Mixed Mode (DEFAULT)**: 20% AI images, 80% videos — best speed/creativity balance
        - **Single Image Mode**: Ideal for long podcasts or narration
        - **Circle Overlay**: Add branding or presenter PIP
        """)
        def generate_wrapper(text, speaker, bg_hex, keyword, enable_music, music_selection, music_vol,
                             enable_intro, enable_cta, random_voices, use_sd_pref, media_type_val,
                             use_single_image, enable_overlay, overlay_diameter, overlay_position, 
                             overlay_border, overlay_selection_val, progress=gr.Progress()):
            if not text or not text.strip():
                return None, None, "❌ Error: Please enter some text", "Ready..."
            bg_hex = bg_hex.lstrip('#')
            try:
                bg_color = tuple(int(bg_hex[i:i + 2], 16) for i in (0, 2, 4))
            except:
                bg_color = (74, 144, 226)
            keyword = keyword.strip() if keyword else None
            def update_progress(current, total, message):
                progress((current, total), desc=message)
                return f"Progress: {current}/{total} - {message}"
            result = generator.generate_video(
                text=text,
                speaker_id=speaker,
                bg_color=bg_color,
                pexels_keyword=keyword,
                enable_background_music=enable_music,
                music_selection=music_selection,
                music_volume_db=music_vol,
                add_intro_slide=enable_intro,
                add_call_to_action=enable_cta,
                use_random_voices=random_voices,
                use_stable_diffusion=use_sd_pref,
                media_type=media_type_val,
                use_single_image_mode=use_single_image,
                enable_circle_overlay=enable_overlay,
                circle_diameter=overlay_diameter,
                circle_position=overlay_position,
                circle_border_width=overlay_border,
                overlay_selection=overlay_selection_val,
                progress_callback=update_progress
            )
            if result.get("success"):
                voices_info = ""
                if result.get('random_voices') and result.get('voices_used'):
                    voices_list = result['voices_used']
                    voices_summary = ', '.join([v[:20] + '...' if len(v) > 20 else v for v in voices_list[:5]])
                    if len(voices_list) > 5:
                        voices_summary += f" ... ({len(voices_list)} total)"
                    voices_info = f"\n- Voices Used: {voices_summary}"
                music_info = ""
                if result.get('music_used'):
                    music_info = f"\n- Music Track: {result['music_used']}"
                logo_info = ""
                if result.get('logo_included'):
                    logo_info = "\n- 🖼️ Logo Overlay: Active"
                sd_info = ""
                if result.get('sd_images_used'):
                    mode_desc = " (Single)" if result.get('single_image_mode') else ""
                    sd_info = f"\n- 🖼️ AI Images Generated: {result['sd_images_used']}{mode_desc}"
                media_info = f"\n- Media Type: {result.get('media_type', 'mixed').replace('_', ' ').title()}"
                overlay_info = ""
                if result.get('circle_overlay_enabled'):
                    overlay_pos = result.get('circle_overlay_position', 'top-right')
                    overlay_sel = result.get('circle_overlay_selection', 'Random')
                    overlay_info = f"\n- ⭕ Circle Overlay: Enabled ({overlay_pos}, {overlay_sel})"
                status = f"""✅ **Video Created Successfully!**
**Details:**
- Sentences: {result['sentence_count']}
- Format: {result['video_format']}
- Text Style: {result.get('text_style', 'Subtitle-style')}
- Intro Slide: {'Yes' if result.get('intro_included') else 'No'}
- CTA Outro: {'Yes' if result['cta_included'] else 'No'}
- Transitions: {result.get('transitions', 'Random effects')}{logo_info}{media_info}{sd_info}{overlay_info}
- Background Music: {'Yes' if result['background_music'] else 'No'}{music_info}
- Random Voices: {'Yes' if result.get('random_voices') else 'No'}{voices_info}
- Output: `{result['output_directory']}`
"""
                return result["audio_path"], result["video_path"], status, "✅ Complete!"
            error_msg = f"❌ **Error:** {result.get('error', 'Unknown error occurred')}"
            return None, None, error_msg, "❌ Failed"
        generate_button.click(
            fn=generate_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color_picker, pexels_keyword,
                    enable_music, music_dropdown, music_volume, enable_intro, enable_cta,
                    use_random_voices, use_sd, media_type_radio, use_single_image_mode,
                    enable_circle_overlay, circle_diameter, circle_position, circle_border_width,
                    overlay_selection],
            outputs=[audio_output, video_output, status_output, progress_bar]
        )
    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)

# =============== MAIN ===============
if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("🚨 python-dotenv not installed. Run: pip install python-dotenv")
        exit(1)
    if not MODELS_AVAILABLE:
        print("\n🚨 Missing required libraries. Please install:")
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests spacy python-dotenv diffusers transformers accelerate")
        print("python -m spacy download en_core_web_md")
    else:
        print("\n" + "=" * 80)
        print("🎥 OPTIMIZED PORTRAIT VIDEO GENERATOR (CPU-ONLY + SQLITE CACHING)")
        print("=" * 80)
        print("🧠 CPU-Only Mode: Enabled")
        if SPACY_AVAILABLE:
            print("✅ spaCy NLP: Enabled (via External API)")
        else:
            print("⚠️ spaCy NLP: Disabled")
        if SD_AVAILABLE:
            print("✅ Stable Diffusion: Enabled")
        else:
            print("⚠️ Stable Diffusion: Disabled (install with: pip install diffusers transformers accelerate)")
        print("\n✅ SQLITE CACHING ACTIVE:")
        print("   - TTS: Reuses audio if (text + speaker) matches")
        print("   - Videos: Reuses full output if all parameters match")
        print("\n⭕ NEW FEATURE: Circle Video Overlay")
        generator = TextToVideoGenerator()
        print(f"\n🗣️  Available voices: {len(generator.available_voices)}")
        for voice in generator.available_voices:
            print(f"   - {voice}")
        print(f"\n🎵 Available music tracks: {len(generator.available_music)}")
        for music in generator.available_music:
            print(f"   - {music}")
        print(f"\n⭕ Circle overlay status: {generator.available_overlays[0]}")
        print(f"\n🖼️ Logo status: {'✅ Loaded' if generator.video_generator.logo_clip else '❌ Not found (place image in background_images/)'}")
        print("\n💡 Ensure Ollama is running (`ollama serve`) and API keys are in `.env`!")
        setup_ui(generator)