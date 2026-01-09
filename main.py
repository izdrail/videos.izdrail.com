import os
import re
import random
import shutil
import traceback
import platform
import subprocess
import uuid
import abc
import yt_dlp
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import torch
import torchaudio

# Core imports
from core.config import Config
from core.database import GenerationDB, DB
from core.nlp.keyword_extractor import KeywordExtractor
from core.ai.stable_diffusion import StableDiffusionManager, SD_AVAILABLE
from core.media.manager import MediaManager
from core.tts.manager import TTSManager
from core.utils.audio import improve_audio_quality, remove_metallic_artifacts
from core.utils.video import get_video_duration, has_audio_stream, is_video_file, get_random_middle_frame, get_smart_thumbnail_frame

# Availability flags
MODELS_AVAILABLE = True # Assumed true since imports above succeeded
SPACY_AVAILABLE = True  # Used in main block

import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
import textwrap
from dotenv import load_dotenv

# Enforce CPU globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)
load_dotenv()

# Get shared config
config_instance = Config()
SUPPORTED_LANGUAGES = config_instance.SUPPORTED_LANGUAGES

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# =============== MEDIA SOURCE CACHE ===============
_media_source_cache = {}

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


# =============== VIDEO GENERATOR WITH FFMPEG ===============
class FFmpegVideoGenerator:
    def __init__(self, config: Config, keyword_extractor: Optional[KeywordExtractor] = None):
        self.config = config
        self.font_path = self._discover_fonts()
        self.media_manager = MediaManager()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.logo_path = self._find_logo()
        self.sd_manager = None
        if SD_AVAILABLE:
            try:
                self.sd_manager = StableDiffusionManager()
                print("[SD] Enabled")
            except Exception as e:
                print(f"[SD] Init error: {e}")

    def _find_logo(self) -> Optional[Path]:
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            if self.config.IMAGES_DIR.exists():
                files = list(self.config.IMAGES_DIR.glob(ext))
                if files:
                    return files[0]
        return None

    def _clean_text(self, text: str) -> str:
        """Clean text of custom metadata tags for visual display."""
        # This mirrors the logic in TTSManager to ensure consistency
        text = re.sub(r'\[\d+\s+levels?\](?:\([^)]+\))?', '', text)
        return text.strip()

    def _discover_fonts(self) -> str:
        font_paths = []
        system = platform.system()
        if system == "Windows":
            font_paths.append(Path("C:/Windows/Fonts"))
        elif system == "Darwin":
            font_paths.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts")])
        else:
            font_paths.extend([
                Path("/usr/share/fonts/truetype"),
                Path("/usr/share/fonts/truetype/dejavu"),
                Path("/usr/share/fonts/TTF"),
            ])
        common_fonts = ["DejaVuSans-Bold.ttf", "arial.ttf", "FreeSansBold.ttf", "NotoSans-Bold.ttf"]
        for path in font_paths:
            if path.is_dir():
                for font_name in common_fonts:
                    if (path / font_name).exists():
                        return str((path / font_name).resolve())
        return "DejaVuSans"

    def get_available_music_files(self) -> List[Dict[str, str]]:
        music_files = []
        if self.config.MUSIC_DIR.exists():
            for ext in ['*.mp3', '*.wav', '*.m4a']:
                for file_path in self.config.MUSIC_DIR.glob(ext):
                    music_files.append({
                        'name': file_path.name,
                        'path': str(file_path)
                    })
        return sorted(music_files, key=lambda x: x['name'])

    def get_music_by_name(self, music_name: str) -> Optional[Path]:
        if not music_name or music_name == "Random":
            music_files = self.get_available_music_files()
            if music_files:
                return Path(random.choice(music_files)['path'])
            return None
        music_files = self.get_available_music_files()
        for mf in music_files:
            if mf['name'] == music_name:
                return Path(mf['path'])
        return None

    def get_background_video(self, keyword: Optional[str], sentence: Optional[str], language: str = 'en', preferred_source: Optional[str] = None) -> Optional[Path]:
        # Collect all candidate keywords
        search_keywords = []
        if keyword:
            search_keywords.append(self.keyword_extractor.sanitize_keyword(keyword))
        elif sentence:
            extracted = self.keyword_extractor.extract_keywords(sentence, 10, language)
            # Sort: unused first
            extracted.sort(key=lambda kw: kw in self.keyword_extractor.used_keywords)
            search_keywords.extend(extracted)

        for kw in search_keywords:
            if not kw: continue
            
            # Skip if already used (unless it's the only option)
            if kw in self.keyword_extractor.used_keywords and len(search_keywords) > 1:
                continue
                
            video = self.media_manager.get_random_media(kw, preferred_source)
            if video:
                self.keyword_extractor.used_keywords.add(kw)
                return video
                
        for ext in ['*.mp4', '*.mov', '*.avi']:
            if self.config.VIDEOS_DIR.exists():
                files = list(self.config.VIDEOS_DIR.glob(ext))
                if files:
                    selected = random.choice(files)
                    print(f"📁 [Local] Using local background video: {selected.name}")
                    return selected
        
        print("💡 [Fallback] No video found from APIs or local folder. Slide will use generated image if SD available.")
        return None

    def get_circle_overlay_video(self) -> Optional[Path]:
        videos = []
        for ext in ['*.mp4', '*.mov', '*.avi', '*.webm']:
            if self.config.CIRCLE_OVERLAYS_DIR.exists():
                videos.extend(self.config.CIRCLE_OVERLAYS_DIR.glob(ext))
        return random.choice(videos) if videos else None

    def _create_text_overlay_png(self, text: str, output_path: Path) -> Path:
        # Clean text for display
        text = self._clean_text(text)
        
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
        padding = 10
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new('L', mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        text_pos = (padding, padding)
        mask_draw.text(text_pos, wrapped_text, font=font, fill=255)
        
        # 4. Create gradient image for the text
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")
        
        # 5. Composite text onto final frame
        final_frame = Image.new('RGBA', img_size, (0, 0, 0, 0))
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
        
        final_frame.save(str(output_path), "PNG")
        return output_path

    def _create_intro_text_png(self, output_path: Path, language: str = 'en') -> Path:
        img_size = self.config.VIDEO_SIZE
        final_frame = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(final_frame)
        
        try:
            font_large = ImageFont.truetype(self.font_path, 120) if self.font_path and os.path.exists(self.font_path) else ImageFont.load_default()
            font_small = ImageFont.truetype(self.font_path, 60) if self.font_path and os.path.exists(self.font_path) else ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        intro_msg = self.config.INTRO_MESSAGES.get(language, self.config.INTRO_MESSAGES['en'])
        lines = intro_msg.upper().split()
        main_text = "\n".join(lines[:2]) if len(lines) > 1 else lines[0]
        
        # Calculate bbox
        temp_draw = ImageDraw.Draw(Image.new('L', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), main_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create mask and gradient for main text
        padding = 20
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new('L', mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.text((padding, padding), main_text, font=font_large, fill=255)
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")
        
        # Position
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2 - 150
        
        # Draw shadow
        for adj in range(-4, 5):
            if adj != 0:
                draw.text((x + adj, y), main_text, font=font_large, fill=(0, 0, 0, 180))
                draw.text((x, y + adj), main_text, font=font_large, fill=(0, 0, 0, 180))
        
        # Paste gradient text
        text_layer = Image.new('RGBA', mask_size, (0, 0, 0, 0))
        text_layer.paste(grad_img, (0, 0), mask_img)
        final_frame.paste(text_layer, (x - padding, y - padding), text_layer)
        
        # Secondary text
        if len(lines) > 2:
            sec_text = " ".join(lines[2:]).upper()
            bbox2 = temp_draw.textbbox((0, 0), sec_text, font=font_small)
            text_width2 = bbox2[2] - bbox2[0]
            x2 = (self.config.VIDEO_WIDTH - text_width2) // 2
            y2 = y + text_height + 100
            for adj in range(-2, 3):
                if adj != 0:
                    draw.text((x2 + adj, y2), sec_text, font=font_small, fill=(0,0,0,150))
            # Gradient styling for secondary text
            mask_size2 = (text_width2 + padding * 2, text_height + padding * 2) # Use main text height approx or recalc
            mask_img2 = Image.new('L', mask_size2, 0)
            mask_draw2 = ImageDraw.Draw(mask_img2)
            mask_draw2.text((padding, padding), sec_text, font=font_small, fill=255)
            grad_img2 = create_gradient_image(mask_size2, LARAVEL_ACCENT_GRADIENT, "to_right")
            
            text_layer2 = Image.new('RGBA', mask_size2, (0, 0, 0, 0))
            text_layer2.paste(grad_img2, (0, 0), mask_img2)
            final_frame.paste(text_layer2, (x2 - padding, y2 - padding), text_layer2)
            
        final_frame.save(str(output_path), "PNG")
        return output_path

    def _create_cta_text_png(self, output_path: Path, language: str = 'en') -> Path:
        img_size = self.config.VIDEO_SIZE
        final_frame = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(final_frame)
        
        try:
            font_large = ImageFont.truetype(self.font_path, 130) if self.font_path and os.path.exists(self.font_path) else ImageFont.load_default()
            font_small = ImageFont.truetype(self.font_path, 70) if self.font_path and os.path.exists(self.font_path) else ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        cta_msg = self.config.CTA_MESSAGES.get(language, self.config.CTA_MESSAGES['en'])
        parts = cta_msg.split(',')
        main_text = parts[0].strip().upper()
        if len(parts) > 1:
            main_text += "\n" + parts[1].strip().upper()
            
        # Calculate bbox
        temp_draw = ImageDraw.Draw(Image.new('L', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), main_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create mask and gradient for main text
        padding = 20
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new('L', mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.text((padding, padding), main_text, font=font_large, fill=255)
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")
        
        # Position
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2
        
        # Draw shadow
        for adj in range(-4, 5):
            if adj != 0:
                draw.text((x + adj, y), main_text, font=font_large, fill=(0, 0, 0, 180))
                draw.text((x, y + adj), main_text, font=font_large, fill=(0, 0, 0, 180))
        
        # Paste gradient text
        text_layer = Image.new('RGBA', mask_size, (0, 0, 0, 0))
        text_layer.paste(grad_img, (0, 0), mask_img)
        final_frame.paste(text_layer, (x - padding, y - padding), text_layer)
        
        final_frame.save(str(output_path), "PNG")
        return output_path

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r'\bDr\.', 'Dr<dot>', text)
        text = re.sub(r'\bMr\.', 'Mr<dot>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<dot>', text)
        text = re.sub(r'\bMs\.', 'Ms<dot>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)
        raw_sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        sentences = [s.replace('<dot>', '.').strip() for s in raw_sentences if s.strip()]
        for i, sentence in enumerate(sentences):
            if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
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
        
    def _generate_overlay_mask(self, shape: str, diameter: int) -> Path:
        """Generates a high-quality PNG mask for the specified shape using Pillow."""
        mask_path = self.config.TEMP_DIR / f"mask_{shape.lower()}_{uuid.uuid4().hex[:8]}.png"
        
        # Create high-res mask (2x size) for anti-aliasing
        size = diameter * 2
        # Use RGBA mode (transparent background)
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        if shape == "Circle":
            draw.ellipse((0, 0, size, size), fill=(255, 255, 255, 255))
        elif shape == "Square":
            draw.rectangle((0, 0, size, size), fill=(255, 255, 255, 255))
        elif shape == "Rectangle":
             # "Rectangle" in this context is just a square crop that fills the bounding box
             # To make it distinct, let's give it rounded corners
            corner_radius = size // 10
            draw.rounded_rectangle((0, 0, size, size), radius=corner_radius, fill=(255, 255, 255, 255))
        elif shape == "Star":
            # 5-pointed star
            cx, cy = size // 2, size // 2
            outer_radius = size // 2
            inner_radius = outer_radius * 0.4 # Ratio for star thickness
            points = []
            import math
            angle = -math.pi / 2 # Start at top
            step = math.pi / 5 # 36 deg
            
            for i in range(10):
                r = outer_radius if i % 2 == 0 else inner_radius
                x = cx + math.cos(angle) * r
                y = cy + math.sin(angle) * r
                points.append((x, y))
                angle += step
            draw.polygon(points, fill=(255, 255, 255, 255))
        else:
            # Fallback to full white square
            draw.rectangle((0, 0, size, size), fill=(255, 255, 255, 255))
            
        # Downscale for smooth edges
        image = image.resize((diameter, diameter), Image.Resampling.LANCZOS)
        image.save(str(mask_path))
        return mask_path

    def _create_slide_with_ffmpeg(self, sentence: str, audio_path: Path, video_path: Optional[Path],
                                  output_path: Path, slide_num: int, is_intro: bool = False,
                                  is_cta: bool = False, circle_video: Optional[Path] = None,
                                  circle_config: Optional[Dict] = None, language: str = 'en',
                                  hide_text: bool = False,
                                  export_fps: int = 30,
                                  overlay_shape: str = "Circle") -> Optional[Path]:
        try:
            if audio_path is None or not Path(audio_path).exists():
                print(f"❌ [FFmpeg] Slide {slide_num} error: audio_path is None or does not exist")
                return None
            
            source_info = f"Video: {video_path.name}" if video_path else "Background: Image/Color"
            # Get audio duration
            duration = get_video_duration(audio_path)
            print(f"🎬 [FFmpeg] Creating slide {slide_num} ({language}) - {source_info} - duration: {duration:.2f}s")

            if not hide_text:
                text_overlay_path = self.config.TEMP_DIR / f"text_{slide_num}_{uuid.uuid4().hex[:8]}.png"
                if is_intro:
                    self._create_intro_text_png(text_overlay_path, language)
                elif is_cta:
                    self._create_cta_text_png(text_overlay_path, language)
                else:
                    self._create_text_overlay_png(sentence, text_overlay_path)
            else:
                text_overlay_path = None

            inputs = []
            filter_parts = []
            input_count = 0

            if video_path and video_path.exists():
                inputs.extend(['-stream_loop', '-1', '-i', str(video_path)])
                filter_parts.append(
                    f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,"
                    f"fps={export_fps},"
                    f"trim=duration={duration},"
                    f"setpts=PTS-STARTPTS[bg_scaled]"
                )
                input_count = 1
            else:
                print(f"🎨 [FFmpeg] Slide {slide_num}: Using branded gradient background")
                grad_path = self.config.TEMP_DIR / f"grad_{slide_num}_{uuid.uuid4().hex[:8]}.png"
                create_gradient_image(self.config.VIDEO_SIZE, LARAVEL_BG_GRADIENT, "135deg").save(str(grad_path))
                inputs.extend(['-loop', '1', '-i', str(grad_path)])
                filter_parts.append(f"[0:v]fps={export_fps},trim=duration={duration}[bg_scaled]")
                input_count = 1

            filter_parts.append("[bg_scaled]format=rgba,colorchannelmixer=aa=0.6[dimmed]")

            if text_overlay_path:
                inputs.extend(['-loop', '1', '-i', str(text_overlay_path)])
                filter_parts.append(f"[dimmed][{input_count}:v]overlay=0:0:format=auto[with_text]")
                input_count += 1
            else:
                # No text overlay; keep dimmed background as final layer
                logo_label = "dimmed"
                # No increment of input_count needed

            logo_label = "with_text" if text_overlay_path else "dimmed"
            if self.logo_path and self.logo_path.exists():
                inputs.extend(['-loop', '1', '-i', str(self.logo_path)])
                cfg = self.config.LOGO_CONFIG
                pos_map = {
                    'top-left': f'{cfg["margin"]}:{cfg["margin"]}',
                    'top-right': f'W-w-{cfg["margin"]}:{cfg["margin"]}',
                    'bottom-left': f'{cfg["margin"]}:H-h-{cfg["margin"]}',
                    'bottom-right': f'W-w-{cfg["margin"]}:H-h-{cfg["margin"]}',
                }
                pos = pos_map.get(cfg['position'], pos_map['top-left'])
                filter_parts.append(f"[{input_count}:v]scale=150:150[logo_scaled]")
                filter_parts.append(f"[{logo_label}][logo_scaled]overlay={pos}:format=auto[final]")
                logo_label = "final"
                input_count += 1

            if circle_video and circle_video.exists() and circle_config:
                inputs.extend(['-stream_loop', '-1', '-i', str(circle_video)])
                diameter = circle_config.get('diameter', 300)
                position = circle_config.get('position', 'top-right')
                pos_map = {
                    'top-left': '50:50',
                    'top-right': f'W-w-50:50',
                    'bottom-left': '50:H-h-50',
                    'bottom-right': 'W-w-50:H-h-50',
                    'center': '(W-w)/2:(H-h)/2',
                }
                overlay_pos = pos_map.get(position, pos_map['top-right'])
                # Generate procedural mask
                mask_path = self._generate_overlay_mask(overlay_shape, diameter)
                inputs.extend(['-loop', '1', '-i', str(mask_path)])
                mask_idx = input_count + 1 # video is input_count, mask is input_count+1
                
                # Scale video to COVER the shape (increase + crop)
                filter_parts.append(
                    f"[{input_count}:v]scale={diameter}:{diameter}:force_original_aspect_ratio=increase,"
                    f"crop={diameter}:{diameter},"
                    f"fps={export_fps},"
                    f"trim=duration={duration},"
                    f"setpts=PTS-STARTPTS,"
                    f"format=rgba[circle_sized]"
                )
                
                # Apply mask
                filter_parts.append(f"[{mask_idx}:v]alphaextract[mask_alpha]")
                filter_parts.append(f"[circle_sized][mask_alpha]alphamerge[circle_masked]")
                
                filter_parts.append(f"[{logo_label}][circle_masked]overlay={overlay_pos}:format=auto[final_with_circle]")
                logo_label = "final_with_circle"
                input_count += 2 # Incremented by 2 (video + mask)


            print(f"🎬 [FFmpeg] Slide {slide_num}: Composition started. Layers: bg={video_path.name if video_path else 'Branded Gradient'}, text={text_overlay_path.name if text_overlay_path else 'None'}")

            audio_idx = input_count
            inputs.extend(['-i', str(audio_path)])

            filter_complex = ";".join(filter_parts)
            cmd = [
                'ffmpeg', '-y',
                *inputs,
                '-filter_complex', filter_complex,
                '-map', f'[{logo_label}]',
                '-map', f'{audio_idx}:a',
                '-c:v', 'libx264',
                '-preset', self.config.VIDEO_PRESET,
                '-crf', str(self.config.VIDEO_CRF),
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-r', str(export_fps),
                '-shortest',
                '-movflags', '+faststart',
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ [FFmpeg] Slide {slide_num}: Successfully composed.")
            if not output_path.exists():
                print(f"[FFmpeg] ERROR: Output not created for slide {slide_num}")
                return None
            file_size = output_path.stat().st_size
            print(f"[FFmpeg] Slide {slide_num} created: {file_size / 1024 / 1024:.2f} MB")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[FFmpeg] Slide {slide_num} failed with return code {e.returncode}")
            print(f"[FFmpeg] Command: {' '.join(e.cmd)}")
            print(f"[FFmpeg] Stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"[FFmpeg] Slide {slide_num} error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if 'text_overlay_path' in locals() and text_overlay_path and text_overlay_path.exists():
                text_overlay_path.unlink(missing_ok=True)

    def create_final_video(self, sentences: List[str], audio_paths: List[Path],
                          keywords: List[Optional[str]], intro_audio: Optional[Path] = None,
                          cta_audio: Optional[Path] = None,
                          music_path: Optional[Path] = None,
                          music_volume_db: int = -20,
                          circle_video: Optional[Path] = None,
                          circle_config: Optional[Dict] = None,
                          circle_selection: str = "Random",
                          language: str = 'en',
                          preferred_media_source: Optional[str] = None,
                          selected_background_video: Optional[Path] = None,
                          hide_text: bool = False,
                          export_fps: int = 30,
                          overlay_shape: str = "Circle",
                          progress_callback=None) -> Path:
        temp_dir = self.config.TEMP_DIR / f"final_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(exist_ok=True)
        
        slide_paths = []
        source_videos = []
        
        # Parallel slide generation
        with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_SLIDES) as executor:
            futures = []
            
            # Submit intro slide if needed
            if intro_audio:
                # If a specific background video is selected, use it for all slides including intro/cta
                intro_video_bg = selected_background_video if selected_background_video else self.get_background_video("intro", "Welcome", language, preferred_media_source)
                if intro_video_bg:
                    source_videos.append(intro_video_bg)
                intro_output = temp_dir / "slide_intro.mp4"
                futures.append(executor.submit(
                    self._create_slide_with_ffmpeg,
                    "", intro_audio, intro_video_bg, intro_output, -1, # -1 for intro slide_num
                    True, False, None, None, language, hide_text, export_fps, overlay_shape # No circle for intro/cta but passing shape for consistency if needed or ignored
                ))

            # Submit main content slides
            for i, (sentence, audio_path, keyword) in enumerate(zip(sentences, audio_paths, keywords)):
                # Get background video for this slide
                video_bg = selected_background_video if selected_background_video else self.get_background_video(keyword, sentence, language, preferred_media_source)
                if video_bg:
                    source_videos.append(video_bg)
                
                output_path = temp_dir / f"slide_{i:03d}.mp4"
                futures.append(executor.submit(
                    self._create_slide_with_ffmpeg,
                    sentence, audio_path, video_bg, output_path, i,
                    False, False, circle_video, circle_config, language, hide_text, export_fps, overlay_shape
                ))
                
            # Submit CTA slide if needed
            if cta_audio:
                cta_video_bg = selected_background_video if selected_background_video else self.get_background_video("outro", "Goodbye", language, preferred_media_source)
                if cta_video_bg:
                    source_videos.append(cta_video_bg)
                cta_output = temp_dir / "slide_cta.mp4"
                futures.append(executor.submit(
                    self._create_slide_with_ffmpeg,
                    "", cta_audio, cta_video_bg, cta_output, 999, # 999 for cta slide_num
                    False, True, None, None, language, hide_text, export_fps, overlay_shape # No circle for intro/cta
                ))
                


            # Collect results in order of slide_num
            results_map = {}
            for i, future in enumerate(as_completed(futures)):
                try:
                    path_created = future.result()
                    if path_created:
                        slide_paths.append(path_created)
                    if progress_callback:
                        progress_callback(i + 1, len(futures), "Generating slides...")
                except Exception as e:
                    print(f"[FFmpeg] Slide error: {e}")
        
        # Sort slide_paths based on their slide_num
        slide_paths.sort(key=lambda p: int(re.search(r'slide_(\w+)\.mp4', p.name).group(1)) if re.search(r'slide_(\d+)\.mp4', p.name) else (0 if 'intro' in p.name else (9999 if 'cta' in p.name else 0)))


        if not slide_paths:
            raise ValueError("No slides created")

        if progress_callback:
            progress_callback(len(sentences), len(sentences), "Concatenating slides...")

        concat_file = self.config.TEMP_DIR / f"concat_{uuid.uuid4().hex[:8]}.txt"
        with open(concat_file, 'w') as f:
            for slide_path in slide_paths:
                f.write(f"file '{slide_path.absolute()}'\n")

        output_path = self.config.TEMP_DIR / f"final_{uuid.uuid4().hex[:8]}.mp4"
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(output_path)
        ]
        print("[FFmpeg] Concatenating slides...")
        subprocess.run(concat_cmd, check=True, capture_output=True)
        concat_file.unlink(missing_ok=True)
        
        # Cleanup rendered slides
        for slide_path in slide_paths:
            try:
                slide_path.unlink(missing_ok=True)
            except:
                pass
        
        # Cleanup source background videos (only those downloaded/sourced for this session)
        print(f"🧹 [Cleanup] Removing {len(source_videos)} source background videos...")
        for sv_path in source_videos:
            try:
                # Check if it's in the background_videos directory (don't delete user/permanent assets elsewhere)
                if sv_path.exists() and str(self.config.VIDEOS_DIR.absolute()) in str(sv_path.absolute()):
                    sv_path.unlink(missing_ok=True)
                    # Also try to remove the parent directory if it's empty (keyword folders)
                    parent = sv_path.parent
                    if parent != self.config.VIDEOS_DIR and parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
            except Exception as e:
                print(f"[Cleanup] Error removing source video {sv_path}: {e}")

        return output_path

# =============== TEXT TO VIDEO GENERATOR ===============
class TextToVideoGenerator:
    def __init__(self):
        self.config = Config()
        self.keyword_extractor = KeywordExtractor()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = FFmpegVideoGenerator(self.config, keyword_extractor=self.keyword_extractor)
        self.available_voices = self._get_available_voices()
        self.available_music = self._get_available_music()
        self.available_circles = self._get_available_circles()
        self.available_languages = list(SUPPORTED_LANGUAGES.keys())
        self.available_languages = list(SUPPORTED_LANGUAGES.keys())
        self.available_models = self.keyword_extractor.get_available_models()
        self.available_background_videos = self._get_available_background_videos()

    def _get_available_voices(self) -> List[str]:
        # 1. Start with Standard voice
        voices = [self.config.STANDARD_VOICE_NAME]
        
        # 2. Add Kokoro Preset Voices
        try:
            kokoro_voices = self.tts_manager.get_available_voices("kokoro")
            voices.extend(kokoro_voices)
        except Exception:
            pass
            
        # 3. Add XTTS Cloned Voices (from folders)
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
            
        # Deduplicate and sort
        return sorted(list(set(voices)))

    def _get_available_music(self) -> List[str]:
        music_files = self.video_generator.get_available_music_files()
        return ["Random"] + [m['name'] for m in music_files]

    def _get_available_circles(self) -> List[str]:
        circles = []
        for ext in ['*.mp4', '*.mov', '*.avi', '*.webm']:
            if self.config.CIRCLE_OVERLAYS_DIR.exists():
                circles.extend(list(self.config.CIRCLE_OVERLAYS_DIR.glob(ext)))
        return ["Random"] + [v.name for v in sorted(circles)]

    def _get_available_background_videos(self) -> List[str]:
        videos = []
        for ext in ['*.mp4', '*.mov', '*.avi', '*.webm']:
            if self.config.BACKGROUND_VIDEOS_DIR.exists():
                videos.extend(list(self.config.BACKGROUND_VIDEOS_DIR.glob(ext)))
        return ["Auto-select (Pexels/Giphy/Local)", "Branded Gradient"] + [v.name for v in sorted(videos)]

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                          language: str = 'en',
                          pexels_keyword: Optional[str] = None,
                          preferred_media_source: Optional[str] = None,
                          selected_background_video_name: Optional[str] = None,  # New parameter
                          enable_background_music: bool = True,
                          music_selection: str = "Random",
                          music_volume_db: int = -15,
                          add_intro_slide: bool = True,
                          add_call_to_action: bool = True,
                          use_random_voices: bool = False,
                          enable_circle_overlay: bool = False,
                          circle_diameter: int = 300,
                          circle_position: str = "top-right",
                           circle_border_width: int = 5,
                           circle_selection: str = "Random",
                           circle_upload_path: Optional[str] = None,
                           hide_text: bool = False,
                           export_fps: int = 30,
                           overlay_shape: str = "Circle",
                           ai_model: str = "mistral:7b",
                           stress_level: float = 1.0,
                           progress_callback=None) -> Dict:
        # Reset keywords for this session
        self.keyword_extractor.clear_used()
        input_params = {
            'text': text,
            'speaker_id': speaker_id,
            'language': language,
            'pexels_keyword': pexels_keyword,
            'preferred_media_source': preferred_media_source,
            'selected_background_video_name': selected_background_video_name, # New parameter
            'enable_background_music': enable_background_music,
            'music_selection': music_selection,
            'music_volume_db': music_volume_db,
            'add_intro_slide': add_intro_slide,
            'add_call_to_action': add_call_to_action,
            'use_random_voices': use_random_voices,
            'enable_circle_overlay': enable_circle_overlay,
            'circle_diameter': circle_diameter,
            'circle_position': circle_position,
            'circle_border_width': circle_border_width,
            'circle_selection': circle_selection,
            'circle_upload_path': str(circle_upload_path) if circle_upload_path else None,
            'hide_text': hide_text,
            'export_fps': export_fps,
            'overlay_shape': overlay_shape,
            'ai_model': ai_model,
            'stress_level': stress_level
        }

        # Update model if changed
        if ai_model and ai_model != self.keyword_extractor.model:
            print(f"[Ollama] Switching model from {self.keyword_extractor.model} to {ai_model}")
            self.keyword_extractor.model = ai_model

        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}
        if len(text) > 10000:
            return {"error": "Text too long (max 10,000 chars)", "success": False}

        sentences = self.video_generator.split_into_sentences(text)
        if len(sentences) > 100:
            return {"error": "Too many sentences (max 100)", "success": False}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}_{language}"
        session_dir.mkdir(exist_ok=True)

        sentence_keywords = []
        for sent in sentences:
            kw = self.keyword_extractor.get_best_unique_keyword(sent, language)
            sentence_keywords.append(kw)

        audio_paths = []
        intro_audio_path = None
        cta_audio_path = None
        music_path = None

        try:
            if enable_background_music:
                music_path = self.video_generator.get_music_by_name(music_selection)

            voices_for_sentences = [random.choice(self.available_voices) for _ in sentences] if use_random_voices else [speaker_id] * len(sentences)

            if add_intro_slide:
                intro_voice = speaker_id if not use_random_voices else random.choice(self.available_voices)
                intro_msg = self.config.INTRO_MESSAGES.get(language, self.config.INTRO_MESSAGES['en'])
                intro_audio_path = self.tts_manager.generate_speech(intro_msg, intro_voice, language)

            for i, (sentence, voice) in enumerate(zip(sentences, voices_for_sentences)):
                if progress_callback:
                    progress_callback(i + 1, len(sentences) * 2, f"Generating audio {i + 1}/{len(sentences)}")
                audio_path = self.tts_manager.generate_speech(sentence, voice, language)
                audio_paths.append(audio_path)

            if add_call_to_action:
                cta_voice = speaker_id if not use_random_voices else voices_for_sentences[-1]
                cta_msg = self.config.CTA_MESSAGES.get(language, self.config.CTA_MESSAGES['en'])
                cta_audio_path = self.tts_manager.generate_speech(cta_msg, cta_voice, language)

            def video_progress(current, total, message):
                if progress_callback:
                    progress_callback(len(sentences) + current, len(sentences) * 2, message)

            circle_config = {
                'diameter': circle_diameter,
                'position': circle_position,
                'border_width': circle_border_width,
            }
            circle_video_path = None
            if enable_circle_overlay:
                if circle_upload_path and Path(circle_upload_path).exists():
                    # Move uploaded file to session dir just in case
                    uploaded_path = Path(circle_upload_path)
                    circle_video_path = session_dir / f"uploaded_circle_{uploaded_path.name}"
                    shutil.copy(circle_upload_path, circle_video_path)
                elif circle_selection and circle_selection != "Random":
                    circle_video_path = self.config.CIRCLE_OVERLAYS_DIR / circle_selection
                    if not circle_video_path.exists():
                        print(f"[Circle] Selected overlay {circle_selection} not found, falling back to random")
                        circle_video_path = self.video_generator.get_circle_overlay_video()
                else:
                    circle_video_path = self.video_generator.get_circle_overlay_video()

            selected_bg_video_path = None
            print(f"[Debug] Selected background video name from UI: '{selected_background_video_name}'")
            if selected_background_video_name and selected_background_video_name not in ["Auto-select (Pexels/Giphy/Local)", "Branded Gradient"]:
                selected_bg_video_path = self.config.BACKGROUND_VIDEOS_DIR / selected_background_video_name
                print(f"[Debug] Resolving path: {selected_bg_video_path}")
                if not selected_bg_video_path.exists():
                    print(f"[Background Video] Selected background video {selected_background_video_name} not found. Falling back to auto-select.")
                    selected_bg_video_path = None
                else:
                    print(f"[Debug] Confirmed background video exists: {selected_bg_video_path}")
            elif selected_background_video_name == "Branded Gradient":
                print("[Debug] Using Branded Gradient background")
                selected_bg_video_path = None # This will trigger the gradient background in _create_slide_with_ffmpeg
            else:
                print("[Debug] Auto-select enabled (default behavior)")

            # Final Video Generation
            video_temp_path = self.video_generator.create_final_video(
                sentences=sentences,
                audio_paths=audio_paths,
                keywords=sentence_keywords,
                intro_audio=intro_audio_path,
                cta_audio=cta_audio_path,
                music_volume_db=music_volume_db,
                circle_video=circle_video_path,
                circle_config=circle_config,
                circle_selection=circle_selection,
                language=language,
                preferred_media_source=preferred_media_source,
                selected_background_video=selected_bg_video_path,
                hide_text=hide_text,
                export_fps=export_fps,
                overlay_shape=overlay_shape,
                progress_callback=video_progress
            )

            if enable_background_music and music_path and music_path.exists():
                if progress_callback:
                    progress_callback(len(sentences) * 2 - 1, len(sentences) * 2, "Adding background music...")
                audio_extract = self.config.TEMP_DIR / f"extracted_{uuid.uuid4().hex[:8]}.wav"
                subprocess.run(['ffmpeg', '-y', '-i', str(video_temp_path), '-vn', '-acodec', 'pcm_s16le', str(audio_extract)],
                             check=True, capture_output=True)
                voice_seg = AudioSegment.from_file(str(audio_extract))
                music = AudioSegment.from_file(str(music_path)) + music_volume_db
                if len(music) < len(voice_seg):
                    loops = (len(voice_seg) // len(music)) + 2
                    music = music * loops
                music = music[:len(voice_seg)]
                music = music.fade_in(1000).fade_out(1000)
                mixed = voice_seg.overlay(music)
                mixed_audio = self.config.TEMP_DIR / f"mixed_{uuid.uuid4().hex[:8]}.wav"
                mixed.export(str(mixed_audio), format="wav")
                video_with_music = self.config.TEMP_DIR / f"with_music_{uuid.uuid4().hex[:8]}.mp4"
                subprocess.run(['ffmpeg', '-y', '-i', str(video_temp_path), '-i', str(mixed_audio),
                              '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-map', '0:v:0', '-map', '1:a:0', '-shortest',
                              str(video_with_music)], check=True, capture_output=True)
                audio_extract.unlink(missing_ok=True)
                mixed_audio.unlink(missing_ok=True)
                video_temp_path.unlink(missing_ok=True)
                video_temp_path = video_with_music

            lang_name = SUPPORTED_LANGUAGES.get(language, {}).get('name', language)
            video_final = session_dir / f"video_{timestamp}_{language}.mp4"
            shutil.move(str(video_temp_path), str(video_final))

            audio_final = session_dir / f"audio_{timestamp}_{language}.mp3"
            subprocess.run(['ffmpeg', '-y', '-i', str(video_final), '-vn', '-c:a', 'libmp3lame', '-b:a', '192k', str(audio_final)],
                         check=True, capture_output=True)

            # Generate Thumbnail
            thumbnail_final = session_dir / f"thumbnail_{timestamp}_{language}.jpg"
            try:
                get_smart_thumbnail_frame(video_final, thumbnail_final)
                print(f"✅ Thumbnail generated: {thumbnail_final}")
            except Exception as e:
                print(f"❌ Thumbnail generation failed: {e}")
                thumbnail_final = None

            result = {
                "success": True,
                "audio_path": str(audio_final),
                "video_path": str(video_final),
                "thumbnail_path": str(thumbnail_final) if thumbnail_final and thumbnail_final.exists() else None,
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "language": lang_name,
                "language_code": language,
                "background_music": enable_background_music and music_path is not None,
                "music_used": music_path.name if music_path else None,
                "intro_included": add_intro_slide,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "video_backgrounds": selected_background_video_name if selected_background_video_name else "Pexels/Giphy API + Local",
                "random_voices": use_random_voices,
                "circle_overlay_enabled": enable_circle_overlay,
                "circle_position": circle_position if enable_circle_overlay else None,
                "circle_selection": circle_selection if enable_circle_overlay else None,
                "hide_text_overlay": hide_text,
            }

            DB.save_video(input_params, result)
            return result

        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "success": False}
        finally:
            for ap in audio_paths:
                try:
                    ap.unlink(missing_ok=True)
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

# =============== GRADIO UI ===============
def setup_ui(generator: TextToVideoGenerator):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="AI Video Generator Pro") as demo:
        gr.Markdown("# 🎬 AI Video Generator Pro")
        gr.Markdown("Create stunning videos with multi-language TTS, auto-backgrounds, and dynamic overlays.")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("📝 Content"):
                        text_input = gr.Textbox(
                            label="Text Content",
                            placeholder="Enter your script here...",
                            lines=10
                        )
                        
                        # Emoticon Toolbar for Intonation Control
                        gr.Markdown("### 🎭 Intonation Controls - Click to insert:")
                        with gr.Row():
                            btn_semicolon = gr.Button("😐 ;", size="sm", variant="secondary")
                            btn_colon = gr.Button("🙂 :", size="sm", variant="secondary")
                            btn_comma = gr.Button("⏸️ ,", size="sm", variant="secondary")
                            btn_period = gr.Button("⏹️ .", size="sm", variant="secondary")
                        
                        with gr.Row():
                            btn_exclaim = gr.Button("❗😄 !", size="sm", variant="secondary")
                            btn_question = gr.Button("❓🤔 ?", size="sm", variant="secondary")
                            btn_dash = gr.Button("➖😶 —", size="sm", variant="secondary")
                            btn_ellipsis = gr.Button("⏳😌 …", size="sm", variant="secondary")
                        
                        with gr.Row():
                            btn_quote = gr.Button("💬 \"", size="sm", variant="secondary")
                            btn_stress1 = gr.Button("🔊 ˈ", size="sm", variant="secondary")
                            btn_stress2 = gr.Button("🔉 ˌ", size="sm", variant="secondary")
                        
                        gr.Markdown("""
                        💡 **Advanced Pronunciation:**
                        - Link syntax: `[Word](/pronunciation/)` e.g., `[Kokoro](/kˈOkəɹO/)`
                        - Adjust stress: `[1 level](-1)` or `[2 levels](-2)`
                        """)
                        with gr.Row():
                            ai_model_dropdown = gr.Dropdown(
                                label="🤖 AI Model",
                                choices=generator.available_models,
                                value=generator.available_models[0] if generator.available_models else "mistral:7b",
                                info="Select LLM for keyword extraction"
                            )
                        
                        stress_level = gr.Slider(0.5, 1.5, 1.0, 0.1, label="Stress Level (Speed)", info="0.5 = Slow/Relaxed, 1.5 = Fast/Stressed")

                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                label="🌐 Language",
                                choices=[(SUPPORTED_LANGUAGES[k]['name'], k) for k in generator.available_languages],
                                value='en'
                            )
                            speaker_dropdown = gr.Dropdown(
                                label="🎙️ Voice",
                                choices=generator.available_voices,
                                value=generator.config.STANDARD_VOICE_NAME
                            )
                        use_random_voices = gr.Checkbox(label="🎲 Random voice per sentence", value=False)

                    with gr.TabItem("🎥 Media"):
                        media_source_dropdown = gr.Dropdown(
                            label="🎞️ Preferred Media Source",
                            choices=["Random", "Pexels", "YouTube", "Giphy", "Dailymotion", "Vimeo", "Twitch", "PeerTube", "api.video", "Cloudflare Stream", "Mux", "Kaltura", "JSON2Video"],
                            value="Random",
                            info="Select your primary source for background videos (Random shuffles available APIs)"
                        )
                        pexels_keyword = gr.Textbox(
                            label="🔍 Custom Search Keyword",
                            placeholder="e.g., 'cyberpunk city', 'peaceful forest'",
                            info="Leave empty for auto-extraction"
                        )
                        background_video_dropdown = gr.Dropdown(
                            label="🏞️ Select Background Video",
                            choices=generator.available_background_videos,
                            value="Auto-select (Pexels/Giphy/Local)",
                            info="Choose a specific video from your 'background_videos' folder or let the system auto-select."
                        )
                        with gr.Row():
                            enable_music = gr.Checkbox(label="🎵 Add background music", value=True)
                            music_dropdown = gr.Dropdown(
                                label="Music Track",
                                choices=generator.available_music,
                                value="Random"
                            )
                        music_volume = gr.Slider(-40, -5, -15, 1, label="Music Volume (dB)")

                    with gr.TabItem("⭕ Overlays"):
                        enable_circle = gr.Checkbox(label="Enable Picture-in-Picture Circle", value=False)
                        circle_selection = gr.Dropdown(
                            label="Circle Content (Local Folder)",
                            choices=generator.available_circles,
                            value="Random",
                            info="Select a video from the local 'circle_overlays' folder"
                        )
                        circle_upload = gr.File(
                            label="📤 Upload Custom Circle Video",
                            file_types=["video"],
                            type="filepath"
                        )
                        with gr.Row():
                            circle_diameter = gr.Slider(150, 600, 300, 25, label="Diameter (px)")
                            circle_border_width = gr.Slider(0, 20, 5, 1, label="Border Width (px)")
                            circle_position = gr.Dropdown(
                                ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                                value="top-right",
                                label="Position"
                            )
                            overlay_shape = gr.Dropdown(
                                ["Circle", "Rectangle", "Square", "Star"],
                                value="Circle",
                                label="Overlay Shape",
                                info="Shape of the PIP overlay"
                            )

                    with gr.TabItem("⚙️ Advanced"):
                        with gr.Row():
                            enable_intro = gr.Checkbox(label="📢 Add Intro Slide", value=True)
                            enable_cta = gr.Checkbox(label="📣 Add CTA Outro", value=True)
                        hide_text = gr.Checkbox(label="🛑 Hide Text Overlay", value=False)
                        export_fps = gr.Slider(10, 60, 30, 1, label="🎞️ Export FPS", info="Target frame rate for the final video (default: 30)")
                        gr.Markdown("""
                        ### 💡 Pronunciation & Stress Control
                        You can control pronunciation and stress in Kokoro TTS using the following syntax:
                        
                        **Link Syntax (Phonemes):**  `[Word](/pronunciation/)`  
                        Example: `[Kokoro](/kˈOkəɹO/)`
                        
                        **Neutral / Pause:**
                        - `;` → 😐 (Neutral pause)
                        - `:` → 🙂 (Slight pause with continuation)
                        - `,` → ⏸️ (Brief pause)
                        - `.` → ⏹️ (Full stop)
                        
                        **Emphasis / Emotion:**
                        - `!` → ❗😄 (Excitement/Emphasis)
                        - `?` → ❓🤔 (Question/Curiosity)
                        - `—` → ➖😶 (Long pause/Interruption)
                        - `…` → ⏳😌 (Trailing off/Thinking)
                        
                        **Speech / Quotation:**
                        - `"` → 💬 (Quoted speech)
                        
                        **Stress Markers:**  
                        - `ˈ` (Primary stress)
                        - `ˌ` (Secondary stress)
                        
                        **Adjust Stress Level:**  
                        - `[1 level](-1)` (Reduce stress by 1 level)
                        - `[2 levels](-2)` (Reduce stress by 2 levels)
                        
                        *Tip: Use the slider below to adjust overall speaking speed ('Stress Level').*
                        """)

                generate_button = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                engine_status_output = gr.Textbox(label="TTS Engine Status", value="Idle", interactive=False)
                progress_bar = gr.Textbox(label="⚡ Status", value="Ready", interactive=False)

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video", height=600)
                thumbnail_output = gr.Image(label="Last Frame Thumbnail", type="filepath")
                audio_output = gr.Audio(label="Extracted Voiceover")
                status_output = gr.Markdown(value="*Your video will appear here after generation.*")


        def generate_wrapper(text, language, speaker, use_random, media_source, keyword,
                            selected_background_video_name,
                            enable_music, music_select, music_vol,
                            enable_circle, circle_sel, circle_upload_path, circle_diam, circle_border, circle_pos, overlay_shape_val,
                            enable_intro, enable_cta, hide_text, export_fps_val, ai_model_val, stress_level_val, progress=gr.Progress()):
            
            if not text or not text.strip():
                return None, None, None, "Idle", "❌ **Error:** Please enter some text.", "Ready"

            def update_progress(current, total, message):
                progress((current, total), desc=message)
                return f"{current}/{total}: {message}"

            result = generator.generate_video(
                text=text,
                language=language,
                speaker_id=speaker,
                pexels_keyword=keyword.strip() if keyword else None,
                preferred_media_source=media_source,
                enable_background_music=enable_music,
                music_selection=music_select,
                music_volume_db=music_vol,
                add_intro_slide=enable_intro,
                add_call_to_action=enable_cta,
                use_random_voices=use_random,
                enable_circle_overlay=enable_circle,
                circle_diameter=circle_diam,
                circle_position=circle_pos,
                circle_border_width=circle_border,
                circle_selection=circle_sel,
                circle_upload_path=circle_upload_path,
                hide_text=hide_text,
                export_fps=export_fps_val,
                overlay_shape=overlay_shape_val,
                ai_model=ai_model_val,
                stress_level=stress_level_val,
                progress_callback=update_progress
            )

            engine_status = generator.tts_manager.last_status_message
            
            if result.get("success"):
                status_md = f"""### ✅ Generation Complete!
- **Video:** {result['video_path']}
- **Duration:** {result.get('duration', 'N/A')}s
- **Source:** {media_source}
"""
                return result["video_path"], result.get("thumbnail_path"), result["audio_path"], engine_status, status_md, "Complete!"
            
            return None, None, None, engine_status, f"❌ **Error:** {result.get('error', 'Unknown error')}", "Failed"
        
        # Helper function to insert text at cursor position
        def insert_symbol(current_text, symbol):
            return current_text + symbol if current_text else symbol
        
        # Connect emoticon buttons to text input
        btn_semicolon.click(lambda txt: insert_symbol(txt, ";"), inputs=[text_input], outputs=[text_input])
        btn_colon.click(lambda txt: insert_symbol(txt, ":"), inputs=[text_input], outputs=[text_input])
        btn_comma.click(lambda txt: insert_symbol(txt, ","), inputs=[text_input], outputs=[text_input])
        btn_period.click(lambda txt: insert_symbol(txt, "."), inputs=[text_input], outputs=[text_input])
        btn_exclaim.click(lambda txt: insert_symbol(txt, "!"), inputs=[text_input], outputs=[text_input])
        btn_question.click(lambda txt: insert_symbol(txt, "?"), inputs=[text_input], outputs=[text_input])
        btn_dash.click(lambda txt: insert_symbol(txt, "—"), inputs=[text_input], outputs=[text_input])
        btn_ellipsis.click(lambda txt: insert_symbol(txt, "…"), inputs=[text_input], outputs=[text_input])
        btn_quote.click(lambda txt: insert_symbol(txt, '"'), inputs=[text_input], outputs=[text_input])
        btn_stress1.click(lambda txt: insert_symbol(txt, "ˈ"), inputs=[text_input], outputs=[text_input])
        btn_stress2.click(lambda txt: insert_symbol(txt, "ˌ"), inputs=[text_input], outputs=[text_input])

        generate_button.click(
            fn=generate_wrapper,
            inputs=[
                text_input, language_dropdown, speaker_dropdown, use_random_voices,
                media_source_dropdown, pexels_keyword, background_video_dropdown,
                enable_music, music_dropdown, music_volume,
                enable_circle, circle_selection, circle_upload, circle_diameter, circle_border_width, circle_position, overlay_shape,
                enable_intro, enable_cta, hide_text, export_fps, ai_model_dropdown, stress_level
            ],
            outputs=[video_output, thumbnail_output, audio_output, engine_status_output, status_output, progress_bar]
        )
    return demo

# =============== MAIN ===============
if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("❌ python-dotenv not found. Install: pip install python-dotenv")
        exit(1)
    if not MODELS_AVAILABLE:
        print("❌ Missing TTS libraries. Install with:")
        print("pip install TTS speechbrain pydub Pillow num2words torch torchaudio gradio requests spacy python-dotenv")
        exit(1)

    print("\n" + "="*80)
    print("🌍 MULTI-LANGUAGE VIDEO GENERATOR")
    print("="*80)
    print("\n✅ FEATURES:")
    print("  ✓ Multi-language TTS (English, Chinese, Spanish, Hindi, Arabic, Romanian)")
    print("  ✓ Video backgrounds (Pexels API + Giphy API + Local)")
    print("  ✓ Circle overlay videos (PIP style)")
    print("  ✓ FFmpeg native processing (5-10x faster)")
    print("  ✓ SQLite caching (TTS + videos)")
    print("  ✓ Parallel slide generation")
    print("  ✓ Background music mixing")
    print("  ✓ Intro/CTA slides with localized text")
    print("  ✓ Random voice assignment")
    print("  ✓ CPU-safe operation")

    print("\n📦 INSTALLED COMPONENTS:")
    if SPACY_AVAILABLE:
        print("  ✅ spaCy NLP - Available (via External API)")
    else:
        print("  ⚠️  spaCy NLP - Not available")
    if SD_AVAILABLE:
        print("  ✅ Stable Diffusion - Available")
    else:
        print("  ⚠️  Stable Diffusion - Not available")

    print("\n🌐 SUPPORTED LANGUAGES:")
    for code, info in SUPPORTED_LANGUAGES.items():
        print(f"  • {info['name']} ({code})")

    try:
        generator = TextToVideoGenerator()
        print("\n📊 RESOURCES:")
        print(f"  🗣️  Voices: {len(generator.available_voices)}")
        print(f"  🎵 Music tracks: {len(generator.available_music) - 1}")
        print(f"  ⭕ Circle overlays: {generator.available_circles[0]}")

        print("\n💡 SETUP CHECKLIST:")
        print("  1. Set PEXELS_API_KEY in .env file")
        print("  2. Set GIPHY_API_KEY in .env file")
        print("  3. Run: ollama serve (for keyword extraction)")
        print("  4. Add circle overlay videos to circle_overlays/ folder")
        print("  5. Add background music to background_music/ folder")
        print("  6. Add logo image to background_images/ folder")

        print("\n🚀 STARTING SERVER...")
        print("   Access at: http://localhost:1603")
        print("="*80 + "\n")

        demo = setup_ui(generator)
        demo.launch(
            server_name="0.0.0.0",
            server_port=1603,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)