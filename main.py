import os
import re
import sys
import random
import shutil
import traceback
import platform
import subprocess
import uuid
import abc
import argparse
import yt_dlp
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import warnings
import torch

# Suppress specific deprecation warnings that are out of our control (internal to libraries like transformers)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.generic"
)
warnings.filterwarnings(
    "ignore", message=".*torch.utils._pytree._register_pytree_node.*"
)

import torchaudio

# Core imports
from core.config import Config
from core.database import GenerationDB, DB
from core.nlp.keyword_extractor import KeywordExtractor
from core.nlp.ollama_client import DEFAULT_FALLBACK_KEYWORDS
from core.ai.stable_diffusion import StableDiffusionManager, SD_AVAILABLE
from core.media.manager import MediaManager
from core.media.youtube_audio import YouTubeAudioLibraryAPI
from core.tts.manager import TTSManager
from core.utils.audio import improve_audio_quality, remove_metallic_artifacts
from core.utils.video import (
    get_video_duration,
    has_audio_stream,
    is_video_file,
    get_random_middle_frame,
    get_smart_thumbnail_frame,
)

# Availability flags
MODELS_AVAILABLE = True  # Assumed true since imports above succeeded
SPACY_AVAILABLE = True  # Used in main block

import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment

try:
    from langdetect import detect as detect_lang
except ImportError:
    detect_lang = None
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

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

# =============== MEDIA SOURCE CACHE ===============
_media_source_cache = {}

# =============== LARAVEL COMPANY THEME ===============
LARAVEL_BG_GRADIENT = ("#0f172a", "#2a1030")  # Dark Blue to Dark Purple
LARAVEL_ACCENT_GRADIENT = ("#7c3aed", "#ec4899")  # Purple to Pink


def create_gradient_image(
    size: Tuple[int, int], colors: Tuple[str, str], direction: str = "135deg"
) -> Image.Image:
    """Creates a linear gradient image using Pillow."""
    base = Image.new("RGB", size, colors[0])
    top = Image.new("RGB", size, colors[1])
    mask = Image.new("L", size)
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
    else:  # to_bottom
        for y in range(h):
            for x in range(w):
                mask_data.append(int(255 * (y / h)))

    mask.putdata(mask_data)
    return Image.composite(top, base, mask)


# =============== VIDEO GENERATOR WITH FFMPEG ===============
class FFmpegVideoGenerator:
    def __init__(
        self, config: Config, keyword_extractor: Optional[KeywordExtractor] = None
    ):
        self.config = config
        self.video_width = config.VIDEO_WIDTH
        self.video_height = config.VIDEO_HEIGHT
        self.video_preset = config.VIDEO_PRESET
        self.video_crf = config.VIDEO_CRF
        self.font_path = self._discover_fonts()
        self.media_manager = MediaManager(self.config)
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
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            if self.config.IMAGES_DIR.exists():
                files = list(self.config.IMAGES_DIR.glob(ext))
                if files:
                    return files[0]
        return None

    def _clean_text(self, text: str) -> str:
        """Clean text of custom metadata tags for visual display."""
        # This mirrors the logic in TTSManager to ensure consistency
        text = re.sub(r"\[\d+\s+levels?\](?:\([^)]+\))?", "", text)
        return text.strip()

    def _discover_fonts(self) -> str:
        font_paths = []
        system = platform.system()
        if system == "Windows":
            font_paths.append(Path("C:/Windows/Fonts"))
        elif system == "Darwin":
            font_paths.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts")])
        else:
            font_paths.extend(
                [
                    Path("/usr/share/fonts/truetype"),
                    Path("/usr/share/fonts/truetype/dejavu"),
                    Path("/usr/share/fonts/TTF"),
                ]
            )
        common_fonts = [
            "DejaVuSans-Bold.ttf",
            "arial.ttf",
            "FreeSansBold.ttf",
            "NotoSans-Bold.ttf",
        ]
        for path in font_paths:
            if path.is_dir():
                for font_name in common_fonts:
                    if (path / font_name).exists():
                        return str((path / font_name).resolve())
        return "DejaVuSans"

    def get_available_music_files(self) -> List[Dict[str, str]]:
        music_files = []
        if self.config.MUSIC_DIR.exists():
            for ext in ["*.mp3", "*.wav", "*.m4a"]:
                for file_path in self.config.MUSIC_DIR.glob(ext):
                    music_files.append({"name": file_path.name, "path": str(file_path)})
        return sorted(music_files, key=lambda x: x["name"])

    def get_music_by_name(self, music_name: str) -> Optional[Path]:
        if not music_name or music_name == "Random":
            music_files = self.get_available_music_files()
            if music_files:
                return Path(random.choice(music_files)["path"])
            return None
        music_files = self.get_available_music_files()
        for mf in music_files:
            if mf["name"] == music_name:
                return Path(mf["path"])
        return None

    def get_background_video(
        self,
        keyword: Optional[str],
        sentence: Optional[str],
        language: str = "en",
        preferred_source: Optional[str] = None,
        use_snn: bool = False,
        theme: Optional[str] = None,
        entity: Optional[str] = None,
        script_id: Optional[str] = None,
        sentence_idx: Optional[int] = None,
        candidate_keywords: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """Return a background video Path.
        Preference order:
        1. Video from keyword/media API (checking provided keyword only).
        2. Availability fallback: next-ranked candidate keyword that actually has
           stock results, then the generic ``generate_fallback_keywords`` map.
        3. Random local video file.
        4. Circle overlay video as full-screen fallback.

        The search query is disambiguated with sentence context (polysemy) and,
        if ``entity`` is supplied, enriched with that entity. Every finalized
        decision is written to the keyword-selection audit (``DB.log_keyword_selection``)
        keyed by ``script_id`` / ``sentence_idx`` for later calibration.
        """
        search_keywords = []
        if keyword:
            sanitized = self.keyword_extractor.sanitize_keyword(keyword)
            if sanitized:
                enriched = self.keyword_extractor.enrich_keyword_context(
                    sanitized, sentence
                )
                search_keywords.append(enriched)
        else:
            sanitized = self.keyword_extractor.sanitize_keyword(sentence or "")
            if sanitized:
                search_keywords.append(sanitized)

        if not search_keywords:
            print("💡 [Fallback] No keyword available; gradient will be used.")
            DB.log_keyword_selection(
                script_id,
                sentence_idx,
                keyword,
                context_preview=sentence or "",
                was_used=False,
            )
            return None

        video, used_kw = self.media_manager.get_random_media(
            search_keywords,
            preferred_source,
            context=sentence,
            use_snn=use_snn,
            return_keyword=True,
            theme=theme,
            entity=entity,
        )

        # Availability fallback: if the chosen keyword yields no footage, try the
        # next-ranked candidate(s) that actually have stock results before falling
        # back to a generic query. Only the first *available* candidate is fetched.
        if not video and candidate_keywords:
            remaining = [c for c in candidate_keywords if c and c != keyword][:3]
            for cand in remaining:
                if self.media_manager.is_keyword_available(cand, preferred_source):
                    cand_q = [
                        self.keyword_extractor.enrich_keyword_context(cand, sentence)
                    ]
                    v, uk = self.media_manager.get_random_media(
                        cand_q,
                        preferred_source,
                        context=sentence,
                        use_snn=use_snn,
                        return_keyword=True,
                        theme=theme,
                        entity=entity,
                    )
                    if v:
                        video, used_kw = v, uk
                        print(
                            f"♻️ [Fallback] Keyword '{keyword}' unavailable; "
                            f"used candidate '{cand}' instead."
                        )
                        break
            # If every candidate is unavailable, broaden to the fallback map.
            if not video:
                fb = self.keyword_extractor.generate_fallback_keywords(
                    keyword or (sentence or "")[:20]
                )
                if fb:
                    v, uk = self.media_manager.get_random_media(
                        fb[:4],
                        preferred_source,
                        context=sentence,
                        use_snn=use_snn,
                        return_keyword=True,
                        theme=theme,
                        entity=entity,
                    )
                    if v:
                        video, used_kw = v, uk

        if video:
            if used_kw:
                self.keyword_extractor.add_used_keyword(used_kw)
            DB.log_keyword_selection(
                script_id,
                sentence_idx,
                used_kw,
                context_preview=sentence or "",
                was_used=True,
            )
            return video

        circle_video = self.get_circle_overlay_video()
        if circle_video:
            print(
                f"🎨 [Fallback] Using circle overlay video as background: {circle_video.name}"
            )
            # Circle overlay is a generic fallback, not a keyword-based match.
            DB.log_keyword_selection(
                script_id,
                sentence_idx,
                keyword,
                context_preview=sentence or "",
                was_used=False,
            )
            return circle_video
        print("💡 [Fallback] No video found; gradient will be used.")
        DB.log_keyword_selection(
            script_id,
            sentence_idx,
            keyword,
            context_preview=sentence or "",
            was_used=False,
        )
        return None

    def get_circle_overlay_video(self) -> Optional[Path]:
        videos = []
        for ext in ["*.mp4", "*.mov", "*.avi", "*.webm"]:
            if self.config.CIRCLE_OVERLAYS_DIR.exists():
                videos.extend(self.config.CIRCLE_OVERLAYS_DIR.glob(ext))
        return random.choice(videos) if videos else None

    def _create_text_overlay_png(
        self,
        text: str,
        output_path: Path,
        vw: Optional[int] = None,
        vh: Optional[int] = None,
    ) -> Path:
        text = self._clean_text(text)
        vw = vw or self.video_width
        vh = vh or self.video_height
        img_size = (vw, vh)
        base_font_size = self.config.TEXT_SIZE_CONFIG["font_size"]
        if len(text) > 60:
            font_size = max(30, int(base_font_size * (1.0 - (len(text) - 60) / 200)))
        else:
            font_size = base_font_size

        try:
            font = (
                ImageFont.truetype(self.font_path, font_size)
                if self.font_path and os.path.exists(self.font_path)
                else ImageFont.load_default()
            )
        except:
            font = ImageFont.load_default()

        # 2. Wrap text and calculate bounding box
        wrapped_text = textwrap.fill(text, width=35)
        # Create a temp image to calculate bbox
        temp_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 3. Create the text mask (white text on black background)
        padding = 10
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new("L", mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        text_pos = (padding, padding)
        mask_draw.text(text_pos, wrapped_text, font=font, fill=255)

        # 4. Create gradient image for the text
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")

        # 5. Composite text onto final frame
        final_frame = Image.new("RGBA", img_size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(final_frame)
        x = (vw - text_width) // 2
        y = vh - text_height - self.config.TEXT_SIZE_CONFIG["bottom_margin"]

        stroke_width = 3
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                if adj_x != 0 or adj_y != 0:
                    shadow_draw.text(
                        (x + adj_x, y + adj_y),
                        wrapped_text,
                        font=font,
                        fill=(0, 0, 0, 200),
                    )

        # Composite the gradient text
        text_layer = Image.new("RGBA", mask_size, (0, 0, 0, 0))
        text_layer.paste(grad_img, (0, 0), mask_img)
        final_frame.paste(text_layer, (x - padding, y - padding), text_layer)

        final_frame.save(str(output_path), "PNG")
        return output_path

    def _create_intro_text_png(
        self,
        output_path: Path,
        language: str = "en",
        vw: Optional[int] = None,
        vh: Optional[int] = None,
    ) -> Path:
        vw = vw or self.video_width
        vh = vh or self.video_height
        img_size = (vw, vh)
        final_frame = Image.new("RGBA", img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(final_frame)

        try:
            font_large = (
                ImageFont.truetype(self.font_path, 120)
                if self.font_path and os.path.exists(self.font_path)
                else ImageFont.load_default()
            )
            font_small = (
                ImageFont.truetype(self.font_path, 60)
                if self.font_path and os.path.exists(self.font_path)
                else ImageFont.load_default()
            )
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        intro_msg = self.config.INTRO_MESSAGES.get(
            language, self.config.INTRO_MESSAGES["en"]
        )
        lines = intro_msg.upper().split()
        main_text = "\n".join(lines[:2]) if len(lines) > 1 else lines[0]

        # Calculate bbox
        temp_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        bbox = temp_draw.textbbox((0, 0), main_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Create mask and gradient for main text
        padding = 20
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new("L", mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.text((padding, padding), main_text, font=font_large, fill=255)
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")

        # Position
        x = (vw - text_width) // 2
        y = (vh - text_height) // 2 - 150

        # Draw shadow
        for adj in range(-4, 5):
            if adj != 0:
                draw.text((x + adj, y), main_text, font=font_large, fill=(0, 0, 0, 180))
                draw.text((x, y + adj), main_text, font=font_large, fill=(0, 0, 0, 180))

        # Paste gradient text
        text_layer = Image.new("RGBA", mask_size, (0, 0, 0, 0))
        text_layer.paste(grad_img, (0, 0), mask_img)
        final_frame.paste(text_layer, (x - padding, y - padding), text_layer)

        # Secondary text
        if len(lines) > 2:
            sec_text = " ".join(lines[2:]).upper()
            bbox2 = temp_draw.textbbox((0, 0), sec_text, font=font_small)
            text_width2 = bbox2[2] - bbox2[0]
            x2 = (vw - text_width2) // 2
            y2 = y + text_height + 100
            for adj in range(-2, 3):
                if adj != 0:
                    draw.text(
                        (x2 + adj, y2), sec_text, font=font_small, fill=(0, 0, 0, 150)
                    )
            # Gradient styling for secondary text
            mask_size2 = (
                text_width2 + padding * 2,
                text_height + padding * 2,
            )  # Use main text height approx or recalc
            mask_img2 = Image.new("L", mask_size2, 0)
            mask_draw2 = ImageDraw.Draw(mask_img2)
            mask_draw2.text((padding, padding), sec_text, font=font_small, fill=255)
            grad_img2 = create_gradient_image(
                mask_size2, LARAVEL_ACCENT_GRADIENT, "to_right"
            )

            text_layer2 = Image.new("RGBA", mask_size2, (0, 0, 0, 0))
            text_layer2.paste(grad_img2, (0, 0), mask_img2)
            final_frame.paste(text_layer2, (x2 - padding, y2 - padding), text_layer2)

        final_frame.save(str(output_path), "PNG")
        return output_path

    def _create_cta_text_png(
        self,
        output_path: Path,
        language: str = "en",
        vw: Optional[int] = None,
        vh: Optional[int] = None,
    ) -> Path:
        vw = vw or self.video_width
        vh = vh or self.video_height
        img_size = (vw, vh)
        final_frame = Image.new("RGBA", img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(final_frame)

        try:
            font_large = (
                ImageFont.truetype(self.font_path, 130)
                if self.font_path and os.path.exists(self.font_path)
                else ImageFont.load_default()
            )
            font_small = (
                ImageFont.truetype(self.font_path, 70)
                if self.font_path and os.path.exists(self.font_path)
                else ImageFont.load_default()
            )
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        cta_msg = self.config.CTA_MESSAGES.get(language, self.config.CTA_MESSAGES["en"])
        parts = cta_msg.split(",")
        main_text = parts[0].strip().upper()
        if len(parts) > 1:
            main_text += "\n" + parts[1].strip().upper()

        # Calculate bbox
        temp_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        bbox = temp_draw.textbbox((0, 0), main_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Create mask and gradient for main text
        padding = 20
        mask_size = (text_width + padding * 2, text_height + padding * 2)
        mask_img = Image.new("L", mask_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.text((padding, padding), main_text, font=font_large, fill=255)
        grad_img = create_gradient_image(mask_size, LARAVEL_ACCENT_GRADIENT, "to_right")

        # Position
        x = (vw - text_width) // 2
        y = (vh - text_height) // 2

        # Draw shadow
        for adj in range(-4, 5):
            if adj != 0:
                draw.text((x + adj, y), main_text, font=font_large, fill=(0, 0, 0, 180))
                draw.text((x, y + adj), main_text, font=font_large, fill=(0, 0, 0, 180))

        # Paste gradient text
        text_layer = Image.new("RGBA", mask_size, (0, 0, 0, 0))
        text_layer.paste(grad_img, (0, 0), mask_img)
        final_frame.paste(text_layer, (x - padding, y - padding), text_layer)

        final_frame.save(str(output_path), "PNG")
        return output_path

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r"\bDr\.", "Dr<dot>", text)
        text = re.sub(r"\bMr\.", "Mr<dot>", text)
        text = re.sub(r"\bMrs\.", "Mrs<dot>", text)
        text = re.sub(r"\bMs\.", "Ms<dot>", text)
        text = re.sub(r"\b([A-Z])\.", r"\1<dot>", text)
        raw_sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        sentences = [
            s.replace("<dot>", ".").strip() for s in raw_sentences if s.strip()
        ]
        for i, sentence in enumerate(sentences):
            if not sentence.endswith((".", "!", "?", "。", "！", "？")):
                sentences[i] = sentence + "."
        filtered = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            if (
                len(current.split()) < 3
                and len(current) < 15
                and i + 1 < len(sentences)
            ):
                merged = current + " " + sentences[i + 1]
                filtered.append(merged)
                i += 2
            else:
                filtered.append(current)
                i += 1
        return filtered

    def _generate_overlay_mask(self, shape: str, diameter: int) -> Path:
        """Generates a high-quality PNG mask for the specified shape using Pillow."""
        mask_path = (
            self.config.TEMP_DIR / f"mask_{shape.lower()}_{uuid.uuid4().hex[:8]}.png"
        )

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
            draw.rounded_rectangle(
                (0, 0, size, size), radius=corner_radius, fill=(255, 255, 255, 255)
            )
        elif shape == "Star":
            # 5-pointed star
            cx, cy = size // 2, size // 2
            outer_radius = size // 2
            inner_radius = outer_radius * 0.4  # Ratio for star thickness
            points = []
            import math

            angle = -math.pi / 2  # Start at top
            step = math.pi / 5  # 36 deg

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

    # Intro video path - permanent asset that should never be deleted
    INTRO_VIDEO_PATH = Path(__file__).resolve().parent / "intro" / "intro-tv-noise.mp4"

    def _create_slide_with_ffmpeg(
        self,
        sentence: str,
        audio_path: Path,
        video_path: Optional[Path],
        output_path: Path,
        slide_num: int,
        is_intro: bool = False,
        is_cta: bool = False,
        circle_video: Optional[Path] = None,
        circle_config: Optional[Dict] = None,
        language: str = "en",
        hide_text: bool = False,
        export_fps: int = 30,
        overlay_shape: str = "Circle",
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> Optional[Path]:
        try:
            vw = video_width or self.video_width
            vh = video_height or self.video_height
            if audio_path is None:
                print(
                    f"❌ [FFmpeg] Slide {slide_num} error: audio_path is None. Check TTS logs for generation failures."
                )
                return None

            if not Path(audio_path).exists():
                print(
                    f"❌ [FFmpeg] Slide {slide_num} error: audio_path does not exist: {audio_path}"
                )
                return None

            # Force intro video for intro slides
            if is_intro:
                if self.INTRO_VIDEO_PATH.exists():
                    video_path = self.INTRO_VIDEO_PATH
                    print(f"🎬 [Intro] Using fixed background: {self.INTRO_VIDEO_PATH}")
                else:
                    print(
                        f"⚠️ [Intro] Fixed background not found: {self.INTRO_VIDEO_PATH}. Falling back to default."
                    )

            source_info = (
                f"Video: {video_path.name}" if video_path else "Background: Image/Color"
            )
            # Get audio duration
            duration = get_video_duration(audio_path)
            print(
                f"🎬 [FFmpeg] Creating slide {slide_num} ({language}) - {source_info} - duration: {duration:.2f}s"
            )

            if not hide_text:
                text_overlay_path = (
                    self.config.TEMP_DIR
                    / f"text_{slide_num}_{uuid.uuid4().hex[:8]}.png"
                )
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

            is_split_screen = (
                overlay_shape == "Split Screen"
                and circle_video
                and circle_video.exists()
            )

            if is_split_screen:
                # Top part: Background video or gradient fallback
                if video_path and video_path.exists():
                    inputs.extend(
                        ["-stream_loop", "-1", "-i", str(video_path)]
                    )  # Input 0
                    filter_parts.append(
                        f"[0:v]scale={vw}:{vh // 2}:force_original_aspect_ratio=increase,crop={vw}:{vh // 2},setsar=1[top]"
                    )
                else:
                    grad_path = (
                        self.config.TEMP_DIR
                        / f"grad_top_{slide_num}_{uuid.uuid4().hex[:8]}.png"
                    )
                    create_gradient_image(
                        (vw, vh // 2), LARAVEL_BG_GRADIENT, "135deg"
                    ).save(str(grad_path))
                    inputs.extend(["-loop", "1", "-i", str(grad_path)])  # Input 0
                    filter_parts.append(f"[0:v]scale={vw}:{vh // 2},setsar=1[top]")

                # Bottom part: Video Overlay
                inputs.extend(
                    ["-stream_loop", "-1", "-i", str(circle_video)]
                )  # Input 1
                filter_parts.append(
                    f"[1:v]scale={vw}:{vh // 2}:force_original_aspect_ratio=increase,crop={vw}:{vh // 2},setsar=1[bottom]"
                )

                # Combine
                filter_parts.append(
                    f"[top][bottom]vstack=inputs=2,"
                    f"fps={export_fps},"
                    f"trim=duration={duration},"
                    f"setpts=PTS-STARTPTS[bg_scaled]"
                )
                input_count = 2
            elif video_path and video_path.exists():
                is_image = video_path.suffix.lower() in [".jpg", ".jpeg", ".png"]
                if is_image:
                    inputs.extend(["-loop", "1", "-i", str(video_path)])
                else:
                    inputs.extend(["-stream_loop", "-1", "-i", str(video_path)])

                filter_parts.append(
                    f"[0:v]scale={vw}:{vh}:force_original_aspect_ratio=decrease,"
                    f"pad={vw}:{vh}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,"
                    f"fps={export_fps},"
                    f"trim=duration={duration},"
                    f"setpts=PTS-STARTPTS[bg_scaled]"
                )
                input_count = 1
            else:
                # No video found – try using circle overlay as full-screen background
                circle_bg = self.get_circle_overlay_video()
                if circle_bg and circle_bg.exists():
                    print(
                        f"🔁 [FFmpeg] Slide {slide_num}: Using circle overlay video as full background"
                    )
                    inputs.extend(["-stream_loop", "-1", "-i", str(circle_bg)])
                    filter_parts.append(
                        f"[0:v]scale={vw}:{vh}:force_original_aspect_ratio=decrease,"
                        f"pad={vw}:{vh}:(ow-iw)/2:(oh-ih)/2,"
                        f"setsar=1,"
                        f"fps={export_fps},"
                        f"trim=duration={duration},"
                        f"setpts=PTS-STARTPTS[bg_scaled]"
                    )
                    input_count = 1
                else:
                    print(
                        f"🎨 [FFmpeg] Slide {slide_num}: Using branded gradient background (fallback)"
                    )
                    grad_path = (
                        self.config.TEMP_DIR
                        / f"grad_{slide_num}_{uuid.uuid4().hex[:8]}.png"
                    )
                    create_gradient_image((vw, vh), LARAVEL_BG_GRADIENT, "135deg").save(
                        str(grad_path)
                    )
                    inputs.extend(["-loop", "1", "-i", str(grad_path)])
                    filter_parts.append(
                        f"[0:v]fps={export_fps},trim=duration={duration}[bg_scaled]"
                    )
                    input_count = 1

            filter_parts.append(
                "[bg_scaled]format=rgba,colorchannelmixer=aa=0.6[dimmed]"
            )

            if text_overlay_path:
                inputs.extend(["-loop", "1", "-i", str(text_overlay_path)])
                filter_parts.append(
                    f"[dimmed][{input_count}:v]overlay=0:0:format=auto[with_text]"
                )
                input_count += 1
            else:
                # No text overlay; keep dimmed background as final layer
                logo_label = "dimmed"
                # No increment of input_count needed

            logo_label = "with_text" if text_overlay_path else "dimmed"
            if self.logo_path and self.logo_path.exists():
                inputs.extend(["-loop", "1", "-i", str(self.logo_path)])
                cfg = self.config.LOGO_CONFIG
                pos_map = {
                    "top-left": f"{cfg['margin']}:{cfg['margin']}",
                    "top-right": f"W-w-{cfg['margin']}:{cfg['margin']}",
                    "bottom-left": f"{cfg['margin']}:H-h-{cfg['margin']}",
                    "bottom-right": f"W-w-{cfg['margin']}:H-h-{cfg['margin']}",
                }
                pos = pos_map.get(cfg["position"], pos_map["top-left"])
                filter_parts.append(f"[{input_count}:v]scale=150:150[logo_scaled]")
                filter_parts.append(
                    f"[{logo_label}][logo_scaled]overlay={pos}:format=auto[final]"
                )
                logo_label = "final"
                input_count += 1

            if (
                circle_video
                and circle_video.exists()
                and circle_config
                and not is_split_screen
            ):
                inputs.extend(["-stream_loop", "-1", "-i", str(circle_video)])
                diameter = circle_config.get("diameter", 300)
                position = circle_config.get("position", "top-right")
                pos_map = {
                    "top-left": "50:50",
                    "top-right": f"W-w-50:50",
                    "bottom-left": "50:H-h-50",
                    "bottom-right": "W-w-50:H-h-50",
                    "center": "(W-w)/2:(H-h)/2",
                }
                overlay_pos = pos_map.get(position, pos_map["top-right"])
                # Generate procedural mask
                mask_path = self._generate_overlay_mask(overlay_shape, diameter)
                inputs.extend(["-loop", "1", "-i", str(mask_path)])
                mask_idx = (
                    input_count + 1
                )  # video is input_count, mask is input_count+1

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
                filter_parts.append(
                    f"[circle_sized][mask_alpha]alphamerge[circle_masked]"
                )

                filter_parts.append(
                    f"[{logo_label}][circle_masked]overlay={overlay_pos}:format=auto[final_with_circle]"
                )
                logo_label = "final_with_circle"
                input_count += 2  # Incremented by 2 (video + mask)

            print(
                f"🎬 [FFmpeg] Slide {slide_num}: Composition started. Layers: bg={video_path.name if video_path else 'Branded Gradient'}, text={text_overlay_path.name if text_overlay_path else 'None'}"
            )

            audio_idx = input_count
            inputs.extend(["-i", str(audio_path)])

            filter_complex = ";".join(filter_parts)
            cmd = [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                f"[{logo_label}]",
                "-map",
                f"{audio_idx}:a",
                "-c:v",
                "libx264",
                "-preset",
                self.video_preset,
                "-crf",
                str(self.video_crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-r",
                str(export_fps),
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ [FFmpeg] Slide {slide_num}: Successfully composed.")
            if not output_path.exists():
                print(f"[FFmpeg] ERROR: Output not created for slide {slide_num}")
                return None
            file_size = output_path.stat().st_size
            print(
                f"[FFmpeg] Slide {slide_num} created: {file_size / 1024 / 1024:.2f} MB"
            )
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
            if (
                "text_overlay_path" in locals()
                and text_overlay_path
                and text_overlay_path.exists()
            ):
                text_overlay_path.unlink(missing_ok=True)

    def create_final_video(
        self,
        sentences: List[str],
        audio_paths: List[Path],
        keywords: List[Optional[str]],
        intro_audio: Optional[Path] = None,
        cta_audio: Optional[Path] = None,
        music_path: Optional[Path] = None,
        music_volume_db: int = -20,
        circle_video: Optional[Path] = None,
        circle_config: Optional[Dict] = None,
        circle_selection: str = "Random",
        language: str = "en",
        preferred_media_source: Optional[str] = None,
        selected_background_video: Optional[Path] = None,
        pre_selected_videos: Optional[Dict[int, str]] = None,
        hide_text: bool = False,
        export_fps: int = 30,
        overlay_shape: str = "Circle",
        intro_text: Optional[str] = None,
        use_snn: bool = False,
        enable_crossfade: bool = False,
        crossfade_duration: float = 0.3,
        progress_callback=None,
        theme: Optional[str] = None,
        entity: Optional[str] = None,
        script_id: Optional[str] = None,
        candidate_map: Optional[Dict[int, List[str]]] = None,
    ) -> Path:

        # Apply aspect ratio and quality from instance vars (set before calling)
        vw = self.video_width
        vh = self.video_height

        # Setup temp directory
        temp_dir = self.config.TEMP_DIR / f"final_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(exist_ok=True)
        source_videos_to_cleanup = set()

        # --- Stage 1: Prepare Slide Data ---
        # We organize all slides (Intro, Main Sentences, CTA) into a uniform structure
        slides_data = []

        # Content Slides
        for i, (sentence, a_path, kw) in enumerate(
            zip(sentences, audio_paths, keywords)
        ):
            slides_data.append(
                {
                    "type": "content",
                    "text": sentence,
                    "audio_path": a_path,
                    "keyword": kw,
                    "sentence": sentence,
                    "slide_num": i,
                    "is_intro": False,
                    "is_cta": False,
                }
            )

        # Intro
        if intro_audio:
            intro_slide = {
                "type": "intro",
                "text": intro_text if intro_text else "Welcome",
                "audio_path": intro_audio,
                "keyword": "intro",
                "sentence": intro_text if intro_text else "Welcome",
                "slide_num": -1,  # Special ID
                "is_intro": True,
                "is_cta": False,
            }
            if len(slides_data) >= 2:
                # Random position between 2 and 5 (or end)
                max_idx = min(len(slides_data), 5)
                start_idx = 2
                if max_idx < start_idx:
                    max_idx = start_idx

                insert_idx = random.randint(start_idx, max_idx)
                print(f"🎲 [Pipeline] Inserting intro at index {insert_idx}")
                slides_data.insert(insert_idx, intro_slide)
            else:
                # Fallback to start
                print(
                    f"🎲 [Pipeline] Not enough slides for random insert. Placing intro at start."
                )
                slides_data.insert(0, intro_slide)

        # CTA
        if cta_audio:
            slides_data.append(
                {
                    "type": "cta",
                    "text": "",
                    "audio_path": cta_audio,
                    "keyword": "outro",
                    "sentence": "Goodbye",
                    "slide_num": 999,
                    "is_intro": False,
                    "is_cta": True,
                }
            )

        total_slides = len(slides_data)
        print(
            f"🚀 [Pipeline] Starting parallel generation for {total_slides} slides..."
        )

        # --- Stage 2: Parallel Resource Fetching ---
        total_slides = len(slides_data)
        print(f"🚀 [Pipeline] Starting parallel fetching for {total_slides} slides...")
        slide_videos = {}

        pre_selected_videos = pre_selected_videos or {}
        print(
            f"[DEBUG] create_final_video: pre_selected_videos has {len(pre_selected_videos)} entries: {dict(list(pre_selected_videos.items())[:5])}"
        )

        if pre_selected_videos:
            for s_num, s_path in pre_selected_videos.items():
                if isinstance(s_path, str) and s_path == "__gradient__":
                    slide_videos[s_num] = None
                    print(
                        f"🎨 [Pipeline] User chose gradient background for slide {s_num}"
                    )
                else:
                    p = Path(s_path) if not isinstance(s_path, Path) else s_path
                    if p.exists():
                        slide_videos[s_num] = p
                        source_videos_to_cleanup.add(p)
                        print(
                            f"📁 [Pipeline] Using user-selected video for slide {s_num}: {p.name}"
                        )

        if selected_background_video and selected_background_video.exists():
            for slide in slides_data:
                if (
                    not slide["is_intro"]
                    and not slide["is_cta"]
                    and slide["slide_num"] not in slide_videos
                ):
                    slide_videos[slide["slide_num"]] = selected_background_video

        # Fetch remaining backgrounds
        with ThreadPoolExecutor(
            max_workers=self.config.WORKER_POOL_MEDIA
        ) as fetch_executor:
            fetch_futures = {}
            for slide in slides_data:
                # Skip if already assigned
                if slide["slide_num"] in slide_videos:
                    continue

                # Skip intro and CTA slides - they use fixed backgrounds
                if slide.get("is_intro") or slide.get("is_cta"):
                    continue

                future = fetch_executor.submit(
                    self.get_background_video,
                    slide["keyword"],
                    slide["sentence"],
                    language,
                    preferred_media_source,
                    use_snn=use_snn,
                    theme=theme,
                    entity=entity,
                    script_id=script_id,
                    sentence_idx=slide["slide_num"],
                    candidate_keywords=(
                        candidate_map.get(slide["slide_num"]) if candidate_map else None
                    ),
                )
                fetch_futures[future] = slide["slide_num"]

            completed_fetches = 0
            for future in as_completed(fetch_futures):
                s_num = fetch_futures[future]
                try:
                    video_path = future.result()
                    if video_path:
                        slide_videos[s_num] = video_path
                        source_videos_to_cleanup.add(video_path)

                    completed_fetches += 1
                    if progress_callback:
                        progress_callback(
                            completed_fetches,
                            total_slides * 2 + 1,
                            f"Fetching background videos ({completed_fetches}/{total_slides})...",
                        )
                except Exception as e:
                    print(f"⚠️ [Pipeline] Resource fetch failed for slide {s_num}: {e}")
                    # Edge case: Fallback to local random video if it fails
                    try:
                        fallback_video = self.media_manager.get_random_media(
                            ["cityscape", "abstract", "office"]
                        )  # Broad generic queries
                        if fallback_video:
                            print(
                                f"📁 [Pipeline] Applied emergency local fallback for slide {s_num}: {fallback_video.name}"
                            )
                            slide_videos[s_num] = fallback_video
                            source_videos_to_cleanup.add(fallback_video)
                    except Exception as e2:
                        print(
                            f"❌ [Pipeline] Emergency fallback also failed for slide {s_num}: {e2}"
                        )

        # --- Stage 3: Parallel Rendering ---
        print(
            f"⚡ [Pipeline] Resources ready. Starting render of {total_slides} slides..."
        )

        # We need to map result path back to the slide object to preserve order
        render_results = {}

        with ThreadPoolExecutor(
            max_workers=self.config.WORKER_POOL_RENDERING
        ) as render_executor:
            future_to_slide_idx = {}

            for idx_in_list, slide in enumerate(slides_data):
                slide_display_index = idx_in_list + 1
                video_bg = slide_videos.get(slide["slide_num"])
                output_path = (
                    temp_dir / f"slide_{slide['type']}_{slide_display_index}.mp4"
                )

                future = render_executor.submit(
                    self._create_slide_with_ffmpeg,
                    slide["text"],
                    slide["audio_path"],
                    video_bg,
                    output_path,
                    slide_display_index,
                    slide["is_intro"],
                    slide["is_cta"],
                    circle_video,
                    circle_config,
                    language,
                    hide_text,
                    export_fps,
                    overlay_shape,
                    vw,
                    vh,
                )
                future_to_slide_idx[future] = idx_in_list

            # Collect Render Results
            completed_renders = 0
            for future in as_completed(future_to_slide_idx):
                idx = future_to_slide_idx[future]
                try:
                    path_created = future.result()
                    if path_created:
                        render_results[idx] = path_created
                    completed_renders += 1
                    if progress_callback:
                        progress_callback(
                            total_slides + completed_renders,
                            total_slides * 2 + 1,
                            "Rendering slides...",
                        )
                except Exception as e:
                    print(f"❌ [Pipeline] Render failed for slide index {idx}: {e}")

        # Assemble and validate paths in the correct order based on slides_data list
        slide_paths = []
        valid_slide_paths = []
        for i in range(len(slides_data)):
            if i in render_results:
                sp = render_results[i]
                slide_paths.append(sp)
                valid, reason = validate_slide(sp)
                if valid:
                    valid_slide_paths.append(sp)
                    print(f"[Pipeline] Slide {i+1} validation: OK ({sp.name})")
                else:
                    print(f"❌ [Pipeline] Slide {i+1} validation failed ({sp.name}): {reason}")

        if len(valid_slide_paths) != len(slides_data):
            print(f"⚠️ [Pipeline] Validation warning: {len(valid_slide_paths)} of {len(slides_data)} slides passed pre-concatenation validation.")

        slide_paths = valid_slide_paths

        if not slide_paths:
            raise ValueError("No valid slides available for final composition")

        # --- Stage 4: Concatenation (with optional crossfade) ---
        if progress_callback:
            progress_callback(
                total_slides * 2, total_slides * 2, "Concatenating final video..."
            )

        output_path = self.config.TEMP_DIR / f"final_{uuid.uuid4().hex[:8]}.mp4"

        if enable_crossfade and len(slide_paths) > 1:
            print(
                f"[FFmpeg] Concatenating {len(slide_paths)} slides with crossfade ({crossfade_duration}s)..."
            )
            inputs = []
            filter_parts = []
            prev_label = None
            for i, sp in enumerate(slide_paths):
                inputs.extend(["-i", str(sp)])
                label = f"s{i}"
                filter_parts.append(
                    f"[{i}:v]settb=1/AVTB,setpts=PTS-STARTPTS[{label}_v]"
                )
                filter_parts.append(f"[{i}:a]aresample=async=1[{label}_a]")
                if i == 0:
                    prev_label = label
                else:
                    dur = get_video_duration(slide_paths[i - 1])
                    offset = dur - crossfade_duration
                    filter_parts.append(
                        f"[{prev_label}_v][{label}_v]xfade=transition=fade:duration={crossfade_duration}:offset={offset}[v{i}]"
                    )
                    filter_parts.append(
                        f"[{prev_label}_a][{label}_a]acrossfade=d={crossfade_duration}[a{i}]"
                    )
                    prev_label = f"v{i}"

            last_suffix = len(slide_paths) - 1
            filter_complex = ";".join(filter_parts)
            cmd = [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                f"[v{last_suffix}]",
                "-map",
                f"[a{last_suffix}]",
                "-c:v",
                "libx264",
                "-preset",
                self.video_preset,
                "-crf",
                str(self.video_crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-r",
                str(export_fps),
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)
                print(
                    f"✅ [Pipeline] Final video created (with crossfade): {output_path}"
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"❌ [FFmpeg] Crossfade failed: {e.stderr[:500]}. Falling back to simple concat."
                )
                enable_crossfade = False
        if not enable_crossfade:
            concat_file = self.config.TEMP_DIR / f"concat_{uuid.uuid4().hex[:8]}.txt"
            with open(concat_file, "w") as f:
                for slide_path in slide_paths:
                    f.write(f"file '{slide_path.absolute()}'\n")

            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output_path),
            ]
            print("[FFmpeg] Concatenating slides...")
            try:
                subprocess.run(concat_cmd, check=True, capture_output=True)
                print(f"✅ [Pipeline] Final video created: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ [FFmpeg] Concat failed: {e}")
                pass
            concat_file.unlink(missing_ok=True)

        # Cleanup rendered slides
        for slide_path in slide_paths:
            try:
                slide_path.unlink(missing_ok=True)
            except:
                pass

        # Cleanup source background videos (only those downloaded/sourced for this session)
        print(
            f"🧹 [Cleanup] Removing {len(source_videos_to_cleanup)} source background videos..."
        )
        for sv_path in source_videos_to_cleanup:
            try:
                # Check if it's in the background_videos directory (don't delete user/permanent assets elsewhere)
                if sv_path.exists() and str(self.config.VIDEOS_DIR.absolute()) in str(
                    sv_path.absolute()
                ):
                    sv_path.unlink(missing_ok=True)
                    # Also try to remove the parent directory if it's empty (keyword folders)
                    parent = sv_path.parent
                    if (
                        parent != self.config.VIDEOS_DIR
                        and parent.exists()
                        and not any(parent.iterdir())
                    ):
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
        self.video_generator = FFmpegVideoGenerator(
            self.config, keyword_extractor=self.keyword_extractor
        )
        self.yt_audio_library = YouTubeAudioLibraryAPI(self.config)
        self.available_voices = self._get_available_voices()
        self.available_music = self._get_available_music()
        self.available_circles = self._get_available_circles()
        self.available_languages = list(SUPPORTED_LANGUAGES.keys())
        self.available_models = self.keyword_extractor.get_available_models()
        self.available_background_videos = self._get_available_background_videos()

    def _audit_keyword_decision(
        self,
        script_id: Optional[str],
        sentence_idx: int,
        keyword: Optional[str],
        sentence: Optional[str],
        was_used: bool = True,
    ) -> None:
        """Log a finalized keyword decision with its raw neuron signals + score.

        Captures the neural evaluation at the exact point a keyword is committed
        for a slide, so weights can later be calibrated by regression. Best-effort
        and non-blocking (failures are swallowed by ``DB.log_keyword_selection``).
        """
        signals = None
        decision_score = None
        if keyword and sentence:
            try:
                sig = self.keyword_extractor.neuron_extractor._local_evaluate_signals(
                    sentence, keyword
                )
                if sig:
                    signals = sig
                    decision_score = self.keyword_extractor.neuron_extractor._calculate_decision_score(
                        sig
                    )
            except Exception as e:  # pragma: no cover - best-effort audit
                print(f"⚠️ [Audit] signal eval failed: {e}")
        DB.log_keyword_selection(
            script_id,
            sentence_idx,
            keyword,
            context_preview=sentence or "",
            signals=signals,
            decision_score=decision_score,
            was_used=was_used,
        )

    def detect_language(self, text: str) -> str:
        """Detect language of the input text"""
        if not detect_lang or not text or len(text.strip()) < 5:
            return "en"
        try:
            detected = detect_lang(text)
            # Map detected code to supported ones
            if detected in self.config.SUPPORTED_LANGUAGES:
                return detected
            # Fallback for common mismatches
            if detected.startswith("zh"):
                return "zh"
            return "en"
        except:
            return "en"

    def preview_voice(
        self, voice_id: str, language: str = "en", speed: float = 1.0
    ) -> Path:
        """Generate a short preview of the selected voice"""
        preview_text = "This is a preview of the selected voice. How does it sound?"
        if language == "zh":
            preview_text = "这是所选声音的预览。听起来怎么样？"
        elif language == "ro":
            preview_text = "Aceasta este o previzualizare a vocii selectate. Cum sună?"

        return self.tts_manager.generate_speech(
            preview_text, voice_id, language, speed=speed
        )

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
            voices.extend(
                [d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()]
            )

        # Deduplicate and sort
        return sorted(list(set(voices)))

    def _get_available_music(self) -> List[str]:
        music_files = self.video_generator.get_available_music_files()
        return ["Random", "Auto (YouTube Library)"] + [m["name"] for m in music_files]

    def _get_available_circles(self) -> List[str]:
        circles = []
        for ext in ["*.mp4", "*.mov", "*.avi", "*.webm"]:
            if self.config.CIRCLE_OVERLAYS_DIR.exists():
                circles.extend(list(self.config.CIRCLE_OVERLAYS_DIR.glob(ext)))
        return ["Random"] + [v.name for v in sorted(circles)]

    def _get_available_background_videos(self) -> List[str]:
        videos = []
        for ext in ["*.mp4", "*.mov", "*.avi", "*.webm"]:
            if self.config.BACKGROUND_VIDEOS_DIR.exists():
                videos.extend(list(self.config.BACKGROUND_VIDEOS_DIR.glob(ext)))
        return ["Auto-select (Pexels/Giphy/Local)", "Branded Gradient"] + [
            v.name for v in sorted(videos)
        ]

    def search_audio_library(self, query: str) -> List[Tuple[str, str]]:
        """Search the library and return (display_name, track_id) tuples"""
        results = self.yt_audio_library.search(query)
        # Format for Gradio dropdown: list of (label, value)
        return [(t.get("name", "Unknown"), t.get("id", "")) for t in results]

    def download_library_track(self, track_id: str, track_name: str) -> Optional[str]:
        """Download a track from the library and return its local name"""
        if not track_id:
            return None

        # Clean track name for filename
        safe_name = "".join(
            [c if c.isalnum() or c in " ._-" else "_" for c in track_name]
        )
        if not safe_name.lower().endswith(".mp3"):
            safe_name += ".mp3"

        output_path = self.config.MUSIC_DIR / safe_name

        if output_path.exists():
            print(f"[YouTubeAudio] Track already exists locally: {safe_name}")
            return safe_name

        print(f"[YouTubeAudio] Downloading track '{track_name}'...")
        if self.yt_audio_library.download_track(track_id, output_path):
            # Refresh available music
            self.available_music = self._get_available_music()
            return safe_name

        return None

    def generate_video(
        self,
        text: str,
        speaker_id: str = "Standard Voice (Non-Cloned)",
        language: str = "en",
        pexels_keyword: Optional[str] = None,
        preferred_media_source: Optional[str] = None,
        selected_background_video_name: Optional[str] = None,
        pre_selected_videos: Optional[Dict[int, str]] = None,
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
        ai_model: str = "gemma4:e2b",
        ai_api_url: Optional[str] = None,
        stress_level: float = 1.0,
        use_snn: bool = False,
        audio_only: bool = False,
        normalize_audio: bool = True,
        aspect_ratio: str = "9:16 Portrait (TikTok/Shorts)",
        quality: str = "Medium (Balanced)",
        enable_crossfade: bool = False,
        entity: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        # Language Detection
        if language == "auto":
            language = self.detect_language(text)
            print(f"✨ [NLP] Auto-detected language: {language}")

        # Reset keywords for this session
        self.keyword_extractor.clear_used()
        input_params = {
            "text": text,
            "speaker_id": speaker_id,
            "language": language,
            "pexels_keyword": pexels_keyword,
            "preferred_media_source": preferred_media_source,
            "selected_background_video_name": selected_background_video_name,  # New parameter
            "enable_background_music": enable_background_music,
            "music_selection": music_selection,
            "music_volume_db": music_volume_db,
            "add_intro_slide": add_intro_slide,
            "add_call_to_action": add_call_to_action,
            "use_random_voices": use_random_voices,
            "enable_circle_overlay": enable_circle_overlay,
            "circle_diameter": circle_diameter,
            "circle_position": circle_position,
            "circle_border_width": circle_border_width,
            "circle_selection": circle_selection,
            "circle_upload_path": str(circle_upload_path)
            if circle_upload_path
            else None,
            "hide_text": hide_text,
            "export_fps": export_fps,
            "overlay_shape": overlay_shape,
            "ai_model": ai_model,
            "ai_api_url": ai_api_url,
            "stress_level": stress_level,
            "use_snn": use_snn,
            "audio_only": audio_only,
            "normalize_audio": normalize_audio,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "enable_crossfade": enable_crossfade,
            "pre_selected_videos": pre_selected_videos,
        }

        # Update API URL if changed
        if ai_api_url and ai_api_url != self.keyword_extractor.api_url:
            self.keyword_extractor.api_url = ai_api_url

        # Update model if changed
        if ai_model and ai_model != self.keyword_extractor.model:
            print(
                f"[Ollama] Switching model from {self.keyword_extractor.model} to {ai_model}"
            )
            self.keyword_extractor.model = ai_model

        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}
        if len(text) > 10000:
            return {"error": "Text too long (max 10,000 chars)", "success": False}

        sentences = self.video_generator.split_into_sentences(text)
        if len(sentences) > 100:
            return {"error": "Too many sentences (max 100)", "success": False}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}_{language}"
        session_dir.mkdir(exist_ok=True)
        # Stable id grouping all audit rows (selection + fetch) for this run.
        run_script_id = uuid.uuid4().hex[:12]

        # Parallel Keyword Extraction
        sentence_keywords = []
        theme = None
        if not audio_only:
            # Extract a global theme to bias keyword selection + media fallback
            try:
                theme = self.keyword_extractor.extract_theme(text, language)
                if theme:
                    print(f"🎯 [NLP] Detected global theme: '{theme}'")
            except Exception as e:
                print(f"⚠️ [NLP] Theme extraction failed: {e}")
                theme = None

            print(
                f"🧠 [NLP] Extracting keywords for {len(sentences)} sentences in parallel..."
            )
            extraction_futures = {}
            sentence_keywords_map = {}

            with ThreadPoolExecutor(
                max_workers=self.config.WORKER_POOL_NLP
            ) as executor:
                for i, sent in enumerate(sentences):
                    # Request multiple candidates to ensure we can pick a unique one.
                    # theme biases the LLM prompt and neural scoring context.
                    future = executor.submit(
                        self.keyword_extractor.extract_keywords,
                        sent,
                        10,
                        language,
                        True,
                        False,
                        theme,
                        entity,
                    )
                    extraction_futures[future] = i

                for future in as_completed(extraction_futures):
                    idx = extraction_futures[future]
                    try:
                        candidates = future.result()
                        sentence_keywords_map[idx] = candidates
                    except Exception as e:
                        print(
                            f"⚠️ [NLP] Keyword extraction failed for sentence {idx}: {e}"
                        )
                        sentence_keywords_map[idx] = []

            if use_snn:
                # Use the spiking-neural-network coherence optimizer to select a
                # globally coherent keyword sequence (balances engagement + flow).
                try:
                    print(
                        "🧠 [SNN] Optimizing keyword sequence for global coherence..."
                    )
                    chosen = self.keyword_extractor.optimize_keyword_sequence(
                        sentences, sentence_keywords_map, theme=theme, use_snn=use_snn
                    )
                    sentence_keywords = []
                    for i in range(len(sentences)):
                        kw = chosen.get(i)
                        if not kw:
                            # LLM/neuron selection failed for this sentence — fall
                            # back to a generic keyword so the render is never blocked.
                            kw = random.choice(DEFAULT_FALLBACK_KEYWORDS)
                            print(
                                f"  - Sentence {i}: no SNN keyword; using fallback '{kw}'"
                            )
                        sentence_keywords.append(kw)
                    try:
                        coherence, _ = (
                            self.keyword_extractor.evaluate_keyword_sequence_coherence(
                                sentences, sentence_keywords, use_snn=use_snn
                            )
                        )
                        print(f"🧠 [SNN] Selected sequence coherence: {coherence:.3f}")
                    except Exception as e:
                        print(f"⚠️ [SNN] Coherence scoring skipped: {e}")
                    for i in range(len(sentences)):
                        print(f"  - Sentence {i}: '{sentence_keywords[i]}'")
                        try:
                            self._audit_keyword_decision(
                                run_script_id,
                                i,
                                sentence_keywords[i],
                                sentences[i],
                                was_used=True,
                            )
                        except Exception as audit_err:
                            print(
                                f"⚠️ [Audit] Keyword decision logging skipped: {audit_err}"
                            )
                except Exception as e:
                    print(f"⚠️ [SNN] Sequence optimization failed, falling back: {e}")
                    use_snn = False  # fall through to standard selection below

            if not use_snn:
                # Assign unique keywords sequentially, using *semantic* similarity
                # (not just exact match) to avoid visually redundant footage.
                for i in range(len(sentences)):
                    candidates = sentence_keywords_map.get(i, [])
                    selected_kw = None
                    for kw in candidates:
                        if self.keyword_extractor.is_semantically_unique(kw):
                            selected_kw = kw
                            self.keyword_extractor.add_used_keyword(kw)
                            break

                    # If all candidates are too similar to earlier ones, fall back
                    # to the first candidate (will still fallback in pipeline).
                    if not selected_kw and candidates:
                        selected_kw = candidates[0]
                        self.keyword_extractor.add_used_keyword(selected_kw)

                    # LLM/keyword-extraction can fail outright (e.g. Ollama down).
                    # Never let that abort the render — fall back to a generic
                    # keyword so the background finder still produces footage.
                    if not selected_kw:
                        selected_kw = random.choice(DEFAULT_FALLBACK_KEYWORDS)
                        print(
                            f"  - Sentence {i}: no keyword extracted; using fallback '{selected_kw}'"
                        )

                    sentence_keywords.append(selected_kw)
                    print(f"  - Sentence {i}: '{selected_kw}'")
                    try:
                        self._audit_keyword_decision(
                            run_script_id,
                            i,
                            selected_kw,
                            sentences[i],
                            was_used=True,
                        )
                    except Exception as audit_err:
                        print(
                            f"⚠️ [Audit] Keyword decision logging skipped: {audit_err}"
                        )
        else:
            # Placeholder for audio-only
            sentence_keywords = [None] * len(sentences)

        audio_paths = []
        intro_audio_path = None
        cta_audio_path = None
        music_path = None

        try:
            if enable_background_music:
                if music_selection == "Auto (YouTube Library)":
                    print(
                        f"🎵 [Music] Auto-selection mode active. Analyzing mood for '{text[:40]}...'"
                    )
                    try:
                        mood_kw = self.keyword_extractor.extract_mood_keyword(text)
                        print(f"🎵 [Music] AI detected mood: '{mood_kw}'")
                    except Exception as mood_err:
                        print(
                            f"⚠️ [Music] Mood detection failed, using fallback: {mood_err}"
                        )
                        mood_kw = os.getenv("OLLAMA_FALLBACK_MOOD", "Cinematic")

                    yt_results = self.search_audio_library(mood_kw)
                    if not yt_results:
                        backup_kw = next(
                            (k for k in sentence_keywords if k), "Cinematic"
                        )
                        print(
                            f"⚠️ [Music] No track found for mood '{mood_kw}', trying visual keyword: '{backup_kw}'"
                        )
                        yt_results = self.search_audio_library(backup_kw)

                    if yt_results:
                        # Pick a random one from top 3 to avoid same track over and over
                        picked_idx = random.randint(0, min(2, len(yt_results) - 1))
                        track_label, track_id = yt_results[picked_idx]
                        print(
                            f"🎵 [Music] Auto-selected YouTube Library track: {track_label} (ID: {track_id})"
                        )

                        local_name = self.download_library_track(track_id, track_label)
                        if local_name:
                            music_path = self.config.MUSIC_DIR / local_name
                            print(f"✅ [Music] Track ready at: {music_path.name}")
                        else:
                            print(f"❌ [Music] Download failed for '{track_label}'")

                    if not music_path:
                        print(
                            f"⚠️ [Music] Fully failed to find automatic track. Falling back to Random local music."
                        )
                        music_path = self.video_generator.get_music_by_name("Random")
                else:
                    print(f"🎵 [Music] Manual selection: {music_selection}")
                    music_path = self.video_generator.get_music_by_name(music_selection)

                if music_path and music_path.exists():
                    print(f"🎶 [Music] Using background track: {music_path.name}")
                else:
                    print(
                        f"🔇 [Music] No background track available / track not found."
                    )

            voices_for_sentences = (
                [random.choice(self.available_voices) for _ in sentences]
                if use_random_voices
                else [speaker_id] * len(sentences)
            )

            # Prepare all audio tasks
            audio_tasks = []

            # 1. Intro Task
            if add_intro_slide:
                intro_voice = (
                    speaker_id
                    if not use_random_voices
                    else random.choice(self.available_voices)
                )
                intro_msg = self.config.INTRO_MESSAGES.get(
                    language, self.config.INTRO_MESSAGES["en"]
                )
                audio_tasks.append(
                    {
                        "type": "intro",
                        "text": intro_msg,
                        "voice": intro_voice,
                        "index": -1,
                    }
                )

            # 2. Sentence Tasks
            for i, (sentence, voice) in enumerate(zip(sentences, voices_for_sentences)):
                audio_tasks.append(
                    {"type": "sentence", "text": sentence, "voice": voice, "index": i}
                )

            # 3. CTA Task
            if add_call_to_action:
                cta_voice = (
                    speaker_id if not use_random_voices else voices_for_sentences[-1]
                )
                cta_msg = self.config.CTA_MESSAGES.get(
                    language, self.config.CTA_MESSAGES["en"]
                )
                audio_tasks.append(
                    {"type": "cta", "text": cta_msg, "voice": cta_voice, "index": 999}
                )

            # Execute Audio Generation in Parallel
            if progress_callback:
                progress_callback(
                    1, len(sentences) * 2, "Starting parallel audio generation..."
                )

            audio_paths_map = {}  # Map index -> path
            completed_audio = 0

            with ThreadPoolExecutor(
                max_workers=self.config.WORKER_POOL_TTS
            ) as audio_executor:
                future_to_task = {}
                for task in audio_tasks:
                    future = audio_executor.submit(
                        self.tts_manager.generate_speech,
                        task["text"],
                        task["voice"],
                        language,
                        speed=stress_level,
                    )
                    future_to_task[future] = task

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        path_result = future.result()
                        print(
                            f"🔈 [Audio] Task {task['type']} (idx: {task['index']}) completed: {path_result}"
                        )

                        if task["type"] == "intro":
                            intro_audio_path = path_result
                        elif task["type"] == "cta":
                            cta_audio_path = path_result
                        elif task["type"] == "sentence":
                            audio_paths_map[task["index"]] = path_result

                        completed_audio += 1
                        if progress_callback:
                            progress_callback(
                                completed_audio,
                                len(audio_tasks) * 2,
                                f"Generating Audio {completed_audio}/{len(audio_tasks)}",
                            )
                    except Exception as e:
                        print(
                            f"❌ [Audio] Unexpected task error for {task['type']}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        # Graceful Fallback handled inside generate_speech usually,
                        # but if generate_speech crashed before creating fallback:
                        path_result = None
                        # Here we just mark it as None and skip it in rendering if needed
                        if task["type"] == "intro":
                            intro_audio_path = None
                        elif task["type"] == "cta":
                            cta_audio_path = None
                        elif task["type"] == "sentence":
                            audio_paths_map[task["index"]] = None

            # Reconstruct ordered list for sentences
            audio_paths = [audio_paths_map.get(i) for i in range(len(sentences))]
            # Ensure no Nones remain in audio_paths to prevent rendering failures
            for i in range(len(audio_paths)):
                if audio_paths[i] is None:
                    print(
                        f"⚠️ [Audio] Emergency! audio_paths[{i}] is None. Creating late fallback..."
                    )
                    fallback_path = (
                        self.config.TEMP_AUDIO_DIR
                        / f"late_fallback_{uuid.uuid4().hex[:8]}.wav"
                    )
                    try:
                        AudioSegment.silent(duration=1000).export(
                            str(fallback_path), format="wav"
                        )
                        audio_paths[i] = fallback_path
                        print(f"📁 [Audio] Late fallback created: {fallback_path}")
                    except Exception as eLate:
                        print(f"💀 [Audio] Late fallback FAILED: {eLate}")

            # Define video_progress callback here, as it's used in the video generation path
            def video_progress(current, total, message):
                if progress_callback:
                    progress_callback(
                        len(sentences) + current, len(sentences) * 2, message
                    )

            # Define selected_bg_video_path here, as it's used in the video generation path
            selected_bg_video_path = None
            print(
                f"[Debug] Selected background video name from UI: '{selected_background_video_name}'"
            )
            if (
                selected_background_video_name
                and selected_background_video_name
                not in ["Auto-select (Pexels/Giphy/Local)", "Branded Gradient"]
            ):
                selected_bg_video_path = (
                    self.config.BACKGROUND_VIDEOS_DIR / selected_background_video_name
                )
                print(f"[Debug] Resolving path: {selected_bg_video_path}")
                if not selected_bg_video_path.exists():
                    print(
                        f"[Background Video] Selected background video {selected_background_video_name} not found. Falling back to auto-select."
                    )
                    selected_bg_video_path = None
                else:
                    print(
                        f"[Debug] Confirmed background video exists: {selected_bg_video_path}"
                    )
            elif selected_background_video_name == "Branded Gradient":
                print("[Debug] Using Branded Gradient background")
                selected_bg_video_path = None  # This will trigger the gradient background in _create_slide_with_ffmpeg
            else:
                print("[Debug] Auto-select enabled (default behavior)")

            if audio_only:
                # --- AUDIO ONLY STITCHING ---
                if progress_callback:
                    progress_callback(
                        len(sentences) + 1,
                        len(sentences) * 2,
                        "Stitching audio files...",
                    )

                final_audio_segments = AudioSegment.empty()

                # Intro
                if intro_audio_path and Path(intro_audio_path).exists():
                    final_audio_segments += AudioSegment.from_file(
                        str(intro_audio_path)
                    )
                    # Small pause
                    final_audio_segments += AudioSegment.silent(duration=300)

                # Sentences
                for i, p in enumerate(audio_paths):
                    if p and Path(p).exists():
                        seg = AudioSegment.from_file(str(p))
                        final_audio_segments += seg
                        # Natural pause between sentences
                        if i < len(audio_paths) - 1:
                            final_audio_segments += AudioSegment.silent(duration=500)

                # CTA
                if cta_audio_path and Path(cta_audio_path).exists():
                    final_audio_segments += AudioSegment.silent(duration=500)
                    final_audio_segments += AudioSegment.from_file(str(cta_audio_path))

                # Add background music if enabled
                if enable_background_music and music_path and music_path.exists():
                    if progress_callback:
                        progress_callback(
                            len(sentences) + 2,
                            len(sentences) * 2,
                            f"Adding background music: {music_path.name}...",
                        )
                    print(
                        f"🔊 [Mixing] Overlaying background music (Audio-Only): {music_path.name}"
                    )

                    bg_music = AudioSegment.from_file(str(music_path)) + music_volume_db
                    if len(bg_music) < len(final_audio_segments):
                        loops = (len(final_audio_segments) // len(bg_music)) + 2
                        bg_music = bg_music * loops
                    bg_music = bg_music[: len(final_audio_segments)]
                    bg_music = bg_music.fade_in(1000).fade_out(1000)
                    final_audio_segments = final_audio_segments.overlay(bg_music)

                audio_final = session_dir / f"audio_only_{timestamp}_{language}.mp3"
                if normalize_audio:
                    from pydub.effects import normalize

                    final_audio_segments = normalize(final_audio_segments, headroom=0.1)
                final_audio_segments.export(
                    str(audio_final), format="mp3", bitrate="192k"
                )

                video_final = None
                thumbnail_final = None
                lang_name = SUPPORTED_LANGUAGES.get(language, {}).get("name", language)
            else:
                # --- FULL VIDEO GENERATION ---
                # Circle logic
                circle_config = {
                    "diameter": circle_diameter,
                    "position": circle_position,
                    "border_width": circle_border_width,
                }
                circle_video_path = None
                if enable_circle_overlay:
                    if circle_upload_path and Path(circle_upload_path).exists():
                        # Move uploaded file to session dir just in case
                        uploaded_path = Path(circle_upload_path)
                        circle_video_path = (
                            session_dir / f"uploaded_circle_{uploaded_path.name}"
                        )
                        shutil.copy(circle_upload_path, circle_video_path)
                    elif circle_selection and circle_selection != "Random":
                        circle_video_path = (
                            self.config.CIRCLE_OVERLAYS_DIR / circle_selection
                        )
                        if not circle_video_path.exists():
                            print(
                                f"[Circle] Selected overlay {circle_selection} not found, falling back to random"
                            )
                            circle_video_path = (
                                self.video_generator.get_circle_overlay_video()
                            )
                    else:
                        circle_video_path = (
                            self.video_generator.get_circle_overlay_video()
                        )

                # Apply aspect ratio and quality to generator
                ar_config = self.config.get_aspect_ratio(aspect_ratio)
                q_config = self.config.get_quality(quality)
                self.video_generator.video_width = ar_config["width"]
                self.video_generator.video_height = ar_config["height"]
                self.video_generator.video_preset = q_config["preset"]
                self.video_generator.video_crf = q_config["crf"]
                effective_fps = export_fps or q_config["fps"]
                print(
                    f"📐 [Pipeline] Aspect: {ar_config['label']} ({ar_config['width']}x{ar_config['height']}), Quality: {q_config['label']} ({q_config['preset']}, CRF {q_config['crf']}, {effective_fps}fps)"
                )

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
                    pre_selected_videos=pre_selected_videos,
                    hide_text=hide_text,
                    export_fps=effective_fps,
                    overlay_shape=overlay_shape,
                    intro_text=intro_msg if add_intro_slide else None,
                    use_snn=use_snn,
                    enable_crossfade=enable_crossfade,
                    crossfade_duration=0.3,
                    progress_callback=video_progress,
                    theme=theme,
                    entity=entity,
                    script_id=run_script_id,
                    candidate_map=sentence_keywords_map,
                )

                if enable_background_music and music_path and music_path.exists():
                    if progress_callback:
                        progress_callback(
                            len(sentences) * 2 - 1,
                            len(sentences) * 2,
                            f"Adding background music: {music_path.name}...",
                        )
                    print(f"🔊 [Mixing] Overlaying background music: {music_path.name}")
                    audio_extract = (
                        self.config.TEMP_DIR / f"extracted_{uuid.uuid4().hex[:8]}.wav"
                    )
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(video_temp_path),
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            str(audio_extract),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    voice_seg = AudioSegment.from_file(str(audio_extract))
                    music = AudioSegment.from_file(str(music_path)) + music_volume_db
                    if len(music) < len(voice_seg):
                        loops = (len(voice_seg) // len(music)) + 2
                        music = music * loops
                    music = music[: len(voice_seg)]
                    music = music.fade_in(1000).fade_out(1000)
                    mixed = voice_seg.overlay(music)
                    if normalize_audio:
                        from pydub.effects import normalize

                        mixed = normalize(mixed, headroom=0.1)
                    mixed_audio = (
                        self.config.TEMP_DIR / f"mixed_{uuid.uuid4().hex[:8]}.wav"
                    )
                    mixed.export(str(mixed_audio), format="wav")
                    video_with_music = (
                        self.config.TEMP_DIR / f"with_music_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(video_temp_path),
                            "-i",
                            str(mixed_audio),
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-b:a",
                            "192k",
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-shortest",
                            str(video_with_music),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    audio_extract.unlink(missing_ok=True)
                    mixed_audio.unlink(missing_ok=True)
                    video_temp_path.unlink(missing_ok=True)
                    video_temp_path = video_with_music

                lang_name = SUPPORTED_LANGUAGES.get(language, {}).get("name", language)

                # Determine keyword for filename
                filename_keyword = "generated"
                if pexels_keyword and pexels_keyword.strip():
                    # Use provided Pexels keyword
                    filename_keyword = "".join(
                        [
                            c if c.isalnum() else "_"
                            for c in pexels_keyword.strip().lower()
                        ]
                    )
                elif (
                    sentence_keywords
                    and len(sentence_keywords) > 0
                    and sentence_keywords[0]
                ):
                    # Use first extract keyword
                    filename_keyword = "".join(
                        [
                            c if c.isalnum() else "_"
                            for c in sentence_keywords[0].lower()
                        ]
                    )

                video_final = (
                    session_dir / f"video_{filename_keyword}_{timestamp}_{language}.mp4"
                )
                shutil.move(str(video_temp_path), str(video_final))

                audio_final = session_dir / f"audio_{timestamp}_{language}.mp3"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(video_final),
                        "-vn",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "192k",
                        str(audio_final),
                    ],
                    check=True,
                    capture_output=True,
                )

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
                "video_path": str(video_final) if video_final else None,
                "thumbnail_path": str(thumbnail_final)
                if thumbnail_final and thumbnail_final.exists()
                else None,
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "language": lang_name,
                "language_code": language,
                "background_music": enable_background_music and music_path is not None,
                "music_used": music_path.name if music_path else None,
                "intro_included": add_intro_slide,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "video_backgrounds": selected_background_video_name
                if selected_background_video_name
                else "Pexels/Giphy API + Local",
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
    with gr.Blocks(
        title="AI Video Generator Pro", theme=gr.themes.Soft(primary_hue="blue")
    ) as demo:
        gr.Markdown("# 🎬 Shorts Generator")
        gr.Markdown(
            "Create stunning videos with multi-language TTS, auto-backgrounds, and dynamic overlays."
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("📝 Content"):
                        text_input = gr.Textbox(
                            label="Text Content",
                            placeholder="Enter your script here...",
                            lines=10,
                        )
                        with gr.Row():
                            btn_generate_script = gr.Button(
                                "✨ AI Clean & Generate Script (No Pauses)",
                                variant="secondary",
                                size="sm",
                            )

                        with gr.Row():
                            ai_api_url = gr.Textbox(
                                label="🌐 AI API URL",
                                value=generator.keyword_extractor.api_url,
                                placeholder="https://ai.izdrail.com/api/generate",
                                info="Endpoint for Ollama keyword extraction",
                            )
                        with gr.Row():
                            ai_model_dropdown = gr.Dropdown(
                                label="🤖 AI Model",
                                choices=generator.available_models,
                                value="gemma4:e2b",
                                info="Select LLM for keyword extraction",
                            )
                            btn_refresh_models = gr.Button(
                                "🔄 Refresh Models", size="sm"
                            )

                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                label="🌐 Language",
                                choices=[
                                    (SUPPORTED_LANGUAGES[k]["name"], k)
                                    for k in generator.available_languages
                                ],
                                value="auto",
                            )
                            speaker_dropdown = gr.Dropdown(
                                label="🎙️ Voice",
                                choices=generator.available_voices,
                                value=generator.config.STANDARD_VOICE_NAME,
                            )
                        use_random_voices = gr.Checkbox(
                            label="🎲 Random voice per sentence", value=False
                        )
                        with gr.Row():
                            preview_voice_btn = gr.Button("👂 Preview Voice", size="sm")
                            preview_audio = gr.Audio(
                                label="Voice Preview", interactive=False
                            )

                    with gr.TabItem("🎥 Media"):
                        media_source_dropdown = gr.Dropdown(
                            label="🎞️ Preferred Media Source",
                            choices=[
                                "Random",
                                "Pexels",
                                "Pixabay",
                                "YouTube",
                                "Giphy",
                                "SearXNG",
                                "Dailymotion",
                                "Vimeo",
                                "Twitch",
                                "PeerTube",
                                "api.video",
                                "Cloudflare Stream",
                                "Mux",
                                "Kaltura",
                                "JSON2Video",
                            ],
                            value="YouTube",
                            info="Select your primary source for background videos (Random shuffles available APIs)",
                        )
                        pexels_keyword = gr.Textbox(
                            label="🔍 Custom Search Keyword",
                            placeholder="e.g., 'cyberpunk city', 'peaceful forest'",
                            info="Leave empty for auto-extraction",
                        )
                        entity_input = gr.Textbox(
                            label="🏷️ Entity (optional)",
                            placeholder="e.g., 'Tesla', 'Paris', 'NASA'",
                            info="Person, brand, location or concept to bias keywords & searches",
                        )
                        background_video_dropdown = gr.Dropdown(
                            label="🏞️ Select Background Video",
                            choices=generator.available_background_videos,
                            value="Auto-select (Pexels/Giphy/Local)",
                            info="Choose a specific video from your 'background_videos' folder or let the system auto-select.",
                        )
                        with gr.Row():
                            enable_music = gr.Checkbox(
                                label="🎵 Add background music", value=True
                            )
                            music_dropdown = gr.Dropdown(
                                label="Music Track",
                                choices=generator.available_music,
                                value="Random",
                            )

                        with gr.Accordion(
                            "🔎 Search YouTube Free Audio Library", open=False
                        ):
                            with gr.Row():
                                yt_audio_query = gr.Textbox(
                                    label="Search Tracks",
                                    placeholder="e.g. 'Epic', 'Chill'...",
                                )
                                yt_audio_search_btn = gr.Button("🔍 Search")
                            yt_audio_results = gr.Dropdown(
                                label="Library Results",
                                choices=[],
                                info="Search for tracks and select one to use.",
                            )
                            yt_audio_download_info = gr.Markdown(
                                "*Search and select a track to download it to your local 'background_music' folder.*"
                            )

                        music_volume = gr.Slider(
                            -40, -5, -22, 1, label="Music Volume (dB)"
                        )

                    with gr.TabItem("⭕ Overlays"):
                        enable_circle = gr.Checkbox(
                            label="Enable Picture-in-Picture Circle", value=False
                        )
                        circle_selection = gr.Dropdown(
                            label="Circle Content (Local Folder)",
                            choices=generator.available_circles,
                            value="Random",
                            info="Select a video from the local 'circle_overlays' folder",
                        )
                        circle_upload = gr.File(
                            label="📤 Upload Custom Circle Video", file_types=["video"]
                        )
                        with gr.Row():
                            circle_diameter = gr.Slider(
                                150, 600, 300, 25, label="Diameter (px)"
                            )
                            circle_border_width = gr.Slider(
                                0, 20, 5, 1, label="Border Width (px)"
                            )
                            circle_position = gr.Dropdown(
                                [
                                    "top-left",
                                    "top-right",
                                    "bottom-left",
                                    "bottom-right",
                                    "center",
                                ],
                                value="top-right",
                                label="Position",
                            )
                            overlay_shape = gr.Dropdown(
                                [
                                    "Circle",
                                    "Rectangle",
                                    "Square",
                                    "Star",
                                    "Split Screen",
                                ],
                                value="Circle",
                                label="Overlay Shape",
                                info="Shape of the PIP overlay",
                            )

                    with gr.TabItem("⚙️ Advanced"):
                        with gr.Row():
                            aspect_ratio_dropdown = gr.Dropdown(
                                label="📐 Aspect Ratio",
                                choices=list(generator.config.ASPECT_RATIOS.keys()),
                                value="9:16 Portrait (TikTok/Shorts)",
                                info="Output video dimensions",
                            )
                            quality_dropdown = gr.Dropdown(
                                label="🎯 Quality",
                                choices=list(generator.config.QUALITY_PRESETS.keys()),
                                value="Medium (Balanced)",
                                info="Encoding quality vs speed tradeoff",
                            )
                        preset_dropdown = gr.Dropdown(
                            label="📋 Quick Presets",
                            choices=[
                                "Default",
                                "TikTok Viral",
                                "YouTube Shorts High-Energy",
                                "Instagram Reels Aesthetic",
                            ],
                            value="Default",
                            info="Select a preset to automatically adjust speed, FPS, and volume.",
                        )
                        with gr.Row():
                            enable_intro = gr.Checkbox(
                                label="📢 Add Intro Slide", value=True
                            )
                            enable_cta = gr.Checkbox(
                                label="📣 Add CTA Outro", value=True
                            )
                            enable_crossfade_checkbox = gr.Checkbox(
                                label="✨ Crossfade Slides", value=False
                            )
                        hide_text = gr.Checkbox(
                            label="🛑 Hide Text Overlay", value=False
                        )
                        export_fps = gr.Slider(
                            10,
                            60,
                            30,
                            1,
                            label="🎞️ Export FPS",
                            info="0 = auto based on quality preset (default: 30)",
                        )
                        stress_level = gr.Slider(
                            0.8,
                            1.5,
                            1.0,
                            0.1,
                            label="🗣️ Voice Speed / Stress",
                            info="1.0 is normal, higher is faster/more energetic",
                        )
                        use_snn_checkbox = gr.Checkbox(
                            label="🧠 Use SNN Biological Evaluation (Slow but Realistic)",
                            value=False,
                        )
                        audio_only_checkbox = gr.Checkbox(
                            label="🔊 Only generate audio (skip video rendering)",
                            value=False,
                        )
                        normalize_audio_checkbox = gr.Checkbox(
                            label="🔊 Normalize Audio (LUFS)", value=True
                        )
                        with gr.Row():
                            clear_cache_btn = gr.Button(
                                "🗑️ Clear Cache", variant="secondary", size="sm"
                            )
                            cache_status = gr.Textbox(
                                label="Cache",
                                value="",
                                interactive=False,
                                visible=False,
                            )

                    with gr.TabItem("🎬 Background Selection"):
                        with gr.Row():
                            preview_btn = gr.Button(
                                "🔍 Preview & Find Backgrounds",
                                variant="primary",
                                scale=2,
                            )
                            clear_preview_btn = gr.Button("🗑️ Clear", scale=1, size="sm")

                        preview_html = gr.HTML(
                            value="<p style='text-align:center;color:#888;padding:40px'>Click <b>Preview & Find Backgrounds</b> to search for videos per slide.</p>",
                        )

                        preview_js = gr.HTML(
                            value="""<style>
 .hidden-textbox { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0; }
</style>
<script>
document.addEventListener('click',function(e){var c=e.target.closest('.vid-card');if(c){c.classList.toggle('selected');var cb=c.querySelector('input[type=checkbox]');if(cb)cb.checked=c.classList.contains('selected');b();}});
document.addEventListener('change',function(e){if(e.target.matches('.vid-card input[type=checkbox]')){var c=e.target.closest('.vid-card');if(c){c.classList.toggle('selected');b();}}});
function b(){var s={};document.querySelectorAll('.vid-card.selected').forEach(function(c){var i=c.getAttribute('data-slide');var r=c.getAttribute('data-source');if(!s[i])s[i]=[];s[i].push(r);});var x=document.getElementById('js-selections-storage');if(!x)return;var t=x.tagName==='INPUT'||x.tagName==='TEXTAREA'?x:x.querySelector('input,textarea');if(!t)return;t.value=JSON.stringify(s);t.dispatchEvent(new Event('input',{bubbles:true}));t.dispatchEvent(new Event('change',{bubbles:true}));}
</script>"""
                        )

                        js_selections = gr.Textbox(
                            value="{}",
                            elem_id="js-selections-storage",
                            elem_classes="hidden-textbox",
                        )

                        custom_selections_input = gr.Textbox(
                            label="✏️ Per-Slide Override Rules",
                            placeholder=(
                                "Format: slide_num:action, one per line\n\n"
                                "Examples:\n"
                                "  1:gradient\n"
                                "  3:skip\n"
                                "  5:/path/to/video.mp4\n\n"
                                "Leave empty to use suggested videos."
                            ),
                            lines=2,
                        )
                        apply_btn = gr.Button("💾 Apply Overrides", variant="secondary")

                        selection_status = gr.Markdown(
                            "*No backgrounds selected. System will auto-select.*"
                        )

                        pre_selected_videos_state = gr.State(None)
                        preview_data_state = gr.State([])

                generate_button = gr.Button(
                    "🚀 Generate Video", variant="primary", size="lg"
                )
                engine_status_output = gr.Textbox(
                    label="TTS Engine Status", value="Idle", interactive=False
                )
                progress_bar = gr.Textbox(
                    label="⚡ Status", value="Ready", interactive=False
                )

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🎬 Video"):
                        video_output = gr.Video(label="Generated Video", height=600)
                    with gr.TabItem("🖼️ Thumbnail"):
                        thumbnail_output = gr.Image(
                            label="Last Frame Thumbnail", type="filepath"
                        )
                    with gr.TabItem("🎵 Audio"):
                        audio_output = gr.Audio(label="Extracted Voiceover")
                    with gr.TabItem("📢 Social"):
                        social_output = gr.Textbox(
                            label="Social Media Descriptions", lines=15
                        )
                char_count_display = gr.Markdown(
                    value="**Characters:** 0 | **TikTok:** ✅ | **Shorts:** ✅"
                )
                status_output = gr.Markdown(
                    value="*Your video will appear here after generation.*"
                )

        def generate_wrapper(
            text,
            language,
            speaker,
            use_random,
            media_source,
            keyword,
            selected_background_video_name,
            enable_music,
            music_select,
            music_vol,
            enable_circle,
            circle_sel,
            circle_upload_path,
            circle_diam,
            circle_border,
            circle_pos,
            overlay_shape_val,
            enable_intro,
            enable_cta,
            hide_text,
            export_fps_val,
            ai_model_val,
            ai_api_url_val,
            stress_level_val,
            use_snn_val,
            audio_only_val,
            normalize_audio_val,
            aspect_ratio_val,
            quality_val,
            enable_crossfade_val,
            pre_selected_videos,
            js_json,
            override_text,
            slide_data,
            entity,
            progress=gr.Progress(),
        ):

            if not text or not text.strip():
                return (
                    None,
                    None,
                    None,
                    None,
                    "Idle",
                    "❌ **Error:** Please enter some text.",
                    "Ready",
                )

            def update_progress(current, total, message):
                progress((current, total), desc=message)
                return f"{current}/{total}: {message}"

            # Process visually-selected backgrounds inline (no Apply step needed)
            final_pre_selected = {}
            if pre_selected_videos is not None and isinstance(
                pre_selected_videos, dict
            ):
                final_pre_selected.update(pre_selected_videos)
                print(
                    f"[DEBUG] generate_wrapper: pre_selected_videos state has {len(pre_selected_videos)} entries: {dict(list(pre_selected_videos.items())[:3])}..."
                )
            else:
                print(
                    f"[DEBUG] generate_wrapper: pre_selected_videos state is {type(pre_selected_videos).__name__}: {pre_selected_videos}"
                )

            media_mgr = generator.video_generator.media_manager
            print(
                f"[DEBUG] generate_wrapper: js_json='{js_json[:200] if js_json else None}', override_text='{override_text[:200] if override_text else None}', slide_data={'present' if slide_data else 'None'}"
            )
            if js_json and js_json.strip() and js_json.strip() != "{}" and slide_data:
                try:
                    import json as _json

                    visual_sel = _json.loads(js_json)
                    for slide_num_str, sources in visual_sel.items():
                        try:
                            s_num = int(slide_num_str)
                        except ValueError:
                            continue
                        if not sources:
                            continue
                        src = sources[0]
                        kw = None
                        for sd in slide_data:
                            if sd["slide_num"] == s_num:
                                kw = sd.get("keyword")
                                break
                        if kw and kw != "N/A":
                            update_progress(
                                0, 1, f"Downloading {src} video for slide {s_num}..."
                            )
                            print(
                                f"⬇️ [Generate] Downloading {src} video for slide {s_num} ('{kw}')..."
                            )
                            try:
                                result = media_mgr.get_random_media(
                                    [kw], preferred_source=src
                                )
                                if result and isinstance(result, tuple):
                                    result = result[0]
                                if result:
                                    final_pre_selected[s_num] = str(result)
                                    print(
                                        f"✅ [Generate] Slide {s_num} <- {src}: {Path(result).name}"
                                    )
                            except Exception as e:
                                print(
                                    f"⚠️ [Generate] Download failed slide {s_num} from {src}: {e}"
                                )
                except Exception as e:
                    print(f"⚠️ [Generate] Failed to parse visual selections: {e}")

            # Apply text overrides (overwrites visual selections)
            print(
                f"[DEBUG] generate_wrapper: after JS processing, final_pre_selected has {len(final_pre_selected)} entries"
            )
            if override_text and override_text.strip():
                for line in override_text.strip().split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        s_num = int(parts[0].strip())
                    except ValueError:
                        continue
                    action = parts[1].strip().lower()
                    if action == "gradient":
                        final_pre_selected[s_num] = "__gradient__"
                    elif action == "skip":
                        final_pre_selected.pop(s_num, None)
                    elif action.startswith("/") or action.startswith("."):
                        p = Path(action)
                        if p.exists():
                            final_pre_selected[s_num] = str(p)
                        else:
                            print(f"⚠️ [Generate] Override path not found: {action}")

            if final_pre_selected:
                gc = sum(1 for v in final_pre_selected.values() if v == "__gradient__")
                vc = len(final_pre_selected) - gc
                print(
                    f"🎯 [Generate] Using {vc} custom video(s) + {gc} gradient(s) from visual selections."
                )

            # Gradio 3/4 compatibility for file upload
            final_circle_path = None
            if circle_upload_path:
                if isinstance(circle_upload_path, (list, tuple)):
                    circle_upload_path = circle_upload_path[0]

                if hasattr(circle_upload_path, "name"):  # Gradio 3 File object
                    final_circle_path = circle_upload_path.name
                elif hasattr(circle_upload_path, "path"):  # Gradio 4 FileData object
                    final_circle_path = circle_upload_path.path
                elif isinstance(circle_upload_path, str):  # Raw path string
                    final_circle_path = circle_upload_path

            result = generator.generate_video(
                text=text,
                language=language,
                speaker_id=speaker,
                pexels_keyword=keyword.strip() if keyword else None,
                preferred_media_source=media_source,
                selected_background_video_name=selected_background_video_name,
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
                circle_upload_path=final_circle_path,
                hide_text=hide_text,
                export_fps=export_fps_val,
                overlay_shape=overlay_shape_val,
                ai_model=ai_model_val,
                ai_api_url=ai_api_url_val,
                stress_level=stress_level_val,
                use_snn=use_snn_val,
                audio_only=audio_only_val,
                normalize_audio=normalize_audio_val,
                aspect_ratio=aspect_ratio_val,
                quality=quality_val,
                enable_crossfade=enable_crossfade_val,
                pre_selected_videos=final_pre_selected,
                entity=entity,
                progress_callback=update_progress,
            )

            engine_status = generator.tts_manager.last_status_message

            if result.get("success"):
                # Generate social media descriptions using keywords
                keywords_used = generator.keyword_extractor.used_keywords
                progress((99, 100), desc="Generating social media descriptions...")
                try:
                    social_desc = (
                        generator.keyword_extractor.generate_social_media_descriptions(
                            text, list(keywords_used), language
                        )
                    )
                except Exception as e:
                    print(f"⚠️ [Social] Failed to generate descriptions: {e}")
                    social_desc = "⚠️ Social media descriptions could not be generated (Ollama/LLM timeout or error)."

                status_md = f"""### ✅ Generation Complete!
- **Video:** {result["video_path"]}
- **Duration:** {result.get("duration", "N/A")}s
- **Source:** {media_source}
"""
                return (
                    result["video_path"],
                    result.get("thumbnail_path"),
                    result["audio_path"],
                    social_desc,
                    engine_status,
                    status_md,
                    "Complete!",
                )

            return (
                None,
                None,
                None,
                None,
                engine_status,
                f"❌ **Error:** {result.get('error', 'Unknown error')}",
                "Failed",
            )

        def refresh_models_action(url):
            from core.nlp.keyword_extractor import OllamaKeywordExtractor

            try:
                models = OllamaKeywordExtractor.fetch_models_static(url)
                return gr.Dropdown(
                    choices=models, value=models[0] if models else "gemma4:e2b"
                )
            except Exception as e:
                print(f"Error refreshing models: {e}")
                return gr.Dropdown(choices=["gemma4:e2b"], value="gemma4:e2b")

        btn_refresh_models.click(
            fn=refresh_models_action, inputs=[ai_api_url], outputs=[ai_model_dropdown]
        )

        def generate_script_action(text, ai_model, api_url):
            if not text or not text.strip():
                return text

            # Update connection settings just in case
            if api_url and api_url != generator.keyword_extractor.api_url:
                generator.keyword_extractor.api_url = api_url
            if ai_model and ai_model != generator.keyword_extractor.model:
                generator.keyword_extractor.model = ai_model

            print(f"Generating clean script for: {text[:50]}...")
            return generator.keyword_extractor.generate_script_from_text(text)

        btn_generate_script.click(
            fn=generate_script_action,
            inputs=[text_input, ai_model_dropdown, ai_api_url],
            outputs=[text_input],
        )

        # Audio handlers
        preview_voice_btn.click(
            fn=generator.preview_voice,
            inputs=[speaker_dropdown, language_dropdown, stress_level],
            outputs=[preview_audio],
        )

        # State to store search results for unpacking
        yt_search_results_state = gr.State([])

        def yt_audio_search_action(query):
            if not query:
                return gr.Dropdown(choices=[]), []
            results = generator.search_audio_library(query)
            # results is a list of (label, value) tuples
            return gr.Dropdown(choices=results, value=None), results

        yt_audio_search_btn.click(
            fn=yt_audio_search_action,
            inputs=[yt_audio_query],
            outputs=[yt_audio_results, yt_search_results_state],
        )

        def yt_audio_download_action(track_id, all_results):
            if not track_id or not all_results:
                return gr.Dropdown(), gr.Markdown()

            # Find the track name from the results list in state
            track_name = "Library Track"
            for label, val in all_results:
                if val == track_id:
                    track_name = label
                    break

            local_name = generator.download_library_track(track_id, track_name)
            if local_name:
                # Update main music dropdown choices and select the new track
                new_choices = generator.available_music
                return gr.Dropdown(
                    choices=new_choices, value=local_name
                ), f"✅ Downloaded: {local_name}"
            return gr.Dropdown(), "❌ Download failed."

        yt_audio_results.change(
            fn=yt_audio_download_action,
            inputs=[yt_audio_results, yt_search_results_state],
            outputs=[music_dropdown, yt_audio_download_info],
        )

        generate_button.click(
            fn=generate_wrapper,
            inputs=[
                text_input,
                language_dropdown,
                speaker_dropdown,
                use_random_voices,
                media_source_dropdown,
                pexels_keyword,
                background_video_dropdown,
                enable_music,
                music_dropdown,
                music_volume,
                enable_circle,
                circle_selection,
                circle_upload,
                circle_diameter,
                circle_border_width,
                circle_position,
                overlay_shape,
                enable_intro,
                enable_cta,
                hide_text,
                export_fps,
                ai_model_dropdown,
                ai_api_url,
                stress_level,
                use_snn_checkbox,
                audio_only_checkbox,
                normalize_audio_checkbox,
                aspect_ratio_dropdown,
                quality_dropdown,
                enable_crossfade_checkbox,
                pre_selected_videos_state,
                js_selections,
                custom_selections_input,
                preview_data_state,
                entity_input,
            ],
            outputs=[
                video_output,
                thumbnail_output,
                audio_output,
                social_output,
                engine_status_output,
                status_output,
                progress_bar,
            ],
        )

        def preview_backgrounds_action(
            text, language, media_source, pexels_keyword, entity=None
        ):
            if not text or not text.strip():
                return (
                    "<p style='color:red'>Please enter text first.</p>",
                    [],
                    "*Please enter text first.*",
                )

            kw_lang = language if language != "auto" else "en"
            kw_extractor = generator.keyword_extractor
            kw_extractor.clear_used()

            sentences = generator.video_generator.split_into_sentences(text)
            media_mgr = generator.video_generator.media_manager

            ordered_sources = media_mgr._get_ordered_sources(media_source)

            slide_data = []
            cards_html = ""

            for i, sentence in enumerate(sentences[:20]):
                candidates = kw_extractor.extract_keywords(
                    sentence, top_n=5, language=kw_lang, entity=entity
                )
                kw = None
                for c in candidates:
                    if kw_extractor.is_semantically_unique(c):
                        kw = c
                        kw_extractor.add_used_keyword(c)
                        break
                if not kw and candidates:
                    kw = candidates[0]
                    kw_extractor.add_used_keyword(kw)

                source_cards = ""
                available_sources = []

                if kw:
                    # Enrich the preview search with the entity for sharper,
                    # on-topic thumbnails when an entity is supplied.
                    search_kw = kw
                    if entity and entity.strip():
                        search_kw = f"{kw} {entity.strip()}"
                    for src in ordered_sources:
                        api = media_mgr.apis.get(src)
                        if not api or not hasattr(api, "search_videos"):
                            continue
                        if (
                            hasattr(api, "api_key")
                            and src not in ("YouTube", "SearXNG")
                            and not api.api_key
                        ):
                            continue
                        try:
                            results = api.search_videos(search_kw, per_page=3)
                            if results:
                                available_sources.append(src)
                                r = results[0]
                                thumb = ""
                                thumb_style = "background:#f0f0f0;display:flex;align-items:center;justify-content:center"
                                if src == "YouTube":
                                    vid = r.get("id", "")
                                    if vid:
                                        thumb = f"https://img.youtube.com/vi/{vid}/default.jpg"
                                elif r.get("thumbnail"):
                                    thumb = r["thumbnail"]
                                elif src == "Unsplash":
                                    thumb = r.get("url", "")

                                if thumb:
                                    thumb_style = (
                                        f"background-image:url({thumb});"
                                        f"background-size:cover;background-position:center"
                                    )

                                source_cards += f"""
                                <div class="vid-card" data-slide="{i}" data-source="{src}">
                                    <div class="vid-thumb" style="{thumb_style}">
                                        {"" if thumb else f'<span style="font-size:24px">🎬</span>'}
                                    </div>
                                    <div class="vid-source">{src}</div>
                                    <div class="vid-check">
                                        <input type="checkbox">
                                        <span>Use this</span>
                                    </div>
                                </div>"""
                        except Exception:
                            continue

                if not source_cards:
                    source_cards = (
                        "<p style='color:#999;padding:12px'>No videos found</p>"
                    )

                display_sentence = (
                    sentence[:80] + "..." if len(sentence) > 80 else sentence
                )
                kw_display = kw or "N/A"

                cards_html += f"""
                <div class="slide-section">
                    <div class="slide-hdr">
                        <span class="slide-num">Slide {i}</span>
                        <span class="slide-kw">Keyword: {kw_display}</span>
                    </div>
                    <div class="slide-txt">{display_sentence}</div>
                    <div class="vid-grid">
                        {source_cards}
                    </div>
                </div>"""

                slide_data.append(
                    {
                        "slide_num": i,
                        "keyword": kw or "N/A",
                        "found_source": (
                            ", ".join(available_sources) if available_sources else "N/A"
                        ),
                    }
                )

            html = f"""<style>
.slide-section {{ margin:12px 0; border:1px solid #2a2a3e; border-radius:10px; padding:14px; background:#1a1a2e; }}
.slide-hdr {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }}
.slide-num {{ font-weight:700; font-size:15px; color:#e0e0e0; }}
.slide-kw {{ font-size:12px; color:#aaa; }}
.slide-txt {{ font-size:13px; color:#ccc; margin-bottom:10px; line-height:1.4; }}
.vid-grid {{ display:flex; flex-wrap:wrap; gap:10px; }}
.vid-card {{ border:2px solid #3a3a4e; border-radius:8px; width:160px; cursor:pointer; transition:all .15s; background:#2a2a3e; overflow:hidden; }}
.vid-card:hover {{ border-color:#7c3aed; box-shadow:0 2px 12px rgba(124,58,237,.25); }}
.vid-card.selected {{ border-color:#4CAF50; background:#1a3a1a; }}
.vid-thumb {{ width:100%; height:90px; background:#333; }}
.vid-source {{ font-size:12px; font-weight:600; padding:4px 8px 0 8px; color:#ccc; }}
.vid-check {{ display:flex; align-items:center; gap:4px; padding:2px 8px 6px 8px; font-size:11px; color:#aaa; }}
.vid-check input {{ margin:0; }}
</style>
{cards_html}"""

            status_text = f"✅ {len(slide_data)} slides previewed. Click cards to select videos, then click Apply."

            return html, slide_data, status_text

        preview_btn.click(
            fn=preview_backgrounds_action,
            inputs=[
                text_input,
                language_dropdown,
                media_source_dropdown,
                pexels_keyword,
                entity_input,
            ],
            outputs=[
                preview_html,
                preview_data_state,
                selection_status,
            ],
        )

        def clear_preview_action():
            return (
                "<p style='text-align:center;color:#888;padding:40px'>Click <b>Preview & Find Backgrounds</b> to search for videos per slide.</p>",
                [],
                None,
                "*No backgrounds selected. System will auto-select.*",
            )

        clear_preview_btn.click(
            fn=clear_preview_action,
            inputs=[],
            outputs=[
                preview_html,
                preview_data_state,
                pre_selected_videos_state,
                selection_status,
            ],
        )

        # ── Debug panel: keyword selection audit ──
        def _format_selection_history(limit: int = 50) -> str:
            try:
                history = generator.keyword_extractor.get_selection_history(limit)
            except Exception as e:
                return f"⚠️ Could not read selection history: {e}"
            if not history:
                return (
                    "*No keyword selections recorded yet. Run a preview or "
                    "generation first.*"
                )
            lines = []
            for h in history[-limit:]:
                ctx = h.get("context", {})
                text = ctx.get("text", "")
                lines.append(
                    f"[{h.get('timestamp', '?')}] text='{text}' "
                    f"entity={h.get('entity')!r} ({h.get('entity_type')})"
                )
                lines.append(
                    f"  source={h.get('source')} fallback={h.get('fallback_used')}"
                )
                lines.append(f"  candidates={h.get('candidates')}")
                lines.append(f"  selected={h.get('selected')}")
                for r in h.get("reasoning", []):
                    lines.append(f"  ↳ {r}")
            return "\n".join(lines)

        def refresh_debug_action():
            return _format_selection_history(50)

        def set_debug_action(enabled: bool):
            try:
                generator.keyword_extractor.set_debug_mode(bool(enabled))
            except Exception:
                pass
            return (
                f"🔍 Debug mode {'ON' if enabled else 'OFF'}. "
                "Keyword selection audit logging "
                f"{'enabled' if enabled else 'disabled'}."
            )

        with gr.Accordion("🔍 Debug: Keyword Selection", open=False):
            with gr.Row():
                refresh_debug_btn = gr.Button("🔄 Refresh Debug")
                debug_toggle = gr.Checkbox(label="Debug Mode", value=False)
            debug_status = gr.Markdown(
                "*Toggle debug mode to log every keyword decision.*"
            )
            debug_output = gr.Textbox(
                label="Selection History",
                lines=20,
                interactive=False,
                placeholder=(
                    "Keyword selection history will appear here after you "
                    "preview or generate..."
                ),
            )

        refresh_debug_btn.click(
            fn=refresh_debug_action,
            inputs=[],
            outputs=[debug_output],
        )
        debug_toggle.change(
            fn=set_debug_action,
            inputs=[debug_toggle],
            outputs=[debug_status],
        )

        def apply_overrides_action(override_text, slide_data, js_json):
            if not slide_data:
                return (
                    None,
                    "*No slide data. Click 'Preview & Find Backgrounds' first.*",
                )

            pre_selected = {}
            total = len(slide_data)
            media_mgr = generator.video_generator.media_manager

            # 1. Parse visual checkbox selections from the HTML UI
            if js_json and js_json.strip() and js_json.strip() != "{}":
                try:
                    import json as _json

                    visual_sel = _json.loads(js_json)
                    for slide_num_str, sources in visual_sel.items():
                        try:
                            s_num = int(slide_num_str)
                        except ValueError:
                            continue
                        if not sources:
                            continue
                        src = sources[0]
                        kw = None
                        for sd in slide_data:
                            if sd["slide_num"] == s_num:
                                kw = sd.get("keyword")
                                break
                        if kw and kw != "N/A":
                            print(
                                f"⬇️ Downloading {src} video for slide {s_num} ('{kw}')..."
                            )
                            try:
                                result = media_mgr.get_random_media(
                                    [kw], preferred_source=src
                                )
                                if result and isinstance(result, tuple):
                                    result = result[0]
                                if result:
                                    pre_selected[s_num] = str(result)
                                    print(
                                        f"✅ Slide {s_num} <- {src}: {Path(result).name}"
                                    )
                            except Exception as e:
                                print(
                                    f"⚠️ Download failed slide {s_num} from {src}: {e}"
                                )
                except Exception as e:
                    print(f"⚠️ Failed to parse visual selections: {e}")

            # 2. Parse text overrides (overwrites visual selections)
            if override_text and override_text.strip():
                for line in override_text.strip().split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        s_num = int(parts[0].strip())
                    except ValueError:
                        continue
                    action = parts[1].strip().lower()

                    if action == "gradient":
                        pre_selected[s_num] = "__gradient__"
                    elif action == "skip":
                        pass  # Not in dict = auto-select during generation
                    elif action.startswith("/") or action.startswith("."):
                        p = Path(action)
                        if p.exists():
                            pre_selected[s_num] = str(p)
                        else:
                            print(f"⚠️ Override: path not found: {action}")

            count = len(pre_selected)
            gradient_count = len(
                [v for v in pre_selected.values() if v == "__gradient__"]
            )
            video_count = count - gradient_count
            auto_count = total - video_count - gradient_count
            msg = f"✅ {video_count} custom video(s), {gradient_count} gradient(s). {auto_count} slide(s) will auto-select during generation."
            return pre_selected, msg

        apply_btn.click(
            fn=apply_overrides_action,
            inputs=[custom_selections_input, preview_data_state, js_selections],
            outputs=[
                pre_selected_videos_state,
                selection_status,
            ],
        )

        def clear_cache_action():
            from core.database import DB

            try:
                import sqlite3

                with sqlite3.connect(str(DB.db_path)) as conn:
                    conn.execute("DELETE FROM tts_cache")
                    conn.execute("DELETE FROM video_logs")
                    conn.commit()
                return "✅ Cache cleared"
            except Exception as e:
                return f"❌ Failed: {e}"

        clear_cache_btn.click(
            fn=clear_cache_action, inputs=[], outputs=[cache_status]
        ).then(fn=lambda: gr.Textbox(visible=True), inputs=[], outputs=[cache_status])

        def update_char_counts(text):
            if not text:
                return "**Characters:** 0"
            count = len(text)
            tiktok_limit = 2200
            shorts_limit = 5000
            reels_limit = 2200
            return f"**Characters:** {count} | **TikTok:** {'✅' if count <= tiktok_limit else '❌'} | **Shorts:** {'✅' if count <= shorts_limit else '❌'} | **Reels:** {'✅' if count <= reels_limit else '❌'}"

        text_input.change(
            fn=update_char_counts, inputs=[text_input], outputs=[char_count_display]
        )

        social_output.change(
            fn=update_char_counts, inputs=[social_output], outputs=[char_count_display]
        )

        def apply_preset(preset_name):
            if preset_name == "TikTok Viral":
                return 1.2, 30, -18, True  # Speed, FPS, Music Vol, CTA
            elif preset_name == "YouTube Shorts High-Energy":
                return 1.4, 60, -15, True
            elif preset_name == "Instagram Reels Aesthetic":
                return 1.0, 30, -25, False
            return 1.0, 30, -22, True

        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[stress_level, export_fps, music_volume, enable_cta],
        )
    return demo


# =============== CLI MODE ===============
def run_cli(args: argparse.Namespace):
    """Run video generation from command line without Gradio UI."""
    generator = TextToVideoGenerator()
    text = args.text
    if not text and args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        print("❌ No text provided. Use --text or --file")
        return
    print(f"🚀 CLI Generation: {len(text)} chars, language={args.language}")
    result = generator.generate_video(
        text=text,
        language=args.language,
        speaker_id=args.voice,
        preferred_media_source=args.media_source or "YouTube",
        enable_background_music=not args.no_music,
        music_selection=args.music or "Random",
        add_intro_slide=not args.no_intro,
        add_call_to_action=not args.no_cta,
        hide_text=args.hide_text,
        aspect_ratio=args.aspect_ratio,
        quality=args.quality,
        enable_crossfade=args.crossfade,
        progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}"),
    )
    if result.get("success"):
        print(f"\n✅ Video: {result['video_path']}")
        print(f"🎵 Audio: {result['audio_path']}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown')}")
        sys.exit(1)


# =============== MAIN ===============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Language Video Generator")
    parser.add_argument(
        "--cli", action="store_true", help="Run in CLI mode (no Gradio UI)"
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Text to convert to video"
    )
    parser.add_argument("--file", type=str, default=None, help="Read text from file")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument(
        "--voice", type=str, default="Standard Voice (Non-Cloned)", help="Voice ID"
    )
    parser.add_argument(
        "--media-source", type=str, default=None, help="Preferred media source"
    )
    parser.add_argument("--music", type=str, default=None, help="Music track name")
    parser.add_argument(
        "--no-music", action="store_true", help="Disable background music"
    )
    parser.add_argument("--no-intro", action="store_true", help="Skip intro slide")
    parser.add_argument("--no-cta", action="store_true", help="Skip CTA slide")
    parser.add_argument("--hide-text", action="store_true", help="Hide text overlay")
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="9:16 Portrait (TikTok/Shorts)",
        help="Aspect ratio preset",
    )
    parser.add_argument(
        "--quality", type=str, default="Medium (Balanced)", help="Quality preset"
    )
    parser.add_argument(
        "--crossfade", action="store_true", help="Enable crossfade transitions"
    )
    args, _ = parser.parse_known_args()

    if args.cli:
        run_cli(args)
        sys.exit(0)

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("❌ python-dotenv not found. Install: pip install python-dotenv")
        exit(1)
    if not MODELS_AVAILABLE:
        print("❌ Missing TTS libraries. Install with:")
        print(
            "pip install TTS speechbrain pydub Pillow num2words torch torchaudio gradio requests spacy python-dotenv"
        )
        exit(1)

    print("\n" + "=" * 80)
    print("🌍 MULTI-LANGUAGE VIDEO GENERATOR")
    print("=" * 80)

    # Run config validation
    cfg = Config()
    cfg_warnings = cfg.validate()
    if cfg_warnings:
        print("\n⚠️  CONFIGURATION WARNINGS:")
        for w in cfg_warnings:
            print(f"  ⚠️  {w}")
    else:
        print("\n✅ Configuration OK")

    print("\n✅ FEATURES:")
    print("  ✓ Multi-language TTS (English, Chinese, Spanish, Hindi, Arabic, Romanian)")
    print("  ✓ Video backgrounds (Pexels, Pixabay, Giphy, SearXNG, Local)")
    print("  ✓ Aspect ratio: 9:16, 16:9, 1:1, 4:5")
    print("  ✓ Quality presets: Low/Medium/High/Ultra")
    print("  ✓ Crossfade transitions between slides")
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

    # Show cache stats
    try:
        import sqlite3

        with sqlite3.connect("generation_cache.db") as conn:
            tts_count = conn.execute("SELECT COUNT(*) FROM tts_cache").fetchone()[0]
            video_count = conn.execute("SELECT COUNT(*) FROM video_logs").fetchone()[0]
            print(f"\n📊 CACHE: {tts_count} TTS entries, {video_count} video logs")
    except Exception:
        pass

    try:
        generator = TextToVideoGenerator()
        print("\n📊 RESOURCES:")
        print(f"  🗣️  Voices: {len(generator.available_voices)}")
        print(f"  🎵 Music tracks: {len(generator.available_music) - 1}")
        print(f"  ⭕ Circle overlays: {generator.available_circles[0]}")

        print("\n💡 SETUP CHECKLIST:")
        print("  1. Set PEXELS_API_KEY in .env file")
        print("  2. Set PIXABAY_API_KEY in .env file")
        print("  3. Set GIPHY_API_KEY in .env file")
        print("  4. Run: ollama serve (for keyword extraction)")
        print("  5. Add circle overlay videos to circle_overlays/ folder")
        print("  6. Add background music to background_music/ folder")
        print("  7. Add logo image to background_images/ folder")

        print("\n🚀 STARTING SERVER...")
        print("   Access at: http://localhost:1603")
        print("   CLI mode: python main.py --cli --text 'Your script here'")
        print("=" * 80 + "\n")

        demo = setup_ui(generator)
        demo.queue()
        demo.launch(
            server_name="0.0.0.0", server_port=1603, share=False, show_error=True
        )
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
