"""FFmpeg rendering and background-selection service."""
import logging, math, os, platform, random, re, subprocess, textwrap, traceback, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from core.ai.stable_diffusion import SDTurboGenerator, SD_AVAILABLE
from core.config import Config
from core.database import DB
from core.media.identity import unique_slide_assets
from core.media.manager import MediaManager
from core.nlp.keyword_extractor import KeywordExtractor
from core.utils.video import get_video_duration, validate_background_asset, validate_slide
from core.visual import VisualProviderFactory
LARAVEL_BG_GRADIENT = ("#0f172a", "#2a1030")
LARAVEL_ACCENT_GRADIENT = ("#7c3aed", "#ec4899")

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
                self.sd_manager = SDTurboGenerator(config=self.config)
                print("[SD-Turbo] Enabled")
            except Exception as e:
                print(f"[SD-Turbo] Init error: {e}")

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
        3. SD-generated image when configured.
        4. Branded gradient when no visual is available.

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

        if self.sd_manager:
            try:
                sd_prompt = sentence or keyword or "abstract cinematic background"
                sd_img = self.sd_manager.generate_image(
                    sd_prompt,
                    keyword=keyword,
                    size=(self.video_width, self.video_height),
                )
                if sd_img and sd_img.exists():
                    print(
                        f"🎨 [SD] Generated fallback image: {sd_img.name} for '{keyword}'"
                    )
                    DB.log_keyword_selection(
                        script_id,
                        sentence_idx,
                        keyword,
                        context_preview=sentence or "",
                        was_used=False,
                    )
                    return sd_img
            except Exception as e:
                print(f"⚠️ [SD] Generation failed: {e}")
        print("💡 [Fallback] No video found; gradient will be used.")
        DB.log_keyword_selection(
            script_id,
            sentence_idx,
            keyword,
            context_preview=sentence or "",
            was_used=False,
        )
        return None

    def _create_text_overlay_png(
        self,
        text: str,
        output_path: Path,
        vw: Optional[int] = None,
        vh: Optional[int] = None,
        gradient_fallback: bool = False,
    ) -> Path:
        text = self._clean_text(text)
        vw = vw or self.video_width
        vh = vh or self.video_height
        img_size = (vw, vh)
        base_font_size = self.config.TEXT_SIZE_CONFIG["font_size"]
        if gradient_fallback:
            base_font_size = int(base_font_size * 1.35)
            print(f"🔠 [Fallback] Enlarging text overlay for gradient-only slide ({base_font_size}px)")
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
        wrapped_text = textwrap.fill(text, width=28 if gradient_fallback else 35)
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
        import logging

        logger = logging.getLogger("split")
        original = text
        text = text.strip()
        if not text:
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        all_sentences: List[str] = []
        for para in paragraphs:
            para = para.strip().replace("\n", " ")
            if not para:
                continue
            tmp = re.sub(r"\bDr\.", "Dr<dot>", para)
            tmp = re.sub(r"\bMr\.", "Mr<dot>", tmp)
            tmp = re.sub(r"\bMrs\.", "Mrs<dot>", tmp)
            tmp = re.sub(r"\bMs\.", "Ms<dot>", tmp)
            tmp = re.sub(r"\b([A-Z])\.", r"\1<dot>", tmp)
            raw_sentences = re.split(r"(?<=[.!?。！？])\s+", tmp)
            sentences = [
                s.replace("<dot>", ".").strip() for s in raw_sentences if s.strip()
            ]
            for i, sentence in enumerate(sentences):
                if not sentence.endswith((".", "!", "?", "。", "！", "？")):
                    sentences[i] = sentence + "."
            all_sentences.extend(sentences)
        logger.info(
            f"[split] paragraphs={len(paragraphs)} raw_sentences={len(all_sentences)}"
        )
        for idx, s in enumerate(all_sentences):
            logger.info(
                f"[split] {idx}: '{s[:80]}' ({len(s)} chars, {len(s.split())} words)"
            )
        return all_sentences

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

            # --- Pre-flight input validation ---
            if audio_path is None or not Path(audio_path).exists():
                print(
                    f"⚠️ [FFmpeg] Slide {slide_num}: audio missing, creating silent fallback (1s)"
                )
                fallback = (
                    self.config.TEMP_DIR
                    / f"silent_{slide_num}_{uuid.uuid4().hex[:6]}.wav"
                )
                try:
                    AudioSegment.silent(duration=1500).export(
                        str(fallback), format="wav"
                    )
                    audio_path = fallback
                except Exception as e:
                    print(f"❌ [FFmpeg] Slide {slide_num} silent fallback failed: {e}")
                    return None

            audio_obj = Path(audio_path)
            if not audio_obj.exists():
                print(
                    f"❌ [FFmpeg] Slide {slide_num} error: audio_path does not exist: {audio_path}"
                )
                return None

            if audio_obj.stat().st_size == 0:
                print(
                    f"❌ [FFmpeg] Slide {slide_num} error: audio_path is empty (0 bytes): {audio_path}"
                )
                return None

            # Force intro video for intro slides
            if is_intro:
                if (
                    self.INTRO_VIDEO_PATH.exists()
                    and self.INTRO_VIDEO_PATH.stat().st_size > 0
                ):
                    video_path = self.INTRO_VIDEO_PATH
                    print(f"🎬 [Intro] Using fixed background: {self.INTRO_VIDEO_PATH}")
                else:
                    print(
                        f"⚠️ [Intro] Fixed background not found or empty: {self.INTRO_VIDEO_PATH}. Falling back to default."
                    )
                    video_path = None

            # Pre-flight check on video_path
            if video_path:
                valid_bg, bg_reason = validate_background_asset(video_path)
                if not valid_bg:
                    print(
                        f"⚠️ [FFmpeg] Slide {slide_num}: Background video validation failed ({bg_reason}) for path: '{video_path}'. Falling back to gradient background."
                    )
                    video_path = None

            # Pre-flight check on circle_video
            if circle_video:
                valid_circle, circle_reason = validate_background_asset(circle_video)
                if not valid_circle:
                    print(
                        f"⚠️ [FFmpeg] Slide {slide_num}: Circle overlay video validation failed ({circle_reason}) for path: '{circle_video}'. Disabling circle overlay."
                    )
                    circle_video = None

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
                    self._create_text_overlay_png(sentence, text_overlay_path, gradient_fallback=not bool(video_path))
            else:
                text_overlay_path = None

            inputs = []
            filter_parts = []
            input_count = 0

            is_split_screen = (
                overlay_shape == "Split Screen"
                and circle_video
                and Path(circle_video).exists()
            )
            if overlay_shape == "Split Screen" and not is_split_screen:
                print(
                    f"⚠️ [SplitScreen] Requested but no circle video available, falling back to normal"
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
                print(
                    f"🎨 [FFmpeg] Slide {slide_num}: No visual available; using branded gradient background"
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
                f"🎬 [FFmpeg] Slide {slide_num}: Composition started. Layers: bg={video_path.name if video_path else 'Branded Gradient'}, text={text_overlay_path.name if text_overlay_path else 'None'} split={is_split_screen}"
            )
            if is_split_screen:
                print(
                    f"🎬 [FFmpeg] SplitScreen filter: top={video_path} bottom={circle_video}"
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
            print(f"🎬 [FFmpeg] Slide {slide_num} executing command:\n{' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            print(f"✅ [FFmpeg] Slide {slide_num}: Successfully composed.")
            if not output_path.exists():
                print(f"❌ [FFmpeg] ERROR: Output not created for slide {slide_num}")
                return None
            file_size = output_path.stat().st_size
            print(
                f"✅ [FFmpeg] Slide {slide_num} created: {file_size / 1024 / 1024:.2f} MB"
            )
            return output_path
        except subprocess.TimeoutExpired as e:
            print(f"❌ [FFmpeg] Slide {slide_num} TIMED OUT after {e.timeout}s.")
            cmd_str = " ".join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
            print(f"❌ [FFmpeg] Slide {slide_num} command that timed out: {cmd_str}")
            return None
        except subprocess.CalledProcessError as e:
            print(
                f"❌ [FFmpeg] Slide {slide_num} failed with return code {e.returncode}"
            )
            cmd_str = " ".join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
            print(f"❌ [FFmpeg] Slide {slide_num} command: {cmd_str}")
            print(f"❌ [FFmpeg] Slide {slide_num} Stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ [FFmpeg] Slide {slide_num} error: {e}")
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

        # Visual provider setup for background media
        visual_provider = VisualProviderFactory.create(
            source_type=getattr(self.config, "VISUAL_SOURCE", "stock"),
            config=self.config,
            sd_generator=self.sd_manager,
            media_manager=self.media_manager,
            background_video_fetcher=self.get_background_video,
        )

        def _fetch_slide_visual_task(slide):
            ctx = {
                "keyword": slide.get("keyword"),
                "sentence": slide.get("sentence"),
                "preferred_source": preferred_media_source,
                "theme": theme,
                "entity": entity,
                "script_id": script_id,
                "sentence_idx": slide["slide_num"],
                "candidate_keywords": (
                    candidate_map.get(slide["slide_num"]) if candidate_map else None
                ),
            }
            asset = visual_provider.get_visual(ctx, target_size=(vw, vh))
            return asset.get_path_obj()

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
                    _fetch_slide_visual_task,
                    slide,
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
                            total_slides * 2,
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

        # Final safety boundary: a generated video cannot contain the same source
        # asset twice, including symlink/path aliases supplied by UI overrides.
        slide_videos, duplicate_count = unique_slide_assets(slide_videos)
        if duplicate_count:
            print(f"🛡️ [Selection] Removed {duplicate_count} duplicate slide asset(s); affected slides use the normal gradient fallback.")

        # --- Stage 2.5: Apply Temporal Coherence Optimization ---
        try:
            from core.visual.temporal_coherence import TemporalCoherenceOptimizer
            tc_opt = TemporalCoherenceOptimizer()
            content_slides = [s for s in slides_data if not s.get("is_intro") and not s.get("is_cta")]
            content_slide_nums = [s["slide_num"] for s in content_slides]
            initial_clips = [slide_videos.get(s_num) for s_num in content_slide_nums]

            # Build candidate pools per slide from local videos & alternative fetched assets
            candidate_clips_map = {}
            local_videos = []
            if self.config and self.config.VIDEOS_DIR.exists():
                for ext in ["*.mp4", "*.mov", "*.avi"]:
                    local_videos.extend(list(self.config.VIDEOS_DIR.rglob(ext)))

            for i, s_num in enumerate(content_slide_nums):
                cands = []
                if slide_videos.get(s_num):
                    cands.append(slide_videos[s_num])
                if local_videos:
                    cands.extend(random.sample(local_videos, min(len(local_videos), 5)))
                candidate_clips_map[i] = list(dict.fromkeys(cands))

            optimized_clips = tc_opt.optimize_slide_clips(
                initial_clips, candidate_clips_per_slide=candidate_clips_map
            )
            for s_num, opt_clip in zip(content_slide_nums, optimized_clips):
                if opt_clip and opt_clip != slide_videos.get(s_num):
                    print(f"🎬 [TemporalCoherence] Updated slide {s_num} background to {getattr(opt_clip, 'name', str(opt_clip))}")
                    slide_videos[s_num] = opt_clip
        except Exception as e:
            print(f"⚠️ [TemporalCoherence] Post-processing skipped: {e}")

        # Temporal optimisation may replace clips, so re-assert the invariant at
        # the last boundary before rendering.
        slide_videos, post_optimization_duplicates = unique_slide_assets(slide_videos)
        if post_optimization_duplicates:
            duplicate_count += post_optimization_duplicates
            print(f"🛡️ [Selection] Removed {post_optimization_duplicates} duplicate asset(s) introduced during post-processing.")

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
                            total_slides * 2,
                            f"Rendering slides ({completed_renders}/{total_slides})...",
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
                    print(f"[Pipeline] Slide {i + 1} validation: OK ({sp.name})")
                else:
                    print(
                        f"❌ [Pipeline] Slide {i + 1} validation failed ({sp.name}): {reason}"
                    )

        if len(valid_slide_paths) != len(slides_data):
            print(
                f"⚠️ [Pipeline] Validation warning: {len(valid_slide_paths)} of {len(slides_data)} slides passed pre-concatenation validation."
            )

        slide_paths = valid_slide_paths

        if not slide_paths:
            raise ValueError("No valid slides available for final composition")

        # Log slide file details before concatenation
        print(
            f"🔍 [Pipeline] Verifying {len(slide_paths)} slide files before concatenation:"
        )
        for idx, sp in enumerate(slide_paths, 1):
            exists = sp.exists()
            size_mb = (sp.stat().st_size / 1024 / 1024) if exists else 0
            print(
                f"   Slide {idx}: {sp.name} | Exists: {exists} | Size: {size_mb:.2f} MB | Path: {sp.absolute()}"
            )

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
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                err_detail = getattr(e, "stderr", str(e))
                print(
                    f"❌ [FFmpeg] Crossfade failed or timed out ({err_detail[:500] if isinstance(err_detail, str) else err_detail}). Falling back to simple concat."
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
                res = subprocess.run(
                    concat_cmd, check=True, capture_output=True, text=True, timeout=300
                )
                print(f"✅ [Pipeline] Final video created: {output_path}")
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr if hasattr(e, "stderr") and e.stderr else str(e)
                print(
                    f"❌ [FFmpeg] Concat failed with returncode {e.returncode}: {err_msg[:500]}"
                )
                raise RuntimeError(f"FFmpeg concat failed: {err_msg[:500]}") from e
            except subprocess.TimeoutExpired as e:
                print(f"❌ [FFmpeg] Concat timed out after 300s")
                raise RuntimeError("FFmpeg concat process timed out") from e
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
