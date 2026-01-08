#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme Video Generator — Multi-language with Kokoro-82M TTS
- Replaced XTTS2 with Kokoro-82M (faster, lighter, better quality)
- Support for multiple languages and voices
- Fixed all MoviePy issues
- Offline model support
"""
import os
import re
import random
import shutil
import uuid
import platform
import requests
import sqlite3
import hashlib
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter
import threading
import torch
import textwrap
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
from dotenv import load_dotenv
import time
import mimetypes
from urllib.parse import urlparse
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip
)

# Core imports
from core.config import Config
from core.database import GenerationDB, DB
from core.nlp.keyword_extractor import KeywordExtractor
from core.ai.stable_diffusion import StableDiffusionManager, SD_AVAILABLE
from core.media.manager import MediaManager
from core.tts.manager import TTSManager
from core.utils.audio import improve_audio_quality, remove_metallic_artifacts
from core.utils.video import get_video_duration, has_audio_stream, is_video_file, sanitize_url_filename

# Handle PIL compatibility
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)
load_dotenv()

# Get shared config
config_instance = Config()
SUPPORTED_LANGUAGES = config_instance.SUPPORTED_LANGUAGES

# DB instance for compatibility
DB_INSTANCE = DB


# ================= KOKORO TTS MANAGER (Legacy Wrapper) =================

# ================= KOKORO TTS MANAGER =================
class KokoroTTSManager:
    """Wrapper to maintain compatibility with meme-main.py logic while using core TTSManager"""
    def __init__(self, config: Config, language: str = "en-US"):
        self.config = config
        self.language = language
        self.tts_manager = TTSManager(config)
        self.current_voice = "af_heart"
        
    def get_available_voices(self) -> List[str]:
        return self.tts_manager.get_available_voices(engine="kokoro")
        
    def set_voice(self, voice_id: str):
        self.current_voice = voice_id
        
    def generate_speech(self, text: str, voice_id: Optional[str] = None) -> Path:
        return self.tts_manager.generate_speech(text, voice_id or self.current_voice, self.language, engine="kokoro")

# ================= CUSTOM MEME API =================
class CustomMemeAPI:
    def __init__(self, base_url: str = "https://trending.izdrail.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.search_cache = {}

    def scrape_memes(self, keyword: str, network: str = "9gag", max_retries: int = 3) -> List[Dict]:
        cache_key = f"{network}_{keyword.lower()}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        url = f"{self.base_url}/api/v1/api/v1/run/scraper"
        payload = {"network": network, "query": keyword}

        for attempt in range(max_retries):
            try:
                r = self.session.post(url, json=payload, timeout=20)
                r.raise_for_status()
                data = r.json()
                items = data.get("scraped_data", {}).get("items", [])
                if items:
                    self.search_cache[cache_key] = items
                    return items
            except Exception:
                time.sleep(2 ** attempt)
        return []

    def download_media(self, media_url: str, output_path: Path) -> bool:
        try:
            response = self.session.get(media_url, stream=True, timeout=30)
            response.raise_for_status()
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(tmp, "wb") as fh:
                for chunk in response.iter_content(8192):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(output_path)
            return output_path.exists()
        except Exception:
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            return False



# ================= OLLAMA HUMOUR REWRITER =================
class OllamaHumourRewriter:
    def __init__(self, model: str = "mistral:7b", url: Optional[str] = None, timeout: int = 20):
        self.model = model
        self.url = url or os.getenv("OLLAMA_API_URL", "https://ai.izdrail.com/api/generate")
        self.timeout = timeout
        self.cache: Dict[str, str] = {}

    def _call_ollama(self, prompt: str, temperature: float = 0.6) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 120}
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                return (data.get("response") or "").strip()
            return str(data).strip()
        except Exception as e:
            print(f"[OllamaHumourRewriter] call error: {e}")
            return ""

    def _make_prompt(self, original_text: str, mode: str) -> str:
        txt = original_text.strip()
        if mode == "sarcastic_news":
            return (
                "You are a sharp-tongued, slightly posh British female news presenter who doesn't suffer fools. "
                "Rewrite the following news snippet in your signature sarcastic, eye-rolling style. "
                "Keep it short, punchy, and dripping with dry humour. One or two sentences max. "
                "Never explain, never apologise.\n"
                f"NEWS: {txt}"
            )
        elif mode == "light":
            return (
                "Rewrite the following text so it is slightly funnier and punchier. "
                "Do NOT list options, do NOT explain. Return only the rewritten text.\n"
                f"TEXT:\n{txt}"
            )
        elif mode == "joke":
            return (
                "Turn the following text into a short, witty joke or punchline. "
                "Keep it to 1 or 2 short sentences. Do NOT explain. Return only the joke.\n"
                f"TEXT:\n{txt}"
            )
        elif mode in ("commentary", "comment"):
            return (
                "Write a short, humorous commentator remark about the following text. "
                "Return only the comment (one sentence). Do NOT restate the original text.\n"
                f"TEXT:\n{txt}"
            )
        else:  # mix
            return (
                "Rewrite the following text into one single funny line by combining: "
                "a slight rewrite, a short punchline, and a witty commentary. "
                "Return exactly one short funny sentence/line. No lists, no explanations.\n"
                f"TEXT:\n{txt}"
            )

    def rewrite(self, text: str, mode: str = "mix") -> str:
        if not text or not text.strip():
            return text

        mode = (mode or "mix").lower()
        cache_key = f"{mode}_{text[:200]}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = self._make_prompt(text, mode)
        temps = {"light": 0.4, "joke": 0.7, "commentary": 0.6, "mix": 0.7, "sarcastic_news": 0.5}
        temp = temps.get(mode, 0.6)

        out = self._call_ollama(prompt, temperature=temp)
        out = (out or "").strip().strip('"').strip()

        if not out:
            out = text

        self.cache[cache_key] = out
        return out

# ================= MEME MEDIA MANAGER =================




# ================= SENTENCE SPLITTER =================
class SentenceSplitter:
    ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "inc", "e.g", "i.e", "vs", "sr", "jr", "rev", "st", "etc", "fig"}

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        t = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        sentences = []
        start = 0

        for m in re.finditer(r'[.!?]+', t):
            end = m.end()
            piece = t[start:end].strip()
            last_token = re.findall(r'([A-Za-z0-9]+)$', piece)
            last_token = last_token[0].lower() if last_token else ""

            if last_token and last_token.rstrip('.').lower() in self.ABBREVIATIONS:
                continue

            next_char_idx = end
            if next_char_idx >= len(t):
                candidate = t[start:end].strip()
                if candidate:
                    sentences.append(candidate if candidate.endswith(('.', '!', '?')) else candidate + '.')
                    start = end
                continue

            rest = t[next_char_idx:]
            m2 = re.match(r"\s+([A-Z0-9\"'])", rest)
            if m2:
                candidate = t[start:end].strip()
                if candidate:
                    sentences.append(candidate if candidate.endswith(('.', '!', '?')) else candidate + '.')
                    start = end
            else:
                continue

        if start < len(t):
            rem = t[start:].strip()
            if rem:
                sentences.append(rem if rem.endswith(('.', '!', '?')) else rem + '.')

        return [s.strip() for s in sentences if s.strip()]

# ================= MOVIEPY VIDEO GENERATOR =================
class MoviePyVideoGenerator:
    def __init__(self, config: Config, custom_api: CustomMemeAPI):
        self.config = config
        self.custom_api = custom_api
        self.keyword_extractor = KeywordExtractor()
        self.font_path = self._discover_font()
        self.media_manager = MediaManager(config)
        self.font_cache = {}
        self.session = custom_api.session
        self.media_cache_dir = self.config.TEMP_DIR / "media_cache"
        self.media_cache_dir.mkdir(parents=True, exist_ok=True)

    def download_media(self, url: Optional[str]) -> Optional[Path]:
        """Download media from URL to cache"""
        if not url:
            return None
        local = self.media_cache_dir / sanitize_url_filename(url)
        if local.exists():
            return local
        try:
            with self.session.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                tmp = local.with_suffix(local.suffix + ".tmp")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(local)
            return local
        except Exception as e:
            print(f"[MoviePy] download failed for {url}: {e}")
            return None

    def prepare_media_for_item(self, media_info: Any) -> Tuple[Optional[Path], bool]:
        """Prepare media from info (URL or dict)"""
        if not media_info:
            return (None, False)
        url = media_info.get("url") if isinstance(media_info, dict) else str(media_info)
        if not url:
            return (None, False)
        local = self.download_media(url)
        if not local:
            return (None, False)
        return (local, is_video_file(local))

    def get_font(self, font_path: str, font_size: int):
        """Cache and return font objects"""
        cache_key = f"{font_path}_{font_size}"
        if cache_key not in self.font_cache:
            try:
                self.font_cache[cache_key] = ImageFont.truetype(font_path, font_size)
            except Exception:
                self.font_cache[cache_key] = ImageFont.load_default()
        return self.font_cache[cache_key]

    def _discover_font(self) -> str:
        system = platform.system()
        if system == "Windows":
            path = Path("C:/Windows/Fonts/arial.ttf")
        elif system == "Darwin":
            path = Path("/System/Library/Fonts/Arial.ttf")
        else:
            path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

        return str(path) if path.exists() else "DejaVuSans"

    def split_into_sentences(self, text: str) -> List[str]:
        return SentenceSplitter().split(text)

    def _create_text_overlay_pil(self, text: str, duration: float) -> ImageClip:
        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = self.get_font(self.font_path, self.config.TEXT_SIZE_CONFIG['font_size'])
        except Exception:
            font = ImageFont.load_default()

        wrapped = textwrap.fill(text, width=35)
        bbox = draw.textbbox((0, 0), wrapped, font=font)
        x = (self.config.VIDEO_WIDTH - (bbox[2] - bbox[0])) // 2
        y = self.config.VIDEO_HEIGHT - (bbox[3] - bbox[1]) - self.config.TEXT_SIZE_CONFIG['bottom_margin']

        # Draw text with stroke
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx or dy:
                    draw.text((x+dx, y+dy), wrapped, font=font, fill='black')
        draw.text((x, y), wrapped, font=font, fill='white')

        return ImageClip(np.array(img)).set_duration(duration)

    def _mix_audio_tracks(self, original_path: Optional[Path], tts_path: Path, output_path: Path, slide_duration: float):
        tts_audio = AudioSegment.from_file(str(tts_path))

        if original_path:
            try:
                original_audio = AudioSegment.from_file(str(original_path))
                if len(original_audio) > slide_duration * 1000:
                    original_audio = original_audio[:int(slide_duration * 1000)]
                if len(original_audio) < slide_duration * 1000:
                    loops_needed = int((slide_duration * 1000) / len(original_audio)) + 1
                    original_audio = original_audio * loops_needed
                    original_audio = original_audio[:int(slide_duration * 1000)]
                original_audio = original_audio - 10
            except Exception as e:
                print(f"Could not process original audio: {e}")
                original_audio = None
        else:
            original_audio = None

        if len(tts_audio) < slide_duration * 1000:
            silence_duration = int(slide_duration * 1000) - len(tts_audio)
            tts_audio = tts_audio + AudioSegment.silent(duration=silence_duration)
        elif len(tts_audio) > slide_duration * 1000:
            tts_audio = tts_audio[:int(slide_duration * 1000)]

        if original_audio:
            mixed_audio = original_audio.overlay(tts_audio)
        else:
            mixed_audio = tts_audio

        if len(mixed_audio) < slide_duration * 1000:
            silence = AudioSegment.silent(duration=int(slide_duration * 1000) - len(mixed_audio))
            mixed_audio += silence
        elif len(mixed_audio) > slide_duration * 1000:
            mixed_audio = mixed_audio[:int(slide_duration * 1000)]

        mixed_audio.export(str(output_path), format="wav")

    def _create_image_clip(self, image_path: Path, duration: float) -> ImageClip:
        img = Image.open(image_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (0, 0, 0))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background

        clip = ImageClip(np.array(img)).set_duration(duration)
        clip = clip.resize(width=self.config.VIDEO_WIDTH)
        clip = clip.on_color(size=self.config.VIDEO_SIZE, color=(0,0,0), pos=('center','center'))

        def zoom_effect(t):
            return 1 + 0.05 * (t / duration)

        clip = clip.resize(zoom_effect)
        return clip

    def _create_slide_moviepy(self,
                            sentence: str,
                            audio_path: Path,
                            media_path: Optional[Path],
                            media_is_video: bool,
                            output_path: Path) -> Tuple[bool, bool]:
        try:
            tts_audio = AudioSegment.from_file(str(audio_path))
            tts_duration = len(tts_audio) / 1000.0

            if media_path and media_path.exists() and media_is_video:
                video_duration = get_video_duration(media_path)
                slide_duration = video_duration
            else:
                slide_duration = max(tts_duration, self.config.MIN_IMAGE_DURATION)

            original_audio_path = None
            has_original_audio = False

            if media_path and media_path.exists() and media_is_video and has_audio_stream(media_path):
                has_original_audio = True
                try:
                    clip = VideoFileClip(str(media_path), audio=True)
                    if clip.audio:
                        temp_audio_path = self.config.get_temp_audio_file("original")
                        clip.audio.subclip(0, min(slide_duration, clip.audio.duration)).write_audiofile(
                            str(temp_audio_path), codec="pcm_s16le", logger=None, fps=22050
                        )
                        original_audio_path = temp_audio_path
                    clip.close()
                except Exception as e:
                    print(f"[MoviePy] Could not extract original audio: {e}")

            mixed_audio_path = self.config.get_temp_audio_file("mixed")
            self._mix_audio_tracks(original_audio_path, audio_path, mixed_audio_path, slide_duration)

            if media_path and media_path.exists():
                if media_is_video:
                    bg_clip = VideoFileClip(str(media_path), audio=False)
                    bg_clip = bg_clip.resize(width=self.config.VIDEO_WIDTH)
                    bg_clip = bg_clip.on_color(size=self.config.VIDEO_SIZE, color=(0,0,0), pos=('center','center'))
                    if bg_clip.duration > slide_duration:
                        bg_clip = bg_clip.subclip(0, slide_duration)
                else:
                    bg_clip = self._create_image_clip(media_path, slide_duration)
            else:
                bg_clip = ColorClip(self.config.VIDEO_SIZE, color=(0,0,0), duration=slide_duration)

            text_clip = self._create_text_overlay_pil(sentence, slide_duration)

            final_clip = CompositeVideoClip([bg_clip, text_clip]).set_duration(slide_duration)
            final_clip = final_clip.set_audio(AudioFileClip(str(mixed_audio_path)))

            # Write the video file without the progress_bar parameter
            final_clip.write_videofile(
                str(output_path),
                fps=30,
                audio_codec='aac',
                preset=self.config.VIDEO_PRESET,
                logger=None
            )

            # Ensure the file is properly saved before proceeding
            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file {output_path} may not have been properly written")

            # Create a backup of the slide immediately after creation
            backup_path = self.config.get_backup_path(output_path)
            try:
                shutil.copy2(output_path, backup_path)
                print(f"Backup of slide created at: {backup_path}")
            except Exception as e:
                print(f"Could not create backup of slide: {e}")

            bg_clip.close()
            text_clip.close()
            final_clip.close()

            if original_audio_path and original_audio_path.exists():
                original_audio_path.unlink(missing_ok=True)
            mixed_audio_path.unlink(missing_ok=True)

            return True, has_original_audio
        except Exception as e:
            print(f"[MoviePy] Slide error: {e}")
            import traceback
            traceback.print_exc()
            return False, False

    def _mix_background_music(self, video_path: Path, has_original_audio: bool) -> Path:
        music_files = list(self.config.MUSIC_DIR.glob("*.mp3")) + list(self.config.MUSIC_DIR.glob("*.wav"))
        if not music_files:
            print("No background music files found, skipping music mix")
            return video_path

        music_path = random.choice(music_files)
        video_clip = VideoFileClip(str(video_path))
        if not video_clip.audio:
            video_clip.close()
            print("Video has no audio track, skipping music mix")
            return video_path

        try:
            temp_voice = self.config.get_temp_audio_file("voice")
            video_clip.audio.write_audiofile(str(temp_voice), codec="pcm_s16le", logger=None, fps=22050)

            voice_seg = AudioSegment.from_file(str(temp_voice))
            music = AudioSegment.from_file(str(music_path))

            base_music_volume = -15
            if has_original_audio:
                base_music_volume = -25

            loops = int(len(voice_seg) / len(music)) + 2
            looped = music
            for _ in range(loops - 1):
                looped = looped.append(music, crossfade=0)
            looped = looped[:len(voice_seg)]
            looped = looped + base_music_volume

            mixed = voice_seg.overlay(looped)
            temp_mixed = self.config.get_temp_audio_file("final_mixed")
            mixed.export(str(temp_mixed), format="wav")

            output_path = video_path.with_stem(video_path.stem + "_music")

            # Add music to video
            video_with_music = VideoFileClip(str(video_path)).set_audio(AudioFileClip(str(temp_mixed)))

            # Write the final video without the progress_bar parameter
            video_with_music.write_videofile(
                str(output_path),
                fps=30,
                audio_codec='aac',
                preset=self.config.VIDEO_PRESET,
                logger=None,
                threads=4
            )

            # Ensure file is properly saved
            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file with music {output_path} may not have been properly written")
            else:
                # Create backup immediately after creation
                backup_path = self.config.get_backup_path(output_path)
                try:
                    shutil.copy2(output_path, backup_path)
                    print(f"Backup of final video created at: {backup_path}")
                except Exception as e:
                    print(f"Could not create backup of final video: {e}")

            video_with_music.close()
            return output_path
        finally:
            # Clean up
            video_clip.close()
            for temp_file in [temp_voice, temp_mixed]:
                try:
                    if temp_file and temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    print(f"Could not delete temp file {temp_file}: {e}")

            try:
                if video_path != output_path and video_path.exists():
                    video_path.unlink()
            except Exception as e:
                print(f"Could not delete original video {video_path}: {e}")

    def create_final_video(self,
                         sentences: List[str],
                         audio_paths: List[Path],
                         keywords: List[Optional[str]],
                         media_infos: List[Tuple[Optional[Path], bool]]) -> Path:
        slide_paths = []
        has_video_with_audio = False

        # Process slides sequentially to avoid duplication issues
        for i, (sent, audio, kw, media_info) in enumerate(zip(sentences, audio_paths, keywords, media_infos)):
            media_path, media_is_video = media_info
            out = self.config.TEMP_DIR / f"slide_{i}_{uuid.uuid4().hex[:8]}.mp4"

            success, has_audio = self._create_slide_moviepy(sent, audio, media_path, media_is_video, out)
            if success:
                slide_paths.append(out)
                has_video_with_audio |= has_audio
            else:
                # Try fallback with just the sentence and audio, no background
                fallback_out = self.config.TEMP_DIR / f"slide_{i}_fallback_{uuid.uuid4().hex[:8]}.mp4"
                fallback_success, _ = self._create_slide_moviepy(sent, audio, None, False, fallback_out)
                if fallback_success:
                    slide_paths.append(fallback_out)
                    print(f"Used fallback slide for sentence {i} due to error with original media")
                else:
                    print(f"WARNING: Could not create slide for sentence {i}, even with fallback")

        if not slide_paths:
            raise RuntimeError("No slides created successfully.")

        clips = []
        for p in slide_paths:
            try:
                clip = VideoFileClip(str(p))
                clips.append(clip)
            except Exception as e:
                print(f"Error loading slide {p}: {e}")
                continue

        if not clips:
            raise RuntimeError("No slides could be loaded for concatenation")

        final_clip = concatenate_videoclips(clips, method="compose")
        output = self.config.OUTPUT_DIR / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        # Write the final concatenated video without the progress_bar parameter
        final_clip.write_videofile(
            str(output),
            fps=30,
            audio_codec='aac',
            preset=self.config.VIDEO_PRESET,
            logger=None,
            threads=4
        )

        # Verify the file was created
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Final video file was not properly created at {output}")

        # Create a backup immediately
        backup_path = self.config.get_backup_path(output)
        try:
            shutil.copy2(output, backup_path)
            print(f"Backup of concatenated video created at: {backup_path}")
        except Exception as e:
            print(f"Could not create backup of concatenated video: {e}")

        output = self._mix_background_music(output, has_video_with_audio)

        # Clean up
        for clip in clips:
            clip.close()
        final_clip.close()

        for p in slide_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                print(f"Could not delete slide file {p}: {e}")

        return output

# ================= MAIN GENERATOR =================
class TextToVideoGenerator:
    def __init__(self, language: str = "en-US"):
        self.config = config_instance
        self.language = language
        self.tts = KokoroTTSManager(self.config, language)
        self.custom_api = CustomMemeAPI()
        self.video_gen = MoviePyVideoGenerator(self.config, self.custom_api)
        self.keyword_extractor = KeywordExtractor()
        self.rewriter = OllamaHumourRewriter()
        self.media_manager = MediaManager(self.config)
        self.voices = self.tts.get_available_voices()

    def set_language(self, language: str):
        """Switch to a different language"""
        if language != self.language:
            self.language = language
            self.tts = KokoroTTSManager(self.config, language)
            self.voices = self.tts.get_available_voices()

    def generate_from_keyword(self,
                            keyword: str,
                            network: str = "9gag",
                            max_items: int = 5,
                            humour_mode: str = "light",
                            speaker: str = None) -> Dict:
        start_time = time.time()

        if not keyword or not keyword.strip():
            return {"error": "Keyword empty", "success": False}

        speaker = speaker or self.voices[0]
        self.tts.set_voice(speaker)

        try:
            items = self.custom_api.scrape_memes(keyword, network)
            if not items:
                return {"error": f"No items scraped for '{keyword}'", "success": False}

            items = items[:max_items]

            per_sentence_texts = []
            per_sentence_media_infos = []
            per_sentence_keywords = []

            for item in items:
                orig = item.get("original_item", {})
                orig_text = (orig.get("text") or "").strip()
                if not orig_text:
                    continue

                try:
                    rewritten = self.rewriter.rewrite(orig_text, mode=humour_mode)
                except Exception:
                    rewritten = orig_text

                media_list = orig.get("media", [])
                media_url = None
                if media_list:
                    media = media_list[0]
                    media_url = media.get("url")

                local_media, is_video = self.video_gen.prepare_media_for_item(media_url)

                sentences = self.video_gen.split_into_sentences(rewritten)
                if not sentences:
                    continue

                for s in sentences:
                    per_sentence_texts.append(s)
                    per_sentence_media_infos.append((local_media, is_video))
                    per_sentence_keywords.append(self.keyword_extractor.get_best_unique_keyword(s))

            if not per_sentence_texts:
                return {"error": "No usable sentences from scraped items", "success": False}

            audios = []
            try:
                for s in per_sentence_texts:
                    audio = self.tts.generate_speech(s, speaker)
                    audios.append(audio)

                video_temp = self.video_gen.create_final_video(
                    per_sentence_texts,
                    audios,
                    per_sentence_keywords,
                    per_sentence_media_infos
                )

                # Ensure the video file exists and create a backup
                if not video_temp.exists() or video_temp.stat().st_size == 0:
                    raise RuntimeError(f"Generated video file is invalid: {video_temp}")

                # Create a backup in case the database operation fails
                backup_path = self.config.get_backup_path(video_temp)
                try:
                    shutil.copy2(video_temp, backup_path)
                    print(f"Emergency backup created at: {backup_path}")
                except Exception as e:
                    print(f"Could not create emergency backup: {e}")

                processing_time = time.time() - start_time

                result = {
                    "success": True,
                    "video_path": str(video_temp),
                    "output_directory": str(video_temp.parent),
                    "sentence_count": len(per_sentence_texts),
                    "video_backgrounds": "item media",
                    "language": self.language,
                    "voice": speaker,
                    "processing_time": processing_time
                }

                # Save to database with the processing time
                DB.save_video({
                    "keyword": keyword,
                    "language": self.language,
                    "network": network,
                    "max_items": max_items,
                    "humour_mode": humour_mode,
                    "speaker": speaker
                }, result, processing_time)

                print(f"Video generation completed successfully in {processing_time:.2f} seconds")
                print(f"Final video saved to: {video_temp}")
                print(f"Backup available at: {backup_path}")

                return result

            except Exception as e:
                # In case of error, ensure we have a reference to the backup
                error_msg = str(e)
                print(f"Error during video generation: {error_msg}")

                # Try to find any partial backups
                backup_files = list(self.config.BACKUP_OUTPUT_DIR.glob("backup_*"))
                if backup_files:
                    latest_backup = sorted(backup_files, key=lambda x: x.stat().st_mtime)[-1]
                    error_msg += f"\n\nA partial backup may be available at: {latest_backup}"

                return {"error": error_msg, "success": False}

        except Exception as e:
            return {"error": str(e), "success": False}

        finally:
            # Clean up audio files
            for a in audios:
                try:
                    if a and a.exists():
                        a.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Could not delete audio file {a}: {e}")

# ================= GRADIO UI =================
def setup_ui():
    generator = TextToVideoGenerator()
    with gr.Blocks(title="Meme Video Generator (Multi-Language Kokoro)") as demo:
        gr.Markdown("# 🎭 Meme Video Generator — Kokoro-82M Multi-Language")
        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=["en-US", "en-GB", "ja", "zh", "es", "fr", "hi", "it", "pt-BR"],
                    value="en-US",
                    label="Language"
                )
                keyword = gr.Textbox(label="Search keyword", lines=1, placeholder="e.g. trending, cats")
                network = gr.Dropdown(choices=["9gag", "reddit", "instagram"], value="9gag", label="Network")
                max_items = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Max items")
                humour = gr.Dropdown(
                    choices=["light", "joke", "commentary", "mix", "sarcastic_news"],
                    value="sarcastic_news",
                    label="Humour mode"
                )
                voice = gr.Dropdown(choices=generator.voices, value=generator.voices[0], label="Voice")
                btn = gr.Button("🚀 Generate Video")
                status = gr.Markdown()
            with gr.Column(scale=1):
                video = gr.Video(label="Generated Video")
                preview = gr.Textbox(label="Preview", lines=12, interactive=False)

        def update_voices(lang):
            generator.set_language(lang)
            return gr.Dropdown(choices=generator.voices, value=generator.voices[0])

        def gen_from_keyword(lang, k, net, mx, hum, v):
            generator.set_language(lang)
            if not k or not k.strip():
                return None, "🚫 Keyword empty", ""

            try:
                items = generator.custom_api.scrape_memes(k, net)
            except Exception as e:
                return None, f"⚠️ Scraper error: {e}", ""

            if not items:
                return None, f"⚠️ No items for '{k}'", ""

            preview_lines = []
            for i, it in enumerate(items[:mx]):
                orig = it.get("original_item", {})
                txt = (orig.get("text") or "").strip()
                try:
                    rewritten = generator.rewriter.rewrite(txt, mode=hum)
                except Exception:
                    rewritten = txt
                preview_lines.append(f"ITEM {i+1}:\nREWRITTEN: {rewritten}\n---")

            preview_text = "\n".join(preview_lines)

            res = generator.generate_from_keyword(k, network=net, max_items=mx, humour_mode=hum, speaker=v)

            if res.get("success"):
                return res["video_path"], f"✅ Done! {res['sentence_count']} slides ({lang})", preview_text

            return None, f"⚠️ Error: {res.get('error', 'unknown')}", preview_text

        language.change(update_voices, inputs=language, outputs=voice)
        btn.click(
            fn=gen_from_keyword,
            inputs=[language, keyword, network, max_items, humour, voice],
            outputs=[video, status, preview]
        )

    return demo

# ================= Entrypoint =================
if __name__ == "__main__":
    print("Install: pip install kokoro moviepy pydub numpy pillow gradio requests python-dotenv num2words scipy torch")
    print("🎬 Starting Meme Video Generator — Kokoro-82M Multi-Language")

    try:
        demo = setup_ui()
        demo.launch(
            server_port=1604,
            server_name="0.0.0.0",
            show_error=True
        )
    except Exception as e:
        print(f"Application failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)