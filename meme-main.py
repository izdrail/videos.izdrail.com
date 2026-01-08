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

# Handle PIL compatibility
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)
load_dotenv()

# ================= DATABASE =================
class GenerationDB:
    def __init__(self, db_path: Path = Path("generation_cache.db")):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Check if old tts_cache table exists with old schema
            try:
                cursor = conn.execute("PRAGMA table_info(tts_cache)")
                columns = [row[1] for row in cursor.fetchall()]
                if "speaker_id" in columns and "voice_id" not in columns:
                    print("[DB] Migrating old tts_cache schema...")
                    # Create new table with correct schema
                    conn.execute("""CREATE TABLE tts_cache_new (
                        text_hash TEXT PRIMARY KEY,
                        voice_id TEXT,
                        language TEXT,
                        audio_path TEXT,
                        created_at TEXT
                    )""")
                    # Copy data
                    conn.execute("""
                        INSERT INTO tts_cache_new (text_hash, voice_id, language, audio_path, created_at)
                        SELECT text_hash, speaker_id, 'en-US', audio_path, created_at FROM tts_cache
                    """)
                    # Drop old table and rename
                    conn.execute("DROP TABLE tts_cache")
                    conn.execute("ALTER TABLE tts_cache_new RENAME TO tts_cache")
                    conn.commit()
                    print("[DB] Migration complete")
            except Exception as e:
                print(f"[DB] Migration check: {e}")
                # Create fresh table
                pass

            # Ensure tts_cache exists with correct schema
            conn.execute("""CREATE TABLE IF NOT EXISTS tts_cache (
                text_hash TEXT PRIMARY KEY,
                voice_id TEXT,
                language TEXT,
                audio_path TEXT,
                created_at TEXT
            )""")

            # Create or update video_logs table
            cursor = conn.execute("PRAGMA table_info(video_logs)")
            columns = [row[1] for row in cursor.fetchall()]

            # Create table if it doesn't exist
            if not columns:
                conn.execute("""CREATE TABLE video_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT UNIQUE,
                    video_path TEXT,
                    audio_path TEXT,
                    output_dir TEXT,
                    sentence_count INTEGER,
                    created_at TEXT,
                    processing_time REAL,
                    system_stats TEXT
                )""")
            else:
                # Add missing columns if they don't exist
                if "processing_time" not in columns:
                    print("[DB] Adding processing_time column to video_logs")
                    conn.execute("ALTER TABLE video_logs ADD COLUMN processing_time REAL")

                if "system_stats" not in columns:
                    print("[DB] Adding system_stats column to video_logs")
                    conn.execute("ALTER TABLE video_logs ADD COLUMN system_stats TEXT")

            conn.commit()

    def get_cached_tts(self, text: str, voice_id: str, language: str) -> Optional[Path]:
        text_hash = hashlib.sha256(f"{text}_{voice_id}_{language}".encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT audio_path FROM tts_cache WHERE text_hash = ?",
                (text_hash,)
            ).fetchone()
            if row and Path(row[0]).exists():
                return Path(row[0])
        return None

    def save_tts(self, text: str, voice_id: str, language: str, audio_path: Path):
        text_hash = hashlib.sha256(f"{text}_{voice_id}_{language}".encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tts_cache (text_hash, voice_id, language, audio_path, created_at) VALUES (?, ?, ?, ?, ?)",
                (text_hash, voice_id, language, str(audio_path), datetime.now().isoformat())
            )

    def save_video(self, input_params: Dict, result: Dict, processing_time: float):
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()))
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            system_stats = str({
                "memory_used_mb": 0,  # Placeholder - could be enhanced with actual memory usage
                "timestamp": datetime.now().isoformat()
            })
            conn.execute("""
                INSERT OR REPLACE INTO video_logs
                (input_hash, video_path, audio_path, output_dir, sentence_count, created_at, processing_time, system_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                input_hash,
                result.get("video_path"),
                result.get("audio_path"),
                result.get("output_directory"),
                result.get("sentence_count", 0),
                datetime.now().isoformat(),
                processing_time,
                system_stats
            ))

DB = GenerationDB()

# ================= KOKORO TTS MANAGER =================
class KokoroTTSManager:
    """Text-to-Speech using Kokoro-82M with multi-language support"""
    # Voice mappings by language
    VOICE_MAP = {
        "en-US": ["af_heart", "af_bella", "af_nicole", "am_michael", "am_liam", "am_puck"],
        "en-GB": ["bf_emma", "bf_isabella", "bm_fable", "bm_george"],
        "ja": ["jf_alpha", "jf_gongitsune", "jm_kumo"],
        "zh": ["zf_xiaobei", "zf_xiaoni", "zm_yunxi", "zm_yunyang"],
        "es": ["ef_dora", "em_alex"],
        "fr": ["ff_siwis"],
        "hi": ["hf_alpha", "hm_omega"],
        "it": ["if_sara", "im_nicola"],
        "pt-BR": ["pf_dora", "pm_alex"],
    }

    def __init__(self, config: 'Config', language: str = "en-US"):
        self.config = config
        self.language = language
        self.model = None
        self.generate_fn = None
        self.available = False
        self.current_voice = None
        self._load_model()

    def _load_model(self):
        """Load Kokoro KPipeline"""
        try:
            print(f"[Kokoro] Loading KPipeline for {self.language}...")
            os.environ['HF_HUB_OFFLINE'] = '1'
            from kokoro import KPipeline

            # Map language to lang code
            lang_map = {
                "en-US": "a",
                "en-GB": "b",
                "ja": "j",
                "zh": "z",
                "es": "e",
                "fr": "f",
                "hi": "h",
                "it": "i",
                "pt-BR": "p"
            }
            lang_code = lang_map.get(self.language, "a")
            self.model = KPipeline(lang_code=lang_code)
            self.available = True
            print(f"[Kokoro] KPipeline loaded for language {self.language} ({lang_code})")
        except Exception as e:
            print(f"[Kokoro] Load error: {e}")
            import traceback
            traceback.print_exc()
            self.available = False

    def get_available_voices(self) -> List[str]:
        """Get available voices for current language"""
        return self.VOICE_MAP.get(self.language, self.VOICE_MAP["en-US"])

    def set_voice(self, voice_id: str):
        """Set the voice for TTS"""
        if voice_id in self.get_available_voices():
            self.current_voice = voice_id
        else:
            voices = self.get_available_voices()
            self.current_voice = voices[0] if voices else "af_heart"

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text for TTS"""
        # Convert numbers to words
        text = re.sub(r'\d+', lambda m: num2words(int(m.group())), text)
        # Clean whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def improve_audio_quality(self, audio_path: Path) -> Path:
        """Enhance audio quality with normalization and filtering"""
        try:
            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_frame_rate(22050) - 6
            audio = audio.high_pass_filter(100)
            audio = low_pass_filter(audio, 4500)
            try:
                audio = audio.compress_dynamic_range(threshold=-30.0, ratio=1.8)
            except Exception:
                pass
            audio = normalize(audio, headroom=0.3)
            improved = self.config.TEMP_DIR / f"improved_{audio_path.name}"
            audio.export(str(improved), format="wav")
            return improved
        except Exception:
            return audio_path

    def generate_speech(self, text: str, voice_id: Optional[str] = None) -> Path:
        """Generate speech using Kokoro KPipeline"""
        if not voice_id:
            voice_id = self.current_voice or self.get_available_voices()[0]

        # Check cache
        cached = DB.get_cached_tts(text, voice_id, self.language)
        if cached:
            return cached

        processed = self.preprocess_text(text)
        temp_path = self.config.TEMP_DIR / f"tts_{uuid.uuid4().hex}.wav"

        try:
            if not self.available or self.model is None:
                raise ValueError("Kokoro KPipeline not loaded")
            import scipy.io.wavfile as wavfile

            # Generate speech - returns generator of (graphemes, phonemes, audio)
            # Collect all audio chunks
            all_audio = []
            for i, (gs, ps, audio) in enumerate(self.model(processed, voice=voice_id, speed=1.0)):
                all_audio.append(audio)

            if not all_audio:
                raise RuntimeError("No audio generated")

            # Concatenate all audio chunks
            full_audio = np.concatenate(all_audio)

            # Normalize
            full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-7)
            full_audio = (full_audio * 32767).astype(np.int16)

            # Save to file (24000 Hz is Kokoro's sample rate)
            wavfile.write(str(temp_path), 24000, full_audio)

            # Improve quality
            improved = self.improve_audio_quality(temp_path)
            if improved != temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            DB.save_tts(text, voice_id, self.language, improved)
            return improved
        except Exception as e:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[Kokoro] Speech generation error: {e}")
            import traceback
            traceback.print_exc()
            raise e

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

# ================= KEYWORD EXTRACTOR =================
class KeywordExtractor:
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        words = re.findall(r'\b\w{3,}\b', text.lower())
        freq = Counter(words)
        return [w for w, _ in freq.most_common(top_n)]

    def get_best_unique_keyword(self, text: str) -> Optional[str]:
        kws = self.extract_keywords(text, 1)
        return kws[0] if kws else None

# ================= OLLAMA HUMOUR REWRITER =================
class OllamaHumourRewriter:
    def __init__(self, model: str = "gemma3:270m", url: Optional[str] = None, timeout: int = 20):
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
class MemeMediaManager:
    def __init__(self, config: 'Config', session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.media_cache_dir = self.config.TEMP_DIR / "media_cache"
        self.media_cache_dir.mkdir(parents=True, exist_ok=True)
        self.font_cache = {}

    def _sanitize_url_filename(self, url: str) -> str:
        parsed = urlparse(url)
        base = Path(parsed.path).name or hashlib.sha256(url.encode()).hexdigest()[:12]
        ext = Path(base).suffix
        if not ext:
            mime, _ = mimetypes.guess_type(url)
            ext = {
                "video/mp4": ".mp4",
                "video/webm": ".webm",
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif"
            }.get(mime, ".bin")
        name = hashlib.sha256(url.encode()).hexdigest()[:20]
        return f"{name}{ext}"

    def _local_path_for_url(self, url: str) -> Path:
        fname = self._sanitize_url_filename(url)
        return self.media_cache_dir / fname

    def download_media(self, url: Optional[str]) -> Optional[Path]:
        if not url:
            return None

        local = self._local_path_for_url(url)
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
            try:
                if local.exists():
                    local.unlink()
            except Exception:
                pass
            print(f"[MemeMediaManager] download failed for {url}: {e}")
            return None

    def detect_is_video(self, local_path: Path) -> bool:
        if not local_path or not local_path.exists():
            return False

        ext = local_path.suffix.lower()
        if ext in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
            return True
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
            return False

        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "v:0", str(local_path)],
                capture_output=True, text=True, timeout=8
            )
            return bool(probe.stdout.strip())
        except Exception:
            return False

    def get_video_duration(self, video_path: Path) -> float:
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ], capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            return 4.0

    def has_audio_stream(self, video_path: Path) -> bool:
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ], capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except Exception:
            return False

    def prepare_media_for_item(self, media_info: Optional[Dict]) -> Tuple[Optional[Path], bool]:
        if not media_info:
            return (None, False)

        url = media_info.get("url") if isinstance(media_info, dict) else str(media_info)
        if not url:
            return (None, False)

        local = self.download_media(url)
        if not local:
            return (None, False)

        return (local, self.detect_is_video(local))

    def get_font(self, font_path: str, font_size: int):
        """Cache and return font objects to avoid reloading"""
        cache_key = f"{font_path}_{font_size}"
        if cache_key not in self.font_cache:
            try:
                self.font_cache[cache_key] = ImageFont.truetype(font_path, font_size)
            except Exception:
                self.font_cache[cache_key] = ImageFont.load_default()
        return self.font_cache[cache_key]

# ================= CONFIG =================
class Config:
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent.resolve()
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.VIDEOS_DIR = self.ROOT_DIR / "background_videos"
        self.MUSIC_DIR = self.ROOT_DIR / "background_music"
        self.IMAGES_DIR = self.ROOT_DIR / "background_images"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"

        for d in [self.VOICE_SAMPLES_DIR, self.VIDEOS_DIR, self.MUSIC_DIR,
                  self.IMAGES_DIR, self.TEMP_DIR, self.OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self.DEVICE = "cpu"
        self.VIDEO_WIDTH = 1080
        self.VIDEO_HEIGHT = 1920
        self.VIDEO_SIZE = (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        self.TEXT_SIZE_CONFIG = {'font_size': 50, 'line_spacing': 1.2, 'max_width': 900, 'bottom_margin': 150}
        self.VIDEO_PRESET = 'ultrafast'
        self.VIDEO_CRF = 28
        self.MAX_PARALLEL_SLIDES = 3
        self.TEMP_AUDIO_DIR = self.TEMP_DIR / "audio_cache"
        self.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.MIN_IMAGE_DURATION = 10.0
        self.BACKUP_OUTPUT_DIR = self.ROOT_DIR / "backup_output"
        self.BACKUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_temp_audio_file(self, prefix: str = "audio") -> Path:
        return self.TEMP_AUDIO_DIR / f"{prefix}_{uuid.uuid4().hex}.wav"

    def get_backup_path(self, original_path: Path) -> Path:
        """Get a backup path for a file to ensure it's preserved"""
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_path.name}"
        return self.BACKUP_OUTPUT_DIR / backup_name

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
        self.media_manager = MemeMediaManager(config, self.custom_api.session)

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
            font = self.media_manager.get_font(self.font_path, self.config.TEXT_SIZE_CONFIG['font_size'])
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
                video_duration = self.media_manager.get_video_duration(media_path)
                slide_duration = video_duration
            else:
                slide_duration = max(tts_duration, self.config.MIN_IMAGE_DURATION)

            original_audio_path = None
            has_original_audio = False

            if media_path and media_path.exists() and media_is_video and self.media_manager.has_audio_stream(media_path):
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
        self.config = Config()
        self.language = language
        self.tts = KokoroTTSManager(self.config, language)
        self.custom_api = CustomMemeAPI()
        self.video_gen = MoviePyVideoGenerator(self.config, self.custom_api)
        self.keyword_extractor = KeywordExtractor()
        self.rewriter = OllamaHumourRewriter()
        self.media_manager = MemeMediaManager(self.config, self.custom_api.session)
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

                local_media, is_video = self.media_manager.prepare_media_for_item(media_url)

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