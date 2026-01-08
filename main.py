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
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import torch
import threading
import torchaudio

# Fix for PyTorch 2.6+ secure loading - comprehensive TTS allowlist
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.shared_configs import BaseAudioConfig
    torch.serialization.add_safe_globals([
        XttsConfig, XttsAudioConfig, XttsArgs, 
        BaseDatasetConfig, BaseAudioConfig
    ])
except ImportError as e:
    print(f"[WARNING] Could not import TTS classes for PyTorch allowlist: {e}")
import gradio as gr
from moviepy.editor import (
    AudioFileClip, ImageClip, VideoFileClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip, vfx, ImageSequenceClip
)
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
import textwrap
from dotenv import load_dotenv

# Enforce CPU globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)  # Optional: limit CPU threads
load_dotenv()

# Stable Diffusion imports
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    SD_AVAILABLE = True
except ImportError:
    print("[SD] Stable Diffusion not available. Install with: pip install diffusers transformers accelerate")
    SD_AVAILABLE = False

# Spacy AI API Configuration
SPACY_AVAILABLE = True # Always True since we use external API
nlp = None # No local NLP object needed

# Fix PIL.ANTIALIAS deprecation
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

try:
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    from TTS.api import TTS
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"TTS libraries not found: {e}")
    MODELS_AVAILABLE = False

# =============== DATABASE SETUP ===============
class GenerationDB:
    def __init__(self, db_path: Path = Path("generation_cache.db")):
        self.db_path = db_path
        self.init_db()
        self.lock = threading.Lock()
    
    def init_db(self):
        # Ensure directory exists for db
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tts_cache table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tts_cache'")
            table_exists = cursor.fetchone() is not None
            
            speaker_id_exists = False
            if table_exists:
                # Check columns using PRAGMA
                cursor.execute("PRAGMA table_info(tts_cache)")
                columns = [info[1] for info in cursor.fetchall()]
                speaker_id_exists = 'speaker_id' in columns
                
                if not speaker_id_exists:
                    print("[DB] Schema mismatch: 'speaker_id' missing in tts_cache. Recreating table...")
                    cursor.execute("DROP TABLE tts_cache")
                    conn.commit() # Ensure drop is committed before creating
            
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tts_cache (
                    text_hash TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT UNIQUE,
                    video_path TEXT,
                    audio_path TEXT,
                    output_dir TEXT,
                    sentence_count INTEGER,
                    created_at TEXT
                )
            """)
            conn.commit()
    
    def get_cached_tts(self, text: str, speaker_id: str) -> Optional[Path]:
        text_hash = hashlib.sha256(f"{text}_{speaker_id}".encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT audio_path FROM tts_cache WHERE text_hash = ? AND speaker_id = ?",
                (text_hash, speaker_id)
            )
            row = cur.fetchone()
            if row and Path(row[0]).exists():
                return Path(row[0])
        return None

    def save_tts(self, text: str, speaker_id: str, audio_path: Path):
        text_hash = hashlib.sha256(f"{text}_{speaker_id}".encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tts_cache (text_hash, speaker_id, audio_path, created_at) VALUES (?, ?, ?, ?)",
                (text_hash, speaker_id, str(audio_path), datetime.now().isoformat())
            )
    
    def get_cached_video(self, input_params: Dict) -> Optional[Dict]:
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()) if k not in {'progress_callback'})
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT video_path, audio_path, output_dir, sentence_count FROM video_logs WHERE input_hash = ?",
                (input_hash,)
            )
            row = cur.fetchone()
            if row and all(Path(p).exists() for p in row[:2] if p):
                return {
                    "video_path": row[0],
                    "audio_path": row[1],
                    "output_directory": row[2],
                    "sentence_count": row[3],
                    "success": True
                }
        return None

    def save_video(self, input_params: Dict, result: Dict):
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()) if k not in {'progress_callback'})
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO video_logs
                (input_hash, video_path, audio_path, output_dir, sentence_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    input_hash,
                    result.get("video_path"),
                    result.get("audio_path"),
                    result.get("output_directory"),
                    result.get("sentence_count", 0),
                    datetime.now().isoformat()
                )
            )

# Initialize global DB
DB = GenerationDB()

# =============== MEDIA SOURCE CACHE ===============
_media_source_cache = {}
_used_keywords = set()

# =============== STABLE DIFFUSION MANAGER (CPU ONLY) ===============
class StableDiffusionManager:
    def __init__(self, model_path: str = "/models/stable-diffusion-v1-5", device: str = "cpu"):
        self.model_path = model_path
        self.device = "cpu"  # Enforce CPU
        self.pipe = None
        self.generation_cache = {}
        self.cache_dir = Path("sd_generated_images")
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        print(f"[SD] Initializing Stable Diffusion on {self.device}...")
        self._load_model()
    
    def _load_model(self):
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                safety_checker=None
            ).to(self.device)
            print("[SD] Model loaded successfully on CPU.")
        except Exception as e:
            print(f"[SD] Failed to load model from {self.model_path}: {e}")
            self.pipe = None
    
    def create_prompt(self, sentence: str, keyword: Optional[str] = None) -> str:
        if keyword:
            return f"high quality cinematic photo of {keyword}, inspired by '{sentence}', ultra-detailed, 4k lighting"
        return f"high quality cinematic photo inspired by '{sentence}', ultra-detailed, 4k lighting"
    
    def generate_image(
        self,
        sentence: str,
        keyword: Optional[str] = None,
        size: Tuple[int, int] = (1080, 1920)
    ) -> Optional[Path]:
        if not self.pipe:
            print("[SD] Model not loaded, cannot generate image.")
            return None
        cache_key = f"{keyword}_{hash(sentence)}" if keyword else f"{hash(sentence)}"
        if cache_key in self.generation_cache:
            cached_path = self.generation_cache[cache_key]
            if cached_path.exists():
                print(f"[SD] Using cached image for: {keyword or sentence[:40]}...")
                return cached_path
        print(f"[SD] Generating image for: {keyword or sentence[:60]}...")
        prompt = self.create_prompt(sentence, keyword)
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature"
        width, height = 360, 768
        num_inference_steps = 25  # Reduce for CPU
        guidance_scale = 7.5
        with self.lock:
            try:
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                    )
            except Exception as e:
                print(f"[SD] Generation error: {e}")
                return None
        image = result.images[0].resize(size, Image.LANCZOS)
        output_path = self.cache_dir / f"sd_{cache_key}_{uuid.uuid4().hex[:8]}.png"
        image.save(output_path, "PNG", quality=95)
        self.generation_cache[cache_key] = output_path
        print(f"[SD] Image generated and saved: {output_path}")
        return output_path

# =============== OLLAMA KEYWORD EXTRACTOR ===============
class OllamaKeywordExtractor:
    def __init__(self, model: str = "gemma3:270m"):
        self.model = model
        self.url = os.getenv("OLLAMA_API_URL", "https://ai.izdrail.com/api/generate")
        self.cache = {}
        print(f"[Ollama] Using model: {self.model} at {self.url}")
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        cache_key = f"{text[:100]}_{top_n}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        prompt = (
            f"Extract up to {top_n} relevant, concrete, visual keywords from this sentence. "
            "Return only a comma-separated list of lowercase words or short phrases (max 3 words each). "
            "Avoid abstract concepts, stop words, or brand names.\n"
            f"Sentence: \"{text}\"\nKeywords:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 64}
        }
        try:
            response = requests.post(self.url, json=payload, timeout=20)
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
                result = keywords[:top_n]
                self.cache[cache_key] = result
                return result
        except Exception as e:
            print(f"[Ollama] Error: {e}")
        return []

class KeywordExtractor:
    def __init__(self):
        self.ollama_extractor = OllamaKeywordExtractor()
        self.nlp = nlp if SPACY_AVAILABLE else None
        self.relevant_pos = {'NOUN', 'PROPN', 'ADJ'}
        self.exclude_words = {
            'thing', 'things', 'something', 'someone', 'way', 'time', 'day',
            'year', 'week', 'month', 'people', 'person', 'place', 'lot',
            'vodafone', 'apple', 'samsung', 'google', 'microsoft', 'amazon',
            'facebook', 'meta', 'tesla', 'nike', 'adidas', 'coca-cola', 'pepsi'
        }
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        ollama_keywords = self.ollama_extractor.extract_keywords(text, top_n)
        if ollama_keywords:
            return ollama_keywords
        
        if not text.strip():
            return []

        # Use external Spacy API
        spacy_url = os.getenv("SPACY_API_URL", "https://spacy.izdrail.com")
        try:
            # 1. Try POS tagging
            pos_resp = requests.post(f"{spacy_url}/pos", json={"text": text.lower()}, timeout=10)
            candidates = []
            if pos_resp.status_code == 200:
                data = pos_resp.json() # Assuming the API returns a list of token dicts
                for token in data:
                    pos = token.get('pos')
                    word = token.get('text')
                    is_stop = token.get('is_stop', False)
                    if (pos in self.relevant_pos and 
                        not is_stop and 
                        len(word) > 2 and 
                        word.isalpha() and 
                        word not in self.exclude_words):
                        candidates.append(word)

            # 2. Try NER
            ner_resp = requests.post(f"{spacy_url}/ner", json={"sections": [text.lower()]}, timeout=10)
            if ner_resp.status_code == 200:
                # API seems to return result based on common fastapi-spacy patterns
                # Based on /ner summary "Recognize Named Entities"
                ner_data = ner_resp.json()
                if isinstance(ner_data, list) and len(ner_data) > 0:
                    entities = ner_data[0].get('entities', [])
                    for ent in entities:
                        if ent.get('label') in {'GPE', 'LOC', 'EVENT', 'WORK_OF_ART'}:
                            candidates.append(ent.get('text').lower())

            if not candidates:
                return []
                
            keyword_freq = Counter(candidates)
            return [word for word, count in keyword_freq.most_common(top_n)]
        except Exception as e:
            print(f"[NLP] External API error: {e}")
            return []
    
    def get_best_unique_keyword(self, text: str) -> Optional[str]:
        global _used_keywords
        keywords = self.extract_keywords(text, top_n=10)
        for kw in keywords:
            if kw not in _used_keywords:
                _used_keywords.add(kw)
                return kw
        if keywords:
            return keywords[0]
        return None
    
    def sanitize_keyword(self, keyword: str) -> Optional[str]:
        """Clean keyword for API: remove bullets, newlines, extra spaces."""
        if not keyword:
            return None
        keyword = re.sub(r'[\*\-•]|\n', '', keyword)  # Remove bullets/newlines
        keyword = re.sub(r'\s+', ' ', keyword).strip()  # Normalize spaces
        keyword = ' '.join(keyword.split()[:4])  # Keep first 4 words
        return keyword if len(keyword) > 2 else None

# =============== CONFIG ===============
class Config:
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.IMAGES_DIR = self.ROOT_DIR / "background_images"
        self.VIDEOS_DIR = self.ROOT_DIR / "background_videos"
        self.MUSIC_DIR = self.ROOT_DIR / "background_music"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        self.SD_MODEL_DIR = self.ROOT_DIR / "models" / "stable-diffusion-v1-5"
        self.VIDEO_OVERLAYS_DIR = self.ROOT_DIR / "video-overlays"
        for dir_path in [self.VOICE_SAMPLES_DIR, self.IMAGES_DIR, self.VIDEOS_DIR,
                         self.MUSIC_DIR, self.TEMP_DIR, self.OUTPUT_DIR, self.VIDEO_OVERLAYS_DIR]:
            dir_path.mkdir(exist_ok=True)
        self.STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
        self.DEVICE = "cpu"  # Enforced CPU
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.INTRO_MESSAGE = "Welcome to our channel!"
        self.CTA_MESSAGE = "Like, share, and subscribe to our channel!"
        self.VIDEO_WIDTH = 1080
        self.VIDEO_HEIGHT = 1920
        self.VIDEO_SIZE = (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        self.TEXT_SIZE_CONFIG = {
            'font_size': 50,
            'line_spacing': 1.2,
            'max_width': 900,
            'bottom_margin': 150,
        }
        self.LOGO_CONFIG = {
            'max_width': 100,
            'max_height': 100,
            'position': 'top-left',
            'margin': 30,
            'opacity': 0.9,
        }
        self.CIRCLE_OVERLAY_CONFIG = {
            'diameter': 300,
            'position': 'top-right',
            'margin': 50,
            'border_width': 5,
            'border_color': (255, 255, 255),
            'opacity': 1.0,
        }
        self.TRANSITION_CONFIG = {
            'duration': 0.5,
        }
        self.MUSIC_CONFIG = {
            'voice_volume_db': 0,
            'music_volume_db': -5,
            'fade_in_duration': 1000,
            'fade_out_duration': 1000,
            'crossfade_duration': 500,
        }
        self.AUDIO_QUALITY_CONFIG = {
            'sample_rate': 22050,
            'low_pass_cutoff': 6000,
            'normalize_audio': True,
            'remove_silence_threshold': -40,
            'apply_compression': True,
            'high_pass_cutoff': 80,
            'apply_warmth': True,
            'reduce_sibilance': True,
        }
        self.MAX_PARALLEL_SLIDES = 3  # Reduce for CPU
        self.MIXED_MODE_SD_RATIO = 0.2
        self.VIDEO_PRESET = 'ultrafast'
        self.VIDEO_CRF = 28

# =============== APIs (Giphy, Pexels) ===============
class GiphyAPI:
    def __init__(self):
        self.base_url = "https://api.giphy.com/v1/gifs/search"
        self.api_key = os.getenv("GIPHY_API_KEY")
        self.download_cache = {}
        self.search_cache = {}
        self.cache_limit = 30

    def _manage_cache(self):
        if len(self.download_cache) > self.cache_limit:
            items_to_remove = len(self.download_cache) - self.cache_limit
            for key in list(self.download_cache.keys())[:items_to_remove]:
                del self.download_cache[key]

    def search_gif(self, query: str, rating: str = "g", limit: int = 5) -> Optional[str]:
        if not self.api_key:
            print("[GIPHY] GIPHY_API_KEY not set in environment.")
            return None
        # Sanitize query before using
        query = re.sub(r'[\*\-•]|\n', '', query)
        query = re.sub(r'\s+', ' ', query).strip()[:50]
        
        cache_key = f"{query}_{rating}_{limit}"
        if cache_key in self.search_cache:
            gifs = self.search_cache[cache_key]
            if gifs:
                gif = random.choice(gifs)
                mp4_url = gif.get("images", {}).get("original_mp4", {}).get("mp4")
                if mp4_url:
                    return mp4_url
                gif_url = gif.get("images", {}).get("original", {}).get("url")
                if gif_url and gif_url.endswith(".gif"):
                    return gif_url
        params = {"api_key": self.api_key, "q": query, "limit": limit, "rating": rating, "lang": "en"}
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            gifs = data.get("data", [])
            if gifs:
                self.search_cache[cache_key] = gifs
            if not gifs:
                return None
            for gif in gifs:
                mp4_url = gif.get("images", {}).get("original_mp4", {}).get("mp4")
                if mp4_url:
                    return mp4_url
                gif_url = gif.get("images", {}).get("original", {}).get("url")
                if gif_url and gif_url.endswith(".gif"):
                    return gif_url
            return None
        except Exception as e:
            print(f"[GIPHY] Error: {e}")
            return None

    def download_gif_or_mp4(self, url: str, output_path: Path) -> bool:
        if output_path.exists():
            return True
        if url in self.download_cache:
            try:
                cache_path = self.download_cache[url]
                if cache_path.exists():
                    shutil.copy(cache_path, output_path)
                    return True
            except Exception as e:
                print(f"[GIPHY] Cache error: {e}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.download_cache[url] = output_path
            self._manage_cache()
            return True
        except Exception as e:
            print(f"[GIPHY] Download error: {e}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return False

    def get_random_gif_video(self, query: str, size: Tuple[int, int] = (1080, 1920)) -> Optional[Path]:
        # Sanitize query
        query = re.sub(r'[\*\-•]|\n', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        gif_url = self.search_gif(query)
        if not gif_url:
            return None
        ext = ".mp4" if gif_url.endswith(".mp4") else ".gif"
        keyword_folder = Path("background_videos") / query.lower().replace(' ', '_')
        keyword_folder.mkdir(parents=True, exist_ok=True)
        temp_path = keyword_folder / f"giphy_{uuid.uuid4().hex[:8]}{ext}"
        if self.download_gif_or_mp4(gif_url, temp_path):
            if ext == ".gif":
                try:
                    clip = VideoFileClip(str(temp_path))
                    target_duration = max(10.0, clip.duration)
                    loops = int(target_duration / clip.duration) + 1
                    looped_clip = concatenate_videoclips([clip] * loops)
                    looped_clip = looped_clip.subclip(0, target_duration)
                    mp4_path = temp_path.with_suffix(".mp4")
                    looped_clip.write_videofile(str(mp4_path), codec="libx264", audio=False, logger=None, preset="ultrafast")
                    clip.close()
                    looped_clip.close()
                    temp_path.unlink(missing_ok=True)
                    return mp4_path
                except Exception as e:
                    print(f"[GIPHY] GIF-to-MP4 conversion failed: {e}")
                    return temp_path
            return temp_path
        return None

class PexelsAPI:
    def __init__(self):
        self.base_url = "https://api.pexels.com/videos"
        self.api_key = os.getenv("PEXELS_API_KEY")
        self.download_cache = {}
        self.search_cache = {}
        self.cache_limit = 30

    def _manage_cache(self):
        if len(self.download_cache) > self.cache_limit:
            items_to_remove = len(self.download_cache) - self.cache_limit
            for key in list(self.download_cache.keys())[:items_to_remove]:
                del self.download_cache[key]

    def search_videos(self, query: str, orientation: str = "portrait", size: str = "medium", per_page: int = 15,
                      page: int = 1) -> List[Dict]:
        if not self.api_key:
            print("[Pexels] PEXELS_API_KEY not set in environment.")
            return []
        
        # Sanitize query
        query = re.sub(r'[\*\-•]|\n', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        cache_key = f"{query}_{orientation}_{size}_{per_page}_{page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        url = f"{self.base_url}/search"
        headers = {"Authorization": self.api_key}
        params = {"query": query, "orientation": orientation, "size": size, "per_page": min(per_page, 80), "page": page}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", [])
            if videos:
                self.search_cache[cache_key] = videos
            return videos
        except requests.exceptions.RequestException as e:
            print(f"[Pexels] Error: {e}")
            return []

    def download_video(self, video_url: str, output_path: Path) -> bool:
        if output_path.exists():
            return True
        if video_url in self.download_cache:
            try:
                cache_path = self.download_cache[video_url]
                if cache_path.exists():
                    shutil.copy(cache_path, output_path)
                    return True
            except Exception as e:
                print(f"[Pexels] Cache error: {e}")
        try:
            response = requests.get(video_url, timeout=30, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.download_cache[video_url] = output_path
            self._manage_cache()
            return True
        except Exception as e:
            print(f"[Pexels] Download error: {e}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return False

    def get_random_video(self, query: str, size: Tuple[int, int] = (1080, 1920)) -> Optional[Path]:
        if not query:
            return None
            
        # Sanitize query
        query = re.sub(r'[\*\-•]|\n', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        videos = self.search_videos(query, orientation="portrait", size="medium", per_page=15)
        if not videos:
            return None
        video = random.choice(videos)
        video_id = video.get("id")
        video_files = video.get("video_files", [])
        portrait_videos = [v for v in video_files if v.get("width", 0) < v.get("height", 0)]
        if not portrait_videos:
            return None
        portrait_videos.sort(key=lambda v: abs((v.get("width", 0) * v.get("height", 0)) - (size[0] * size[1])))
        selected_video = portrait_videos[0]
        video_url = selected_video.get("link")
        if not video_url:
            return None
        keyword_folder = Path("background_videos") / query.lower().replace(' ', '_')
        keyword_folder.mkdir(parents=True, exist_ok=True)
        temp_video_path = keyword_folder / f"pexels_{video_id}_{uuid.uuid4().hex[:8]}.mp4"
        if self.download_video(video_url, temp_video_path):
            return temp_video_path
        return None

# =============== TTS MANAGER WITH CACHING ===============
class TTSManager:
    def __init__(self, config: Config):
        self.config = config
        self.voice_model = None
        self.standard_models: Dict[str, any] = {}
        self._load_models()
    
    def _load_models(self):
        if not MODELS_AVAILABLE:
            return
        try:
            self.voice_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.config.DEVICE)
            print("[TTS] Coqui XTTS loaded")
        except Exception as e:
            print(f"[TTS] Coqui error loading models:")
            traceback.print_exc()
            self.voice_model = None
        try:
            sb_cache = os.environ.get('SPEECHBRAIN_CACHE', '/opt/speechbrain_models')
            self.standard_models['tacotron2'] = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech", 
                savedir=f"{sb_cache}/tts-tacotron2-ljspeech"
            )
            self.standard_models['hifi_gan'] = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech", 
                savedir=f"{sb_cache}/tts-hifigan-ljspeech"
            )
            print("[TTS] SpeechBrain loaded")
        except Exception as e:
            print(f"[TTS] SpeechBrain error: {e}")
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    
    def remove_metallic_artifacts(self, waveforms: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            waveforms = waveforms * 0.4
            waveforms = torchaudio.functional.lowpass_biquad(waveforms, sample_rate, cutoff_freq=5500, Q=0.707)
            waveforms = torchaudio.functional.lowpass_biquad(waveforms, sample_rate, cutoff_freq=4800, Q=0.707)
            try:
                waveforms = torchaudio.functional.bandreject_biquad(waveforms, sample_rate, central_freq=3000, Q=1.5)
            except Exception:
                pass
            waveforms = torchaudio.functional.highpass_biquad(waveforms, sample_rate, cutoff_freq=100, Q=0.707)
            waveforms = torchaudio.functional.lowpass_biquad(waveforms, sample_rate, cutoff_freq=6200, Q=1.0)
            max_val = waveforms.abs().max()
            if max_val > 0.85:
                waveforms = waveforms / (max_val * 1.15)
            waveforms = torch.tanh(waveforms * 1.2) * 0.85
            try:
                waveforms = torchaudio.functional.spectral_subtract(waveforms, noise_estimate=None, noise_reduction_amount=0.4)
            except Exception:
                pass
            threshold = 0.6
            ratio = 2.5
            above_threshold = waveforms.abs() > threshold
            compressed = waveforms.clone()
            compressed[above_threshold] = torch.sign(waveforms[above_threshold]) * (
                threshold + (waveforms[above_threshold].abs() - threshold) / ratio
            )
            waveforms = compressed
            waveforms = waveforms * 0.85
            return waveforms
        except Exception as e:
            print(f"[TTS] Warning in remove_metallic_artifacts: {e}")
            return waveforms * 0.3

    def improve_audio_quality(self, audio_path: Path) -> Path:
        try:
            audio = AudioSegment.from_file(str(audio_path))
            cfg = self.config.AUDIO_QUALITY_CONFIG
            if audio.frame_rate != cfg['sample_rate']:
                audio = audio.set_frame_rate(cfg['sample_rate'])
            audio = audio - 6
            audio = audio.high_pass_filter(100)
            audio = low_pass_filter(audio, 4500)
            if cfg['reduce_sibilance']:
                audio = audio.low_pass_filter(6000)
            if cfg['apply_warmth']:
                warm_audio = audio.low_pass_filter(350) + 2
                audio = audio.overlay(warm_audio - 20)
            if cfg['apply_compression']:
                audio = audio.compress_dynamic_range(threshold=-30.0, ratio=1.8, attack=20.0, release=200.0)
            if cfg['normalize_audio']:
                audio = normalize(audio, headroom=0.3)
            audio = audio.strip_silence(silence_len=150, silence_thresh=cfg['remove_silence_threshold'], padding=150)
            audio = audio.fade_in(150).fade_out(150)
            improved_path = self.config.TEMP_DIR / f"improved_{audio_path.name}"
            audio.export(str(improved_path), format="wav", parameters=["-q:a", "0"])
            return improved_path
        except Exception as e:
            print(f"[TTS] Audio improvement warning: {e}")
            return audio_path

    def generate_speech(self, text: str, speaker_id: str) -> Path:
        # Check DB cache first
        cached = DB.get_cached_tts(text, speaker_id)
        if cached:
            print(f"[TTS] Using cached audio for speaker '{speaker_id}': {cached}")
            return cached
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        processed_text = self.preprocess_text(text)
        temp_wav_path = self.config.TEMP_DIR / f"tts_{uuid.uuid4()}.wav"
        try:
            if speaker_id == self.config.STANDARD_VOICE_NAME:
                if not self.standard_models:
                    raise ValueError("Standard TTS models unavailable.")
                tacotron2 = self.standard_models['tacotron2']
                hifi_gan = self.standard_models['hifi_gan']
                mel_outputs, mel_lengths, alignments = tacotron2.encode_text(processed_text)
                mel_spec = mel_outputs[0]
                if mel_spec.min() < 0:
                    mel_spec = (mel_spec + 2.5) / 5
                if len(mel_spec.shape) == 2:
                    mel_spec = mel_spec.unsqueeze(0)
                waveforms = hifi_gan.decode_batch(mel_spec)
                sample_rate = 22050
                waveforms = self.remove_metallic_artifacts(waveforms, sample_rate)
                torchaudio.save(str(temp_wav_path), waveforms.squeeze(1).to(torch.float32), sample_rate)
            else:
                if not self.voice_model:
                    raise ValueError("Voice cloning model unavailable.")
                reference_audio = self.config.VOICE_SAMPLES_DIR / speaker_id / "reference.wav"
                if not reference_audio.exists():
                    raise FileNotFoundError(f"Reference audio not found for '{speaker_id}'.")
                self.voice_model.tts_to_file(
                    text=processed_text,
                    file_path=str(temp_wav_path),
                    speaker_wav=str(reference_audio),
                    language="en",
                    split_sentences=False,
                )
            if not temp_wav_path.exists():
                raise RuntimeError("TTS generation failed — no output file created.")
            improved_path = self.improve_audio_quality(temp_wav_path)
            if improved_path != temp_wav_path:
                temp_wav_path.unlink(missing_ok=True)
            # Save to DB
            DB.save_tts(text, speaker_id, improved_path)
            return improved_path
        except Exception as e:
            if temp_wav_path.exists():
                temp_wav_path.unlink(missing_ok=True)
            raise e

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

# =============== VIDEO GENERATOR — with DB video caching ===============
class VideoGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.font_path = self._discover_fonts()
        self.pexels = PexelsAPI()
        self.giphy = GiphyAPI()
        self.keyword_extractor = KeywordExtractor()
        self.logo_clip = self._load_logo()
        self.effects_manager = VideoEffectsManager()
        self.circle_overlay_manager = CircleOverlayManager(config)
        self.sd_manager = None
        if SD_AVAILABLE:
            try:
                sd_model_path = str(config.SD_MODEL_DIR) if config.SD_MODEL_DIR.exists() else "/models/stable-diffusion-v1-5"
                self.sd_manager = StableDiffusionManager(model_path=sd_model_path, device=config.DEVICE)
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
        image_path = self.sd_manager.generate_image(text, keyword)
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
                image_path = self.sd_manager.generate_image(sentence or "abstract art", keyword)
                if image_path:
                    _media_source_cache[keyword or "default"] = 'sd'
                    return image_path
            return None

        # Step 1: Try Pexels/Giphy with explicit keyword
        if pexels_keyword:
            sanitized_kw = self.keyword_extractor.sanitize_keyword(pexels_keyword)
            if sanitized_kw:
                # Try Pexels first
                video_path = self.pexels.get_random_video(sanitized_kw, self.config.VIDEO_SIZE)
                if video_path:
                    _media_source_cache[sanitized_kw] = 'pexels'
                    return video_path
                # Then Giphy
                video_path = self.giphy.get_random_gif_video(sanitized_kw, self.config.VIDEO_SIZE)
                if video_path:
                    _media_source_cache[sanitized_kw] = 'giphy'
                    return video_path

        # Step 2: Try sentence-based keywords
        if sentence:
            candidates = self.keyword_extractor.extract_keywords(sentence, top_n=5)
            for kw in candidates:
                sanitized_kw = self.keyword_extractor.sanitize_keyword(kw)
                if not sanitized_kw:
                    continue
                # Pexels
                video_path = self.pexels.get_random_video(sanitized_kw, self.config.VIDEO_SIZE)
                if video_path:
                    _media_source_cache[sanitized_kw] = 'pexels'
                    return video_path
                # Giphy
                video_path = self.giphy.get_random_gif_video(sanitized_kw, self.config.VIDEO_SIZE)
                if video_path:
                    _media_source_cache[sanitized_kw] = 'giphy'
                    return video_path

        # Step 3: Try local background videos
        video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV']
        video_files = []
        if self.config.VIDEOS_DIR.exists():
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.config.VIDEOS_DIR, ext)))
        if video_files:
            return Path(random.choice(video_files))

        # Step 4: ONLY FALL BACK TO STABLE DIFFUSION IF ALL VIDEO SOURCES FAIL
        if self.sd_manager and (use_sd or media_type == "mixed"):
            keyword = pexels_keyword or (self.keyword_extractor.get_best_unique_keyword(sentence) if sentence else None)
            image_path = self.sd_manager.generate_image(sentence or "abstract background", keyword)
            if image_path:
                _media_source_cache[keyword or "default"] = 'sd'
                return image_path

        return None  # No media found

    def _create_subtitle_overlay_pil(self, text: str, duration: float) -> ImageClip:
        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        base_font_size = self.config.TEXT_SIZE_CONFIG['font_size']
        if len(text) > 60:
            font_size = max(30, int(base_font_size * (1.0 - (len(text) - 60) / 200)))
        else:
            font_size = base_font_size
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            print(f"[Video] Font loading error, using default: {e}")
            font = ImageFont.load_default()
        wrapped_text = textwrap.fill(text, width=35)
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = self.config.VIDEO_HEIGHT - text_height - self.config.TEXT_SIZE_CONFIG['bottom_margin']
        stroke_width = 2
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                draw.text((x + adj_x, y + adj_y), wrapped_text, font=font, fill='black')
        draw.text((x, y), wrapped_text, font=font, fill='white')
        img_clip = ImageClip(np.array(img)).set_duration(duration)
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
                dimming_clip = ColorClip(size=self.config.VIDEO_SIZE, color=(0,0,0), duration=duration_sec).set_opacity(0.4)
                video_clip = CompositeVideoClip([video_clip, dimming_clip])
            else:
                video_clip = ColorClip(size=self.config.VIDEO_SIZE, color=list(bg_color), duration=duration_sec)
            text_clip = self._create_subtitle_overlay_pil(sentence, duration_sec)
            layers = [video_clip, text_clip]
            if self.logo_clip:
                logo = self.logo_clip.set_duration(duration_sec)
                layers.append(logo)
            final_clip = CompositeVideoClip(layers)
            final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
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
        transition_type = random.choice(['crossfade', 'slide_left', 'slide_right', 'zoom', 'fade_black'])
        w, h = self.config.VIDEO_WIDTH, self.config.VIDEO_HEIGHT
        audio1 = clip1.audio
        audio2 = clip2.audio
        clip1_main = clip1.subclip(0, clip1.duration - duration)
        clip1_tail = clip1.subclip(clip1.duration - duration)
        clip2_head = clip2.subclip(0, duration)
        clip2_main = clip2.subclip(duration)
        if transition_type == 'crossfade':
            clip1_tail = clip1_tail.crossfadeout(duration)
            clip2_head = clip2_head.crossfadein(duration)
            middle = CompositeVideoClip([clip1_tail, clip2_head.set_start(0)], size=(w, h))
        elif transition_type == 'slide_left':
            clip1_tail = clip1_tail.set_position((0, 0))
            def slide_pos(t): return (w * (1 - t / duration), 0)
            clip2_head = clip2_head.set_position(slide_pos).set_start(0)
            middle = CompositeVideoClip([clip1_tail, clip2_head], size=(w, h))
        elif transition_type == 'slide_right':
            clip1_tail = clip1_tail.set_position((0, 0))
            def slide_pos(t): return (-w * (1 - t / duration), 0)
            clip2_head = clip2_head.set_position(slide_pos).set_start(0)
            middle = CompositeVideoClip([clip1_tail, clip2_head], size=(w, h))
        elif transition_type == 'zoom':
            def zoom_out(t): return 1 + 0.3 * (t / duration)
            def zoom_in(t): return 1.3 - 0.3 * (t / duration)
            clip1_tail = clip1_tail.fx(vfx.resize, zoom_out)
            clip2_head = clip2_head.fx(vfx.resize, zoom_in).set_start(0)
            middle = CompositeVideoClip([clip1_tail, clip2_head], size=(w, h))
        elif transition_type == 'fade_black':
            black = ColorClip((w, h), color=(0, 0, 0), duration=duration)
            clip1_tail = clip1_tail.fadeout(duration / 2)
            black_clip = black.set_start(duration / 2).fadein(duration / 2).fadeout(duration / 2)
            clip2_head = clip2_head.set_start(duration).fadein(duration / 2)
            middle = CompositeVideoClip([clip1_tail, black_clip, clip2_head], size=(w, h))
        else:
            middle = concatenate_videoclips([clip1_tail, clip2_head], method="compose")
        result = concatenate_videoclips([clip1_main, middle, clip2_main], method="compose")
        full_audio = concatenate_videoclips([clip1, clip2], method="compose").audio
        if abs(full_audio.duration - result.duration) > 0.1:
            if full_audio.duration > result.duration:
                full_audio = full_audio.subclip(0, result.duration)
            else:
                from pydub import AudioSegment
                temp_path = self.config.TEMP_DIR / f"temp_audio_fix_{uuid.uuid4()}.wav"
                full_audio.write_audiofile(str(temp_path), logger=None)
                audio_seg = AudioSegment.from_file(str(temp_path))
                silence_needed = int((result.duration - full_audio.duration) * 1000)
                if silence_needed > 0:
                    audio_seg = audio_seg + AudioSegment.silent(duration=silence_needed)
                    extended_path = self.config.TEMP_DIR / f"extended_fix_{uuid.uuid4()}.wav"
                    audio_seg.export(str(extended_path), format="wav")
                    full_audio = AudioFileClip(str(extended_path))
                    temp_path.unlink(missing_ok=True)
        result = result.set_audio(full_audio)
        return result

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
        global _used_keywords
        _used_keywords.clear()
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
        final_clip.write_videofile(
            str(output_path),
            fps=30,
            codec='libx264',
            audio_codec='aac',
            logger=None,
            preset=self.config.VIDEO_PRESET,
            threads=4,
            ffmpeg_params=["-crf", str(self.config.VIDEO_CRF)]
        )
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
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config)
        self.keyword_extractor = KeywordExtractor()
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
            print("✅ spaCy NLP: Enabled")
        else:
            print("⚠️ spaCy NLP: Disabled (install with: pip install spacy)")
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