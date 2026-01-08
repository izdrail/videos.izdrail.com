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
import subprocess
import threading
import traceback
import textwrap
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

# Language configuration - Top 5 + Romanian
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'code': 'en', 'tts_code': 'en'},
    'zh': {'name': 'Chinese (Mandarin)', 'code': 'zh-cn', 'tts_code': 'zh-cn'},
    'es': {'name': 'Spanish', 'code': 'es', 'tts_code': 'es'},
    'hi': {'name': 'Hindi', 'code': 'hi', 'tts_code': 'hi'},
    'ar': {'name': 'Arabic', 'code': 'ar', 'tts_code': 'ar'},
    'ro': {'name': 'Romanian', 'code': 'ro', 'tts_code': 'ro'}
}

# Stable Diffusion imports
try:
    from diffusers import StableDiffusionPipeline
    SD_AVAILABLE = True
except ImportError:
    print("[SD] Stable Diffusion not available.")
    SD_AVAILABLE = False

# SpaCy import with multi-language support
# Spacy AI API Configuration
SPACY_AVAILABLE = True # Always True since we use external API
SPACY_MODELS = {} # No local models needed

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

try:
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    from TTS.api import TTS
    MODELS_AVAILABLE = True
except ImportError:
    print("[TTS] TTS libraries not found.")
    MODELS_AVAILABLE = False

# =============== DATABASE SETUP ===============
class GenerationDB:
    def __init__(self, db_path: Path = Path("generation_cache.db")):
        self.db_path = db_path
        self.init_db()
        self.lock = threading.Lock()

    def _get_tables(self, conn) -> List[str]:
        """Get list of existing tables"""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]

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
                # Check columns for speaker_id
                cursor.execute("PRAGMA table_info(tts_cache)")
                columns = [info[1] for info in cursor.fetchall()]
                speaker_id_exists = 'speaker_id' in columns
                
                if not speaker_id_exists:
                    print("[DB] Schema mismatch: 'speaker_id' missing in tts_cache. Recreating table...")
                    cursor.execute("DROP TABLE tts_cache")
                    conn.commit()
            
            # === tts_cache table ===
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tts_cache (
                    text_hash TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en',
                    audio_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # === video_logs table ===
            if 'video_logs' not in self._get_tables(conn):
                conn.execute("""
                    CREATE TABLE video_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_hash TEXT UNIQUE,
                        video_path TEXT,
                        audio_path TEXT,
                        output_dir TEXT,
                        sentence_count INTEGER,
                        language TEXT,
                        created_at TEXT
                    )
                """)
                print("[DB] Created video_logs table")
            else:
                columns = self.get_columns_helper(conn, 'video_logs')
                if 'language' not in columns:
                    try:
                        conn.execute("ALTER TABLE video_logs ADD COLUMN language TEXT DEFAULT 'en'")
                        print("[DB] Added 'language' column to video_logs")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            print(f"[DB] video_logs migration error: {e}")
            conn.commit()

    def get_columns_helper(self, conn, table_name):
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

    def get_cached_tts(self, text: str, speaker_id: str, language: str) -> Optional[Path]:
        text_hash = hashlib.sha256(f"{text}_{speaker_id}_{language}".encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            try:
                cur = conn.execute(
                    "SELECT audio_path FROM tts_cache WHERE text_hash = ? AND speaker_id = ? AND language = ?",
                    (text_hash, speaker_id, language)
                )
                row = cur.fetchone()
                if row and Path(row[0]).exists():
                    return Path(row[0])
            except sqlite3.OperationalError:
                pass
        return None

    def save_tts(self, text: str, speaker_id: str, language: str, audio_path: Path):
        text_hash = hashlib.sha256(f"{text}_{speaker_id}_{language}".encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO tts_cache (text_hash, speaker_id, language, audio_path, created_at) VALUES (?, ?, ?, ?, ?)",
                    (text_hash, speaker_id, language, str(audio_path), datetime.now().isoformat())
                )
            except sqlite3.OperationalError as e:
                print(f"[DB] Warning: Could not save TTS cache: {e}")

    def get_cached_video(self, input_params: Dict) -> Optional[Dict]:
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()) if k not in {'progress_callback'})
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            try:
                cur = conn.execute(
                    "SELECT video_path, audio_path, output_dir, sentence_count, language FROM video_logs WHERE input_hash = ?",
                    (input_hash,)
                )
                row = cur.fetchone()
                if row and all(Path(p).exists() for p in row[:2] if p):
                    return {
                        "video_path": row[0],
                        "audio_path": row[1],
                        "output_directory": row[2],
                        "sentence_count": row[3],
                        "language": row[4],
                        "success": True
                    }
            except sqlite3.OperationalError:
                pass
        return None

    def save_video(self, input_params: Dict, result: Dict):
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()) if k not in {'progress_callback'})
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO video_logs
                    (input_hash, video_path, audio_path, output_dir, sentence_count, language, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        input_hash,
                        result.get("video_path"),
                        result.get("audio_path"),
                        result.get("output_directory"),
                        result.get("sentence_count", 0),
                        result.get("language", "en"),
                        datetime.now().isoformat()
                    )
                )
            except sqlite3.OperationalError as e:
                print(f"[DB] Warning: Could not save video cache: {e}")

DB = GenerationDB()

# =============== MEDIA SOURCE CACHE ===============
_media_source_cache = {}
_used_keywords = set()

# =============== STABLE DIFFUSION MANAGER ===============
class StableDiffusionManager:
    def __init__(self, model_path: str = "/models/stable-diffusion-v1-5", device: str = "cpu"):
        self.model_path = model_path
        self.device = "cpu"
        self.pipe = None
        self.generation_cache = {}
        self.cache_dir = Path("sd_generated_images")
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        print(f"[SD] Initializing on {self.device}...")
        self._load_model()

    def _load_model(self):
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                safety_checker=None
            ).to(self.device)
            print("[SD] Model loaded successfully.")
        except Exception as e:
            print(f"[SD] Failed to load model: {e}")
            self.pipe = None

    def generate_image(self, sentence: str, keyword: Optional[str] = None,
                      size: Tuple[int, int] = (1080, 1920)) -> Optional[Path]:
        if not self.pipe:
            return None
        cache_key = f"{keyword}_{hash(sentence)}" if keyword else f"{hash(sentence)}"
        if cache_key in self.generation_cache:
            cached_path = self.generation_cache[cache_key]
            if cached_path.exists():
                return cached_path
        prompt = f"high quality cinematic photo of {keyword}, ultra-detailed, 4k" if keyword else "abstract art, cinematic"
        negative_prompt = "blurry, low quality, distorted, ugly, watermark"
        with self.lock:
            try:
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        width=360,
                        height=768,
                    )
            except Exception as e:
                print(f"[SD] Generation error: {e}")
                return None
        image = result.images[0].resize(size, Image.LANCZOS)
        output_path = self.cache_dir / f"sd_{cache_key}_{uuid.uuid4().hex[:8]}.png"
        image.save(output_path, "PNG", quality=95)
        self.generation_cache[cache_key] = output_path
        print(f"[SD] Image generated: {output_path}")
        return output_path

# =============== KEYWORD EXTRACTOR ===============
class OllamaKeywordExtractor:
    def __init__(self, model: str = "gemma3:270m"):
        self.model = model
        self.url = os.getenv("OLLAMA_API_URL", "https://ai.izdrail.com/api/generate")
        self.cache = {}

    def extract_keywords(self, text: str, top_n: int = 5, language: str = 'en') -> List[str]:
        cache_key = f"{text[:100]}_{top_n}_{language}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        lang_name = SUPPORTED_LANGUAGES.get(language, {}).get('name', 'English')
        prompt = f"Extract {top_n} visual keywords from this {lang_name} text: {text}\nReturn only comma-separated words:"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 50}
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

    def extract_keywords(self, text: str, top_n: int = 5, language: str = 'en') -> List[str]:
        ollama_keywords = self.ollama_extractor.extract_keywords(text, top_n, language)
        if ollama_keywords:
            return ollama_keywords
            
        if not text.strip():
            return []

        # Use external Spacy API
        spacy_url = os.getenv("SPACY_API_URL", "https://spacy.izdrail.com")
        try:
            # Try POS tagging via API
            pos_resp = requests.post(f"{spacy_url}/pos", json={"text": text.lower()}, timeout=10)
            candidates = []
            if pos_resp.status_code == 200:
                data = pos_resp.json()
                for token in data:
                    pos = token.get('pos')
                    word = token.get('text')
                    is_stop = token.get('is_stop', False)
                    # Simple heuristic mapping for externalized NLP
                    if pos in {'NOUN', 'PROPN'} and not is_stop and len(word) > 2:
                        candidates.append(word)

            if not candidates:
                return []
                
            freq = Counter(candidates)
            return [word for word, count in freq.most_common(top_n)]
        except Exception as e:
            print(f"[NLP] External API error: {e}")
            return []

    def get_best_unique_keyword(self, text: str, language: str = 'en') -> Optional[str]:
        global _used_keywords
        keywords = self.extract_keywords(text, top_n=10, language=language)
        for kw in keywords:
            if kw not in _used_keywords:
                _used_keywords.add(kw)
                return kw
        return keywords[0] if keywords else None

# =============== CONFIG ===============
class Config:
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.VIDEOS_DIR = self.ROOT_DIR / "background_videos"
        self.MUSIC_DIR = self.ROOT_DIR / "background_music"
        self.CIRCLE_OVERLAYS_DIR = self.ROOT_DIR / "video-overlays"
        self.IMAGES_DIR = self.ROOT_DIR / "background_images"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        for dir_path in [self.VOICE_SAMPLES_DIR, self.VIDEOS_DIR, self.MUSIC_DIR,
                         self.CIRCLE_OVERLAYS_DIR, self.IMAGES_DIR, self.TEMP_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(exist_ok=True)
        self.STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
        self.DEVICE = "cpu"
        os.environ["COQUI_TOS_AGREED"] = "1"

        self.INTRO_MESSAGES = {
            'en': "Welcome to our channel!",
            'zh': "欢迎来到我们的频道！",
            'es': "¡Bienvenido a nuestro canal!",
            'hi': "हमारे चैनल में आपका स्वागत है!",
            'ar': "مرحبا بكم في قناتنا!",
            'ro': "Bine ați venit pe canalul nostru!"
        }
        self.CTA_MESSAGES = {
            'en': "Like, share, and subscribe!",
            'zh': "点赞、分享和订阅！",
            'es': "¡Dale me gusta, comparte y suscríbete!",
            'hi': "लाइक करें, शेयर करें और सब्सक्राइब करें!",
            'ar': "أعجبني، شارك، واشترك!",
            'ro': "Apreciază, distribuie și abonează-te!"
        }

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
            'max_width': 50,
            'max_height': 50,
            'position': 'top-left',
            'margin': 20,
            'opacity': 0.9,
        }
        self.MUSIC_CONFIG = {
            'voice_volume_db': 0,
            'music_volume_db': -5,
            'fade_in_duration': 1000,
            'fade_out_duration': 1000,
            'crossfade_duration': 500,
        }
        self.MAX_PARALLEL_SLIDES = 3
        self.VIDEO_PRESET = 'ultrafast'
        self.VIDEO_CRF = 28

# =============== MEDIA SOURCE BASE ===============
class BaseMediaAPI(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def download(self, item: Dict[str, Any], output_path: Path) -> bool:
        pass

    @abc.abstractmethod
    def get_random(self, query: str) -> Optional[Path]:
        pass

# =============== PEXELS API ===============
class PexelsAPI(BaseMediaAPI):
    def __init__(self):
        self.base_url = "https://api.pexels.com/videos"
        self.api_key = os.getenv("PEXELS_API_KEY")
        self.download_cache = {}
        self.search_cache = {}

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        return self.search_videos(query)

    def download(self, item: Dict[str, Any], output_path: Path) -> bool:
        video_files = item.get("video_files", [])
        if not video_files: return False
        # Prefer portrait
        portrait_videos = [v for v in video_files if v.get("width", 0) < v.get("height", 0)]
        if portrait_videos:
            portrait_videos.sort(key=lambda v: v.get("width", 0) * v.get("height", 0), reverse=True)
            url = portrait_videos[0].get("link")
        else:
            video_files.sort(key=lambda v: v.get("width", 0) * v.get("height", 0), reverse=True)
            url = video_files[0].get("link")
        
        if not url: return False
        return self.download_video(url, output_path)

    def get_random(self, query: str) -> Optional[Path]:
        path = self.get_random_video(query)
        if path:
            print(f"✨ [Pexels] Successfully sourced video for: {query}")
        return path

    def search_videos(self, query: str) -> List[Dict]:
        if not self.api_key:
            return []
        query = re.sub(r'[\*\-•\n]+', '', query).strip()
        if not query:
            return []
        cache_key = f"{query}_portrait"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        url = f"{self.base_url}/search"
        headers = {"Authorization": self.api_key}
        params = {"query": query, "orientation": "portrait", "per_page": 10}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            videos = response.json().get("videos", [])
            self.search_cache[cache_key] = videos
            return videos
        except Exception as e:
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
            except:
                pass
        try:
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            self.download_cache[video_url] = output_path
            return True
        except Exception as e:
            print(f"[Pexels] Download failed: {e}")
            return False

    def get_random_video(self, query: str) -> Optional[Path]:
        videos = self.search_videos(query)
        if not videos:
            return None
        
        # Shuffle results to be random but try more than one
        random.shuffle(videos)
        for video in videos[:3]: # Try up to 3 different results
            video_files = video.get("video_files", [])
            portrait_videos = [v for v in video_files if v.get("width", 0) < v.get("height", 0)]
            if not portrait_videos:
                continue
            
            portrait_videos.sort(key=lambda v: v.get("width", 0) * v.get("height", 0), reverse=True)
            selected = portrait_videos[0]
            video_url = selected.get("link")
            if not video_url:
                continue
                
            keyword_folder = Path("background_videos") / query.lower().replace(' ', '_')
            keyword_folder.mkdir(parents=True, exist_ok=True)
            output_path = keyword_folder / f"pexels_{uuid.uuid4().hex[:8]}.mp4"
            
            if self.download_video(video_url, output_path):
                return output_path
        return None

# =============== GIPHY API ===============
class GiphyAPI(BaseMediaAPI):
    def __init__(self):
        self.base_url = "https://api.giphy.com/v1/gifs/search"
        self.api_key = os.getenv("GIPHY_API_KEY")
        self.download_cache = {}
        self.search_cache = {}

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        if not self.api_key: return []
        query = re.sub(r'[\*\-•\n]+', '', query).strip()
        if not query: return []
        
        params = {"api_key": self.api_key, "q": query, "limit": 10, "rating": "g"}
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("data", [])
        except:
            return []

    def download(self, item: Dict[str, Any], output_path: Path) -> bool:
        mp4_url = item.get("images", {}).get("original_mp4", {}).get("mp4")
        if not mp4_url: return False
        return self.download_gif(mp4_url, output_path)

    def get_random(self, query: str) -> Optional[Path]:
        path = self.get_random_gif_video(query)
        if path:
            print(f"✨ [Giphy] Successfully sourced GIF video for: {query}")
        return path

    def search_gifs(self, query: str) -> Optional[str]:
        if not self.api_key:
            return None
        query = re.sub(r'[\*\-•\n]+', '', query).strip()
        if not query:
            return None
        cache_key = query
        if cache_key in self.search_cache:
            result = self.search_cache[cache_key]
            if result:
                return result
        params = {"api_key": self.api_key, "q": query, "limit": 5, "rating": "g"}
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            gifs = response.json().get("data", [])
            if gifs:
                gif = gifs[0]
                mp4_url = gif.get("images", {}).get("original_mp4", {}).get("mp4")
                if mp4_url:
                    self.search_cache[cache_key] = mp4_url
                    return mp4_url
        except Exception as e:
            print(f"[Giphy] Error: {e}")
        return None

    def download_gif(self, url: str, output_path: Path) -> bool:
        if output_path.exists():
            return True
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"[Giphy] Download failed: {e}")
            return False

    def get_random_gif_video(self, query: str) -> Optional[Path]:
        items = self.search(query)
        if not items:
            return None
        
        random.shuffle(items)
        for item in items[:3]: # Try up to 3 results
            mp4_url = item.get("images", {}).get("original_mp4", {}).get("mp4")
            if not mp4_url: continue
            
            keyword_folder = Path("background_videos") / query.lower().replace(' ', '_')
            keyword_folder.mkdir(parents=True, exist_ok=True)
            output_path = keyword_folder / f"giphy_{uuid.uuid4().hex[:8]}.mp4"
            
            if self.download_gif(mp4_url, output_path):
                return output_path
        return None

# =============== YOUTUBE API ===============
class YouTubeAPI(BaseMediaAPI):
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        self.search_cache = {}

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        if query in self.search_cache:
            return self.search_cache[query]
            
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "maxResults": 10,
            "type": "video",
            "videoCaption": "closedCaption",
            "key": self.api_key
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            items = response.json().get("items", [])
            self.search_cache[query] = items
            return items
        except Exception as e:
            print(f"[YouTube] Search error: {e}")
            return []

    def download(self, item: Dict[str, Any], output_path: Path) -> bool:
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            return False
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        except Exception as e:
            print(f"[YouTube] Download error: {e}")
            return False

    def get_random(self, query: str) -> Optional[Path]:
        videos = self.search(query)
        if not videos:
            return None
        
        random.shuffle(videos)
        for video in videos[:3]: # Try up to 3 results
            keyword_folder = Path("background_videos") / query.lower().replace(' ', '_')
            keyword_folder.mkdir(parents=True, exist_ok=True)
            output_path = keyword_folder / f"youtube_{uuid.uuid4().hex[:8]}.mp4"
            
            if self.download(video, output_path):
                print(f"✨ [YouTube] Successfully sourced video for: {query}")
                return output_path
        return None

# =============== OTHER API STUBS ===============
class DailymotionAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class VimeoAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class TwitchAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class PeerTubeAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class ApiVideoAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class CloudflareStreamAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class MuxAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class KalturaAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

class JSON2VideoAPI(BaseMediaAPI):
    def search(self, query, **kwargs): return []
    def download(self, item, output_path): return False
    def get_random(self, query): return None

# =============== MEDIA MANAGER ===============
class MediaManager:
    def __init__(self):
        self.apis = {
            "Pexels": PexelsAPI(),
            "Giphy": GiphyAPI(),
            "YouTube": YouTubeAPI(),
            "Dailymotion": DailymotionAPI(),
            "Vimeo": VimeoAPI(),
            "Twitch": TwitchAPI(),
            "PeerTube": PeerTubeAPI(),
            "api.video": ApiVideoAPI(),
            "Cloudflare Stream": CloudflareStreamAPI(),
            "Mux": MuxAPI(),
            "Kaltura": KalturaAPI(),
            "JSON2Video": JSON2VideoAPI()
        }
        self.preferred_order = ["Pexels", "YouTube", "Giphy"]

    def get_random_media(self, query: str, preferred_source: Optional[str] = None) -> Optional[Path]:
        all_sources = list(self.apis.keys())
        
        if preferred_source == "Random" or not preferred_source:
            random.shuffle(all_sources)
        else:
            # Build priority list: preferred -> preferred_order -> others
            priority_list = []
            if preferred_source in self.apis:
                priority_list.append(preferred_source)
            for s in self.preferred_order:
                if s not in priority_list and s in self.apis:
                    priority_list.append(s)
            for s in all_sources:
                if s not in priority_list:
                    priority_list.append(s)
            all_sources = priority_list

        for source_name in all_sources:
            try:
                # Only log if it's not a stub or if it actually has an API key
                api = self.apis[source_name]
                if hasattr(api, 'api_key') and not api.api_key:
                    continue
                
                path = api.get_random(query)
                if path:
                    print(f"🎯 [MediaManager] Selected {source_name} for query: '{query}'")
                    return path
            except Exception as e:
                print(f"[MediaManager] Error with {source_name}: {e}")
            
        return None

# =============== TTS MANAGER WITH MULTI-LANGUAGE ===============
class TTSManager:
    def __init__(self, config: Config):
        self.config = config
        self.voice_model = None
        self.standard_models = {}
        self._load_models()

    def _load_models(self):
        if not MODELS_AVAILABLE:
            return
        try:
            self.voice_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.config.DEVICE)
            print("[TTS] Coqui XTTS loaded (multi-language)")
        except Exception as e:
            print(f"[TTS] Coqui error loading models:")
            traceback.print_exc()
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
            print("[TTS] SpeechBrain loaded (English only)")
        except Exception as e:
            print(f"[TTS] SpeechBrain error: {e}")

    @staticmethod
    def preprocess_text(text: str, language: str = 'en') -> str:
        if language == 'en':
            return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
        return text

    def remove_metallic_artifacts(self, waveforms: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Minimal normalization to prevent metallic/crunchy sounds from tanh/over-processing"""
        try:
            # Simple peak normalization
            max_val = waveforms.abs().max()
            if max_val > 0.01:
                waveforms = waveforms / max_val * 0.95
            return waveforms
        except Exception as e:
            print(f"[TTS] Warning in remove_metallic_artifacts: {e}")
            return waveforms

    def improve_audio_quality(self, audio_path: Path) -> Path:
        """Minimal post-processing: 24kHz native rate, 20ms fades, peak normalization."""
        try:
            audio = AudioSegment.from_file(str(audio_path))
            # Use 24kHz (XTTS native) or 44.1kHz (High quality)
            audio = audio.set_frame_rate(24000)
            # Remove compression and low-pass which cause dullness
            audio = normalize(audio, headroom=0.1)
            # Minimal fades to prevent digital pops
            audio = audio.fade_in(20).fade_out(20)
            improved_path = self.config.TEMP_DIR / f"improved_{audio_path.name}"
            audio.export(str(improved_path), format="wav")
            return improved_path
        except Exception as e:
            print(f"[TTS] Audio quality warning: {e}")
            return audio_path

    def generate_speech(self, text: str, speaker_id: str, language: str = 'en') -> Path:
        cached = DB.get_cached_tts(text, speaker_id, language)
        if cached:
            print(f"[TTS] Using cached audio ({language})")
            return cached
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        processed_text = self.preprocess_text(text, language)
        temp_wav_path = self.config.TEMP_DIR / f"tts_{uuid.uuid4()}.wav"
        try:
            if speaker_id == self.config.STANDARD_VOICE_NAME:
                if language != 'en':
                    print(f"[TTS] Warning: Standard voice only supports English, using XTTS for {language}")
                    if not self.voice_model:
                        raise ValueError("Multi-language TTS unavailable")
                    lang_code = SUPPORTED_LANGUAGES.get(language, {}).get('tts_code', 'en')
                    self.voice_model.tts_to_file(
                        text=processed_text,
                        file_path=str(temp_wav_path),
                        language=lang_code,
                    )
                else:
                    if not self.standard_models:
                        # Attempt to reload or use backup
                        print("[TTS] Standard models missing, attempting re-init...")
                        self._load_models()
                        
                    if not self.standard_models:
                         # Fallback to XTTS if available, even for English
                        if self.voice_model:
                             print("[TTS] Standard models failed, falling back to XTTS for English...")
                             self.voice_model.tts_to_file(
                                text=processed_text,
                                file_path=str(temp_wav_path),
                                language="en",
                            )
                        else:
                            raise ValueError("Standard TTS models unavailable and XTTS fallback failed.")
                    else:
                        tacotron2 = self.standard_models['tacotron2']
                        hifi_gan = self.standard_models['hifi_gan']
                        mel_outputs, _, _ = tacotron2.encode_text(processed_text)
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
                lang_code = SUPPORTED_LANGUAGES.get(language, {}).get('tts_code', 'en')
                self.voice_model.tts_to_file(
                    text=processed_text,
                    file_path=str(temp_wav_path),
                    speaker_wav=str(reference_audio),
                    language=lang_code,
                )

            if not temp_wav_path.exists():
                raise RuntimeError("TTS generation failed.")
            improved_path = self.improve_audio_quality(temp_wav_path)
            if improved_path != temp_wav_path:
                temp_wav_path.unlink(missing_ok=True)
            DB.save_tts(text, speaker_id, language, improved_path)
            return improved_path
        except Exception as e:
            if temp_wav_path.exists():
                temp_wav_path.unlink(missing_ok=True)
            raise e

# =============== VIDEO GENERATOR WITH FFMPEG ===============
class FFmpegVideoGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.font_path = self._discover_fonts()
        self.media_manager = MediaManager()
        self.keyword_extractor = KeywordExtractor()
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
        if keyword:
            video = self.media_manager.get_random_media(keyword, preferred_source)
            if video: return video
            
        if sentence:
            keywords = self.keyword_extractor.extract_keywords(sentence, 5, language)
            for kw in keywords:
                video = self.media_manager.get_random_media(kw, preferred_source)
                if video: return video
                
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
        except:
            font = ImageFont.load_default()
        wrapped_text = textwrap.fill(text, width=35)
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = self.config.VIDEO_HEIGHT - text_height - self.config.TEXT_SIZE_CONFIG['bottom_margin']
        stroke_width = 3
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.text((x + adj_x, y + adj_y), wrapped_text, font=font, fill='black')
        draw.text((x, y), wrapped_text, font=font, fill='white')
        img.save(str(output_path), "PNG")
        return output_path

    def _create_intro_text_png(self, output_path: Path, language: str = 'en') -> Path:
        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            if self.font_path and os.path.exists(self.font_path):
                font_large = ImageFont.truetype(self.font_path, 100)
                font_small = ImageFont.truetype(self.font_path, 60)
            else:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        intro_msg = self.config.INTRO_MESSAGES.get(language, self.config.INTRO_MESSAGES['en'])
        lines = intro_msg.upper().split()
        main_text = "\n".join(lines[:2]) if len(lines) > 1 else lines[0]
        bbox = draw.textbbox((0, 0), main_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2 - 80
        for adj in range(-3, 4):
            if adj != 0:
                draw.text((x + adj, y), main_text, font=font_large, fill='black')
                draw.text((x, y + adj), main_text, font=font_large, fill='black')
        draw.text((x, y), main_text, font=font_large, fill='cyan')
        if len(lines) > 2:
            sec_text = " ".join(lines[2:]).upper()
            bbox2 = draw.textbbox((0, 0), sec_text, font=font_small)
            text_width2 = bbox2[2] - bbox2[0]
            x2 = (self.config.VIDEO_WIDTH - text_width2) // 2
            y2 = int(self.config.VIDEO_HEIGHT * 0.6)
            for adj in range(-2, 3):
                if adj != 0:
                    draw.text((x2 + adj, y2), sec_text, font=font_small, fill='black')
                    draw.text((x2, y2 + adj), sec_text, font=font_small, fill='black')
            draw.text((x2, y2), sec_text, font=font_small, fill='white')
        img.save(str(output_path), "PNG")
        return output_path

    def _create_cta_text_png(self, output_path: Path, language: str = 'en') -> Path:
        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            if self.font_path and os.path.exists(self.font_path):
                font_large = ImageFont.truetype(self.font_path, 110)
                font_small = ImageFont.truetype(self.font_path, 70)
            else:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        cta_msg = self.config.CTA_MESSAGES.get(language, self.config.CTA_MESSAGES['en'])
        parts = cta_msg.split(',')
        main_text = parts[0].strip().upper()
        if len(parts) > 1:
            main_text += "\n" + parts[1].strip().upper()
        bbox = draw.textbbox((0, 0), main_text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2 - 100
        for adj in range(-3, 4):
            if adj != 0:
                draw.text((x + adj, y), main_text, font=font_large, fill='black')
                draw.text((x, y + adj), main_text, font=font_large, fill='black')
        draw.text((x, y), main_text, font=font_large, fill='yellow')
        if len(parts) > 2:
            sec_text = parts[2].strip().upper()
            bbox2 = draw.textbbox((0, 0), sec_text, font=font_small)
            text_width2 = bbox2[2] - bbox2[0]
            x2 = (self.config.VIDEO_WIDTH - text_width2) // 2
            y2 = int(self.config.VIDEO_HEIGHT * 0.67)
            for adj in range(-2, 3):
                if adj != 0:
                    draw.text((x2 + adj, y2), sec_text, font=font_small, fill='black')
                    draw.text((x2, y2 + adj), sec_text, font=font_small, fill='black')
            draw.text((x2, y2), sec_text, font=font_small, fill='white')
        img.save(str(output_path), "PNG")
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

    def _create_slide_with_ffmpeg(self, sentence: str, audio_path: Path, video_path: Optional[Path],
                                  output_path: Path, slide_num: int, is_intro: bool = False,
                                  is_cta: bool = False, circle_video: Optional[Path] = None,
                                  circle_config: Optional[Dict] = None, language: str = 'en') -> Optional[Path]:
        try:
            source_info = f"Video: {video_path.name}" if video_path else "Background: Image/Color"
            print(f"🎬 [FFmpeg] Creating slide {slide_num} ({language}) - {source_info} - duration: {audio_path.stat().st_size} bytes")
            # Get audio duration
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            duration = float(result.stdout.strip())

            text_overlay_path = self.config.TEMP_DIR / f"text_{slide_num}_{uuid.uuid4().hex[:8]}.png"
            if is_intro:
                self._create_intro_text_png(text_overlay_path, language)
            elif is_cta:
                self._create_cta_text_png(text_overlay_path, language)
            else:
                self._create_text_overlay_png(sentence, text_overlay_path)

            inputs = []
            filter_parts = []
            input_count = 0

            if video_path and video_path.exists():
                inputs.extend(['-stream_loop', '-1', '-i', str(video_path)])
                filter_parts.append(
                    f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,"
                    f"fps=30,"
                    f"trim=duration={duration},"
                    f"setpts=PTS-STARTPTS[bg_scaled]"
                )
                input_count = 1
            else:
                inputs.extend(['-f', 'lavfi', '-i', f'color=c=0x4A90E2:s=1080x1920:d={duration}:r=30'])
                filter_parts.append("[0:v]null[bg_scaled]")
                input_count = 1

            filter_parts.append("[bg_scaled]format=rgba,colorchannelmixer=aa=0.6[dimmed]")

            inputs.extend(['-loop', '1', '-i', str(text_overlay_path)])
            filter_parts.append(f"[dimmed][{input_count}:v]overlay=0:0:format=auto[with_text]")
            input_count += 1

            logo_label = "with_text"
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
                filter_parts.append(f"[{input_count}:v]scale=50:50[logo_scaled]")
                filter_parts.append(f"[with_text][logo_scaled]overlay={pos}:format=auto[final]")
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
                filter_parts.append(
                    f"[{input_count}:v]scale={diameter}:{diameter}:force_original_aspect_ratio=decrease,"
                    f"pad={diameter}:{diameter}:(ow-iw)/2:(oh-ih)/2,"
                    f"fps=30,"
                    f"trim=duration={duration},"
                    f"setpts=PTS-STARTPTS,"
                    f"format=rgba[circle_sized]"
                )
                filter_parts.append(
                    f"color=black:s={diameter}x{diameter}:d={duration}[black];"
                    f"[black]geq=lum='if(gt(sqrt((X-{diameter/2})^2+(Y-{diameter/2})^2),{diameter/2}),0,255)',"
                    f"format=gray[mask]"
                )
                filter_parts.append(f"[circle_sized][mask]alphamerge[circle_masked]")
                filter_parts.append(f"[{logo_label}][circle_masked]overlay={overlay_pos}:format=auto[final_with_circle]")
                logo_label = "final_with_circle"
                input_count += 1

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
                '-r', '30',
                '-shortest',
                '-movflags', '+faststart',
                str(output_path)
            ]

            print(f"[FFmpeg] Creating slide {slide_num} ({language}) - duration: {duration:.2f}s")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not output_path.exists():
                print(f"[FFmpeg] ERROR: Output not created for slide {slide_num}")
                return None
            file_size = output_path.stat().st_size
            print(f"[FFmpeg] Slide {slide_num} created: {file_size / 1024 / 1024:.2f} MB")
            return output_path
        except Exception as e:
            print(f"[FFmpeg] Slide {slide_num} error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if 'text_overlay_path' in locals() and text_overlay_path.exists():
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
                          progress_callback=None) -> Path:
        global _used_keywords
        _used_keywords.clear()
        
        temp_dir = self.config.TEMP_DIR / f"final_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(exist_ok=True)
        
        slide_paths = []
        
        # Parallel slide generation
        with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_SLIDES) as executor:
            futures = []
            
            # Submit intro slide if needed
            if intro_audio:
                intro_video_bg = self.get_background_video("intro", "Welcome", language, preferred_media_source)
                intro_output = temp_dir / "slide_intro.mp4"
                futures.append(executor.submit(
                    self._create_slide_with_ffmpeg,
                    "", intro_audio, intro_video_bg, intro_output, -1, # -1 for intro slide_num
                    True, False, None, None, language # No circle for intro/cta
                ))

            # Submit main content slides
            for i, (sentence, audio_path, keyword) in enumerate(zip(sentences, audio_paths, keywords)):
                # Get background video for this slide
                video_bg = self.get_background_video(keyword, sentence, language, preferred_media_source)
                
                output_path = temp_dir / f"slide_{i:03d}.mp4"
                futures.append(executor.submit(
                    self._create_slide_with_ffmpeg,
                    sentence, audio_path, video_bg, output_path, i,
                    False, False, circle_video, circle_config, language
                ))
                
            # Submit CTA slide if needed
            if cta_audio:
                cta_video_bg = self.get_background_video("outro", "Goodbye", language, preferred_media_source)
                cta_output = temp_dir / "slide_cta.mp4"
                futures.append(executor.submit(
                    self._create_slide_with_ffmpeg,
                    "", cta_audio, cta_video_bg, cta_output, 999, # 999 for cta slide_num
                    False, True, None, None, language # No circle for intro/cta
                ))

            # Collect results in order of slide_num
            results_map = {}
            for i, future in enumerate(as_completed(futures)):
                try:
                    path_created = future.result()
                    if path_created:
                        # Extract slide_num from path_created (e.g., slide_000.mp4 -> 0, slide_intro.mp4 -> -1, slide_cta.mp4 -> 999)
                        # This assumes _create_slide_with_ffmpeg returns True on success, and output_path is passed.
                        # We need the actual output_path from the future to sort.
                        # A better way is to return (slide_num, output_path) from _create_slide_with_ffmpeg
                        # For now, let's assume the future.result() is the output_path if successful.
                        # The slide_num is passed to _create_slide_with_ffmpeg, but not returned.
                        # Let's modify _create_slide_with_ffmpeg to return output_path on success.
                        # For now, we'll just append and sort later.
                        slide_paths.append(path_created)
                    if progress_callback:
                        progress_callback(i + 1, len(futures), "Generating slides...")
                except Exception as e:
                    print(f"[FFmpeg] Slide error: {e}")
        
        # Sort slide_paths based on their slide_num (encoded in filename for now)
        # This is a temporary workaround. A better approach would be to return (slide_num, path) from the future.
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
        for slide_path in slide_paths:
            try:
                slide_path.unlink(missing_ok=True)
            except:
                pass
        return output_path

# =============== TEXT TO VIDEO GENERATOR ===============
class TextToVideoGenerator:
    def __init__(self):
        self.config = Config()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = FFmpegVideoGenerator(self.config)
        self.keyword_extractor = KeywordExtractor()
        self.available_voices = self._get_available_voices()
        self.available_music = self._get_available_music()
        self.available_circles = self._get_available_circles()
        self.available_languages = list(SUPPORTED_LANGUAGES.keys())

    def _get_available_voices(self) -> List[str]:
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)

    def _get_available_music(self) -> List[str]:
        music_files = self.video_generator.get_available_music_files()
        return ["Random"] + [m['name'] for m in music_files]

    def _get_available_circles(self) -> List[str]:
        circles = []
        for ext in ['*.mp4', '*.mov', '*.avi', '*.webm']:
            if self.config.CIRCLE_OVERLAYS_DIR.exists():
                circles.extend(list(self.config.CIRCLE_OVERLAYS_DIR.glob(ext)))
        return ["Random"] + [v.name for v in sorted(circles)]

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                      language: str = 'en',
                      pexels_keyword: Optional[str] = None,
                      preferred_media_source: Optional[str] = None,
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
                       progress_callback=None) -> Dict:
        input_params = {
            'text': text,
            'speaker_id': speaker_id,
            'language': language,
            'pexels_keyword': pexels_keyword,
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
            'circle_upload_path': str(circle_upload_path) if circle_upload_path else None
        }

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

            result = {
                "success": True,
                "audio_path": str(audio_final),
                "video_path": str(video_final),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "language": lang_name,
                "language_code": language,
                "background_music": enable_background_music and music_path is not None,
                "music_used": music_path.name if music_path else None,
                "intro_included": add_intro_slide,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "video_backgrounds": "Pexels/Giphy API + Local",
                "random_voices": use_random_voices,
                "circle_overlay_enabled": enable_circle_overlay,
                "circle_position": circle_position if enable_circle_overlay else None,
                "circle_selection": circle_selection if enable_circle_overlay else None,
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

                    with gr.TabItem("⚙️ Advanced"):
                        with gr.Row():
                            enable_intro = gr.Checkbox(label="📢 Add Intro Slide", value=True)
                            enable_cta = gr.Checkbox(label="📣 Add CTA Outro", value=True)
                        gr.Markdown("More advanced settings coming soon (resolution, frame rate, etc.)")

                generate_button = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                progress_bar = gr.Textbox(label="⚡ Status", value="Ready", interactive=False)

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video", height=600)
                audio_output = gr.Audio(label="Extracted Voiceover")
                status_output = gr.Markdown(value="*Your video will appear here after generation.*")

        def generate_wrapper(text, language, speaker, use_random, media_source, keyword,
                            enable_music, music_select, music_vol,
                            enable_circle, circle_sel, circle_upload_path, circle_diam, circle_pos, circle_border,
                            enable_intro, enable_cta, progress=gr.Progress()):
            
            if not text or not text.strip():
                return None, None, "❌ **Error:** Please enter some text.", "Ready"

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
                circle_selection=circle_sel,
                circle_upload_path=circle_upload_path,
                circle_border_width=circle_border,
                progress_callback=update_progress
            )

            if result.get("success"):
                status_md = f"""### ✅ Generation Complete!
- **Video:** {result['video_path']}
- **Duration:** {result.get('duration', 'N/A')}s
- **Source:** {media_source}
"""
                return result["video_path"], result["audio_path"], status_md, "Complete!"
            
            return None, None, f"❌ **Error:** {result.get('error', 'Unknown error')}", "Failed"

        generate_button.click(
            fn=generate_wrapper,
            inputs=[
                text_input, language_dropdown, speaker_dropdown, use_random_voices,
                media_source_dropdown, pexels_keyword,
                enable_music, music_dropdown, music_volume,
                enable_circle, circle_selection, circle_upload, circle_diameter, circle_position, circle_border_width,
                enable_intro, enable_cta
            ],
            outputs=[video_output, audio_output, status_output, progress_bar]
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
        print(f"  ✅ spaCy NLP - {len(SPACY_MODELS)} language model(s)")
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