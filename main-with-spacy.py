#!/usr/bin/env python3
import os
import re
import glob
import random
import shutil
import uuid
import platform
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import torch
import torchaudio
import gradio as gr
from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips, \
    TextClip, ColorClip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
import textwrap

# SpaCy import
try:
    import spacy

    SPACY_AVAILABLE = True
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("[NLP] Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_md")
        nlp = spacy.load("en_core_web_md")
except ImportError:
    print("[NLP] spaCy not installed. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    nlp = None

# Fix PIL.ANTIALIAS deprecation for Pillow >= 10.0.0
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

try:
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    from TTS.api import TTS

    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"TTS libraries not found: {e}")
    MODELS_AVAILABLE = False


class KeywordExtractor:
    """Extract relevant keywords from text using spaCy NLP"""

    def __init__(self):
        self.nlp = nlp if SPACY_AVAILABLE else None
        # Visual/video-relevant POS tags
        self.relevant_pos = {'NOUN', 'PROPN', 'ADJ'}
        # Stop words to exclude
        self.exclude_words = {
            'thing', 'things', 'something', 'someone', 'way', 'time', 'day',
            'year', 'week', 'month', 'people', 'person', 'place', 'lot'
        }

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top N keywords from text"""
        if not self.nlp or not text.strip():
            return []

        # Process the text
        doc = self.nlp(text.lower())

        # Extract candidate keywords
        candidates = []

        # Single-word keywords (nouns, proper nouns, adjectives)
        for token in doc:
            if (token.pos_ in self.relevant_pos and
                    not token.is_stop and
                    len(token.text) > 2 and
                    token.text.isalpha() and
                    token.text not in self.exclude_words):
                candidates.append(token.text)

        # Noun phrases (multi-word concepts)
        for chunk in doc.noun_chunks:
            # Clean the chunk
            chunk_text = chunk.text.strip()
            # Get only the head noun or last important word
            if len(chunk_text.split()) <= 3:  # Limit phrase length
                candidates.append(chunk_text)

        # Named entities (often visually interesting)
        for ent in doc.ents:
            if ent.label_ in {'GPE', 'LOC', 'ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}:
                candidates.append(ent.text.lower())

        # Count frequency
        keyword_freq = Counter(candidates)

        # Get top keywords
        top_keywords = [word for word, count in keyword_freq.most_common(top_n)]

        return top_keywords

    def get_best_keyword(self, text: str) -> Optional[str]:
        """Get the single best keyword for video search"""
        keywords = self.extract_keywords(text, top_n=5)

        if not keywords:
            return None

        # Prefer single-word keywords for better search results
        single_words = [kw for kw in keywords if ' ' not in kw]

        if single_words:
            return single_words[0]

        # If no single words, use the first keyword
        return keywords[0]

    def get_keyword_with_fallback(self, text: str, user_keyword: Optional[str] = None) -> Optional[str]:
        """Get keyword with manual override option"""
        # If user provided a keyword, use that
        if user_keyword and user_keyword.strip():
            return user_keyword.strip()

        # Otherwise, extract automatically
        return self.get_best_keyword(text)


class Config:
    def __init__(self):
        self.ROOT_DIR = Path(__file__).parent
        self.VOICE_SAMPLES_DIR = self.ROOT_DIR / "voice_samples"
        self.IMAGES_DIR = self.ROOT_DIR / "background_images"
        self.VIDEOS_DIR = self.ROOT_DIR / "background_videos"
        self.MUSIC_DIR = self.ROOT_DIR / "background_music"
        self.TEMP_DIR = self.ROOT_DIR / "temp"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        for dir_path in [self.VOICE_SAMPLES_DIR, self.IMAGES_DIR, self.VIDEOS_DIR,
                         self.MUSIC_DIR, self.TEMP_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(exist_ok=True)
        self.STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["COQUI_TOS_AGREED"] = "1"

        # CTA MESSAGE
        self.CTA_MESSAGE = "Like, share, and subscribe to our channel!"

        # PORTRAIT VIDEO SETTINGS (9:16 for mobile)
        self.VIDEO_WIDTH = 1080
        self.VIDEO_HEIGHT = 1920
        self.VIDEO_SIZE = (self.VIDEO_WIDTH, self.VIDEO_HEIGHT)

        self.TEXT_SIZE_CONFIG = {
            'font_size': 80,
            'line_spacing': 1.3,
            'max_width': 900,
        }
        self.MUSIC_CONFIG = {
            'voice_volume_db': 0,
            'music_volume_db': -15,
            'fade_in_duration': 2000,
            'fade_out_duration': 2000,
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

        # PERFORMANCE SETTINGS
        self.MAX_PARALLEL_SLIDES = 4
        self.ENABLE_CACHING = True
        self.VIDEO_CACHE_SIZE = 10

        # VIBRANT COLOR PALETTE for text
        self.TEXT_COLORS = [
            'white', 'gold', 'hotpink', 'cyan', 'orange',
            'limegreen', 'deeppink', 'yellow', 'blueviolet',
            'orangered', 'springgreen', 'pink', 'lightblue',
            'peachpuff', 'lightgreen'
        ]


class PexelsAPI:
    """Pexels API client for fetching videos"""

    def __init__(self):
        self.base_url = "https://api.pexels.com/videos"
        self.api_key = None
        self.download_cache = {}
        self.cache_limit = 20

    def set_api_key(self, api_key: str):
        """Set the Pexels API key"""
        self.api_key = api_key

    def _manage_cache(self):
        """Manage download cache size"""
        if len(self.download_cache) > self.cache_limit:
            items_to_remove = len(self.download_cache) - self.cache_limit
            for key in list(self.download_cache.keys())[:items_to_remove]:
                del self.download_cache[key]

    def search_videos(self, query: str, orientation: str = "portrait", size: str = "medium", per_page: int = 15,
                      page: int = 1) -> List[Dict]:
        """Search for videos on Pexels"""
        if not self.api_key:
            return []

        url = f"{self.base_url}/search"
        headers = {"Authorization": self.api_key}
        params = {
            "query": query,
            "orientation": orientation,
            "size": size,
            "per_page": min(per_page, 80),
            "page": page
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", [])
            print(f"[Pexels] Found {len(videos)} videos for '{query}'")
            return videos
        except requests.exceptions.RequestException as e:
            print(f"[Pexels] Error: {e}")
            return []

    def download_video(self, video_url: str, output_path: Path) -> bool:
        """Download a video from URL to local file"""
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
        """Get a random portrait video from Pexels"""
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

    def clear_cache(self):
        """Clear video download cache"""
        self.download_cache.clear()


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
            print(f"[TTS] Coqui error: {e}")
            self.voice_model = None
        try:
            tmp_tts = self.config.ROOT_DIR / "tmpdir_tts"
            tmp_vocoder = self.config.ROOT_DIR / "tmpdir_vocoder"
            self.standard_models['tacotron2'] = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech", savedir=tmp_tts
            )
            self.standard_models['hifi_gan'] = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech", savedir=tmp_vocoder
            )
            print("[TTS] SpeechBrain loaded")
        except Exception as e:
            print(f"[TTS] SpeechBrain error: {e}")

    @staticmethod
    def preprocess_text(text: str) -> str:
        return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

    def improve_audio_quality(self, audio_path: Path) -> Path:
        try:
            audio = AudioSegment.from_file(str(audio_path))
            cfg = self.config.AUDIO_QUALITY_CONFIG

            if audio.frame_rate != cfg['sample_rate']:
                audio = audio.set_frame_rate(cfg['sample_rate'])

            audio = audio.high_pass_filter(cfg['high_pass_cutoff'])
            audio = low_pass_filter(audio, cfg['low_pass_cutoff'])

            if cfg['reduce_sibilance']:
                audio = audio.low_pass_filter(7000)

            if cfg['apply_warmth']:
                warm_audio = audio.low_pass_filter(500) + 2
                audio = audio.overlay(warm_audio - 15)

            if cfg['apply_compression']:
                audio = audio.compress_dynamic_range(
                    threshold=-25.0,
                    ratio=2.5,
                    attack=10.0,
                    release=100.0
                )

            if cfg['normalize_audio']:
                audio = normalize(audio, headroom=0.1)

            audio = audio.strip_silence(
                silence_len=150,
                silence_thresh=cfg['remove_silence_threshold'],
                padding=150
            )

            audio = audio.fade_in(50).fade_out(50)

            improved_path = self.config.TEMP_DIR / f"improved_{audio_path.name}"
            audio.export(str(improved_path), format="wav", parameters=["-q:a", "0"])
            return improved_path
        except Exception as e:
            print(f"[TTS] Audio improvement warning: {e}")
            return audio_path

    def generate_speech(self, text: str, speaker_id: str) -> Path:
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

        improved_path = self.improve_audio_quality(temp_wav_path)
        if improved_path != temp_wav_path:
            try:
                temp_wav_path.unlink(missing_ok=True)
            except:
                pass
        return improved_path


class VideoGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.font_path = self._discover_fonts()
        self.pexels = PexelsAPI()
        self.background_cache = []
        self._preload_in_progress = False

    def _discover_fonts(self) -> str:
        """Discover system fonts and return a font path as string"""
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
                        font_str = str(font_file.resolve())
                        print(f"[Video] Using font: {font_str}")
                        return font_str

        print("[Video] No system fonts found, using fallback font name")
        return "DejaVuSans"

    def preload_backgrounds(self, count: int, pexels_keyword: Optional[str] = None):
        """Preload background videos"""
        if not self.config.ENABLE_CACHING or self._preload_in_progress:
            return

        self._preload_in_progress = True
        self.background_cache = []

        try:
            for i in range(min(count, self.config.VIDEO_CACHE_SIZE)):
                video_path = self._fetch_background_video_direct(pexels_keyword)
                if video_path and isinstance(video_path, Path) and video_path.exists():
                    self.background_cache.append(video_path)
        finally:
            self._preload_in_progress = False

    def _fetch_background_video_direct(self, pexels_keyword: Optional[str] = None) -> Optional[Path]:
        """Direct fetch without cache - used for preloading"""
        if pexels_keyword and self.pexels.api_key:
            video_path = self.pexels.get_random_video(pexels_keyword, self.config.VIDEO_SIZE)
            if video_path and video_path.exists():
                return video_path

        video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV']
        video_files = []

        if pexels_keyword:
            keyword_dir = self.config.VIDEOS_DIR / pexels_keyword.lower().replace(' ', '_')
            if keyword_dir.exists():
                for ext in video_extensions:
                    video_files.extend(glob.glob(os.path.join(keyword_dir, ext)))

        if not video_files and self.config.VIDEOS_DIR.exists():
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.config.VIDEOS_DIR, ext)))

        if video_files:
            return Path(random.choice(video_files))

        return None

    def get_background_video(self, pexels_keyword: Optional[str] = None) -> Optional[Path]:
        """Get a background video from cache or fetch new"""
        if self.background_cache and self.config.ENABLE_CACHING and not self._preload_in_progress:
            cached_path = self.background_cache.pop(0)
            if cached_path and isinstance(cached_path, Path) and cached_path.exists():
                return cached_path

        return self._fetch_background_video_direct(pexels_keyword)

    def get_random_text_color(self) -> str:
        """Get a random vibrant color for text"""
        return random.choice(self.config.TEXT_COLORS)

    def get_random_background_music(self) -> Optional[Path]:
        music_extensions = ['*.mp3', '*.MP3', '*.wav', '*.WAV']
        music_files = []
        if self.config.MUSIC_DIR.exists():
            for ext in music_extensions:
                music_files.extend(glob.glob(os.path.join(self.config.MUSIC_DIR, ext)))
        if music_files:
            selected = random.choice(music_files)
            print(f"[Music] Selected: {os.path.basename(selected)}")
            return Path(selected)
        return None

    def mix_audio_with_music(self, voice_audio_path: Path, total_duration_ms: int,
                             music_path: Optional[Path] = None) -> Path:
        if music_path is None:
            music_path = self.get_random_background_music()
        if music_path is None or not music_path.exists():
            return voice_audio_path

        try:
            voice = AudioSegment.from_file(str(voice_audio_path))
            music = AudioSegment.from_file(str(music_path))
            cfg = self.config.MUSIC_CONFIG

            voice = voice + cfg['voice_volume_db']
            music = music + cfg['music_volume_db']

            if len(music) < len(voice):
                loops_needed = (len(voice) // len(music)) + 2
                looped_music = music
                for _ in range(loops_needed - 1):
                    looped_music = looped_music.append(music, crossfade=cfg['crossfade_duration'])
                music = looped_music

            music = music[:len(voice)]
            music = music.fade_in(cfg['fade_in_duration']).fade_out(cfg['fade_out_duration'])
            mixed = voice.overlay(music)

            output_path = self.config.TEMP_DIR / f"final_mixed_{uuid.uuid4()}.wav"
            mixed.export(str(output_path), format="wav", parameters=["-q:a", "0"])
            return output_path
        except Exception as e:
            print(f"[Music] Error: {e}")
            return voice_audio_path

    def _create_text_overlay_pil(self, text: str, duration: float,
                                 text_color: Optional[str] = None) -> ImageClip:
        """Fallback method: Create text overlay using PIL instead of TextClip"""
        if text_color is None:
            text_color = self.get_random_text_color()

        img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font_size = self.config.TEXT_SIZE_CONFIG['font_size']
        try:
            if self.font_path and isinstance(self.font_path, str) and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            print(f"[Video] Font loading error, using default: {e}")
            font = ImageFont.load_default()

        wrapped_text = textwrap.fill(text, width=20)

        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.config.VIDEO_WIDTH - text_width) // 2
        y = (self.config.VIDEO_HEIGHT - text_height) // 2

        padding = 40
        bg_rect = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 180))

        stroke_width = 3
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                draw.text((x + adj_x, y + adj_y), wrapped_text, font=font, fill='black')

        draw.text((x, y), wrapped_text, font=font, fill=text_color)

        img_clip = ImageClip(np.array(img)).set_duration(duration).set_position('center')

        return img_clip

    def _create_text_overlay_clip(self, text: str, duration: float,
                                  text_color: Optional[str] = None) -> any:
        """Create a text overlay clip - tries TextClip first, falls back to PIL"""
        if text_color is None:
            text_color = self.get_random_text_color()

        if isinstance(self.font_path, str):
            font = self.font_path
        else:
            font = "DejaVuSans"

        wrapped_text = textwrap.fill(text, width=20)

        try:
            text_clip = TextClip(
                wrapped_text,
                fontsize=self.config.TEXT_SIZE_CONFIG['font_size'],
                color=text_color,
                font=font,
                stroke_color='black',
                stroke_width=3,
                method='caption',
                size=(self.config.TEXT_SIZE_CONFIG['max_width'], None),
                align='center',
                bg_color=(0, 0, 0, 180)
            ).set_duration(duration).set_position('center')

            return text_clip

        except Exception as e:
            print(f"[Video] TextClip failed, using PIL fallback: {e}")
            return self._create_text_overlay_pil(text, duration, text_color)

    def create_cta_slide(self, audio_path: Path, bg_color: Tuple[int, int, int] = (74, 144, 226),
                         pexels_keyword: Optional[str] = None) -> Tuple[VideoFileClip, List[Path]]:
        """Create a Call-To-Action slide"""
        temp_files = []

        audio_segment = AudioSegment.from_file(str(audio_path))
        duration_sec = len(audio_segment) / 1000.0

        background_video = self.get_background_video(pexels_keyword)

        if background_video and background_video.exists():
            try:
                video_clip = VideoFileClip(str(background_video))
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
                video_clip = video_clip.fl_image(lambda img: (img * 0.6).astype('uint8'))

            except Exception as e:
                print(f"[Video] CTA error: {e}")
                video_clip = None
        else:
            video_clip = None

        if video_clip is None:
            video_clip = ColorClip(size=self.config.VIDEO_SIZE, color=list(bg_color), duration=duration_sec)

        if isinstance(self.font_path, str):
            font = self.font_path
        else:
            font = "DejaVuSans"

        try:
            main_text = TextClip(
                "LIKE\nSHARE\nSUBSCRIBE",
                fontsize=140,
                color='yellow',
                font=font,
                stroke_color='black',
                stroke_width=3,
                method='caption',
                size=(self.config.VIDEO_WIDTH - 200, None),
                align='center'
            ).set_duration(duration_sec).set_position('center')

            secondary_text = TextClip(
                "TO OUR CHANNEL",
                fontsize=80,
                color='white',
                font=font,
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(self.config.VIDEO_WIDTH - 200, None),
                align='center'
            ).set_duration(duration_sec).set_position(('center', self.config.VIDEO_HEIGHT * 0.65))

            final_clip = CompositeVideoClip([video_clip, main_text, secondary_text])

        except Exception as e:
            print(f"[Video] CTA TextClip failed, using PIL fallback: {e}")
            img = Image.new('RGBA', self.config.VIDEO_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            try:
                if self.font_path and isinstance(self.font_path, str) and os.path.exists(self.font_path):
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
            final_clip = CompositeVideoClip([video_clip, text_clip])

        final_clip = final_clip.set_duration(duration_sec)

        audio_clip = AudioFileClip(str(audio_path))
        final_clip = final_clip.set_audio(audio_clip)

        return (final_clip, temp_files)

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = re.sub(r'\bDr\.', 'Dr<dot>', text)
        text = re.sub(r'\bMr\.', 'Mr<dot>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<dot>', text)
        text = re.sub(r'\bMs\.', 'Ms<dot>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<dot>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<dot>', '.').strip() for s in sentences if s.strip()]
        for i, sentence in enumerate(sentences):
            if not sentence.endswith(('.', '!', '?')):
                sentences[i] = sentence + '.'
        return sentences

    def _create_single_slide(self, sentence: str, audio_path: Path, bg_color: Tuple[int, int, int],
                             pexels_keyword: Optional[str], slide_num: int) -> Tuple[
        Optional[VideoFileClip], List[Path]]:
        """Create a single slide with video background and text overlay"""
        temp_files = []
        try:
            audio_segment = AudioSegment.from_file(str(audio_path))
            duration_sec = len(audio_segment) / 1000.0

            text_color = self.get_random_text_color()
            video_path = self.get_background_video(pexels_keyword)

            if video_path and isinstance(video_path, Path) and video_path.exists():
                try:
                    video_clip = VideoFileClip(str(video_path))
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
                    video_clip = video_clip.fl_image(lambda img: (img * 0.6).astype('uint8'))

                    text_clip = self._create_text_overlay_clip(sentence, duration_sec, text_color)
                    final_clip = CompositeVideoClip([video_clip, text_clip])
                    final_clip = final_clip.set_duration(duration_sec)

                    audio_clip = AudioFileClip(str(audio_path))
                    final_clip = final_clip.set_audio(audio_clip)

                    return (final_clip, temp_files)

                except Exception as e:
                    print(f"[Video] Slide {slide_num} video error: {e}")
                    video_clip = None

            bg_clip = ColorClip(size=self.config.VIDEO_SIZE, color=list(bg_color), duration=duration_sec)
            text_clip = self._create_text_overlay_clip(sentence, duration_sec, text_color)

            final_clip = CompositeVideoClip([bg_clip, text_clip])
            final_clip = final_clip.set_duration(duration_sec)

            audio_clip = AudioFileClip(str(audio_path))
            final_clip = final_clip.set_audio(audio_clip)

            return (final_clip, temp_files)

        except Exception as e:
            print(f"[Video] Slide {slide_num} error: {e}")
            import traceback
            traceback.print_exc()
            return (None, temp_files)

    def create_video_per_sentence(self, sentences: List[str], audio_paths: List[Path],
                                  cta_audio_path: Optional[Path] = None,
                                  bg_color: Tuple[int, int, int] = (74, 144, 226),
                                  pexels_keyword: Optional[str] = None,
                                  add_cta_slide: bool = True,
                                  progress_callback=None) -> Path:
        clips = []
        temp_files = []

        if pexels_keyword:
            self.preload_backgrounds(len(sentences), pexels_keyword)

        with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_SLIDES) as executor:
            futures = {}
            for i, (sentence, audio_path) in enumerate(zip(sentences, audio_paths)):
                future = executor.submit(
                    self._create_single_slide,
                    sentence, audio_path, bg_color, pexels_keyword, i + 1
                )
                futures[future] = i

            slide_results = [None] * len(sentences)
            for future in as_completed(futures):
                i = futures[future]
                try:
                    video_clip, slide_temp_files = future.result()
                    if video_clip:
                        slide_results[i] = video_clip
                        temp_files.extend(slide_temp_files)
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
                cta_clip, cta_temp_files = self.create_cta_slide(
                    cta_audio_path,
                    bg_color=bg_color,
                    pexels_keyword=pexels_keyword
                )
                clips.append(cta_clip)
                temp_files.extend(cta_temp_files)
            except Exception as e:
                print(f"[Video] CTA warning: {e}")

        if progress_callback:
            progress_callback(len(sentences), len(sentences), "Assembling...")

        final_clip = concatenate_videoclips(clips, method="compose")
        output_path = self.config.TEMP_DIR / f"video_{uuid.uuid4()}.mp4"
        final_clip.write_videofile(
            str(output_path),
            fps=30,
            codec='libx264',
            audio_codec='aac',
            logger=None,
            preset='medium',
            threads=4
        )

        for path in temp_files:
            try:
                if path and path.exists():
                    path.unlink(missing_ok=True)
            except:
                pass

        for clip in clips:
            try:
                clip.close()
            except:
                pass

        try:
            final_clip.close()
        except:
            pass

        return output_path


class TextToVideoGenerator:
    def __init__(self):
        self.config = Config()
        self.tts_manager = TTSManager(self.config)
        self.video_generator = VideoGenerator(self.config)
        self.keyword_extractor = KeywordExtractor()
        self.available_voices = self._get_available_voices()

    def _get_available_voices(self) -> List[str]:
        voices = [self.config.STANDARD_VOICE_NAME]
        if self.config.VOICE_SAMPLES_DIR.is_dir():
            voices.extend([d.name for d in self.config.VOICE_SAMPLES_DIR.iterdir() if d.is_dir()])
        return sorted(voices)

    def generate_video(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                       bg_color: Tuple[int, int, int] = (74, 144, 226),
                       pexels_keyword: Optional[str] = None,
                       pexels_api_key: Optional[str] = None,
                       enable_background_music: bool = True,
                       music_volume_db: int = -15,
                       add_call_to_action: bool = True,
                       progress_callback=None) -> Dict:
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}
        if len(text) > 10000:
            return {"error": "Text is too long (max 10,000 characters)", "success": False}

        if music_volume_db != self.config.MUSIC_CONFIG['music_volume_db']:
            self.config.MUSIC_CONFIG['music_volume_db'] = music_volume_db

        if pexels_api_key and pexels_api_key.strip():
            self.video_generator.pexels.set_api_key(pexels_api_key.strip())

        # AUTOMATIC KEYWORD EXTRACTION
        extracted_keyword = self.keyword_extractor.get_keyword_with_fallback(text, pexels_keyword)
        print(extracted_keyword)
        if extracted_keyword:
            print(f"[NLP] Using keyword: '{extracted_keyword}'")
            pexels_keyword = extracted_keyword
        else:
            print("[NLP] No keyword extracted, using local videos")
            pexels_keyword = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        audio_paths = []
        cta_audio_path = None
        music_path = None

        try:
            sentences = self.video_generator.split_into_sentences(text)
            print(f"[Info] Processing {len(sentences)} sentences")

            if len(sentences) > 100:
                return {"error": "Too many sentences (max 100)", "success": False}

            if enable_background_music:
                music_path = self.video_generator.get_random_background_music()

            for i, sentence in enumerate(sentences):
                if progress_callback:
                    progress_callback(i + 1, len(sentences) * 2, f"Audio {i + 1}/{len(sentences)}")
                audio_path = self.tts_manager.generate_speech(sentence, speaker_id)
                audio_paths.append(audio_path)

            if add_call_to_action:
                cta_audio_path = self.tts_manager.generate_speech(self.config.CTA_MESSAGE, speaker_id)

            combined_voice = AudioSegment.empty()
            for audio_path in audio_paths:
                segment = AudioSegment.from_file(str(audio_path))
                combined_voice += segment

            if add_call_to_action and cta_audio_path:
                cta_segment = AudioSegment.from_file(str(cta_audio_path))
                combined_voice += cta_segment

            total_duration_ms = len(combined_voice)
            temp_combined_voice_path = self.config.TEMP_DIR / f"combined_voice_{uuid.uuid4()}.wav"
            combined_voice.export(str(temp_combined_voice_path), format="wav")

            final_audio_path = temp_combined_voice_path
            if enable_background_music and music_path:
                final_audio_path = self.video_generator.mix_audio_with_music(
                    temp_combined_voice_path,
                    total_duration_ms,
                    music_path
                )

            audio_mp3_path = session_dir / f"audio_{timestamp}.mp3"
            final_audio_segment = AudioSegment.from_file(str(final_audio_path))
            final_audio_segment.export(str(audio_mp3_path), format="mp3", bitrate="192k")

            def video_progress(current, total, message):
                if progress_callback:
                    progress_callback(len(sentences) + current, len(sentences) * 2, message)

            video_temp_path = self.video_generator.create_video_per_sentence(
                sentences, audio_paths,
                cta_audio_path=cta_audio_path,
                bg_color=bg_color,
                pexels_keyword=pexels_keyword,
                add_cta_slide=add_call_to_action,
                progress_callback=video_progress
            )

            video_clip = VideoFileClip(str(video_temp_path))
            final_audio_clip = AudioFileClip(str(final_audio_path))
            final_video = video_clip.set_audio(final_audio_clip)

            video_final_path = session_dir / f"video_portrait_{timestamp}.mp4"
            final_video.write_videofile(
                str(video_final_path),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                logger=None,
                preset='medium',
                threads=4
            )

            video_clip.close()
            final_audio_clip.close()
            final_video.close()

            try:
                video_temp_path.unlink(missing_ok=True)
            except:
                pass

            return {
                "success": True,
                "audio_path": str(audio_mp3_path),
                "video_path": str(video_final_path),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "background_music": enable_background_music and music_path is not None,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "text_colors": "Random vibrant colors",
                "video_backgrounds": "Pexels API" if pexels_keyword and pexels_api_key else "Local videos",
                "extracted_keyword": extracted_keyword if extracted_keyword else "None"
            }

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
            if cta_audio_path:
                try:
                    cta_audio_path.unlink(missing_ok=True)
                except:
                    pass
            try:
                if 'temp_combined_voice_path' in locals():
                    temp_combined_voice_path.unlink(missing_ok=True)
            except:
                pass


def setup_ui(generator: TextToVideoGenerator):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   title="Portrait Video Generator with AI Keyword Extraction") as demo:
        gr.Markdown("# 🎥 Portrait Video Generator (9:16) with AI Keyword Extraction")
        gr.Markdown("Create stunning portrait videos with automatic keyword extraction using spaCy NLP!")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter Your Text",
                    placeholder="Type your text here... AI will automatically extract keywords for video backgrounds.",
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

                enable_cta = gr.Checkbox(label="Add Call-to-Action Slide", value=True)
                enable_music = gr.Checkbox(label="Enable Background Music", value=True)
                music_volume = gr.Slider(-40, -5, value=-15, step=1, label="Music Volume (dB)")

                with gr.Row():
                    pexels_keyword = gr.Textbox(
                        label="Manual Keyword Override (Optional)",
                        placeholder="Leave empty for automatic extraction",
                        info="AI will extract keywords automatically if left blank"
                    )
                    pexels_api_key = gr.Textbox(
                        label="Pexels API Key",
                        type="password",
                        placeholder="Get free API key from pexels.com/api",
                        info="Optional - uses local videos if not provided"
                    )

                progress_bar = gr.Textbox(label="Progress", value="Ready...", interactive=False)
                generate_button = gr.Button("🎥 Generate Video with AI Keywords", variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown()

        def generate_wrapper(text, speaker, bg_hex, keyword, api_key,
                             enable_music, music_vol, enable_cta, progress=gr.Progress()):
            if not text or not text.strip():
                return None, None, "❌ Error: Please enter some text", "Ready..."

            bg_hex = bg_hex.lstrip('#')
            try:
                bg_color = tuple(int(bg_hex[i:i + 2], 16) for i in (0, 2, 4))
            except:
                bg_color = (74, 144, 226)

            keyword = keyword.strip() if keyword else None
            api_key = api_key.strip() if api_key else None

            def update_progress(current, total, message):
                progress((current, total), desc=message)
                return f"Progress: {current}/{total} - {message}"

            result = generator.generate_video(
                text, speaker, bg_color, keyword, api_key,
                enable_background_music=enable_music,
                music_volume_db=music_vol,
                add_call_to_action=enable_cta,
                progress_callback=update_progress
            )

            if result.get("success"):
                status = f"""✅ **Video Created Successfully!**

**Details:**
- Sentences: {result['sentence_count']}
- Format: {result['video_format']}
- **AI Extracted Keyword:** `{result.get('extracted_keyword', 'None')}`
- Background Music: {'Yes' if result['background_music'] else 'No'}
- CTA Slide: {'Yes' if result['cta_included'] else 'No'}
- Text Colors: {result['text_colors']}
- Video Source: {result['video_backgrounds']}
- Output: `{result['output_directory']}`
"""
                return result["audio_path"], result["video_path"], status, "✅ Complete!"

            error_msg = f"❌ **Error:** {result.get('error', 'Unknown error occurred')}"
            return None, None, error_msg, "❌ Failed"

        generate_button.click(
            fn=generate_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color_picker, pexels_keyword,
                    pexels_api_key, enable_music, music_volume, enable_cta],
            outputs=[audio_output, video_output, status_output, progress_bar]
        )

        gr.Markdown("""
        ---
        ### 📖 Instructions:
        1. Enter your text (each sentence becomes a slide)
        2. Leave "Manual Keyword Override" blank for **automatic AI extraction**
        3. Add Pexels API key for video backgrounds (or use local videos)
        4. Choose voice and configure settings
        5. Click "Generate Video with AI Keywords"

        ### 🤖 AI Keyword Extraction Features:
        - 🧠 Automatically extracts most relevant keywords using spaCy NLP
        - 🎯 Identifies nouns, proper nouns, adjectives, and named entities
        - 📊 Ranks keywords by frequency and relevance
        - 🌍 Recognizes locations, organizations, products, events
        - 🔄 Falls back to local videos if no keyword found
        - ✏️ Manual override option available

        ### 🎨 Video Features:
        - 🎥 Portrait format (9:16) for mobile/social media
        - 🌈 Random vibrant text colors per slide
        - 🎬 Automatic video backgrounds from Pexels
        - 🎵 Background music mixing
        - 📱 Call-to-action slide
        - ⚡ Parallel processing for speed

        ### 📦 Installation:
        ```bash
        pip install spacy
        python -m spacy download en_core_web_sm
        ```

        ### 🐛 All Bug Fixes Applied:
        - ✅ Automatic keyword extraction with spaCy
        - ✅ Fixed numpy import
        - ✅ Fixed font path type checking
        - ✅ Improved PIL fallback robustness
        """)

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)


if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\n❌ Missing required libraries. Please install:")
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests spacy")
        print("python -m spacy download en_core_web_sm")
    else:
        print("\n" + "=" * 80)
        print("🎥 PORTRAIT VIDEO GENERATOR WITH AI KEYWORD EXTRACTION")
        print("=" * 80)

        if SPACY_AVAILABLE:
            print("✅ spaCy NLP: Enabled")
        else:
            print("⚠️  spaCy NLP: Disabled (install with: pip install spacy)")

        print("\nStarting application...")
        print("=" * 80 + "\n")

        generator = TextToVideoGenerator()
        setup_ui(generator)