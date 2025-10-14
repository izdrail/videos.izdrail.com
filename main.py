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
from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips, ColorClip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from num2words import num2words
import textwrap

# SpaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
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


class KeywordExtractor:
    def __init__(self):
        self.nlp = nlp if SPACY_AVAILABLE else None
        self.relevant_pos = {'NOUN', 'PROPN', 'ADJ'}
        self.exclude_words = {
            'thing', 'things', 'something', 'someone', 'way', 'time', 'day',
            'year', 'week', 'month', 'people', 'person', 'place', 'lot',
            'vodafone', 'apple', 'samsung', 'google', 'microsoft', 'amazon',
            'facebook', 'meta', 'tesla', 'nike', 'adidas', 'coca-cola', 'pepsi'
        }

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        if not self.nlp or not text.strip():
            return []
        doc = self.nlp(text.lower())
        candidates = []

        for token in doc:
            if (token.pos_ in self.relevant_pos and
                not token.is_stop and
                len(token.text) > 2 and
                token.text.isalpha() and
                token.text not in self.exclude_words):
                candidates.append(token.text)

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text.split()) <= 3:
                candidates.append(chunk_text)

        for ent in doc.ents:
            if ent.label_ in {'GPE', 'LOC', 'EVENT', 'WORK_OF_ART'}:
                candidates.append(ent.text.lower())

        keyword_freq = Counter(candidates)
        return [word for word, count in keyword_freq.most_common(top_n)]

    def get_best_keyword(self, text: str) -> Optional[str]:
        keywords = self.extract_keywords(text, top_n=5)
        if not keywords:
            return None
        single_words = [kw for kw in keywords if ' ' not in kw]
        return single_words[0] if single_words else keywords[0]


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
        self.CTA_MESSAGE = "Like, share, and subscribe to our channel!"
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
        self.MAX_PARALLEL_SLIDES = 4
        self.TEXT_COLORS = [
            'white', 'gold', 'hotpink', 'cyan', 'orange',
            'limegreen', 'deeppink', 'yellow', 'blueviolet',
            'orangered', 'springgreen', 'pink', 'lightblue',
            'peachpuff', 'lightgreen'
        ]


class PexelsAPI:
    def __init__(self):
        self.base_url = "https://api.pexels.com/videos"
        self.api_key = None
        self.download_cache = {}
        self.cache_limit = 20

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def _manage_cache(self):
        if len(self.download_cache) > self.cache_limit:
            items_to_remove = len(self.download_cache) - self.cache_limit
            for key in list(self.download_cache.keys())[:items_to_remove]:
                del self.download_cache[key]

    def search_videos(self, query: str, orientation: str = "portrait", size: str = "medium", per_page: int = 15,
                      page: int = 1) -> List[Dict]:
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
        self.keyword_extractor = KeywordExtractor()

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
                        font_str = str(font_file.resolve())
                        print(f"[Video] Using font: {font_str}")
                        return font_str
        print("[Video] No system fonts found, using fallback font name")
        return "DejaVuSans"

    def _try_keywords_for_video(self, sentence: str) -> Optional[Path]:
        candidates = self.keyword_extractor.extract_keywords(sentence, top_n=5)
        for kw in candidates:
            print(f"[Pexels] Trying keyword: '{kw}' for sentence")
            video_path = self.pexels.get_random_video(kw, self.config.VIDEO_SIZE)
            if video_path:
                print(f"[Pexels] Success with keyword: '{kw}'")
                return video_path
        return None

    def get_background_video(self, pexels_keyword: Optional[str] = None, sentence: Optional[str] = None) -> Optional[Path]:
        if pexels_keyword:
            video_path = self.pexels.get_random_video(pexels_keyword, self.config.VIDEO_SIZE)
            if video_path:
                return video_path
        if sentence:
            return self._try_keywords_for_video(sentence)
        video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV']
        video_files = []
        if self.config.VIDEOS_DIR.exists():
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.config.VIDEOS_DIR, ext)))
        if video_files:
            return Path(random.choice(video_files))
        return None

    def get_random_text_color(self) -> str:
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

    def _create_text_overlay_pil(self, text: str, duration: float, text_color: Optional[str] = None) -> ImageClip:
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

    def create_cta_slide(self, audio_path: Path, bg_color: Tuple[int, int, int] = (74, 144, 226),
                         pexels_keyword: Optional[str] = None) -> VideoFileClip:
        audio_clip = AudioFileClip(str(audio_path))
        duration_sec = audio_clip.duration
        background_video = self.get_background_video(pexels_keyword=pexels_keyword)
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
        final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
        return final_clip

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
                             pexels_keyword: Optional[str], slide_num: int) -> Optional[VideoFileClip]:
        try:
            audio_clip = AudioFileClip(str(audio_path))
            duration_sec = audio_clip.duration
            text_color = self.get_random_text_color()
            video_path = self.get_background_video(pexels_keyword=pexels_keyword, sentence=sentence)
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
                    text_clip = self._create_text_overlay_pil(sentence, duration_sec, text_color)
                    final_clip = CompositeVideoClip([video_clip, text_clip])
                    final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
                    return final_clip
                except Exception as e:
                    print(f"[Video] Slide {slide_num} video error: {e}")
                    video_clip = None
            bg_clip = ColorClip(size=self.config.VIDEO_SIZE, color=list(bg_color), duration=duration_sec)
            text_clip = self._create_text_overlay_pil(sentence, duration_sec, text_color)
            final_clip = CompositeVideoClip([bg_clip, text_clip])
            final_clip = final_clip.set_duration(duration_sec).set_audio(audio_clip)
            return final_clip
        except Exception as e:
            print(f"[Video] Slide {slide_num} error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_video_per_sentence(self, sentences: List[str], audio_paths: List[Path],
                                  sentence_keywords: List[Optional[str]],
                                  cta_audio_path: Optional[Path] = None,
                                  bg_color: Tuple[int, int, int] = (74, 144, 226),
                                  add_cta_slide: bool = True,
                                  progress_callback=None) -> Path:
        clips = []
        with ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_SLIDES) as executor:
            futures = {}
            for i, (sentence, audio_path, keyword) in enumerate(zip(sentences, audio_paths, sentence_keywords)):
                future = executor.submit(
                    self._create_single_slide,
                    sentence, audio_path, bg_color, keyword, i + 1
                )
                futures[future] = i
            slide_results = [None] * len(sentences)
            for future in as_completed(futures):
                i = futures[future]
                try:
                    video_clip = future.result()
                    if video_clip:
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
                cta_kw = sentence_keywords[0] if sentence_keywords else None
                cta_clip = self.create_cta_slide(cta_audio_path, bg_color=bg_color, pexels_keyword=cta_kw)
                clips.append(cta_clip)
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
                       use_random_voices: bool = False,
                       progress_callback=None) -> Dict:
        if not text or not text.strip():
            return {"error": "Text cannot be empty", "success": False}
        if len(text) > 10000:
            return {"error": "Text is too long (max 10,000 characters)", "success": False}
        if music_volume_db != self.config.MUSIC_CONFIG['music_volume_db']:
            self.config.MUSIC_CONFIG['music_volume_db'] = music_volume_db
        if pexels_api_key and pexels_api_key.strip():
            self.video_generator.pexels.set_api_key(pexels_api_key.strip())

        sentences = self.video_generator.split_into_sentences(text)
        print(f"[Info] Processing {len(sentences)} sentences")
        if len(sentences) > 100:
            return {"error": "Too many sentences (max 100)", "success": False}

        if pexels_keyword and pexels_keyword.strip():
            sentence_keywords = [pexels_keyword.strip()] * len(sentences)
            print(f"[NLP] Using manual keyword for all: '{pexels_keyword.strip()}'")
        else:
            sentence_keywords = []
            for sent in sentences:
                kw = self.keyword_extractor.get_best_keyword(sent)
                sentence_keywords.append(kw)
                print(f"[NLP] Sentence: '{sent[:50]}...' → Keyword: '{kw}'")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = self.config.OUTPUT_DIR / f"video_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        audio_paths = []
        cta_audio_path = None
        music_path = None
        try:
            if enable_background_music:
                music_path = self.video_generator.get_random_background_music()

            # Generate random voices for each sentence if enabled
            if use_random_voices:
                voices_for_sentences = [random.choice(self.available_voices) for _ in sentences]
                print(f"[Voice] Using random voices per sentence")
                for i, voice in enumerate(voices_for_sentences):
                    print(f"[Voice] Sentence {i+1}: {voice}")
            else:
                voices_for_sentences = [speaker_id] * len(sentences)

            for i, (sentence, voice) in enumerate(zip(sentences, voices_for_sentences)):
                if progress_callback:
                    voice_name = voice if len(voice) < 30 else voice[:27] + "..."
                    progress_callback(i + 1, len(sentences) * 2, f"Audio {i + 1}/{len(sentences)} ({voice_name})")
                audio_path = self.tts_manager.generate_speech(sentence, voice)
                audio_paths.append(audio_path)

            # Use the last voice or the selected speaker_id for CTA
            if add_call_to_action:
                cta_voice = speaker_id if not use_random_voices else voices_for_sentences[-1]
                cta_audio_path = self.tts_manager.generate_speech(self.config.CTA_MESSAGE, cta_voice)

            def video_progress(current, total, message):
                if progress_callback:
                    progress_callback(len(sentences) + current, len(sentences) * 2, message)

            video_temp_path = self.video_generator.create_video_per_sentence(
                sentences=sentences,
                audio_paths=audio_paths,
                sentence_keywords=sentence_keywords,
                cta_audio_path=cta_audio_path,
                bg_color=bg_color,
                add_cta_slide=add_call_to_action,
                progress_callback=video_progress
            )

            # Apply background music to final audio track
            final_video_clip = VideoFileClip(str(video_temp_path))
            if enable_background_music and music_path and music_path.exists():
                voice_audio = final_video_clip.audio
                music = AudioSegment.from_file(str(music_path))
                voice_segment = AudioSegment.from_file(str(video_temp_path), format="mp4")
                cfg = self.config.MUSIC_CONFIG
                voice = voice_segment
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
                preset='medium',
                threads=4
            )

            # Export audio only
            audio_only_path = session_dir / f"audio_{timestamp}.mp3"
            if mixed_audio_path:
                mixed_audio_segment = AudioSegment.from_file(str(mixed_audio_path))
            else:
                original_audio = AudioSegment.from_file(str(video_temp_path), format="mp4")
                mixed_audio_segment = original_audio
            mixed_audio_segment.export(str(audio_only_path), format="mp3", bitrate="192k")

            # Cleanup
            try:
                video_temp_path.unlink(missing_ok=True)
                if mixed_audio_path:
                    mixed_audio_path.unlink(missing_ok=True)
            except:
                pass

            return {
                "success": True,
                "audio_path": str(audio_only_path),
                "video_path": str(video_final_path),
                "output_directory": str(session_dir),
                "sentence_count": len(sentences),
                "background_music": enable_background_music and music_path is not None,
                "cta_included": add_call_to_action,
                "video_format": "9:16 Portrait (1080x1920)",
                "text_colors": "Random vibrant colors",
                "video_backgrounds": "Pexels API" if pexels_api_key else "Local videos",
                "random_voices": use_random_voices,
                "voices_used": voices_for_sentences if use_random_voices else None,
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


def setup_ui(generator: TextToVideoGenerator):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   title="Portrait Video Generator with Random Voices") as demo:
        gr.Markdown("# 🎥 Portrait Video Generator (9:16) with Random Voices Per Sentence")
        gr.Markdown("Each sentence gets its own keyword, background, voice, and **perfect audio sync**!")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter Your Text",
                    placeholder="Each sentence → unique keyword & voice → perfect sync!",
                    lines=8
                )
                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        label="Voice (used when Random Voices disabled)",
                        choices=generator.available_voices,
                        value=generator.config.STANDARD_VOICE_NAME
                    )
                    bg_color_picker = gr.ColorPicker(
                        label="Background Color (Fallback)",
                        value="#4A90E2"
                    )
                enable_cta = gr.Checkbox(label="Add Call-to-Action Slide", value=True)
                enable_music = gr.Checkbox(label="Enable Background Music", value=True)
                use_random_voices = gr.Checkbox(
                    label="🎙️ Use Random Voice Per Sentence",
                    value=False,
                    info="Each sentence will use a different random voice from available voices"
                )
                music_volume = gr.Slider(-40, -5, value=-15, step=1, label="Music Volume (dB)")
                with gr.Row():
                    pexels_keyword = gr.Textbox(
                        label="Manual Keyword Override (Optional)",
                        placeholder="Leave empty for per-sentence AI extraction"
                    )
                    pexels_api_key = gr.Textbox(
                        label="Pexels API Key",
                        type="password",
                        placeholder="Get free API key from pexels.com/api"
                    )
                progress_bar = gr.Textbox(label="Progress", value="Ready...", interactive=False)
                generate_button = gr.Button("🎥 Generate Video", variant="primary", size="lg")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown()

        def generate_wrapper(text, speaker, bg_hex, keyword, api_key,
                             enable_music, music_vol, enable_cta, random_voices, progress=gr.Progress()):
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
                use_random_voices=random_voices,
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

                status = f"""✅ **Video Created Successfully!**
**Details:**
- Sentences: {result['sentence_count']}
- Format: {result['video_format']}
- Background Music: {'Yes' if result['background_music'] else 'No'}
- CTA Slide: {'Yes' if result['cta_included'] else 'No'}
- Random Voices: {'Yes' if result.get('random_voices') else 'No'}{voices_info}
- Output: `{result['output_directory']}`
"""
                return result["audio_path"], result["video_path"], status, "✅ Complete!"
            error_msg = f"❌ **Error:** {result.get('error', 'Unknown error occurred')}"
            return None, None, error_msg, "❌ Failed"

        generate_button.click(
            fn=generate_wrapper,
            inputs=[text_input, speaker_dropdown, bg_color_picker, pexels_keyword,
                    pexels_api_key, enable_music, music_volume, enable_cta, use_random_voices],
            outputs=[audio_output, video_output, status_output, progress_bar]
        )
        gr.Markdown("""
        ---
        ### ✅ Perfect Sync Guaranteed
        - Each slide uses **its own TTS audio** → no drift
        - Background music added **after video assembly**
        - Per-sentence keywords for relevant visuals
        - **NEW**: Random voice per sentence for maximum variety!

        ### 🎙️ Available Voices: {}
        Add more voices by creating folders in `voice_samples/` with a `reference.wav` file.
        """.format(len(generator.available_voices)))

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)


if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("\n❌ Missing required libraries. Please install:")
        print("pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests spacy")
        print("python -m spacy download en_core_web_md")
    else:
        print("\n" + "=" * 80)
        print("🎥 PORTRAIT VIDEO GENERATOR WITH RANDOM VOICES & PERFECT SYNC")
        print("=" * 80)
        if SPACY_AVAILABLE:
            print("✅ spaCy NLP: Enabled")
        else:
            print("⚠️  spaCy NLP: Disabled (install with: pip install spacy)")
        print("\nStarting application...")
        print("=" * 80 + "\n")
        generator = TextToVideoGenerator()
        print(f"🎙️ Available voices: {len(generator.available_voices)}")
        for voice in generator.available_voices:
            print(f"   - {voice}")
        print()
        setup_ui(generator)