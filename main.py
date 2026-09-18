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
from core.media.identity import unique_slide_assets
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
from core.ai.stable_diffusion import (
    SDTurboGenerator,
    StableDiffusionManager,
    SD_AVAILABLE,
)
from core.visual import VisualProviderFactory
from core.media.manager import MediaManager
from core.media.youtube_audio import YouTubeAudioLibraryAPI
from core.tts.manager import TTSManager
from core.job_manager import JobManager
from core.utils.audio import improve_audio_quality, remove_metallic_artifacts
from core.utils.video import (
    get_video_duration,
    has_audio_stream,
    is_video_file,
    get_random_middle_frame,
    get_smart_thumbnail_frame,
    validate_background_asset,
    validate_slide,
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

torch.set_num_threads(4)
load_dotenv()

# Get shared config
config_instance = Config()
SUPPORTED_LANGUAGES = config_instance.SUPPORTED_LANGUAGES

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

# =============== MEDIA SOURCE CACHE ===============
_media_source_cache = {}

from core.video.ffmpeg_generator import FFmpegVideoGenerator


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
        visual_source: str = "Stock Media",
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
        # Reset per-video media selection state (MMR diversity + used URLs)
        try:
            if self.video_generator and self.video_generator.media_manager:
                self.video_generator.media_manager.reset_media_selection()
        except Exception:
            pass
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
                    # Scale video phase (0..total) into the 2nd half of overall progress (audio_tasks_count..audio_tasks_count*2)
                    audio_count = len(audio_tasks)
                    scaled_current = audio_count + int(
                        (current / max(total, 1)) * audio_count
                    )
                    progress_callback(
                        min(scaled_current, audio_count * 2), audio_count * 2, message
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
                        uploaded_path = Path(circle_upload_path)
                        circle_video_path = (
                            session_dir / f"uploaded_circle_{uploaded_path.name}"
                        )
                        shutil.copy(circle_upload_path, circle_video_path)
                    else:
                        print(
                            "⚠️ [Circle] Overlay enabled without an uploaded video; "
                            "continuing without the overlay."
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
                        audio_count = len(audio_tasks)
                        progress_callback(
                            audio_count * 2 - 1,
                            audio_count * 2,
                            f"Adding background music: {music_path.name}...",
                        )
                    print(f"🔊 [Mixing] Overlaying background music: {music_path.name}")
                    try:
                        audio_extract = (
                            self.config.TEMP_DIR
                            / f"extracted_{uuid.uuid4().hex[:8]}.wav"
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
                            text=True,
                            timeout=300,
                        )
                        voice_seg = AudioSegment.from_file(str(audio_extract))
                        music = (
                            AudioSegment.from_file(str(music_path)) + music_volume_db
                        )
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
                            self.config.TEMP_DIR
                            / f"with_music_{uuid.uuid4().hex[:8]}.mp4"
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
                            text=True,
                            timeout=300,
                        )
                        audio_extract.unlink(missing_ok=True)
                        mixed_audio.unlink(missing_ok=True)
                        video_temp_path.unlink(missing_ok=True)
                        video_temp_path = video_with_music
                        print("✅ [Mixing] Background music successfully overlaid.")
                    except Exception as eMix:
                        print(
                            f"⚠️ [Mixing] Failed to mix background music: {eMix}. Proceeding with original video."
                        )

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
job_manager = None

def _jobs_dataframe():
    try:
        jobs = job_manager.get_all_jobs(limit=50) if job_manager else []
    except Exception:
        jobs=[]
    import pandas as pd
    rows=[[j['job_id'][:8], j['status'], f"{j['progress']}%", j['created_at'][:19], (j.get('error_message') or '')[:60]] for j in jobs]
    return pd.DataFrame(rows, columns=["Job ID","Status","Progress","Created","Error"])

def _jobs_html():
    try:
        jobs = job_manager.get_all_jobs(limit=50) if job_manager else []
    except Exception:
        jobs=[]
    if not jobs:
        return "<p style='color:#888;text-align:center;padding:20px'>No jobs yet. Generate a video to create one.</p>"
    html="<table style='width:100%;border-collapse:collapse;font-size:13px'><tr style='background:#2a2a3e;color:#ccc'><th style='padding:6px;border:1px solid #444'>Job ID</th><th style='padding:6px;border:1px solid #444'>Status</th><th style='padding:6px;border:1px solid #444'>Progress</th><th style='padding:6px;border:1px solid #444'>Created</th></tr>"
    color={"queued":"#888","processing":"#4a9","completed":"#4CAF50","failed":"#e55","canceled":"#aa5"}
    for j in jobs:
        c=color.get(j['status'],"#ccc")
        html+=f"<tr><td style='padding:6px;border:1px solid #333'>{j['job_id'][:8]}</td><td style='padding:6px;border:1px solid #333;color:{c}'>{j['status']}</td><td style='padding:6px;border:1px solid #333'><div style='background:#333;border-radius:6px;height:14px'><div style='width:{j['progress']}%;background:{c};height:14px;border-radius:6px'></div></div>{j['progress']}%</td><td style='padding:6px;border:1px solid #333'>{j['created_at'][:19]}</td></tr>"
    html+="</table>"
    return html

def setup_ui(generator: TextToVideoGenerator):
    with gr.Blocks(
        title="AI Video Generator Pro", theme=gr.themes.Soft(primary_hue="blue")
    ) as demo:
        gr.Markdown("# 🎬 Shorts Generator")
        gr.Markdown(
            "Create stunning videos with multi-language TTS, auto-backgrounds, and dynamic overlays."
        )

        with gr.Tabs():
            with gr.Tab("Video Generation"):
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
                                visual_source_radio = gr.Radio(
                                    choices=["Stock Media", "AI Generated Images", "Mixed"],
                                    value="Stock Media",
                                    label="🖼️ Visual Source",
                                    info="Choose background source for scenes: Stock Media, AI Images (SD-Turbo), or Mixed",
                                )
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
            with gr.Tab("📋 Jobs") as jobs_tab:
                gr.Markdown("## 📋 Jobs — persistent queue (auto-refresh every 2s)")
                jobs_html = gr.HTML(value=_jobs_html())
                jobs_df = gr.Dataframe(value=_jobs_dataframe(), headers=["Job ID","Status","Progress","Created","Error"], interactive=False, wrap=True)
                with gr.Row():
                    job_id_dropdown = gr.Dropdown(label="Select Job (completed)", choices=[], value=None)
                    refresh_jobs_btn = gr.Button("🔄 Refresh", size="sm")
                with gr.Row():
                    download_btn = gr.Button("⬇️ Download Video", variant="primary")
                    retry_btn = gr.Button("🔁 Retry Failed", variant="secondary")
                    cancel_btn = gr.Button("⛔ Cancel", size="sm")
                download_file = gr.File(label="Download", interactive=False)
                job_status_msg = gr.Markdown("")
                jobs_timer = gr.Timer(value=2, active=True)

                def _refresh_jobs():
                    import pandas as pd
                    jobs = job_manager.get_all_jobs(limit=50) if job_manager else []
                    html = _jobs_html()
                    df = _jobs_dataframe()
                    choices = [j['job_id'] for j in jobs if j['status']=='completed']
                    labels = [f"{j['job_id'][:8]} — {j['status']} {j['progress']}%" for j in jobs]
                    # dropdown choices as job_ids
                    return html, df, gr.Dropdown(choices=choices)
                jobs_timer.tick(fn=_refresh_jobs, inputs=[], outputs=[jobs_html, jobs_df, job_id_dropdown])
                refresh_jobs_btn.click(fn=_refresh_jobs, inputs=[], outputs=[jobs_html, jobs_df, job_id_dropdown])

                def _download(job_id):
                    if not job_id or not job_manager:
                        return None
                    j = job_manager.get_job(job_id)
                    if not j or j['status']!='completed' or not j.get('video_path'):
                        return None
                    p = j['video_path']
                    import pathlib as _pl
                    return str(_pl.Path(p)) if _pl.Path(p).exists() else None
                download_btn.click(fn=_download, inputs=[job_id_dropdown], outputs=[download_file])

                def _retry(job_id):
                    if not job_id or not job_manager:
                        return "No job selected"
                    j = job_manager.get_job(job_id)
                    if not j:
                        return "Job not found"
                    new_id = job_manager.retry_job(job_id)
                    return f"Retried as {new_id[:8]}" if new_id else "Retry failed"
                retry_btn.click(fn=_retry, inputs=[job_id_dropdown], outputs=[job_status_msg])

                def _cancel(job_id):
                    if not job_id or not job_manager:
                        return "No job selected"
                    ok = job_manager.cancel_job(job_id)
                    return "Canceled" if ok else "Cannot cancel"
                cancel_btn.click(fn=_cancel, inputs=[job_id_dropdown], outputs=[job_status_msg])

        def _submit_job_wrapper(text, language, speaker, use_random, visual_source, media_source, keyword, selected_background_video_name, enable_music, music_select, music_vol, enable_circle, circle_upload_path, circle_diam, circle_border, circle_pos, overlay_shape_val, enable_intro, enable_cta, hide_text, export_fps_val, ai_model_val, ai_api_url_val, stress_level_val, use_snn_val, audio_only_val, normalize_audio_val, aspect_ratio_val, quality_val, enable_crossfade_val, pre_selected_videos, js_json, override_text, slide_data, entity):
            if not text or not text.strip():
                return "❌ Enter text first."
            # build params dict matching generate_video signature
            params = dict(text=text, language=language, speaker_id=speaker, pexels_keyword=keyword.strip() if keyword else None, preferred_media_source=media_source, visual_source=visual_source, selected_background_video_name=selected_background_video_name, pre_selected_videos=pre_selected_videos, enable_background_music=enable_music, music_selection=music_select, music_volume_db=music_vol, add_intro_slide=enable_intro, add_call_to_action=enable_cta, use_random_voices=use_random, enable_circle_overlay=enable_circle, circle_diameter=circle_diam, circle_position=circle_pos, circle_border_width=circle_border, circle_upload_path=circle_upload_path, hide_text=hide_text, export_fps=export_fps_val, overlay_shape=overlay_shape_val, ai_model=ai_model_val, ai_api_url=ai_api_url_val, stress_level=stress_level_val, use_snn=use_snn_val, audio_only=audio_only_val, normalize_audio=normalize_audio_val, aspect_ratio=aspect_ratio_val, quality=quality_val, enable_crossfade=enable_crossfade_val, entity=entity)
            # merge visual selections same as generate_wrapper does
            import json as _json
            final_pre = dict(pre_selected_videos) if isinstance(pre_selected_videos, dict) else {}
            media_mgr = generator.video_generator.media_manager
            if js_json and js_json.strip() not in ("", "{}") and slide_data:
                try:
                    visual_sel = _json.loads(js_json)
                    for k,v in visual_sel.items():
                        try:
                            s_num=int(k)
                        except: continue
                        if not v: continue
                        src=v[0]
                        kw=None
                        for sd in slide_data:
                            if sd["slide_num"]==s_num: kw=sd.get("keyword"); break
                        if kw and kw!="N/A":
                            try:
                                r=media_mgr.get_random_media([kw], preferred_source=src)
                                if isinstance(r,tuple): r=r[0]
                                if r: final_pre[s_num]=str(r)
                            except: pass
                except: pass
            if override_text and override_text.strip():
                for line in override_text.strip().split("\n"):
                    line=line.strip()
                    if not line or line.startswith("#"): continue
                    parts=line.split(":",1)
                    if len(parts)!=2: continue
                    try: s_num=int(parts[0].strip())
                    except: continue
                    act=parts[1].strip().lower()
                    if act=="gradient": final_pre[s_num]="__gradient__"
                    elif act=="skip": final_pre.pop(s_num,None)
                    elif act.startswith("/") or act.startswith("."):
                        from pathlib import Path as _P
                        if _P(act).exists(): final_pre[s_num]=str(_P(act))
            params["pre_selected_videos"]=final_pre
            # handle circle upload object
            cp=params.get("circle_upload_path")
            if cp is not None:
                if isinstance(cp,(list,tuple)): cp=cp[0]
                if hasattr(cp,"name"): cp=cp.name
                elif hasattr(cp,"path"): cp=cp.path
                params["circle_upload_path"]=cp if isinstance(cp,str) else None
            try:
                jid = job_manager.submit_job(params)
                return f"✅ Job queued: {jid[:8]} (ID: {jid}) — switch to Jobs tab to track."
            except Exception as e:
                return f"❌ Queue failed: {e}"

        # hook submit button to job queue (keep original generate_wrapper for direct preview)
        submit_job_btn = gr.Button("📋 Submit as Background Job", variant="secondary")
        job_submit_status = gr.Markdown("")
        submit_job_btn.click(fn=_submit_job_wrapper, inputs=[text_input, language_dropdown, speaker_dropdown, use_random_voices, visual_source_radio, media_source_dropdown, pexels_keyword, background_video_dropdown, enable_music, music_dropdown, music_volume, enable_circle, circle_upload, circle_diameter, circle_border_width, circle_position, overlay_shape, enable_intro, enable_cta, hide_text, export_fps, ai_model_dropdown, ai_api_url, stress_level, use_snn_checkbox, audio_only_checkbox, normalize_audio_checkbox, aspect_ratio_dropdown, quality_dropdown, enable_crossfade_checkbox, pre_selected_videos_state, js_selections, custom_selections_input, preview_data_state, entity_input], outputs=[job_submit_status])

        def generate_wrapper(
            text,
            language,
            speaker,
            use_random,
            visual_source,
            media_source,
            keyword,
            selected_background_video_name,
            enable_music,
            music_select,
            music_vol,
            enable_circle,
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
                visual_source=visual_source,
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
                visual_source_radio,
                media_source_dropdown,
                pexels_keyword,
                background_video_dropdown,
                enable_music,
                music_dropdown,
                music_volume,
                enable_circle,
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
                                    DB.log_clip_performance(
                                        media_url=str(result),
                                        keyword=kw,
                                        source=src,
                                        event_type="select",
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
                        if s_num in pre_selected:
                            DB.log_clip_performance(
                                media_url=pre_selected[s_num],
                                event_type="replace",
                            )
                        pass  # Not in dict = auto-select during generation
                    elif action.startswith("/") or action.startswith("."):
                        p = Path(action)
                        if p.exists():
                            if s_num in pre_selected:
                                DB.log_clip_performance(
                                    media_url=pre_selected[s_num],
                                    event_type="replace",
                                )
                            pre_selected[s_num] = str(p)
                            DB.log_clip_performance(
                                media_url=str(p),
                                event_type="select",
                            )
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
    print("  ✓ Upload-only circle overlay videos (PIP style)")
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

        print("\n💡 SETUP CHECKLIST:")
        print("  1. Set PEXELS_API_KEY in .env file")
        print("  2. Set PIXABAY_API_KEY in .env file")
        print("  3. Set GIPHY_API_KEY in .env file")
        print("  4. Run: ollama serve (for keyword extraction)")
        print("  5. Upload a video in the Overlays tab to use Picture-in-Picture")
        print("  6. Add background music to background_music/ folder")
        print("  7. Add logo image to background_images/ folder")

        print("\n🚀 STARTING SERVER...")
        print("   Access at: http://localhost:1603")
        print("   CLI mode: python main.py --cli --text 'Your script here'")
        print("=" * 80 + "\n")

        import core.job_manager as _jm
        globals()['job_manager'] = _jm.JobManager(config=cfg)
        job_manager.set_generator_factory(lambda: TextToVideoGenerator())
        print(f"📋 Job queue ready: {job_manager.max_workers} workers")
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
