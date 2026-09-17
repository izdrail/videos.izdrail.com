# 🎥 Portrait Video Generator — `videos.izdrail.com`

> **Text → Viral 9:16 Video Pipeline.** Converts arbitrary text / topic into a fully synchronized portrait video (1080×1920 default) with per-sentence AI TTS, stock/AI visuals, background music, overlays, and FFmpeg composition. Single entrypoint `main.py` with Gradio UI; heavy use of caching, parallelism, and graceful AI fallbacks.
>
> **Author:** Stefan Bogdan · **Stack:** Python 3.11, PyTorch, Coqui XTTS v2 / Kokoro-82M / gTTS / MMS-TTS, MoviePy/FFmpeg, Gradio 5-7, spaCy, Diffusers SD-Turbo, Pexels/Pixabay/Giphy/YouTube/Openverse/Wikimedia/Internet Archive/Unsplash, Ollama (gemma4:e2b), SQLite WAL, Docker + Supervisor.

---

## 1. Purpose & Product Definition

- **Input:** Free-form text (pasted), topic prompt (AI script generation), or sentence list. Language `en|zh|es|fr|it|pt|hi|ja|ar|ro|auto` (auto via `langdetect`).
- **Output:** `output/video_<timestamp>_<lang>/` containing :
  - `final_video.mp4` (H.264 + AAC, `yuv420p`, `+faststart`, `-shortest`)
  - `audio_<stamp>.mp3` (merged sentence audios + optional intro/CTA + music ducked)
  - `thumbnail_<stamp>.jpg` (smart middle-frame via `core/utils/video.py`)
  - Per-sentence `slide_<n>.mp4` temp files (concatenated via `core/utils/video.concat_videos`)
- **Positioning:** TikTok / YouTube Shorts / Instagram Reels ready. Generates viral title/description/hashtags via Ollama.
- **Non-goals (current):** Real-time streaming, collaborative editing, cloud queue.

---

## 2. High-Level Architecture

```
User Text / Topic
      │
      ├─► [NLP] KeywordExtractor ──► OllamaKeywordExtractor ──► OllamaClient (retry/cache/fallback)
      │         │                    └─ theme, mood, social descriptions, script generation
      │         ├─► spaCy en_core_web_md|sm (NER, noun_chunks, POS filter)
      │         ├─► NeuronExtractor + BrainSimulator (SNN engagement)
      │         ├─► EntityHandler (entity-aware re-ranking)
      │         └─► Semantic dedup (spaCy vectors or sentence-transformers all-MiniLM-L6-v2)
      │
      ├─► [TTS] TTSManager (Kokoro → XTTS v2 → gTTS → MMS)
      │         └─► GenerationDB tts_cache (SHA256 text+voice+lang+speed) + audio_quality (normalize/filter/compress)
      │
      ├─► [VISUAL] VisualProviderFactory ──► StockProvider | AiProvider (SD-Turbo) | MixedProvider
      │                │                           └─ SDTurboGenerator (diffusers 0.31.0, 1-step turbo)
      │                └─► MediaManager ──► Pexels, Pixabay, Giphy, YouTube, Openverse, Wikimedia, IA, Unsplash
      │                                      └─ File cache + availability check + random selection
      │
      └─► [RENDER] FFmpegVideoGenerator
                   ├─ get_background_video() 4-tier fallback chain
                   ├─ _create_slide_with_ffmpeg() per-slide FFmpeg graph
                   ├─ ThreadPoolExecutor pools (NLP 20, TTS 4, MEDIA 10, RENDER 4)
                   └─ create_final_video() concat + music mix + thumbnail
                            └─ Gradio UI (main.py) progress callbacks
```

**Key design principles:** SOLID + DRY — shared `core/` (config, database, utils, media, nlp, tts, ai, visual). Single `Config` source of truth. Single `GenerationDB` (WAL). PyTorch compat layer.

---

## 3. Repository Layout (authoritative)

```
.
├── main.py                          # SOLE entrypoint: FFmpegVideoGenerator + TTSManager + Gradio app (port 1603)
├── core/
│   ├── config.py                    # Config class — all env vars, paths, presets, validation
│   ├── database.py                  # GenerationDB (tts_cache, video_logs, keyword_selection_audit) WAL + thread lock
│   ├── __init__.py
│   ├── ai/
│   │   ├── stable_diffusion.py      # SDTurboGenerator, StableDiffusionManager, SD_AVAILABLE flag
│   │   └── prompt_generator.py      # SD prompt enrichment
│   ├── media/
│   │   ├── base.py                  # BaseMediaProvider ABC
│   │   ├── manager.py               # MediaManager orchestrator, get_random_media(), is_keyword_available()
│   │   ├── pexels.py / pixabay.py / giphy.py / youtube.py
│   │   ├── unsplash.py / openverse.py / wikimedia.py / internet_archive.py
│   │   ├── searxng.py / youtube_audio.py  # YouTube Free Audio Library
│   │   └── __init__.py
│   ├── nlp/
│   │   ├── ollama_client.py         # OllamaClient (retry, LRU cache, stats, post_or_fallback)
│   │   ├── keyword_extractor.py     # OllamaKeywordExtractor + KeywordExtractor (orchestrator, beam search, audit)
│   │   ├── neuron_extractor.py      # Heuristic + SNN scoring
│   │   ├── brain_simulator.py       # Spiking NN biological scoring (optional)
│   │   ├── entity.py                # EntityHandler parse/rank
│   │   └── __init__.py
│   ├── tts/
│   │   ├── manager.py               # TTSManager: Kokoro > XTTS > gTTS > MMS, voice cloning, speed/stress
│   │   └── __init__.py
│   ├── visual/
│   │   ├── provider.py              # VisualProvider ABC, VisualAsset
│   │   ├── asset.py                 # VisualAsset dataclass (Path, type, metadata)
│   │   ├── factory.py               # VisualProviderFactory.create(source_type)
│   │   ├── stock_provider.py        # Delegates to MediaManager
│   │   ├── ai_provider.py           # Delegates to SDTurboGenerator
│   │   ├── mixed_provider.py        # Ratio-based stock/AI switch
│   │   └── __init__.py
│   └── utils/
│       ├── audio.py                 # improve_audio_quality, remove_metallic_artifacts, normalize, filters
│       ├── video.py                 # get_video_duration, has_audio_stream, is_video_file, get_random_middle_frame, get_smart_thumbnail_frame, validate_background_asset, validate_slide, concat_videos
│       ├── gpu.py                   # Device detection
│       ├── pytorch_compat.py        # torch 2.6+ secure loading shim
│       └── __init__.py
├── background_videos/               # Local fallback stock (gitignored content)
├── background_images/               # Local fallback images
├── background_music/                # .mp3/.wav/.m4a — get_available_music_files()
├── voice_samples/<voice>/reference.wav  # XTTS cloning samples
├── circle_overlays/*.mp4            # PiP / fullscreen fallback overlays (girl-1.mp4 etc.)
├── video-overlays/                  # Additional overlays
├── intro/intro-tv-noise.mp4         # FIXED intro background (never deleted) — intro slides force this
├── temp/                            # All ephemeral: audio_cache/, slide temps, gradients, masks
├── output/                          # Final renders (timestamped subdirs)
├── backup_output/                   # Backup copies via Config.get_backup_path()
├── cache/images/                    # SD-Turbo image cache (IMAGE_GENERATION_CACHE_DIR)
├── models/sd-turbo/                 # Local SD model (SD_MODEL_DIR)
├── tests/
│   ├── test_visual_providers.py / test_media_providers.py / test_video_pipeline.py
│   ├── test_video_progress_and_concat.py / test_sd_turbo.py / test_keyword_selection_overhaul.py
│   ├── test_ollama_resilience.py / test_ollama_script.py / test_parallel_pipeline.py
│   ├── test_youtube_audio.py / test_kokoro_load.py / test_mms_romanian.py / test_neuron_keyword_search.py
│   └── verify_*.py
├── docs/media_providers.md
├── scripts/
│   ├── download_sd_turbo.py / generate_tiktok_patent_video.py / purge_cache.py / clean_project.sh
├── Dockerfile                       # python:3.11, ffmpeg, espeak-ng, venv, spacy models, TTS+Kokoro pre-download
├── docker-compose.yaml              # service videos.izdrail.com, ports 1603+1604, GPU, volumes
├── docker/supervisord.conf          # Runs Gradio app(s)
├── makefile                         # build, dev, prod, down, rebuild, ssh, publish, test-brain
├── requirements.txt                 # Pinned torch 2.3.1, transformers <4.49, diffusers 0.31.0, kokoro 0.7.16, etc.
├── download_model.py / download_kokoro.py / download_sd_model.py
├── .env / .env.example
├── generation_cache.db (+ -wal/-shm)  # SQLite cache (gitignored ideally)
└── readme.md                        # This file
```

---

## 4. Core Modules — Deep Dive

### 4.1 `core/config.py` — `Config`

- **Paths created on init:** `VOICE_SAMPLES_DIR`, `VIDEOS_DIR`, `MUSIC_DIR`, `IMAGES_DIR`, `TEMP_DIR`, `OUTPUT_DIR`, `BACKUP_OUTPUT_DIR`, `IMAGE_GENERATION_CACHE_DIR`, `VIDEO_OVERLAYS_DIR`, `CIRCLE_OVERLAYS_DIR`. Alias `BACKGROUND_VIDEOS_DIR = VIDEOS_DIR`. `TEMP_AUDIO_DIR = TEMP_DIR/audio_cache`.
- **Device:** `DEVICE="cpu"` (override via `utils/gpu.py` detection; Dockerfile reserves GPU).
- **Video defaults:** `VIDEO_WIDTH=1080, VIDEO_HEIGHT=1920, VIDEO_SIZE=(1080,1920), FPS=30, VIDEO_PRESET="ultrafast", VIDEO_CRF=28, VIDEO_CODEC="libx264", AUDIO_CODEC="aac"`. `MIXED_MODE_SD_RATIO=0.2`, `SENTENCE_MERGE_ENABLED=False`, `MAX_PARALLEL_SLIDES=4`, `MIN_IMAGE_DURATION=10.0`.
- **Worker pools:** `WORKER_POOL_NLP=20, WORKER_POOL_TTS=4, WORKER_POOL_MEDIA=10, WORKER_POOL_RENDERING=4`.
- **Text:** `TEXT_SIZE_CONFIG={font_size:50, line_spacing:1.2, max_width:900, bottom_margin:150}`. `CIRCLE_OVERLAY_CONFIG`, `LOGO_CONFIG{position:top-right, margin:20}`, `MUSIC_CONFIG`, `TRANSITION_CONFIG`.
- **Languages:** `SUPPORTED_LANGUAGES` dict with `name/code/tts_code/kokoro_code`. `ar`/`ro` have `kokoro_code=None` (fallback to gTTS/MMS). `auto` = auto-detect. `STANDARD_VOICE_NAME="sexy"`.
- **Keyword/entity:** `KEYWORD_DEBUG_MODE`, `ENTITY_ENABLED`, `ENTITY_WEIGHT_BOOST=1.5`, `KEYWORD_HISTORY_LIMIT=200`, `KEYWORD_EMBEDDING_MODEL=spacy|sentence_transformer`, `SEMANTIC_DUP_THRESHOLD=0.8`, `KEYWORD_CONTEXT_ENRICH=true`.
- **SD-Turbo:** `IMAGE_GENERATION_ENABLED`, `IMAGE_GENERATION_MODEL=stabilityai/sd-turbo`, `IMAGE_GENERATION_DEVICE=auto`, `STEPS=1`, `GUIDANCE_SCALE=0.0`, `WIDTH/HEIGHT=512`. `VISUAL_SOURCE=stock|ai|mixed`, `MIXED_MODE_IMAGE_RATIO=0.5`.
- **Ollama (resilience):** `AI_MODEL=gemma4:e2b`, `OLLAMA_API_URL=https://ai.izdrail.com/api/generate`, `OLLAMA_MAX_RETRIES=3`, `OLLAMA_RETRY_BASE_DELAY=1.0`, `OLLAMA_TIMEOUT=180`, `OLLAMA_CACHE_MAX_SIZE=512`, `OLLAMA_FALLBACK_KEYWORDS=abstract,motion,...`, `OLLAMA_FALLBACK_MOOD=Cinematic`, `OLLAMA_FALLBACK_SCRIPT=...`.
- **Other env:** `UNSPLASH_APP_ID/ACCESS_KEY/SECRET_KEY`, `PEXELS_API_KEY`, `YOUTUBE_AUDIO_API_URL`, `SPACY_API_URL`.
- **Presets:** `ASPECT_RATIOS{9:16 1080x1920, 16:9 1920x1080, 1:1, 4:5}`, `QUALITY_PRESETS{Low ultrafast crf35 fps24, Medium veryfast 28 30, High medium 22 30, Ultra slow 18 60}`.
- **Helpers:** `validate()→warnings`, `get_aspect_ratio()`, `get_quality()`, `get_temp_audio_file()`, `get_backup_path()`.
- **Messages:** `INTRO_MESSAGES` / `CTA_MESSAGES` per language.

### 4.2 `core/database.py` — `GenerationDB` / `DB` singleton

- **Init:** `generation_cache.db`, `threading.Lock`, `PRAGMA journal_mode=WAL`, auto-migration (drops `tts_cache` if `speaker_id`/`voice_id` missing).
- **Tables:**
  - `tts_cache(text_hash PK, speaker_id, voice_id, language, audio_path, created_at)` — `text_hash=SHA256(f"{text}_{identifier}_{language}_{speed}")`, dual-column lookup for XTTS compat.
  - `video_logs(id, input_hash UNIQUE, video_path, audio_path, output_dir, sentence_count, created_at, processing_time, system_stats)` — `input_hash=SHA256(sorted input_params excluding progress_callback)`.
  - `keyword_selection_audit(id, timestamp, script_id, sentence_idx, keyword, context_preview[:200], signals_json, decision_score, was_used)` — calibration audit, best-effort non-blocking.
- **API:** `get_cached_tts()`, `save_tts()`, `get_cached_video()`, `save_video()`, `log_keyword_selection()`.

### 4.3 `core/nlp/ollama_client.py` — `OllamaClient`

- **Constructor:** `model, url (env OLLAMA_API_URL), max_retries, base_delay, timeout, cache_max_size`, `_cache dict`, `_stats{total_calls,cache_hits,retries,failures}`.
- **post():** SHA256 cache key from sorted JSON payload, LRU eviction (FIFO oldest), retries on `ConnectionError|Timeout` + `5xx`, no retry on `4xx`, exponential backoff `base_delay*2^(attempt-1)`, returns parsed JSON or `None`.
- **post_or_fallback():** Returns `fallback` on `None`. `stats`, `clear_cache()`, `set_url()` (clears cache), `set_model()`, `_make_cache_key()`, `_put_cache()`.

### 4.4 `core/nlp/keyword_extractor.py` — `OllamaKeywordExtractor` + `KeywordExtractor`

- **OllamaKeywordExtractor:** `extract_keywords(text, top_n, language, theme)` → prompt with stock-footage rules, `temperature 0.3 num_predict 64`, fallback to `OLLAMA_FALLBACK_KEYWORDS`; `extract_theme()` single phrase; `generate_social_media_descriptions()` title/desc/hashtags/TikTok; `extract_mood_keyword()` → Epic/Relaxing/…; `generate_script_from_text()` + `generate_topic_script()`; `fetch_models_static()` / `get_available_models()` via `/api/tags`.
- **KeywordExtractor (orchestrator):**
  - `__init__`: loads `en_core_web_md`→`en_core_web_sm`→None, `relevant_pos{NOUN,PROPN,ADJ}`, `exclude_words{thing,time,people,…}`, `used_keywords set`, `used_embeddings list`, `semantic_threshold 0.8`, `EntityHandler`, `debug_mode`, `selection_history[limit 200]`, `embedding_model`.
  - `extract_keywords(text, top_n, language, use_neuron_ai, use_snn, theme, entity, entity_type)`: collects candidates (`_extract_spacy_local` NER→noun_chunks→POS, entity keywords appended), Ollama supplement if `<2` candidates, neural vs heuristic ranking, entity re-prioritisation, `_record_audit()`.
  - Semantic: `_embedding()` / `_st_embedding(all-MiniLM-L6-v2)`, `is_semantically_unique()`, `add_used_keyword()`, `get_best_unique_keyword()`, `clear_used()`.
  - Coherence: `_keyword_engagement()` (NeuronExtractor local signals + optional SNN), `_keyword_similarity()`, `optimize_keyword_sequence()` beam search (`beam_width 4`, `coherence_weight 0.5`, `engagement_weight 1.0`, dup penalty 5.0), `_unique_against()`, `evaluate_keyword_sequence_coherence()`.
  - Ranking: `rank_keywords()` stock_categories word-boundary matching, `generate_fallback_keywords()` category_map, `sanitize_keyword()`, `enrich_keyword_context(max_words 3)` polysemy disambiguation.

### 4.5 `core/nlp/neuron_extractor.py` + `brain_simulator.py` + `entity.py`

- Neuron signals → decision score; BrainSimulator optional SNN biological score when `use_snn=True`. EntityHandler parses `entity`/`entity_type` and boosts matching keywords.

### 4.6 `core/media/*` + `core/media/manager.py`

- Providers implement `BaseMediaProvider.search(keyword)→List[Path]` with local file cache. Manager tries `search_keywords` in order, `get_random_media(search_keywords, preferred_source, context, use_snn, return_keyword, theme, entity)` → `(Path, used_keyword)`. `is_keyword_available()` for fallback availability probing. Supported: Pexels (requires `PEXELS_API_KEY`), Pixabay, Giphy, YouTube (yt-dlp), Unsplash (`UNSPLASH_ACCESS_KEY`), Openverse, Wikimedia, Internet Archive, SearxNG, YouTube Audio Library.

### 4.7 `core/visual/*` — Strategy pattern

- `VisualProvider` ABC + `VisualAsset(path, type, metadata)` + `VisualProviderFactory.create(source_type, config, sd_generator, media_manager, background_video_fetcher)`.
- `StockProvider` → `MediaManager`; `AiProvider` → `SDTurboGenerator.generate_image(prompt, keyword, size)`; `MixedProvider` → ratio `MIXED_MODE_IMAGE_RATIO` / `MIXED_MODE_SD_RATIO`.

### 4.8 `core/ai/stable_diffusion.py`

- `SDTurboGenerator` wraps `diffusers` `AutoPipelineForText2Image` with `stabilityai/sd-turbo`, `steps=1`, `guidance 0.0`, device auto, caching under `cache/images`. `SD_AVAILABLE` flag gates init in `FFmpegVideoGenerator`.

### 4.9 `core/tts/manager.py` — `TTSManager`

- **Priority:** Kokoro-82M (fast, multilingual, `kokoro_code` map) → Coqui XTTS v2 (voice cloning via `voice_samples/<name>/reference.wav`, `STANDARD_VOICE_NAME=sexy`) → gTTS → MMS-TTS (for `ar`/`ro`). Per-sentence speed/stress, language-aware routing, `GenerationDB` cache, silence trimming, `improve_audio_quality` pipeline.

### 4.10 `core/utils/*`

- `audio.py`: `improve_audio_quality` (loudness normalize, low/high-pass, dynamic compression, fade), `remove_metallic_artifacts`.
- `video.py`: `get_video_duration(ffprobe)`, `has_audio_stream`, `is_video_file`, `get_random_middle_frame`, `get_smart_thumbnail_frame`, `validate_background_asset` / `validate_slide`, `concat_videos` (FFmpeg concat demuxer, optional crossfade).
- `gpu.py`: CUDA availability, `pytorch_compat.py`: `torch.load` weights_only shim.

### 4.11 `main.py` — `FFmpegVideoGenerator` + Gradio

- **FFmpegVideoGenerator:**
  - `__init__(config, keyword_extractor)`: discovers fonts (DejaVuSans-Bold etc.), `MediaManager`, `KeywordExtractor`, logo from `IMAGES_DIR`, `SDTurboGenerator` if `SD_AVAILABLE`.
  - `LARAVEL_BG_GRADIENT (#0f172a→#2a1030)`, `LARAVEL_ACCENT_GRADIENT (#7c3aed→#ec4899)`, `create_gradient_image(size, colors, direction)` Pillow composite.
  - `split_into_sentences(text)` regex paragraphs + protected `Dr./Mr./Mrs./Ms./X.` placeholders + `[.!?。！？]` split, ensures terminal punctuation, logs counts.
  - `_find_logo()`, `_clean_text()` strips `[\d+ levels](…)` tags, `_discover_fonts()`, `get_available_music_files()` / `get_music_by_name()`.
  - **`get_background_video(keyword, sentence, language, preferred_source, use_snn, theme, entity, script_id, sentence_idx, candidate_keywords)` 4-tier fallback:** (1) enriched keyword via `sanitize+enrich_keyword_context` → `media_manager.get_random_media(return_keyword=True)`; (2) availability fallback: check `candidate_keywords[:3]` via `is_keyword_available()` then generic `generate_fallback_keywords()`; (3) `SDTurboGenerator.generate_image()` if `sd_manager`; (4) `get_circle_overlay_video()`; else gradient. Every decision → `DB.log_keyword_selection()`.
  - Text PNGs: `_create_text_overlay_png()` (wraps 35 cols, font scaling, gradient text via mask, stroke shadow, bottom_margin), `_create_intro_text_png()` / `_create_cta_text_png()` similarly with LARAVEL gradient.
  - `_generate_overlay_mask(shape, diameter)` Pillow high-res 2× then LANCZOS downscale: Circle/Square/Rounded Rectangle/Star(5-point polygon).
  - **`_create_slide_with_ffmpeg(sentence, audio_path, video_path, output_path, slide_num, is_intro, is_cta, circle_video, circle_config, language, hide_text, export_fps, overlay_shape, video_width/height)`** — Pre-flight validates audio (silent 1.5s fallback if missing/empty), forces `INTRO_VIDEO_PATH=intro/intro-tv-noise.mp4` for intro, validates `video_path`/`circle_video`. Builds FFmpeg filter graph:
    - Split Screen (if `overlay_shape=="Split Screen"` and circle exists): top `scale+ crop` background + bottom circle `scale+crop`, `vstack`, `fps`, `trim=duration`, `setpts`.
    - Normal: image (`-loop 1`) vs video (`-stream_loop -1`) → `scale:force_original_aspect_ratio=decrease, pad`, `fps`, `trim`, `setpts`; fallback to circle-as-fullscreen or branded gradient PNG.
    - Dimming: `format=rgba,colorchannelmixer=aa=0.6`.
    - Text overlay: `-loop 1` PNG → `overlay=0:0`.
    - Logo: `scale=150:150` → `overlay=position`.
    - Circle PiP: mask PNG → `alphaextract` → `alphamerge` → `overlay=position` (`top-left|top-right|bottom-left|bottom-right|center` diameter 300 default).
    - Audio: `-map [final]:v -map audio_idx:a -c:v libx264 -preset <preset> -crf <crf> -pix_fmt yuv420p -c:a aac -b:a 192k -r fps -shortest -movflags +faststart`. Timeout 600s, captures stderr.
  - **`create_final_video(sentences, audio_paths, keywords, intro_audio, cta_audio, music_path, music_volume_db, circle_video, circle_config, circle_selection, language, preferred_media_source, selected_background_video, pre_selected_videos, hide_text, export_fps, overlay_shape, intro_text, use_snn, enable_crossfade, crossfade_duration, progress_callback, theme, entity, script_id, candidate_map)`**:
    - Stage 1: Assemble `slides_data` (content + intro inserted at random idx 2-5 + CTA appended).
    - Stage 2: Parallel fetch via `VisualProviderFactory` + `ThreadPoolExecutor(WORKER_POOL_MEDIA)` respecting `pre_selected_videos["__gradient__"]` and `selected_background_video`; emergency fallback `["cityscape","abstract","office"]`.
    - Stage 3: Parallel render via `ThreadPoolExecutor(WORKER_POOL_RENDERING)` calling `_create_slide_with_ffmpeg` per slide; progress callbacks `fetching (completed/total*2)` + `rendering`.
    - Stage 4: Concatenate slides via `concat_videos` (or FFmpeg complex with crossfade), mix background music (`music_volume_db -20`, ducking, fade 3s), generate thumbnail, write `output/video_<stamp>_<lang>/`, log to `video_logs`.

- **Gradio UI (bottom of main.py):** `gr.Blocks` with inputs: text/topic, language dropdown, voice selector (from `VOICE_SAMPLES_DIR`), random voice toggle, speed/stress sliders, visual source `stock|ai|mixed`, preferred media source, theme/entity fields, background video picker, circle overlay shape `Circle|Square|Rectangle|Star|Split Screen`, hide text, aspect ratio `ASPECT_RATIOS`, quality preset `QUALITY_PRESETS`, music file + volume, intro/CTA toggles, `use_snn`, crossfade. Live `progress_callback` bar, video + audio preview, download, viral description textbox. Launch `app.launch(server_name="0.0.0.0", server_port=1603, share=False)`.

---

## 5. End-to-End Pipeline (what happens on Generate click)

1. **Normalize & split:** `split_into_sentences()` → `List[str]`; optional topic → `ollama_extractor.generate_topic_script()` → split.
2. **Language:** `detect_lang` if `auto`, else dropdown.
3. **Theme/entity:** `keyword_extractor.extract_theme(full_text)` once; `entity`/`entity_type` from UI.
4. **Candidates per sentence:** `ThreadPoolExecutor(WORKER_POOL_NLP)` → `extract_keywords(sentence, top_n=5, theme, entity, use_snn)` → `candidate_map[idx]=List[str]`.
5. **Sequence optimization:** `optimize_keyword_sequence(sentences, candidate_map, theme, use_snn, beam_width=4)` → `chosen_keywords{idx:kw}` (beam search coherence + engagement + dup penalty).
6. **TTS:** `ThreadPoolExecutor(WORKER_POOL_TTS)` → `TTSManager.synthesize(sentence, language, voice_id, speed)` → cached `Path` in `TEMP_AUDIO_DIR` + `DB.save_tts`. Intro/CTA audios similarly from `INTRO_MESSAGES`/`CTA_MESSAGES`.
7. **Visual fetch:** `create_final_video` Stage 2 parallel via `VisualProviderFactory` → `MediaManager.get_random_media()` → `slide_videos{idx:Path|None}`.
8. **Render slides:** Stage 3 parallel `_create_slide_with_ffmpeg()` per slide → `temp/slide_<n>_<uuid>.mp4`.
9. **Concat & mix:** `concat_videos` (+ optional crossfade `0.3s`) → mix `music_path` at `music_volume_db` with ducking → `output/video_<stamp>_<lang>/final_video.mp4`.
10. **Thumbnail & social:** `get_smart_thumbnail_frame()` → `thumbnail.jpg`; `generate_social_media_descriptions()` → textbox.
11. **Cache & audit:** `DB.save_video()` + `keyword_selection_audit` rows + return `{"video_path","audio_path","output_directory","sentence_count","success"}` to Gradio.

---

## 6. Configuration Reference (`.env` / env vars)

| Var | Default | Used in | Notes |
|-----|---------|---------|-------|
| `PEXELS_API_KEY` | — | `media/pexels.py` | Required for Pexels |
| `UNSPLASH_ACCESS_KEY` / `UNSPLASH_SECRET_KEY` / `UNSPLASH_APP_ID` | — | `media/unsplash.py` | Unsplash search |
| `OLLAMA_API_URL` | `https://ai.izdrail.com/api/generate` | `OllamaClient`, `KeywordExtractor` | Set to `http://localhost:11434/api/generate` for local |
| `AI_MODEL` | `gemma4:e2b` | `OllamaClient` | `get_available_models()` via `/api/tags` |
| `OLLAMA_MAX_RETRIES` | `3` | `OllamaClient` | |
| `OLLAMA_RETRY_BASE_DELAY` | `1.0` | `OllamaClient` | Exponential |
| `OLLAMA_TIMEOUT` | `180` | `OllamaClient` | Seconds |
| `OLLAMA_CACHE_MAX_SIZE` | `512` | `OllamaClient` | In-memory LRU |
| `OLLAMA_FALLBACK_KEYWORDS` | `abstract,motion,light,texture,landscape,cityscape` | `OllamaClient`/`keyword_extractor` | |
| `OLLAMA_FALLBACK_MOOD` | `Cinematic` | `ollama_client` | |
| `OLLAMA_FALLBACK_SCRIPT` | static msg | `ollama_client` | |
| `IMAGE_GENERATION_ENABLED` | `true` | `Config`, `FFmpegVideoGenerator` | Gate SD |
| `IMAGE_GENERATION_MODEL` | `stabilityai/sd-turbo` | `SDTurboGenerator` | |
| `IMAGE_GENERATION_DEVICE` | `auto` | `SDTurboGenerator` | `cuda|cpu|auto` |
| `IMAGE_GENERATION_STEPS` | `1` | `SDTurboGenerator` | |
| `IMAGE_GENERATION_GUIDANCE_SCALE` | `0.0` | `SDTurboGenerator` | |
| `IMAGE_GENERATION_WIDTH/HEIGHT` | `512` | `SDTurboGenerator` | |
| `IMAGE_GENERATION_CACHE_DIR` | `cache/images` | `Config` | |
| `VISUAL_SOURCE` | `stock` | `Config`, `Factory` | `stock|ai|mixed` |
| `MIXED_MODE_IMAGE_RATIO` | `0.5` | `MixedProvider` | |
| `KEYWORD_DEBUG_MODE` | `False` | `KeywordExtractor` | Verbose audit log |
| `KEYWORD_HISTORY_LIMIT` | `200` | `KeywordExtractor` | |
| `KEYWORD_EMBEDDING_MODEL` | `spacy` | `KeywordExtractor` | `spacy|sentence_transformer` |
| `SEMANTIC_DUP_THRESHOLD` | `0.8` | `KeywordExtractor` | Cosine threshold |
| `KEYWORD_CONTEXT_ENRICH` | `true` | `enrich_keyword_context` | |
| `ENABLE_SD_FALLBACK` | `True` | `Config` | |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `/opt/huggingface_models` | `Dockerfile` | Model cache |
| `YOUTUBE_AUDIO_API_URL` | `https://thibault.../api.json` | `youtube_audio.py` | |
| `SPACY_API_URL` | `https://spacy.izdrail.com` | `_extract_spacy_fallback` | |

`.env.example` ships minimal subset; full list above is authoritative from `core/config.py:__init__`.

---

## 7. External Integrations

- **Ollama:** `POST {url} {model, prompt, stream, options, format}` → `response` field. Resilience: retry 5xx/timeout, LRU cache, fallback keywords/mood/script. Stats in `OllamaClient.stats`.
- **Pexels/Pixabay/Giphy/YouTube/Openverse/Wikimedia/IA/Unsplash:** REST search, random pick, file cache under `temp/`, availability probe.
- **spaCy:** `en_core_web_md` preferred, fallback `en_core_web_sm`, remote `SPACY_API_URL/pos` fallback.
- **Diffusers SD-Turbo:** `diffusers==0.31.0`, `transformers>=4.45,<4.49`, `accelerate`, `torch 2.3.1`.
- **TTS:** `TTS==0.22.0` (XTTS), `kokoro==0.7.16`, `gTTS`, `speechbrain==1.0.3`, `brian2` (SNN).
- **FFmpeg/ffprobe:** Slide composition, duration probe, concat, thumbnail. Must be in `PATH` (`Config.validate()` warns).

---

## 8. Caching & Persistence

- **Ollama in-memory LRU** (`OllamaClient._cache`, SHA256 of payload) — per-process, cleared on URL change.
- **TTS file cache** (`GenerationDB.tts_cache`) — key `SHA256(text_voice_lang_speed)`, file under `temp/audio_cache`, reused across runs.
- **Video logs** (`video_logs`) — `input_hash` dedup, requires both `video_path` + `audio_path` exist.
- **Media file cache** — downloaded stock videos/images under `temp/` (provider-specific).
- **SD image cache** — `cache/images/` keyed by prompt+seed.
- **Keyword audit** — `keyword_selection_audit` for future regression calibration.

Purge: `scripts/purge_cache.py`, `scripts/clean_project.sh`.

---

## 9. Installation & Running

```bash
git clone https://github.com/izdrail/videos.izdrail.com.git
cd videos.izdrail.com
pip install -r requirements.txt
python -m spacy download en_core_web_md
# Optional: python download_model.py ; python download_kokoro.py ; python scripts/download_sd_turbo.py
export PEXELS_API_KEY="..."
export UNSPLASH_ACCESS_KEY="..."
# export OLLAMA_API_URL="https://ai.izdrail.com/api/generate"  # or http://localhost:11434/api/generate
python main.py  # → http://localhost:1603
```

**Docker (recommended):**
```bash
make build   # docker buildx linux/amd64
make dev     # docker-compose up --build (ports 1603,1604, GPU)
make ssh     # exec bash
make down / make rebuild / make publish
# Tests: make test-brain ; docker-compose exec videos.izdrail.com pytest
```

**FFmpeg:** `apt install ffmpeg` (Dockerfile already installs). GPU optional (`torch.cuda.is_available()`).

---

## 10. Gradio UI Contract (for AI agents modifying `main.py`)

- Inputs to `create_final_video` are wired from Gradio components; keep `progress_callback` signature `(completed, total, msg)`.
- `pre_selected_videos: Dict[int, str]` values are filesystem Paths or `"__gradient__"` sentinel; `selected_background_video` applies to all non-intro/CTA slides.
- `overlay_shape` enum `Circle|Square|Rectangle|Star|Split Screen`; `Split Screen` auto-selects `circle_video` if missing.
- `language` codes must match `SUPPORTED_LANGUAGES` keys; `auto` triggers `langdetect`.
- Output components: `gr.Video`, `gr.Audio`, `gr.Textbox` (social descriptions), `gr.File` download. Keep `examples` and `flagging` disabled.

---

## 11. Testing & Quality

```bash
pytest tests/ -v
pytest tests/test_media_providers.py tests/test_visual_providers.py -v
pytest tests/test_video_pipeline.py tests/test_video_progress_and_concat.py -v
pytest tests/test_keyword_selection_overhaul.py tests/test_ollama_resilience.py -v
ruff check . ; ruff format --check .  # if configured
```

Coverage focuses on: media provider selection & fallback, visual factory, FFmpeg slide validation, keyword beam search & audit, Ollama retry/cache/fallback, parallel pipeline thread safety (WAL).

---

## 12. Known Limitations & Gotchas (read before changing code)

- **Intro video is hardcoded:** `intro/intro-tv-noise.mp4` is forced for `is_intro` slides; do not delete/move. Validation falls back to gradient if missing/empty.
- **No `src/` layout:** All imports are `core.*` relative to repo root; `PYTHONPATH=/app` in Docker.
- **Thread safety:** `GenerationDB.lock` + `WAL` required; audit writes are best-effort swallowed. Do not add blocking DB writes in hot path.
- **FFmpeg graphs are string-built:** Any new overlay/mask must correctly increment `input_count` and track `mask_idx = input_count+1`. Missing `-shortest` causes black tail.
- **Font discovery:** `_discover_fonts()` platform-specific; fallback `DejaVuSans`. Missing font → `ImageFont.load_default()`.
- **Disk pressure:** `temp/` + `output/` + `cache/` grow unbounded; purge scripts exist but no automatic rotation. SD model `~10GB`.
- **API keys:** Absence only warns (`Config.validate()`), but Pexels/Unsplash searches silently return empty → fallback chain.
- **Language gaps:** `ar`/`ro` Kokoro `None` → gTTS/MMS only; test with `tests/test_mms_romanian.py`.
- **Transformers pin:** `transformers<4.49` required due to `GlmModel` / `flex_attention` incompat with `torch 2.3.1`; do not bump without testing.

---

## 13. Improvement Backlog (prioritized for AI agents)

**P0 — Reliability & Correctness**
- Add per-slide `validate_slide()` pre-flight unit tests for every `overlay_shape` + missing audio/video combos.
- Make `GenerationDB` paths respect `Config.ROOT_DIR` instead of CWD-relative `generation_cache.db`.
- Add automatic `temp/` rotation (TTL + max size) and `output/` retention policy.
- Surface `OllamaClient.stats` in Gradio footer for observability.

**P1 — Performance**
- Replace `ThreadPoolExecutor` with `asyncio` + `httpx` for Ollama/media (lower overhead than 20 threads).
- Cache `get_available_music_files()` and watch `background_music/` via `watchdog`.
- Pre-warm `SDTurboGenerator` lazily on first `ai`/`mixed` request, not in `__init__`.
- Profile FFmpeg `ultrafast` vs `veryfast` quality/time tradeoff per preset.

**P2 — Features**
- Auto language detection per sentence + per-sentence TTS voice routing.
- Scene-aware transitions (crossfade vs cut based on keyword coherence score).
- OpenCV visual effects (ken burns, zoom) as FFmpeg `zoompan` alternative.
- GPU audio acceleration (CUDA `torchaudio` resample).
- Persist Gradio state (URL params, localStorage) for language/voice/preferred source.

**P3 — Architecture**
- Extract `FFmpegVideoGenerator` from `main.py` into `core/video/generator.py` (currently 1300+ lines in entrypoint).
- Introduce `core/pipeline.py` orchestrator separating NLP→TTS→Visual→Render stages (testable without Gradio).
- Add `pydantic` models for `SlideData`, `RenderConfig` instead of `Dict[str,Any]`.
- Replace stringly-typed `preferred_media_source` with `Enum`.

**P4 — Docs & DX**
- Generate `docs/media_providers.md` coverage for remaining providers (Wikimedia, IA, SearxNG).
- Add `AGENTS.md` with `ruff` / `pytest` commands for future agents.
- Provide `docker-compose.override.yml` for local Ollama (`ollama serve` sidecar).

---

## 14. Prompting Guide for Future AI Agents

When asked to **improve** this codebase, an agent should:

1. **Read `core/config.py` first** — every tunable is there. Prefer adding an env var + default over hardcoding.
2. **Respect the fallback chains** — `get_background_video()` and `OllamaClient.post_or_fallback()` must never raise; log warning and continue.
3. **Keep `main.py` thin** — new logic belongs in `core/`; `main.py` only wires Gradio → pipeline.
4. **Maintain cache keys** — changing TTS cache key format requires migration in `GenerationDB.init_db()`.
5. **Test with `pytest` and manual `python main.py`** — verify Gradio still launches on `:1603` and a 3-sentence English generation succeeds end-to-end (use `Pexels` mock if no key).
6. **Document new env vars** in both `core/config.py` docstring and §6 table of this README.
7. **Never commit secrets** (`.env`, `*.db`, `output/`, `temp/`, `cache/`) — check `.gitignore`.

**One-line TL;DR for agents:** `python main.py` → Gradio `:1603` → text splits → Ollama+spaCy keywords (beam search) → parallel TTS+Media → parallel FFmpeg slides (gradient/mask/logo/circle/split) → concat+music mix → `output/video_<ts>_<lang>/final_video.mp4`.

---

## 15. Tech Stack Table

| Layer | Technology | Version / Notes |
|-------|------------|-----------------|
| NLP | spaCy `en_core_web_md` + Ollama `gemma4:e2b` | via `OllamaClient` retry/cache/fallback |
| TTS | Kokoro-82M, Coqui XTTS v2, gTTS, MMS-TTS | `TTS 0.22.0`, `kokoro 0.7.16`, `speechbrain 1.0.3` |
| Visual AI | Stable Diffusion Turbo | `diffusers 0.31.0`, `transformers <4.49`, `accelerate` |
| Video | FFmpeg + Pillow + NumPy | `ffmpeg-python`, `PIL ImageFilter/Enhance` |
| Audio | PyDub + torchaudio | `pydub 0.25`, `torch 2.3.1` |
| UI | Gradio | `5 ≤ gradio <7` |
| Media APIs | Pexels, Pixabay, Giphy, YouTube, Unsplash, Openverse, Wikimedia, IA | `requests`, `yt-dlp 2025.12.8` |
| DB | SQLite3 WAL | `generation_cache.db` |
| ML Backend | PyTorch | `2.3.1`, `torchaudio 2.3.1` |
| Infra | Docker + Supervisor | `python:3.11`, ports `1603/1604`, NVIDIA runtime |

---

## 16. License & Credits

- **License:** MIT — free to modify/commercialize with credit.
- **Credits:** Stefan Bogdan · Coqui TTS · SpeechBrain · Pexels · PyTorch · Gradio · MoviePy · Diffusers.
- **Feedback:** https://github.com/anomalyco/opencode/issues · In CLI: `/help`.

---

## 🚀 Quick Start (Gen Z TL;DR)

**Input:** Text → **Output:** Viral 9:16 video w/ voice, music, vibe. **Command:** `python main.py` → `http://localhost:1603` → 🔥 TikTok-ready.

## Video selection identity, diversity, and diagnostics

SNN mode sequences keywords; candidate retrieval still uses the normal media providers. Exact video identity is provider + stable asset ID when available, otherwise a canonical source URL. Concurrent selectors reserve that identity atomically, and the final renderer also rejects duplicate real files (including symlink aliases).

Ranking blends keyword and narration relevance and uses MMR to avoid repeatedly choosing visually similar clips. Tune with `MS_GATE_B_WEIGHT` (default `0.5`) and `MS_MMR_LAMBDA` (default `0.85`; set `1` to disable the MMR penalty). See `docs/research/snn-selection-dedup-and-diversity.md` for the investigation, trade-offs and benchmark command.
