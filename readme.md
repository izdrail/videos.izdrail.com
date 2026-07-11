# 🎥 Portrait Video Generator (Text → Video)

> Multi-lingual AI-powered pipeline that converts **text into fully synchronized 9:16 portrait videos** — with per-sentence TTS, dynamic visuals (via Pexels API), random voice cloning, and smooth audio mixing.
> Built with ❤️ using **Python**, **Coqui TTS**, **SpeechBrain**, **Pexels API**, **MoviePy**, and **Gradio**.

---

## 🌐 Multilingual Summary

### 🇬🇧 English

This project transforms any input text into a **portrait video** with synchronized speech, background visuals, and optional background music.
Each sentence can have its **own AI-generated voice**, keyword-matched background, and **unique visual vibe**.

### 🇷🇺 Русский

Этот проект превращает текст в вертикальное видео (9:16) с синхронизированной озвучкой, визуальными фонами и музыкой.
Каждое предложение получает **уникальный голос**, фон и стиль оформления.

### 🇨🇳 中文

这个项目可以将输入的文本转换为带有**配音和背景视频**的竖屏视频。
每个句子都有独立的声音、关键词背景和个性化风格。超炫酷🔥！

---

## ⚙️ Features / 功能特性 / Возможности

* 🎙️ **Per-sentence AI Voice Generation**
    * Supports **Kokoro-82M** (High speed), **Coqui XTTS v2** (Voice cloning), and **MMS-TTS**
    * 🗣️ **Speed & Stress Control**: Adjust voice pace and energy (default 1.0, perfect for shorts)
    * Option to randomize voices per sentence

* 🚀 **Lightning Fast ⚡ (Parallel & Cached)**
    * **Persistent Caching**: Audio and Background Videos are cached (`temp/audio_cache` & `temp/video_cache`)
    * **AI Response Caching**: Ollama responses are cached in-memory (SHA-256 keyed) to avoid redundant calls
    * Reuses existing assets to skip downloads and TTS generation
    * **Parallel Resource Fetching**: Multi-threaded keyword extraction and media downloads

* 🛡️ **Ollama Resilience**
    * **Exponential Backoff**: Automatic retries (3 attempts, 1s→2s→4s) on connection errors, timeouts, and 5xx
    * **Graceful Fallback**: If all retries fail, uses configurable default keywords/mood/script
    * **Video never stops**: Pipeline continues rendering with fallback content when AI is unavailable

* 📢 **Social Media Optimization**
    * Automatically generates viral-ready descriptions/captions for TikTok, Shorts, and Reels
    * Easy-to-copy textbox output with relevant hashtags

* 🎬 **Smart Visuals**
    * Fetches relevant portrait videos via **Pexels, Pixabay, Giphy, and YouTube**
    * Automatically picks best keyword per sentence using **Ollama (Mistral/Llama)** or **spaCy NLP**

* 🎧 **Audio Perfection**

    * Loudness normalization, low/high-pass filtering
    * **Silence trimming** for cleaner speech output
    * Dynamic range compression for consistent volume
    * Automatic fade-in/out & compression
    * Background music mixing with volume control

* 🎨 **Visual Styling**

    * Random vibrant text colors
    * Fallback color backgrounds
    * Smooth transitions & CTA (Call-to-Action) slide

* 🧠 **AI NLP Magic**

    * Keyword extraction powered by spaCy `en_core_web_md`
    * Supports fallback mode if NLP unavailable

* 🧩 **UI via Gradio**

    * Clean interface with live progress updates
    * Audio + video previews
    * Color picker, voice selector, and random voice toggle

---

## 🧰 Installation / 安装步骤 / Установка

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/izdrail/videos.izdrail.com.git
cd videos.izdrail.com
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> **Or manually:**

```bash
pip install TTS speechbrain pydub moviepy Pillow num2words torch torchaudio gradio requests spacy
python -m spacy download en_core_web_md
```

### 3️⃣ (Optional) Set up Pexels API

Get your free key from [https://www.pexels.com/api](https://www.pexels.com/api)

Add it in the UI or via environment:

```bash
export PEXELS_API_KEY="your_api_key_here"
```

### 4️⃣ Ollama / AI Configuration (Optional)

The pipeline works **without Ollama** — fallback keywords, mood, and script content are used automatically when the AI service is unreachable. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API_URL` | `https://ai.izdrail.com/api/generate` | Ollama API endpoint |
| `AI_MODEL` | `mistral:7b` | LLM model name |
| `OLLAMA_MAX_RETRIES` | `3` | Retry attempts before fallback |
| `OLLAMA_RETRY_BASE_DELAY` | `1.0` | Base delay (seconds) for exponential backoff |
| `OLLAMA_TIMEOUT` | `180` | Request timeout in seconds |
| `OLLAMA_CACHE_MAX_SIZE` | `512` | Max cached API responses (in-memory LRU) |
| `OLLAMA_FALLBACK_KEYWORDS` | `abstract,motion,light,texture,landscape,cityscape` | Fallback keywords when AI unavailable |
| `OLLAMA_FALLBACK_MOOD` | `Cinematic` | Fallback mood for music selection |
| `OLLAMA_FALLBACK_SCRIPT` | *(static message)* | Fallback script text when generation fails |

**Fallback behavior**: When Ollama is unavailable, the pipeline logs a WARNING and continues with fallback content. No video generation is interrupted.

---

## 🎤 Directory Structure

```plaintext
project_root/
├── core/                     # 🛡️ Shared core modules (SOLID architecture)
│   ├── config.py             # Centralized configuration
│   ├── database.py           # TTS & video caching (GenerationDB)
│   ├── utils/
│   │   ├── pytorch_compat.py # PyTorch secure loading
│   │   ├── audio.py         # Shared audio processing
│   │   └── video.py         # Shared video processing utilities
│   ├── media/                # Media API clients (Pexels, Giphy, YouTube)
│   ├── nlp/                  # Keyword extraction (Mistral:7b / Spacy)
│   │   ├── ollama_client.py  # Robust Ollama API client (retry, cache, fallback)
│   │   ├── keyword_extractor.py
│   │   └── neuron_extractor.py
│   ├── ai/                   # AI components (Stable Diffusion)
│   └── tts/                  # TTS management (Kokoro, XTTS v2)
├── background_images/        # Local fallback images (optional)
├── background_videos/        # Local fallback videos
├── background_music/         # Music tracks (.mp3/.wav)
├── voice_samples/
│   ├── my_voice/
│   │   └── reference.wav     # Voice clone sample
│   └── another_voice/
│       └── reference.wav
├── temp/                     # Temporary files (caches)
│   ├── audio_cache/          # Persistent TTS audio cache
│   └── video_cache/          # Persistent background video cache
├── output/                   # Generated videos & audio
├── main.py                   # Main video generator (All-in-one)
└── makefile                  # Build & run automation
```

---

## 🖥️ Run the App / 启动应用 / Запуск приложения

### Start the Gradio Interface:

```bash
python main.py
```

Then open:

```
http://localhost:1603  # High-performance Video Generator
```

> **Note:** Port `11434` must be reachable for Ollama integration (Local LLM).

---

## 💡 Usage Tips

| Feature             | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| 🎙️ Voice cloning   | Add your own samples in `voice_samples/<voice_name>/reference.wav` |
| 🧠 Keywords         | Auto-extracted via NLP or manually set in the UI                   |
| 🎵 Background music | Place `.mp3` files in `background_music/`                          |
| 🎨 CTA Slide        | Optional "Like, Share, Subscribe" ending                           |
| 🌀 Random Voices    | Toggle to give each sentence a different vibe                      |

---

## 🧩 Example Workflow

1. Input:

   ```
   The sun is shining bright.  
   Let's explore the world together!  
   Subscribe for more adventure vibes!
   ```
2. Output:

    * Each line → AI speech
    * Relevant Pexels video (e.g. “sun”, “world”, “adventure”)
    * Merged into portrait 9:16 video
    * Background music + CTA slide added

Result: 🔥 A ready-to-upload TikTok/YouTube Short!

---

## 🏗️ Architecture & Code Quality

### SOLID Principles Refactoring

The codebase has been refactored following SOLID principles to improve maintainability and reduce code duplication:

- **Single Responsibility**: Each module has one clear purpose
- **DRY (Don't Repeat Yourself)**: Shared components extracted to `core/` modules
- **Centralized Configuration**: All apps use unified `Config` class
- **Unified Caching**: Single `GenerationDB` for all TTS and video caching
- **PyTorch Compatibility**: Centralized PyTorch 2.6+ secure loading setup

### Benefits

- ✅ **Massive code de-duplication**: Reduced footprint by ~1,500 lines
- ✅ **Improved maintainability** with single source of truth in `core/`
- ✅ **Better testability** with modularized components
- ✅ **Consistent behavior** across all generators
- ✅ **Parallel Pipeline**: Simultaneous processing of NLP, Media Fetching, and TTS reduces generation time by up to 70%.
- ✅ **Persistent Storage**: Intelligent caching system prevents duplicate work, saving bandwidth and compute resources.

---

## 🧠 Tech Stack

| Layer      | Technology                       |
| ---------- | -------------------------------- |
| NLP        | Mistral:7b / spaCy `en_core_web_md` (via OllamaClient with retry/cache/fallback) |
| TTS        | Kokoro-82M, Coqui XTTS v2           |
| Video      | MoviePy, FFmpeg, PIL, NumPy         |
| Audio      | PyDub                               |
| AI Generic | Stable Diffusion                    |
| UI         | Gradio                              |
| API        | Pexels, Giphy, YouTube              |
| DB         | SQLite3                             |
| ML Backend | PyTorch                             |

---

## 🧪 Developer Notes

* Make sure you have **FFmpeg** installed and accessible in your PATH.
* GPU recommended for TTS speed (`torch.cuda.is_available()`).
* Supports both **Linux**, **macOS**, and **Windows**.

---

## 💬 Future Plans / 未来计划 / Будущие планы

* [ ] Auto language detection + multilingual TTS
* [ ] Scene-based transitions
* [ ] Advanced visual effects via OpenCV
* [ ] GPU audio acceleration

---

## ✨ Credits

* **Author:** Stefan Bogdan
* **AI Models:** Coqui TTS, SpeechBrain
* **Media API:** [Pexels](https://pexels.com)
* **Frameworks:** PyTorch, Gradio, MoviePy

---

## 📜 License

MIT License – free to modify and commercialize with credit.

---

## 🚀 Quick TL;DR (for Gen Z devs)

**Input:** Text
**Output:** Viral vertical video w/ voice, music, and AI vibe.
**Command:**

```bash
python main.py
```

**Result:**
🔥 TikTok-ready 9:16 AI-generated video.

