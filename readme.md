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

    * Supports **Coqui XTTS v2** (voice cloning) and **SpeechBrain Tacotron2 + HiFi-GAN**
    * Option to randomize voices per sentence

* 🎬 **Smart Visuals**

    * Fetches relevant portrait videos via **Pexels API**
    * Automatically picks best keyword per sentence using **spaCy NLP**

* 🎧 **Audio Perfection**

    * Loudness normalization, low/high-pass filtering
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

---

## 🎤 Directory Structure

```plaintext
project_root/
├── background_images/        # Local fallback images (optional)
├── background_videos/        # Local fallback videos
├── background_music/         # Music tracks (.mp3/.wav)
├── voice_samples/
│   ├── my_voice/
│   │   └── reference.wav     # Voice clone sample
│   └── another_voice/
│       └── reference.wav
├── temp/                     # Temporary files
├── output/                   # Generated videos & audio
└── main.py                   # Entry script (contains UI + logic)
```

---

## 🖥️ Run the App / 启动应用 / Запуск приложения

### Start the Gradio Interface:

```bash
python main.py
```

Then open:

```
http://localhost:1602
```

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

## 🧠 Tech Stack

| Layer      | Technology                       |
| ---------- | -------------------------------- |
| NLP        | spaCy `en_core_web_md`           |
| TTS        | Coqui TTS (XTTS v2), SpeechBrain |
| Video      | MoviePy, PIL, NumPy              |
| Audio      | PyDub                            |
| UI         | Gradio                           |
| API        | Pexels                           |
| ML Backend | PyTorch                          |

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

