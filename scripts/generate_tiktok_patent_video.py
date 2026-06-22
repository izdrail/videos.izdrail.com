#!/usr/bin/env python3
"""Generate a TikTok-style video about patent disputes.

Usage:
  python scripts/generate_tiktok_patent_video.py
  python scripts/generate_tiktok_patent_video.py --text "Your custom script..."
  python scripts/generate_tiktok_patent_video.py --file script.txt --language ro --voice american-man
  python scripts/generate_tiktok_patent_video.py --topic "TikTok patent disputes"
  python scripts/generate_tiktok_patent_video.py --topic "AI regulation" --model "mistral:7b" --language en
  python scripts/generate_tiktok_patent_video.py --list-voices
"""

import os
import sys
import argparse
from typing import Dict, Optional
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))


def _import_generator():
    from main import TextToVideoGenerator  # noqa: F401
    from core.config import Config  # noqa: F401
    from core.nlp.keyword_extractor import KeywordExtractor  # noqa: F401

    return TextToVideoGenerator, Config, KeywordExtractor


DEFAULT_TEXT = """
TikTok, owned by ByteDance, faces several patent infringement lawsuits and legal challenges concerning its core "For You" feed algorithm and e-commerce features, with claims it copied patented technology for personalized video delivery from companies like 7Echo (David Russek) and VCA, and its "green screen" video tool from Triller, highlighting IP theft concerns amidst its global expansion and regulatory scrutiny. 
Key Patent Disputes & Claims
Personalized Feed (7Echo/VCA): A major lawsuit by VCA (related to 7Echo patents) alleges TikTok infringes on patents for systems that deliver personalized media, store submissions, and reward users, mirroring the "For You" feed and its reward system.
Green Screen Feature (Triller): Triller sued TikTok for its "green screen" video feature, claiming it illegally combined multiple videos synchronized to audio, infringing Triller's patents.
TikTok Shop (ShopSee Inc.): ShopSee accused TikTok Shop of copying its patented system for integrating product links within video content, enabling the popular e-commerce platform. 
TikTok's Own Patents
While dealing with infringement claims, ByteDance also holds patents, including some related to music services like Resso and SoundOn, as part of its broader strategy in music discovery and artist promotion.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a TikTok patent-news video with AI backgrounds, TTS, and music.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --text 'Your script here...' --language ro\n"
            "  %(prog)s --file script.txt --voice american-man --no-intro\n"
            "  %(prog)s --topic 'AI regulation news' --model 'mistral:7b'\n"
            "  %(prog)s --topic 'TikTok patents' --language en --stress 1.2\n"
            "  %(prog)s --list-voices\n"
        ),
    )

    content = parser.add_argument_group("Content")
    content.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text content for the video (overrides default patent text)",
    )
    content.add_argument("--file", type=str, default=None, help="Read text from a file")
    content.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Generate a script from a topic using AI (e.g. 'TikTok patent disputes')",
    )
    content.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model for topic script generation (default: mistral:7b)",
    )

    audio = parser.add_argument_group("Audio / Voice")
    audio.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh", "es", "fr", "it", "pt", "hi", "ja", "ar", "ro", "auto"],
        help="Language code (default: en)",
    )
    audio.add_argument(
        "--voice",
        type=str,
        default="american-man",
        help="Speaker/voice ID (default: american-man)",
    )
    audio.add_argument(
        "--stress",
        type=float,
        default=1.0,
        help="Voice speed multiplier (0.8-1.5, default: 1.0)",
    )
    audio.add_argument(
        "--no-music", action="store_true", help="Disable background music"
    )
    audio.add_argument(
        "--music",
        type=str,
        default="Random",
        help="Music track name or 'Random' (default: Random)",
    )

    video = parser.add_argument_group("Video")
    video.add_argument(
        "--media-source",
        type=str,
        default="YouTube",
        choices=["Random", "Pexels", "Pixabay", "YouTube", "Giphy", "SearXNG"],
        help="Preferred background video source (default: YouTube)",
    )
    video.add_argument("--fps", type=int, default=30, help="Export FPS (default: 30)")
    video.add_argument("--no-intro", action="store_true", help="Skip intro slide")
    video.add_argument("--no-cta", action="store_true", help="Skip CTA outro slide")
    video.add_argument(
        "--crossfade",
        action="store_true",
        help="Enable crossfade transitions between slides",
    )
    video.add_argument(
        "--aspect-ratio",
        type=str,
        default="9:16 Portrait (TikTok/Shorts)",
        help="Aspect ratio preset name",
    )
    video.add_argument(
        "--quality",
        type=str,
        default="Medium (Balanced)",
        help="Quality preset name (Low/Medium/High/Ultra)",
    )
    video.add_argument(
        "--select-backgrounds",
        action="store_true",
        help="Interactively pick background videos for each slide",
    )

    info = parser.add_argument_group("Info")
    info.add_argument(
        "--list-voices", action="store_true", help="Show available voices and exit"
    )
    info.add_argument(
        "--validate", action="store_true", help="Run system validation checks and exit"
    )

    return parser


def interactive_select_backgrounds(
    text: str,
    language: str,
    generator,
    preferred_source: Optional[str] = None,
) -> Dict[int, str]:
    """Let user interactively pick background videos per content slide.

    Returns a dict mapping content slide index (0-based) to video file path.
    """
    sentences = generator.video_generator.split_into_sentences(text)
    kw_extractor = generator.keyword_extractor
    kw_extractor.clear_used()
    pre_selected: Dict[int, str] = {}

    print(f"\n🎬 Interactive background selection for {len(sentences)} slides")
    print("For each slide, you can:")
    print("  [Enter]  — Use the suggested video")
    print("  s        — Skip (let the system auto-select)")
    print("  c        — Enter a custom keyword to search for")
    print("  n        — No video (use gradient background)")
    print()

    for i, sentence in enumerate(sentences):
        print(f"\n{'─' * 60}")
        print(f"📄 Slide {i + 1}/{len(sentences)}")
        print(f"   Text: {sentence[:120]}{'…' if len(sentence) > 120 else ''}")
        print(f"{'─' * 60}")

        candidates = kw_extractor.extract_keywords(
            sentence, top_n=10, language=language
        )
        kw = None
        for c in candidates:
            if c not in kw_extractor.used_keywords:
                kw = c
                kw_extractor.used_keywords.add(c)
                break
        if not kw and candidates:
            kw = candidates[0]

        if not kw:
            print("   ⚠ No keyword could be extracted. Skipping.")
            continue

        print(f"   🔑 Keyword: '{kw}'")

        print("   🔍 Searching for background videos...")
        media_manager = generator.video_generator.media_manager
        result = media_manager.get_random_media(
            [kw], preferred_source, context=sentence, return_keyword=True
        )

        if result and result[0]:
            path, used_kw = result
            print(f"   ✅ Found: {path.name}  (keyword: '{used_kw}')")
        else:
            print(f"   ❌ No video found for '{kw}'")
            path = None

        choice = (
            input("   ▶ Use this? [Enter=yes, s=skip, c=custom kw, n=no video]: ")
            .strip()
            .lower()
        )

        if choice == "" or choice == "y":
            if path:
                pre_selected[i] = str(path)
                print(f"      ✓ Selected: {path.name}")
            else:
                print("      No video was found. Trying auto with a broader search…")
                fallback = media_manager.get_random_media(
                    [kw], preferred_source, return_keyword=True
                )
                if fallback and fallback[0]:
                    pre_selected[i] = str(fallback[0])
                    print(f"      ✓ Fallback selected: {fallback[0].name}")
        elif choice == "s":
            print("      → Skipped (system will auto-select during generation)")
            continue
        elif choice == "n":
            print("      → No video will be used (gradient fallback)")
            pre_selected[i] = "__gradient__"
        elif choice == "c":
            custom = input("   Enter custom keyword: ").strip()
            if custom:
                print(f"   🔍 Searching for '{custom}'…")
                custom_result = media_manager.get_random_media(
                    [custom], preferred_source, context=sentence, return_keyword=True
                )
                if custom_result and custom_result[0]:
                    cpath, _ = custom_result
                    pre_selected[i] = str(cpath)
                    print(f"      ✓ Selected: {cpath.name}")
                else:
                    print("      ❌ No video found for custom keyword. Skipping.")
            else:
                print("      Empty keyword. Skipping.")
        else:
            print("      Unknown option. Skipping.")

    print(
        f"\n✅ Selected {len(pre_selected)}/{len(sentences)} backgrounds interactively."
    )
    return pre_selected


def list_voices():
    """Print available TTS voices."""
    TextToVideoGenerator, _, _ = _import_generator()
    gen = TextToVideoGenerator()
    print("Available voices:")
    for v in gen.available_voices:
        print(f"  - {v}")
    print(f"\nTotal: {len(gen.available_voices)} voices")


def run_validation():
    """Run and display config validation warnings."""
    _, Config, _ = _import_generator()
    cfg = Config()
    warnings = cfg.validate()
    if warnings:
        print("Configuration warnings:")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("All checks passed.")
    return len(warnings)


def generate_tiktok_video(args: argparse.Namespace):
    # Resolve text source
    if args.topic:
        print(f"Generating script for topic: '{args.topic}'...")
        _, _, KeywordExtractor = _import_generator()
        kw_extractor = KeywordExtractor()
        if args.model:
            kw_extractor.model = args.model
        script_lang = args.language if args.language != "auto" else "en"
        text = kw_extractor.generate_topic_script(args.topic, script_lang)
        if not text:
            print("Topic script generation failed.")
            sys.exit(1)
        print(f"Generated {len(text)}-char script:\n{text}\n")
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
        print(f"Read {len(text)} chars from {args.file}")
    elif args.text is not None:
        text = args.text.strip()
    elif env_text := os.getenv("VIDEO_TEXT"):
        text = env_text.strip()
        print("Using text from VIDEO_TEXT env var")
    else:
        text = DEFAULT_TEXT
        print("Using default TikTok patent text")

    if not text:
        print("No text provided. Use --text, --file, --topic, or VIDEO_TEXT env var.")
        sys.exit(1)

    print(f"Language: {args.language}")
    print(f"Voice: {args.voice}")
    print(f"Speed: {args.stress}")
    if args.topic:
        print(f"Topic: {args.topic}")
        if args.model:
            print(f"Model: {args.model}")
    print(f"Media source: {args.media_source}")
    print(f"Intro: {not args.no_intro}, CTA: {not args.no_cta}")
    print(f"Music: {'disabled' if args.no_music else args.music}")
    print(f"Aspect ratio: {args.aspect_ratio}, Quality: {args.quality}")
    print(f"Crossfade: {args.crossfade}")
    if args.select_backgrounds:
        print("Background selection: Interactive")

    try:
        TextToVideoGenerator, _, _ = _import_generator()
        generator = TextToVideoGenerator()

        pre_selected_videos = None
        if args.select_backgrounds:
            if args.no_intro:
                print(
                    "⚠ --select-backgrounds is incompatible with --no-intro, ignoring"
                )
            else:
                print("\n🎬 Entering interactive background selection mode...")
                kw_lang = args.language if args.language != "auto" else "en"
                pre_selected_videos = interactive_select_backgrounds(
                    text=text,
                    language=kw_lang,
                    generator=generator,
                    preferred_source=args.media_source,
                )

        result = generator.generate_video(
            text=text,
            language=args.language,
            speaker_id=args.voice,
            preferred_media_source=args.media_source,
            enable_background_music=not args.no_music,
            music_selection=args.music,
            add_intro_slide=not args.no_intro,
            add_call_to_action=not args.no_cta,
            stress_level=args.stress,
            export_fps=args.fps,
            aspect_ratio=args.aspect_ratio,
            quality=args.quality,
            enable_crossfade=args.crossfade,
            pre_selected_videos=pre_selected_videos,
        )

        if result.get("success"):
            print("\nVideo generated successfully!")
            print(f"  Video: {result['video_path']}")
            print(f"  Thumbnail: {result.get('thumbnail_path')}")
            print(f"  Audio: {result['audio_path']}")
            print(f"  Output: {result['output_directory']}")
        else:
            print(f"\nGeneration failed: {result.get('error')}")
            sys.exit(1)

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if args.validate:
        n_warnings = run_validation()
        sys.exit(0 if n_warnings == 0 else 1)

    # Run validation non-fatally before generation
    n_warnings = run_validation()

    generate_tiktok_video(args)


if __name__ == "__main__":
    main()
