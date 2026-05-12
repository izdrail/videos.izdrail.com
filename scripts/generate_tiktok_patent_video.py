import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    from main import TextToVideoGenerator
    from core.config import Config
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

def generate_tiktok_video():
    print("🚀 Starting TikTok Patent Video Generation...")
    
    text = """
TikTok, owned by ByteDance, faces several patent infringement lawsuits and legal challenges concerning its core "For You" feed algorithm and e-commerce features, with claims it copied patented technology for personalized video delivery from companies like 7Echo (David Russek) and VCA, and its "green screen" video tool from Triller, highlighting IP theft concerns amidst its global expansion and regulatory scrutiny. 
Key Patent Disputes & Claims
Personalized Feed (7Echo/VCA): A major lawsuit by VCA (related to 7Echo patents) alleges TikTok infringes on patents for systems that deliver personalized media, store submissions, and reward users, mirroring the "For You" feed and its reward system.
Green Screen Feature (Triller): Triller sued TikTok for its "green screen" video feature, claiming it illegally combined multiple videos synchronized to audio, infringing Triller's patents.
TikTok Shop (ShopSee Inc.): ShopSee accused TikTok Shop of copying its patented system for integrating product links within video content, enabling the popular e-commerce platform. 
TikTok's Own Patents
While dealing with infringement claims, ByteDance also holds patents, including some related to music services like Resso and SoundOn, as part of its broader strategy in music discovery and artist promotion.
""".strip()

    try:
        generator = TextToVideoGenerator()
        
        # Configuration for the video
        # We'll use English and a standard voice.
        # Neuron AI is enabled by default in KeywordExtractor.
        result = generator.generate_video(
            text=text,
            language='en',
            speaker_id="american-man", # Or another available voice
            preferred_media_source="YouTube",
            enable_background_music=True,
            music_selection="Random",
            add_intro_slide=True,
            add_call_to_action=True,
            stress_level=1.1, # High energy
            export_fps=30
        )
        
        if result.get("success"):
            print("\n✅ Video Generated Successfully!")
            print(f"📍 Video Path: {result['video_path']}")
            print(f"🖼️ Thumbnail: {result.get('thumbnail_path')}")
            print(f"🎵 Audio: {result['audio_path']}")
            print(f"📂 Output Folder: {result['output_directory']}")
        else:
            print(f"\n❌ Generation Failed: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_tiktok_video()
