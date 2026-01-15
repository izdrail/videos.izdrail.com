import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TextToVideoGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_pipeline():
    print("🚀 Starting Pipeline Test...")
    generator = TextToVideoGenerator()
    
    # Text with enough sentences to trigger parallel logic
    text = "The sun rises over the digital city. Data streams flow like rivers of light. The future is now, and it is beautiful. Technology connects us all."
    
    start_time = time.time()
    
    try:
        result = generator.generate_video(
            text=text,
            language='en',
            speaker_id="Standard Voice (Non-Cloned)",
            pexels_keyword="technology", # Provide a keyword to test search
            enable_background_music=False, # Disable music to speed up test
            add_intro_slide=True,
            add_call_to_action=True,
            export_fps=24, # Lower FPS for speed
            progress_callback=lambda current, total, msg: print(f"[{time.strftime('%H:%M:%S')}] Progress {current}/{total}: {msg}")
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result['success']:
            print(f"✅ Test Passed! Video generated in {duration:.2f}s")
            print(f"Output: {result['video_path']}")
        else:
            print(f"❌ Test Failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
