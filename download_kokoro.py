import time
import sys
from huggingface_hub import snapshot_download

MODELS = {
    "Kokoro-82M": "hexgrad/Kokoro-82M",
    "MMS-TTS Romanian": "facebook/mms-tts-ron",
}

def download_model(name: str, repo_id: str, max_retries: int = 5) -> None:
    print(f"\n▶ Pre-downloading {name}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.bin", "*.pt", "*.pth", "*.json", "*.onnx", "*.model"],
                resume_download=True,
            )
            print(f"✔ {name} downloaded successfully after {attempt} attempt(s)")
            print(f"  Location: {path}")
            return
        except Exception as e:
            print(f"⚠️  Attempt {attempt} failed for {name}: {e}")
            if attempt < max_retries:
                wait_time = attempt * 5
                print(f"🔄 Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

def main():
    try:
        for name, repo_id in MODELS.items():
            download_model(name, repo_id)

        print("\n✅ All TTS models downloaded and cached")

    except Exception as e:
        print(f"\n❌ Model download failed after multiple attempts: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
