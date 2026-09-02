import sys
import time
import torch

try:
    from diffusers import AutoPipelineForText2Image
except Exception as e:
    print(f"❌ Error importing diffusers AutoPipelineForText2Image: {e}")
    AutoPipelineForText2Image = None


def download_sd_model(model_id="stabilityai/sd-turbo", max_retries=5):
    if AutoPipelineForText2Image is None:
        print("❌ Cannot download model: diffusers AutoPipelineForText2Image is not available.")
        sys.exit(1)

    print(f"Downloading {model_id} model...")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"▶ Attempt {attempt}...")
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            print("✔ SD-Turbo model downloaded and cached successfully!")
            return pipe
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait_time = attempt * 10
                print(f"🔄 Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("❌ Max retries reached. Download failed.")
                raise e


if __name__ == "__main__":
    download_sd_model()
