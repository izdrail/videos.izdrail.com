import os
import time
import torch
from diffusers import AutoPipelineForText2Image


def download_sd_turbo(model_id="stabilityai/sd-turbo", max_retries=5):
    print(f"Downloading {model_id}...")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"▶ Attempt {attempt}/{max_retries}...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            )
            print("✔ SD-Turbo model download complete and cached successfully!")
            return pipeline
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
    download_sd_turbo()
