import torch
import os
import time
from diffusers import StableDiffusionPipeline

def download_sd_model(max_retries=5):
    print("Downloading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    save_directory = "/models/stable-diffusion-v1-5"
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"▶ Attempt {attempt}...")
            # 1. Download the model (this uses the cache)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )

            # 2. Save the model to your permanent /models directory
            print(f"Saving model to {save_directory}...")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
                
            pipe.save_pretrained(save_directory)
            print("✔ Stable Diffusion model downloaded and saved successfully!")
            return
        except Exception as e:
            print(f"⚠️  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait_time = attempt * 15
                print(f"🔄 Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("❌ Max retries reached. Download failed.")
                raise e

if __name__ == "__main__":
    download_sd_model()