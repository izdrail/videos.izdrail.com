from diffusers import StableDiffusionPipeline
import torch
import os

print("Downloading Stable Diffusion model...")
model_id = "runwayml/stable-diffusion-v1-5"
# Define the permanent save directory inside your Docker image
save_directory = "/models/stable-diffusion-v1-5"

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

print("Stable Diffusion model downloaded and saved successfully!")