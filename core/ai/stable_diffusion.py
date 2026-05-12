"""
Stable Diffusion Manager
Handles AI image generation using Stable Diffusion on CPU
"""
import uuid
import torch
import hashlib
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline
    SD_AVAILABLE = True
except ImportError:
    print("[SD] diffusers library not found. SD image generation will be unavailable.")
    SD_AVAILABLE = False

class StableDiffusionManager:
    """Manages Stable Diffusion image generation (optimized for CPU)"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        from ..config import Config
        cfg = Config()
        self.model_path = model_path or str(cfg.SD_MODEL_DIR)
        self.device = "cpu"
        self.pipe = None
        self.generation_cache = {}
        self.cache_dir = cfg.IMAGES_DIR
        self.cache_dir.mkdir(exist_ok=True)
    
    def _load_pipeline(self):
        """Lazy load the pipeline to save memory"""
        if not SD_AVAILABLE or self.pipe is not None:
            return
            
        try:
            print(f"[SD] Loading model from {self.model_path} on CPU...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            # Optimize for memory
            self.pipe.enable_attention_slicing()
            print("[SD] Pipeline loaded successfuly")
        except Exception as e:
            print(f"[SD] Failed to load pipeline: {e}")
            self.pipe = None

    def generate_image(self, prompt: str, keyword: Optional[str] = None, 
                      size: Tuple[int, int] = (1080, 1920)) -> Optional[Path]:
        """
        Generate an image matching the prompt
        
        Args:
            prompt: Text description
            keyword: Optional keyword for caching
            size: Desired image size
            
        Returns:
            Path to generated image or None
        """
        if not SD_AVAILABLE:
            return None
            
        # Check cache
        cache_key = hashlib.sha256(f"{prompt}_{keyword}_{size}".encode()).hexdigest()
        if cache_key in self.generation_cache:
            cached_path = self.generation_cache[cache_key]
            if cached_path.exists():
                return cached_path
        
        self._load_pipeline()
        if not self.pipe:
            return None
            
        try:
            # Optimize prompt for SD
            full_prompt = (
                f"{prompt}, high quality, detailed, realistic background, "
                f"cinematic lighting, {keyword or ''}"
            ).strip(", ")
            
            negative_prompt = "text, watermark, logo, blurry, low resolution, distorted"
            
            # Generate smaller image and upscale/resize to save CPU time
            gen_width, gen_height = 512, 768 # Portrait aspect
            if size[0] > size[1]: # Landscape
                gen_width, gen_height = 768, 512
                
            print(f"[SD] Generating image for: {keyword or prompt[:30]}...")
            with torch.no_grad():
                result = self.pipe(
                    full_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20, # Low steps for CPU speed
                    guidance_scale=7.5,
                    width=gen_width,
                    height=gen_height
                )
            
            if result.images:
                image = result.images[0].resize(size, Image.LANCZOS)
                output_path = self.cache_dir / f"sd_{uuid.uuid4().hex[:8]}.png"
                image.save(output_path, "PNG", quality=95)
                self.generation_cache[cache_key] = output_path
                print(f"[SD] Image saved: {output_path}")
                return output_path
                
        except Exception as e:
            print(f"[SD] Generation error: {e}")
            
        return None
