"""
Stability AI SD-Turbo Image Generator
Handles AI image generation using SD-Turbo with lazy loading, device detection, SHA256 caching, and aspect ratio crop.
"""

import hashlib
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageOps

from ..config import Config
from ..utils.gpu import get_optimal_device

logger = logging.getLogger(__name__)

try:
    from diffusers import AutoPipelineForText2Image

    SD_TURBO_AVAILABLE = True
except ImportError:
    logger.warning(
        "[SD] diffusers library not found or AutoPipelineForText2Image unavailable."
    )
    SD_TURBO_AVAILABLE = False

SD_AVAILABLE = SD_TURBO_AVAILABLE


class SDTurboGenerator:
    """Stability AI SD-Turbo image generator with lazy loading and memory management."""

    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.config = config or Config()
        self.model_name = (
            model_path
            or getattr(
                self.config, "IMAGE_GENERATION_MODEL", "stabilityai/sd-turbo"
            )
        )
        self.requested_device = getattr(
            self.config, "IMAGE_GENERATION_DEVICE", device
        )
        self._pipeline = None
        self._device = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self.cache_dir = getattr(
            self.config,
            "IMAGE_GENERATION_CACHE_DIR",
            self.config.ROOT_DIR / "cache/images",
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _detect_device(self) -> str:
        return get_optimal_device(self.requested_device)

    def _load_model(self):
        """Lazy load the SD-Turbo pipeline on demand."""
        if self._loaded or not SD_TURBO_AVAILABLE:
            return

        with self._load_lock:
            if self._loaded:
                return

            try:
                self._device = self._detect_device()
                logger.info(
                    f"[SD] Loading model {self.model_name} on device: {self._device}..."
                )
                print(
                    f"[SD] Loading model {self.model_name} on device: {self._device}..."
                )

                if self._device == "cuda":
                    self._pipeline = AutoPipelineForText2Image.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    )
                else:
                    self._pipeline = AutoPipelineForText2Image.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                    )

                self._pipeline.to(self._device)
                if hasattr(self._pipeline, "enable_attention_slicing"):
                    self._pipeline.enable_attention_slicing()

                self._loaded = True
                logger.info("[SD] SD-Turbo pipeline loaded successfully")
                print("[SD] SD-Turbo pipeline loaded successfully")
            except Exception as e:
                logger.error(f"[SD] Failed to load SD-Turbo pipeline: {e}")
                print(f"[SD] Failed to load SD-Turbo pipeline: {e}")
                self._pipeline = None
                self._loaded = False

    def _generate_cache_key(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
    ) -> str:
        raw_key = (
            f"{self.model_name}_{prompt}_{width}_{height}_{steps}_{guidance_scale}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        keyword: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        scene_index: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = (1080, 1920),
        **kwargs,
    ) -> Optional[Path]:
        """Generate an image using SD-Turbo and crop to target_size.

        Args:
            prompt: Text prompt
            keyword: Optional keyword
            width: Raw generation width (default from config or 512)
            height: Raw generation height (default from config or 512)
            steps: Inference steps (default 1 for SD-Turbo)
            guidance_scale: Guidance scale (default 0.0 for SD-Turbo)
            scene_index: Scene/slide index for logging
            target_size: Output resolution (width, height), e.g. (1080, 1920)

        Returns:
            Path to generated image or None on failure.
        """
        if not SD_TURBO_AVAILABLE:
            return None

        w = width or getattr(self.config, "IMAGE_GENERATION_WIDTH", 512)
        h = height or getattr(self.config, "IMAGE_GENERATION_HEIGHT", 512)
        num_steps = steps if steps is not None else getattr(self.config, "IMAGE_GENERATION_STEPS", 1)
        g_scale = (
            guidance_scale
            if guidance_scale is not None
            else getattr(self.config, "IMAGE_GENERATION_GUIDANCE_SCALE", 0.0)
        )

        cache_key = self._generate_cache_key(prompt, w, h, num_steps, g_scale)
        cache_path = self.cache_dir / f"{cache_key}.png"

        if cache_path.exists():
            logger.info(f"[IMAGE] Provider: Stability AI")
            logger.info(f"[IMAGE] Model: {self.model_name}")
            logger.info(f"[IMAGE] Device: {self._device or self.requested_device}")
            logger.info(f"[IMAGE] Scene: {scene_index}")
            logger.info(f"[IMAGE] Resolution: {target_size[0]}x{target_size[1] if target_size else h}")
            logger.info(f"[IMAGE] Steps: {num_steps}")
            logger.info(f"[IMAGE] Cache: HIT")
            logger.info(f"[IMAGE] Generation: 0.00s")
            logger.info(f"[IMAGE] Output: {cache_path}")
            return cache_path

        start_time = time.time()
        self._load_model()
        if not self._pipeline:
            return None

        try:
            with torch.no_grad():
                result = self._pipeline(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=g_scale,
                    width=w,
                    height=h,
                )

            if result.images:
                pil_image = result.images[0].convert("RGB")

                # Format/crop image to target resolution if requested
                if target_size:
                    pil_image = ImageOps.fit(
                        pil_image, target_size, method=Image.Resampling.LANCZOS
                    )

                pil_image.save(cache_path, "PNG", quality=95)
                duration = time.time() - start_time

                logger.info(f"[IMAGE] Provider: Stability AI")
                logger.info(f"[IMAGE] Model: {self.model_name}")
                logger.info(f"[IMAGE] Device: {self._device}")
                logger.info(f"[IMAGE] Scene: {scene_index}")
                logger.info(
                    f"[IMAGE] Resolution: {target_size[0]}x{target_size[1]}"
                )
                logger.info(f"[IMAGE] Steps: {num_steps}")
                logger.info(f"[IMAGE] Cache: MISS")
                logger.info(f"[IMAGE] Generation: {duration:.2f}s")
                logger.info(f"[IMAGE] Output: {cache_path}")
                print(
                    f"[IMAGE] SD-Turbo generation completed in {duration:.2f}s -> {cache_path.name}"
                )

                return cache_path

        except Exception as e:
            logger.error(f"[IMAGE] SD-Turbo generation error: {e}")
            print(f"[IMAGE] SD-Turbo generation error: {e}")

        return None

    def clear_memory(self):
        """Free memory and clear CUDA cache if available."""
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class StableDiffusionManager:
    """Backwards-compatibility wrapper for SDTurboGenerator."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        config: Optional[Config] = None,
    ):
        self.generator = SDTurboGenerator(
            config=config, model_path=model_path, device=device
        )

    def generate_image(
        self,
        prompt: str,
        keyword: Optional[str] = None,
        size: Tuple[int, int] = (1080, 1920),
    ) -> Optional[Path]:
        return self.generator.generate(
            prompt=prompt, keyword=keyword, target_size=size
        )
