"""
CLIP Scorer
Uses a pre-trained CLIP model (clip-ViT-B-32 via sentence-transformers)
to calculate visual semantic similarity between text (sentences/keywords)
and image/video frame thumbnails.
"""

import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import requests

logger = logging.getLogger(__name__)

_CLIP_MODEL_INSTANCE = None


class CLIPScorer:
    """Computes cosine similarity between text and images using clip-ViT-B-32."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name

    def _get_model(self):
        global _CLIP_MODEL_INSTANCE
        if _CLIP_MODEL_INSTANCE is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("[CLIPScorer] Loading CLIP model '%s'...", self.model_name)
                _CLIP_MODEL_INSTANCE = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("[CLIPScorer] Failed to load CLIP model: %s", e)
                _CLIP_MODEL_INSTANCE = False
        return _CLIP_MODEL_INSTANCE if _CLIP_MODEL_INSTANCE else None

    def load_image(self, image_input: Union[str, Path, Image.Image, Any]) -> Optional[Image.Image]:
        """Loads an image from a URL, file path, video thumbnail, or PIL Image object."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (str, Path)):
            image_str = str(image_input).strip()
            if image_str.startswith("http://") or image_str.startswith("https://"):
                try:
                    resp = requests.get(
                        image_str, timeout=5, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    if resp.status_code == 200:
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    logger.debug(
                        "[CLIPScorer] Failed to download image from %s: %s",
                        image_str,
                        e,
                    )
                    return None
            else:
                try:
                    p = Path(image_str)
                    if p.exists():
                        if p.suffix.lower() in [".mp4", ".mov", ".avi", ".webm"]:
                            from core.utils.video import get_random_middle_frame

                            temp_thumb = p.parent / f"clip_thumb_{p.stem}.jpg"
                            if get_random_middle_frame(p, temp_thumb):
                                img = Image.open(temp_thumb).convert("RGB")
                                temp_thumb.unlink(missing_ok=True)
                                return img
                            return None
                        return Image.open(p).convert("RGB")
                except Exception as e:
                    logger.debug(
                        "[CLIPScorer] Failed to load local image %s: %s", image_str, e
                    )
                    return None
        return None

    def compute_similarity(
        self, text: str, image_input: Union[str, Path, Image.Image, Any]
    ) -> float:
        """Computes similarity in range [0.0, 1.0] between text and image_input."""
        if not text or not text.strip():
            return 0.5

        model = self._get_model()
        if not model:
            return 0.5

        img = self.load_image(image_input)
        if img is None:
            return 0.5

        try:
            from sentence_transformers import util

            text_emb = model.encode(text, convert_to_tensor=True)
            image_emb = model.encode(img, convert_to_tensor=True)
            cos_sim = util.cos_sim(text_emb, image_emb).item()
            # Normalize cosine similarity from [-1, 1] to [0, 1]
            score = float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))
            return score
        except Exception as e:
            logger.warning("[CLIPScorer] Similarity calculation failed: %s", e)
            return 0.5

    def score_media_candidates(
        self, text: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Scores candidate media dicts based on CLIP text-image similarity.
        Appends 'clip_score' to each candidate dict.
        """
        for cand in candidates:
            img_src = (
                cand.get("thumbnail")
                or cand.get("url")
                or cand.get("path")
                or cand.get("media", {}).get("thumbnail")
                or cand.get("media", {}).get("url")
            )
            score = self.compute_similarity(text, img_src) if img_src else 0.5
            cand["clip_score"] = score
        return candidates
