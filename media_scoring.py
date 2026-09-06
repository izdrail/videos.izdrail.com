"""
Media Scoring Pipeline
Replaces legacy neuron/bandit evaluation with cross-modal CLIP relevance,
resolution/aspect quality scoring, and global pooled reranking across all media sources.
"""

import time
import io
import os
import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Global model cache for lazy loading
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_CLIP_CONFIGURED_MODEL_NAME = "ViT-B-32"


def _get_clip_model(model_name: str = "ViT-B-32"):
    """
    Lazy loader for open_clip model and preprocessing pipeline.
    Falls back gracefully if CUDA is unavailable or loading fails.
    """
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE, _CLIP_CONFIGURED_MODEL_NAME

    if _CLIP_MODEL is not None and _CLIP_CONFIGURED_MODEL_NAME == model_name:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    try:
        import torch
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {model_name} on device {device}...")

        # Try loading specified model with default pretrained weights
        pretrained_weights = "laion2b_s34b_b79k"
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_weights, device=device
            )
        except Exception:
            # Fallback pretrained weights if laion2b unavailable
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai", device=device
            )

        model.eval()
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
        _CLIP_DEVICE = device
        _CLIP_CONFIGURED_MODEL_NAME = model_name
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    except Exception as e:
        logger.warning(f"Failed to load CLIP model ({model_name}): {e}")
        return None, None, None


@lru_cache(maxsize=512)
def _fetch_thumbnail_image(thumbnail_url: str) -> Optional[Image.Image]:
    """
    Fetch thumbnail image from URL with LRU caching.
    Returns PIL Image or None on failure.
    """
    if not thumbnail_url or not isinstance(thumbnail_url, str):
        return None

    try:
        if thumbnail_url.startswith(("http://", "https://")):
            response = requests.get(thumbnail_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image
        elif os.path.exists(thumbnail_url):
            image = Image.open(thumbnail_url).convert("RGB")
            return image
    except Exception as e:
        logger.debug(f"Failed to fetch thumbnail image from {thumbnail_url}: {e}")

    return None


def clip_relevance_score(text: str, thumbnail_url: Any, model_name: str = "ViT-B-32") -> float:
    """
    Cross-modal text-image relevance score using CLIP.

    Args:
        text: Prompt or narration text segment.
        thumbnail_url: Image URL string, file path, PIL Image, or dictionary containing image/thumbnail info.
        model_name: Configurable CLIP model name (defaults to 'ViT-B-32').

    Returns:
        Float score normalized roughly between 0.0 and 1.0. Returns 0.5 (neutral fallback) on error.
    """
    if not text or not text.strip():
        return 0.5

    # Extract URL or PIL Image if dict passed
    image_input = thumbnail_url
    if isinstance(thumbnail_url, dict):
        image_input = (
            thumbnail_url.get("thumbnail")
            or thumbnail_url.get("image")
            or thumbnail_url.get("url")
        )

    pil_image = None
    if isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    elif isinstance(image_input, str):
        pil_image = _fetch_thumbnail_image(image_input)
    elif isinstance(image_input, Path):
        pil_image = _fetch_thumbnail_image(str(image_input))

    if pil_image is None:
        return 0.5

    try:
        import torch
        import open_clip

        model, preprocess, device = _get_clip_model(model_name)
        if model is None or preprocess is None:
            return 0.5

        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        text_tokens = open_clip.tokenize([text[:256]]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (text_features @ image_features.T).item()

        # Map similarity (typically 0.1 to 0.35 for ViT-B-32) to roughly [0.0, 1.0]
        # Soft min-max scaling where 0.15 is neutral (0.5), range [0.0, 0.30] -> [0.0, 1.0]
        score = max(0.0, min(1.0, (similarity - 0.05) / 0.25))
        return float(score)

    except Exception as e:
        logger.warning(f"Error computing CLIP relevance score: {e}")
        return 0.5


def compute_quality_score(
    media: dict,
    target_w: int = 1080,
    target_h: int = 1920
) -> float:
    """
    Calculate media quality score based on resolution distance and aspect-ratio alignment.

    Args:
        media: Media dictionary containing 'width' and 'height'.
        target_w: Expected target width (default 1080).
        target_h: Expected target height (default 1920).

    Returns:
        Float quality score between 0.0 and 1.0.
    """
    if not isinstance(media, dict):
        return 0.5

    w = media.get("width") or 0
    h = media.get("height") or 0

    if w <= 0 or h <= 0:
        return 0.5

    # Target aspect ratio check
    target_is_portrait = target_h >= target_w
    media_is_portrait = h >= w

    # Aspect alignment penalty: wrong orientation gets heavily penalized
    orientation_factor = 1.0 if (target_is_portrait == media_is_portrait) else 0.3

    # Resolution fit score
    target_pixels = target_w * target_h
    media_pixels = w * h

    # Total pixel coverage ratio capped at 1.0
    coverage_score = min(1.0, media_pixels / target_pixels)

    # Resolution proximity score (penalizes hugely oversized or tiny resolution clips)
    res_diff = abs(w - target_w) / max(target_w, 1) + abs(h - target_h) / max(target_h, 1)
    proximity_score = max(0.2, 1.0 - (res_diff * 0.2))

    quality = (coverage_score * 0.6 + proximity_score * 0.4) * orientation_factor
    return max(0.0, min(1.0, float(quality)))


def rerank_pooled_candidates(
    narration_text: str,
    candidates_by_source: Dict[str, List[dict]],
    used_urls: Optional[set] = None,
    source_usage_count: Optional[Dict[str, int]] = None,
    source_last_used: Optional[Dict[str, float]] = None,
    now: Optional[float] = None,
    preferred_source: Optional[str] = None,
    top_k: int = 1,
    target_width: int = 1080,
    target_height: int = 1920,
) -> List[dict]:
    """
    Pool candidate media items across all sources and globally rerank them in a single stage.

    Scoring formula:
        Relevance (CLIP): 55%
        Quality (resolution/aspect): 25%
        Freshness (time since source used): 10%
        Diversity (source usage count): 10%
        Preferred source boost: +0.15

    Args:
        narration_text: Text snippet to score relevance against.
        candidates_by_source: Map of source_name -> list of media candidate dicts.
        used_urls: Set of previously used media URLs to exclude or penalize.
        source_usage_count: Map of source_name -> integer usage count.
        source_last_used: Map of source_name -> timestamp of last usage.
        now: Current timestamp (defaults to time.time()).
        preferred_source: Preferred source name for boost.
        top_k: Number of top candidate items to return.
        target_width: Desired video width.
        target_height: Desired video height.

    Returns:
        List of top_k candidate dicts, enriched with '_source' and '_score' fields.
    """
    if now is None:
        now = time.time()

    used_urls = used_urls or set()
    source_usage_count = source_usage_count or {}
    source_last_used = source_last_used or {}

    pooled_candidates = []

    for source_name, candidate_list in candidates_by_source.items():
        if not candidate_list:
            continue

        usage = source_usage_count.get(source_name, 0)
        last_used = source_last_used.get(source_name, 0)

        freshness = 1.0 / (now - last_used + 1.0)
        diversity = 1.0 / (usage + 1.0)
        preferred_boost = 0.15 if (preferred_source and source_name == preferred_source) else 0.0

        for candidate in candidate_list:
            if not isinstance(candidate, dict):
                continue

            candidate_url = candidate.get("url")
            if candidate_url and candidate_url in used_urls:
                # Exclude already used media
                continue

            # Create a shallow copy of candidate to enrich with scores
            item = dict(candidate)

            # Calculate individual signal scores
            relevance = clip_relevance_score(
                narration_text,
                item.get("thumbnail") or item.get("image") or candidate_url
            )
            quality = compute_quality_score(item, target_w=target_width, target_h=target_height)

            # DB clip performance score multiplier if available
            perf_score = 0.0
            try:
                from core.database import DB
                if candidate_url:
                    perf_score = DB.get_clip_performance_score(
                        media_url=candidate_url, source=source_name
                    )
            except Exception:
                pass

            total_score = (
                relevance * 0.55
                + quality * 0.25
                + freshness * 0.10
                + diversity * 0.10
                + perf_score * 0.05
                + preferred_boost
            )

            item["_source"] = source_name
            item["_score"] = float(total_score)
            item["_relevance_score"] = float(relevance)
            item["_quality_score"] = float(quality)

            pooled_candidates.append(item)

    # Sort all pooled candidates globally by _score descending
    pooled_candidates.sort(key=lambda x: x["_score"], reverse=True)

    return pooled_candidates[:top_k]
