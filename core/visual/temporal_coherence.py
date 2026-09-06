"""
Temporal Coherence Optimizer
Evaluates brightness and color distribution of clips to avoid jarring visual scene changes between consecutive slides.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TemporalCoherenceOptimizer:
    """Optimizes video sequence selection to maintain smooth brightness and color continuity across slides."""

    def __init__(self, max_brightness_diff: float = 0.35, max_color_diff: float = 0.4):
        self.max_brightness_diff = max_brightness_diff
        self.max_color_diff = max_color_diff

    @staticmethod
    def extract_frame_features(
        image_or_video_path: Union[str, Path, Image.Image]
    ) -> Optional[Dict[str, Any]]:
        """Extracts average brightness (0..1) and 3D RGB color histogram from an image or video file."""
        img = None
        if isinstance(image_or_video_path, Image.Image):
            img = image_or_video_path.convert("RGB")
        elif isinstance(image_or_video_path, (str, Path)):
            p = Path(image_or_video_path)
            if not p.exists():
                return None
            try:
                if p.suffix.lower() in [".mp4", ".mov", ".avi", ".webm"]:
                    from core.utils.video import get_random_middle_frame

                    temp_thumb = p.parent / f"tc_thumb_{p.stem}.jpg"
                    if get_random_middle_frame(p, temp_thumb):
                        img = Image.open(temp_thumb).convert("RGB")
                        temp_thumb.unlink(missing_ok=True)
                else:
                    img = Image.open(p).convert("RGB")
            except Exception as e:
                logger.debug(
                    "[TemporalCoherence] Failed frame extraction for %s: %s", p, e
                )
                return None

        if img is None:
            return None

        try:
            # Resize image to standard size for fast computation
            img_small = img.resize((100, 100))
            arr = np.array(img_small, dtype=np.float32) / 255.0

            # Luminance / average brightness Y = 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            avg_brightness = float(np.mean(luminance))

            # Color distribution: 8-bin histogram per RGB channel
            r_hist, _ = np.histogram(arr[:, :, 0], bins=8, range=(0.0, 1.0))
            g_hist, _ = np.histogram(arr[:, :, 1], bins=8, range=(0.0, 1.0))
            b_hist, _ = np.histogram(arr[:, :, 2], bins=8, range=(0.0, 1.0))

            color_hist = np.concatenate([r_hist, g_hist, b_hist]).astype(np.float32)
            hist_sum = np.sum(color_hist)
            if hist_sum > 0:
                color_hist /= hist_sum

            return {
                "brightness": avg_brightness,
                "color_hist": color_hist,
            }
        except Exception as e:
            logger.warning("[TemporalCoherence] Error extracting features: %s", e)
            return None

    def compute_transition_difference(
        self, feat1: Dict[str, Any], feat2: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates difference in brightness and color distribution between two feature sets."""
        if not feat1 or not feat2:
            return {"brightness_diff": 0.0, "color_diff": 0.0, "total_diff": 0.0}

        brightness_diff = abs(feat1["brightness"] - feat2["brightness"])

        # Histogram distance
        hist1 = feat1["color_hist"]
        hist2 = feat2["color_hist"]
        color_diff = float(0.5 * np.sum(np.abs(hist1 - hist2)))

        total_diff = 0.5 * brightness_diff + 0.5 * color_diff
        return {
            "brightness_diff": brightness_diff,
            "color_diff": color_diff,
            "total_diff": total_diff,
        }

    def optimize_slide_clips(
        self,
        chosen_clips: List[Optional[Path]],
        candidate_clips_per_slide: Optional[Dict[int, List[Path]]] = None,
    ) -> List[Optional[Path]]:
        """Post-processes selected slide clips to eliminate abrupt brightness/color transitions.
        If consecutive clips differ beyond thresholds, picks an alternative candidate for that slide
        that has the smallest visual transition difference to the previous slide.
        """
        if len(chosen_clips) <= 1:
            return chosen_clips

        optimized = list(chosen_clips)
        features = [
            self.extract_frame_features(clip) if clip else None for clip in optimized
        ]

        for i in range(1, len(optimized)):
            prev_feat = features[i - 1]
            curr_feat = features[i]

            if not prev_feat or not curr_feat:
                continue

            diff = self.compute_transition_difference(prev_feat, curr_feat)

            if (
                diff["brightness_diff"] > self.max_brightness_diff
                or diff["color_diff"] > self.max_color_diff
            ):
                logger.info(
                    "[TemporalCoherence] Abrupt transition at slide %d: brightness_diff=%.2f, color_diff=%.2f",
                    i,
                    diff["brightness_diff"],
                    diff["color_diff"],
                )

                cands = (
                    candidate_clips_per_slide.get(i, [])
                    if candidate_clips_per_slide
                    else []
                )

                best_cand = None
                best_cand_feat = None
                min_cand_diff = diff["total_diff"]

                for cand in cands:
                    if cand == optimized[i]:
                        continue
                    cand_feat = self.extract_frame_features(cand)
                    if not cand_feat:
                        continue
                    cand_diff = self.compute_transition_difference(prev_feat, cand_feat)
                    if cand_diff["total_diff"] < min_cand_diff:
                        min_cand_diff = cand_diff["total_diff"]
                        best_cand = cand
                        best_cand_feat = cand_feat

                if best_cand:
                    logger.info(
                        "[TemporalCoherence] Replaced slide %d clip with closer candidate: %s (total_diff reduced from %.2f to %.2f)",
                        i,
                        getattr(best_cand, "name", str(best_cand)),
                        diff["total_diff"],
                        min_cand_diff,
                    )
                    optimized[i] = best_cand
                    features[i] = best_cand_feat

        return optimized
