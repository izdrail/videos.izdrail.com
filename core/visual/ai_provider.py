"""
AI image provider using SD-Turbo.
"""
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional
from .asset import VisualAsset, AssetType
from .provider import VisualProvider
from ..ai.prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)


class AIImageProvider(VisualProvider):
    """AI image provider using SD-Turbo generator."""

    def __init__(self, sd_generator, prompt_generator: Optional[PromptGenerator] = None, fallback_provider: Optional[VisualProvider] = None):
        self.sd_generator = sd_generator
        self.prompt_generator = prompt_generator or PromptGenerator(
            getattr(sd_generator, "config", None)
        )
        self.fallback_provider = fallback_provider

    def get_visual(self, context: Dict[str, Any], **kwargs) -> VisualAsset:
        sentence = context.get("sentence", "")
        keyword = context.get("keyword")
        entities = context.get("entities")
        if not entities and context.get("entity"):
            entities = [context["entity"]]

        duration = kwargs.get("duration", 3.0)
        target_size = kwargs.get("target_size", (1080, 1920))

        prompt = self.prompt_generator.generate(
            sentence=sentence, keyword=keyword, entities=entities
        )

        try:
            image_path = self.sd_generator.generate(
                prompt=prompt,
                keyword=keyword,
                scene_index=context.get("sentence_idx"),
                target_size=target_size,
            ) if self.sd_generator else None
        except Exception as exc:
            raise RuntimeError("AI background image generation failed; retry the render after checking SD-Turbo.") from exc

        if image_path:
            image_path = Path(image_path)
            if image_path.is_file() and image_path.stat().st_size > 0:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as exc:
                    raise RuntimeError("Generated background image is corrupt; retry generation.") from exc
                return VisualAsset(
                    asset_type=AssetType.IMAGE,
                    path=image_path,
                    duration=duration,
                    metadata={"prompt": prompt, "provider": "sd_turbo"},
                )

        # Do not hide unavailable image generation behind a gradient or a
        # recursive stock/AI fallback. The job must be retriable instead.
        raise RuntimeError(
            "No background image was generated. Check IMAGE_GENERATION_ENABLED, "
            "SD-Turbo model availability and the image-generation logs, then retry."
        )
