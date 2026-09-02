"""
AI image provider using SD-Turbo.
"""
import logging
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

        image_path = None
        if self.sd_generator:
            image_path = self.sd_generator.generate(
                prompt=prompt,
                keyword=keyword,
                scene_index=context.get("sentence_idx"),
                target_size=target_size,
            )

        if image_path and image_path.exists():
            return VisualAsset(
                asset_type=AssetType.IMAGE,
                path=image_path,
                duration=duration,
                metadata={"prompt": prompt, "provider": "sd_turbo"},
            )

        logger.warning("[AIImageProvider] AI generation failed or returned None. Triggering fallback.")
        if self.fallback_provider:
            return self.fallback_provider.get_visual(context, **kwargs)

        return VisualAsset(
            asset_type=AssetType.GRADIENT,
            duration=duration,
            metadata={"source": "gradient_fallback"},
        )
