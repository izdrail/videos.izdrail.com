"""
Visual Provider Factory.
"""
from typing import Optional
from .provider import VisualProvider
from .stock_provider import StockMediaProvider
from .ai_provider import AIImageProvider
from .mixed_provider import MixedProvider
from ..ai.prompt_generator import PromptGenerator


class VisualProviderFactory:
    """Factory for instantiating VisualProvider based on source type."""

    @staticmethod
    def create(
        source_type: str,
        config=None,
        sd_generator=None,
        media_manager=None,
        background_video_fetcher=None,
    ) -> VisualProvider:
        st = (source_type or "stock").lower().replace(" ", "_").strip()

        stock_prov = StockMediaProvider(
            media_manager=media_manager,
            background_video_fetcher=background_video_fetcher,
        )

        prompt_gen = PromptGenerator(config)
        ai_prov = AIImageProvider(
            sd_generator=sd_generator,
            prompt_generator=prompt_gen,
            fallback_provider=stock_prov,
        )

        if st in ["ai", "ai_generated_images", "ai_generated", "ai_images"]:
            return ai_prov
        elif st == "mixed":
            ratio = getattr(config, "MIXED_MODE_IMAGE_RATIO", 0.5) if config else 0.5
            return MixedProvider(
                stock_provider=stock_prov,
                ai_provider=ai_prov,
                ratio=ratio,
            )
        else:
            # Default to stock
            return stock_prov
