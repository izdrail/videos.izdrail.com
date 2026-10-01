"""
Visual Provider Factory.
"""
from .provider import VisualProvider
from .stock_provider import StockMediaProvider
from .ai_provider import AIImageProvider
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
        # All scene backgrounds follow the owner's video-first policy. Legacy
        # AI/mixed settings remain accepted, but cannot bypass available footage.
        ai_prov = AIImageProvider(
            sd_generator=sd_generator,
            prompt_generator=PromptGenerator(config),
        )
        return StockMediaProvider(
            media_manager=media_manager,
            background_video_fetcher=background_video_fetcher,
            fallback_provider=ai_prov,
        )
