"""
Stock media provider.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from .asset import VisualAsset, AssetType
from .provider import VisualProvider


class StockMediaProvider(VisualProvider):
    """Stock media provider using MediaManager / local video stock."""

    def __init__(self, media_manager=None, background_video_fetcher=None):
        self.media_manager = media_manager
        self.background_video_fetcher = background_video_fetcher

    def get_visual(self, context: Dict[str, Any], **kwargs) -> VisualAsset:
        duration = kwargs.get("duration", 3.0)

        # If a background_video_fetcher callable is passed, use it
        if self.background_video_fetcher:
            video_path = self.background_video_fetcher(
                keyword=context.get("keyword"),
                sentence=context.get("sentence"),
                preferred_source=context.get("preferred_source"),
                theme=context.get("theme"),
                entity=context.get("entity"),
                script_id=context.get("script_id"),
                sentence_idx=context.get("sentence_idx"),
                candidate_keywords=context.get("candidate_keywords"),
            )
            if video_path:
                return VisualAsset(
                    asset_type=AssetType.VIDEO,
                    path=video_path,
                    duration=duration,
                    metadata={"source": "stock_fetcher"},
                )

        # Fallback to direct media manager call if available
        if self.media_manager and hasattr(self.media_manager, "get_random_media"):
            search_keywords = []
            if context.get("keyword"):
                search_keywords.append(context["keyword"])
            elif context.get("sentence"):
                search_keywords.append(context["sentence"])

            if search_keywords:
                video_path = self.media_manager.get_random_media(
                    search_keywords,
                    preferred_source=context.get("preferred_source"),
                    theme=context.get("theme"),
                    entity=context.get("entity"),
                )
                if video_path:
                    return VisualAsset(
                        asset_type=AssetType.VIDEO,
                        path=video_path,
                        duration=duration,
                        metadata={"source": "media_manager"},
                    )

        # If no video path returned, return gradient fallback
        return VisualAsset(
            asset_type=AssetType.GRADIENT,
            duration=duration,
            metadata={"source": "gradient_fallback"},
        )
