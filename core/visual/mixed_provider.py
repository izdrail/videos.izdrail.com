"""
Mixed provider alternating between stock and AI visuals.
"""
import random
from typing import Dict, Any
from .asset import VisualAsset
from .provider import VisualProvider


class MixedProvider(VisualProvider):
    """Mixed provider combining stock media and AI image generation."""

    def __init__(
        self,
        stock_provider: VisualProvider,
        ai_provider: VisualProvider,
        ratio: float = 0.5,
    ):
        self.stock_provider = stock_provider
        self.ai_provider = ai_provider
        self.ratio = ratio
        self._counter = 0

    def get_visual(self, context: Dict[str, Any], **kwargs) -> VisualAsset:
        # Legacy callers also obey video-first rather than alternating away
        # from available footage. Stock provider normally owns the AI fallback.
        try:
            return self.stock_provider.get_visual(context, **kwargs)
        except RuntimeError:
            return self.ai_provider.get_visual(context, **kwargs)
