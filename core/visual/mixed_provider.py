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
        self._counter += 1

        # Deterministic alternation if ratio is 0.5, else random based on ratio
        if self.ratio == 0.5:
            use_ai = (self._counter % 2 == 0)
        else:
            use_ai = (random.random() < self.ratio)

        if use_ai:
            asset = self.ai_provider.get_visual(context, **kwargs)
            # If AI provider fails / falls back to gradient, try stock provider
            if asset.is_gradient() and self.stock_provider:
                stock_asset = self.stock_provider.get_visual(context, **kwargs)
                if not stock_asset.is_gradient():
                    return stock_asset
            return asset
        else:
            asset = self.stock_provider.get_visual(context, **kwargs)
            if asset.is_gradient() and self.ai_provider:
                ai_asset = self.ai_provider.get_visual(context, **kwargs)
                if not ai_asset.is_gradient():
                    return ai_asset
            return asset
