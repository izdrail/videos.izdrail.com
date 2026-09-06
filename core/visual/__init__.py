"""
Visual Abstraction Module.
"""
from .asset import VisualAsset, AssetType
from .provider import VisualProvider
from .stock_provider import StockMediaProvider
from .ai_provider import AIImageProvider
from .mixed_provider import MixedProvider
from .factory import VisualProviderFactory
from .clip_scorer import CLIPScorer
from .temporal_coherence import TemporalCoherenceOptimizer

__all__ = [
    "VisualAsset",
    "AssetType",
    "VisualProvider",
    "StockMediaProvider",
    "AIImageProvider",
    "MixedProvider",
    "VisualProviderFactory",
    "CLIPScorer",
    "TemporalCoherenceOptimizer",
]
