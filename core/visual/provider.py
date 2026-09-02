"""
Visual provider abstract interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .asset import VisualAsset


class VisualProvider(ABC):
    """Abstract base class for visual source providers."""

    @abstractmethod
    def get_visual(self, context: Dict[str, Any], **kwargs) -> VisualAsset:
        """Get visual asset for given context."""
        pass

    def get_multiple(self, contexts: List[Dict[str, Any]], **kwargs) -> List[VisualAsset]:
        """Get visual assets for multiple contexts."""
        return [self.get_visual(ctx, **kwargs) for ctx in contexts]
