"""
Visual asset representation for video generation pipeline.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union


class AssetType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    COLOR = "color"
    GRADIENT = "gradient"


@dataclass
class VisualAsset:
    asset_type: AssetType
    path: Optional[Union[str, Path]] = None
    color: Optional[tuple] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_video(self) -> bool:
        return self.asset_type == AssetType.VIDEO

    def is_image(self) -> bool:
        return self.asset_type == AssetType.IMAGE

    def is_color(self) -> bool:
        return self.asset_type == AssetType.COLOR

    def is_gradient(self) -> bool:
        return self.asset_type == AssetType.GRADIENT

    def get_path_obj(self) -> Optional[Path]:
        if self.path:
            return Path(self.path)
        return None
