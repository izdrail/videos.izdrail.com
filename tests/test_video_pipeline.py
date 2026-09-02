"""
Integration tests for video generation pipeline with visual provider selection.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.ai.stable_diffusion import SDTurboGenerator
from core.visual import VisualProviderFactory, AssetType, VisualAsset


class TestVideoPipelineVisualIntegration:
    def test_pipeline_stock_visual_source(self, tmp_path):
        v_file = tmp_path / "stock_bg.mp4"
        v_file.touch()

        factory = VisualProviderFactory()
        provider = factory.create(
            "stock",
            background_video_fetcher=lambda **kwargs: v_file,
        )

        ctx = {"keyword": "technology", "sentence": "AI is changing the world"}
        asset = provider.get_visual(ctx)

        assert asset.is_video() is True
        assert asset.get_path_obj() == v_file

    def test_pipeline_ai_visual_source(self, tmp_path):
        i_file = tmp_path / "ai_bg.png"
        i_file.touch()

        mock_sd = MagicMock(spec=SDTurboGenerator)
        mock_sd.generate.return_value = i_file

        provider = VisualProviderFactory.create(
            "ai",
            sd_generator=mock_sd,
        )

        ctx = {"keyword": "space", "sentence": "Rockets flying to Mars"}
        asset = provider.get_visual(ctx)

        assert asset.is_image() is True
        assert asset.get_path_obj() == i_file

    def test_pipeline_mixed_visual_source(self, tmp_path):
        v_file = tmp_path / "stock_bg.mp4"
        v_file.touch()
        i_file = tmp_path / "ai_bg.png"
        i_file.touch()

        mock_sd = MagicMock(spec=SDTurboGenerator)
        mock_sd.generate.return_value = i_file

        provider = VisualProviderFactory.create(
            "mixed",
            sd_generator=mock_sd,
            background_video_fetcher=lambda **kwargs: v_file,
        )

        contexts = [
            {"sentence": "Slide 1"},
            {"sentence": "Slide 2"},
            {"sentence": "Slide 3"},
        ]

        assets = provider.get_multiple(contexts)
        assert len(assets) == 3
        # Mix should contain at least video or image assets
        types = [a.asset_type for a in assets]
        assert AssetType.VIDEO in types or AssetType.IMAGE in types
