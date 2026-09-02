"""
Unit tests for visual providers and VisualProviderFactory.
"""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.visual import (
    AssetType,
    VisualAsset,
    StockMediaProvider,
    AIImageProvider,
    MixedProvider,
    VisualProviderFactory,
)


class TestVisualAsset:
    def test_asset_properties(self, tmp_path):
        f = tmp_path / "test.png"
        f.touch()

        asset = VisualAsset(asset_type=AssetType.IMAGE, path=f, duration=5.0)
        assert asset.is_image() is True
        assert asset.is_video() is False
        assert asset.get_path_obj() == f


class TestStockMediaProvider:
    def test_get_visual_video(self, tmp_path):
        dummy_video = tmp_path / "sample.mp4"
        dummy_video.touch()

        mock_fetcher = MagicMock(return_value=dummy_video)
        provider = StockMediaProvider(background_video_fetcher=mock_fetcher)

        ctx = {"keyword": "nature", "sentence": "A beautiful river in the forest"}
        asset = provider.get_visual(ctx)

        assert asset.is_video() is True
        assert asset.get_path_obj() == dummy_video

    def test_get_visual_fallback(self):
        mock_fetcher = MagicMock(return_value=None)
        provider = StockMediaProvider(background_video_fetcher=mock_fetcher)

        asset = provider.get_visual({"keyword": "nonexistent"})
        assert asset.is_gradient() is True


class TestAIImageProvider:
    def test_get_visual_success(self, tmp_path):
        img_path = tmp_path / "gen.png"
        img_path.touch()

        mock_sd = MagicMock()
        mock_sd.generate.return_value = img_path

        provider = AIImageProvider(sd_generator=mock_sd)
        asset = provider.get_visual({"sentence": "A robotic arm assembling a car"})

        assert asset.is_image() is True
        assert asset.get_path_obj() == img_path

    def test_get_visual_failure_triggers_fallback(self, tmp_path):
        fallback_path = tmp_path / "stock.mp4"
        fallback_path.touch()

        fallback_provider = StockMediaProvider(
            background_video_fetcher=MagicMock(return_value=fallback_path)
        )

        mock_sd = MagicMock()
        mock_sd.generate.return_value = None

        provider = AIImageProvider(
            sd_generator=mock_sd, fallback_provider=fallback_provider
        )
        asset = provider.get_visual({"sentence": "Test sentence"})

        assert asset.is_video() is True
        assert asset.get_path_obj() == fallback_path


class TestMixedProvider:
    def test_alternation(self, tmp_path):
        stock_path = tmp_path / "stock.mp4"
        stock_path.touch()
        ai_path = tmp_path / "ai.png"
        ai_path.touch()

        stock_prov = StockMediaProvider(
            background_video_fetcher=MagicMock(return_value=stock_path)
        )
        mock_sd = MagicMock()
        mock_sd.generate.return_value = ai_path
        ai_prov = AIImageProvider(sd_generator=mock_sd)

        mixed_prov = MixedProvider(
            stock_provider=stock_prov, ai_provider=ai_prov, ratio=0.5
        )

        # Call 1 -> ratio 0.5 alternate: _counter=1 (odd -> stock)
        a1 = mixed_prov.get_visual({"sentence": "First slide"})
        assert a1.is_video() is True

        # Call 2 -> _counter=2 (even -> AI)
        a2 = mixed_prov.get_visual({"sentence": "Second slide"})
        assert a2.is_image() is True


class TestVisualProviderFactory:
    def test_create_stock(self):
        prov = VisualProviderFactory.create("stock")
        assert isinstance(prov, StockMediaProvider)

    def test_create_ai(self):
        prov = VisualProviderFactory.create("ai_generated_images")
        assert isinstance(prov, AIImageProvider)

    def test_create_mixed(self):
        prov = VisualProviderFactory.create("mixed")
        assert isinstance(prov, MixedProvider)
