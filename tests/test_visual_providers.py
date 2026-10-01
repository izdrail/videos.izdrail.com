"""Video-first backgrounds with required AI images, never gradients."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image
from core.visual import AssetType, VisualAsset, StockMediaProvider, AIImageProvider, MixedProvider, VisualProviderFactory


def generated_file(tmp_path):
    path = tmp_path / "generated.png"
    Image.new("RGB", (64, 96), "green").save(path)
    return path


def test_visual_asset(tmp_path):
    path = generated_file(tmp_path)
    asset = VisualAsset(AssetType.IMAGE, path)
    assert asset.is_image() and not asset.is_video()
    assert asset.get_path_obj() == path


@pytest.mark.parametrize("source", ["stock", "ai", "ai_generated_images", "mixed", None])
def test_all_legacy_modes_choose_video_first(source, tmp_path):
    video = tmp_path / "stock.mp4"
    video.write_bytes(b"video")
    sd = MagicMock()
    provider = VisualProviderFactory.create(source, sd_generator=sd, background_video_fetcher=lambda **kw: video)
    asset = provider.get_visual({"sentence": "A city"})
    assert asset.is_video() and asset.get_path_obj() == video
    sd.generate.assert_not_called()


@pytest.mark.parametrize("source", ["stock", "ai", "mixed"])
def test_missing_video_generates_image(source, tmp_path):
    sd = MagicMock()
    sd.generate.return_value = generated_file(tmp_path)
    provider = VisualProviderFactory.create(source, sd_generator=sd, background_video_fetcher=lambda **kw: None)
    asset = provider.get_visual({"sentence": "A river", "sentence_idx": 3}, target_size=(320, 480))
    assert asset.is_image() and not asset.is_gradient()
    sd.generate.assert_called_once()
    assert sd.generate.call_args.kwargs["target_size"] == (320, 480)
    assert sd.generate.call_args.kwargs["scene_index"] == 3


@pytest.mark.parametrize("failure", [None, "missing", "empty", "exception"])
def test_ai_failure_is_explicit_never_gradient(failure, tmp_path):
    sd = MagicMock()
    if failure == "exception":
        sd.generate.side_effect = ValueError("out of memory")
    elif failure == "empty":
        path = tmp_path / "empty.png"
        path.touch()
        sd.generate.return_value = path
    else:
        sd.generate.return_value = tmp_path / "missing.png" if failure == "missing" else None
    with pytest.raises(RuntimeError, match="background image"):
        AIImageProvider(sd).get_visual({"sentence": "A mountain"})


def test_disabled_generator_fails_explicitly():
    with pytest.raises(RuntimeError, match="IMAGE_GENERATION_ENABLED"):
        VisualProviderFactory.create("stock").get_visual({})


def test_stock_direct_call_requests_only_videos(tmp_path):
    manager = MagicMock()
    manager.get_random_media.return_value = tmp_path / "city.mp4"
    asset = StockMediaProvider(media_manager=manager).get_visual({"keyword": "city"})
    assert asset.is_video()
    assert manager.get_random_media.call_args.kwargs["videos_only"] is True


def test_exhausted_fetcher_does_not_repeat_stock_search(tmp_path):
    manager = MagicMock()
    sd = MagicMock()
    sd.generate.return_value = generated_file(tmp_path)
    provider = VisualProviderFactory.create("stock", media_manager=manager, sd_generator=sd, background_video_fetcher=lambda **kw: None)
    assert provider.get_visual({"keyword": "city"}).is_image()
    manager.get_random_media.assert_not_called()


def test_legacy_mixed_always_prefers_stock(tmp_path):
    stock = MagicMock()
    stock.get_visual.return_value = VisualAsset(AssetType.VIDEO, tmp_path / "v.mp4")
    ai = MagicMock()
    provider = MixedProvider(stock, ai)
    for _ in range(3):
        assert provider.get_visual({}).is_video()
    ai.get_visual.assert_not_called()


def test_corrupt_generated_image_is_not_accepted(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    sd = MagicMock()
    sd.generate.return_value = path
    with pytest.raises(RuntimeError, match="corrupt"):
        AIImageProvider(sd).get_visual({})
