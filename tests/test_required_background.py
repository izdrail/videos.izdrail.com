"""Dependency-light tests execute real renderer methods with fake AI only.

AST extraction avoids loading TTS/torch/spacy for FFmpeg boundary tests.
It does not replace method code or the actual FFmpeg process.
"""
import ast
import math
import os
import subprocess
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from PIL import Image
from core.visual import AIImageProvider
from core.utils.video import validate_background_asset, get_video_duration


@pytest.fixture
def renderer(tmp_path):
    module = ast.parse(Path("core/video/ffmpeg_generator.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "FFmpegVideoGenerator")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {
        "_generate_background_image", "_create_slide_with_ffmpeg"}]
    namespace = dict(Path=Path, Dict=dict, Optional=__import__('typing').Optional,
                     AIImageProvider=AIImageProvider, validate_background_asset=validate_background_asset,
                     get_video_duration=get_video_duration, subprocess=subprocess,
                     uuid=uuid, math=math, os=os, traceback=traceback)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "renderer", "exec"), namespace)
    instance = namespace["FFmpegVideoGenerator"]()
    instance.config = SimpleNamespace(TEMP_DIR=tmp_path)
    instance.video_width, instance.video_height = 160, 240
    instance.video_preset, instance.video_crf = "ultrafast", 28
    instance.logo_path = None
    instance.INTRO_VIDEO_PATH = tmp_path / "missing-intro.mp4"
    instance.sd_manager = MagicMock()
    image = tmp_path / "generated.png"
    Image.new("RGB", (160, 240), "lime").save(image)
    instance.sd_manager.generate.return_value = image
    return instance


def audio(tmp_path):
    path = tmp_path / "audio.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=440:d=0.4", str(path)], check=True, capture_output=True)
    return path


@pytest.mark.parametrize("kind", ["missing", "invalid", "intro", "cta", "split"])
def test_missing_visual_uses_generated_image_in_real_ffmpeg(renderer, tmp_path, kind):
    if not __import__('shutil').which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    input_audio = audio(tmp_path)
    background = tmp_path / "invalid.mp4" if kind == "invalid" else None
    overlay = None
    if kind == "split":
        overlay = tmp_path / "overlay.mp4"
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=0.4", "-pix_fmt", "yuv420p", str(overlay)], check=True, capture_output=True)
    output = tmp_path / "output.mp4"
    result = renderer._create_slide_with_ffmpeg(
        "A green forest", input_audio, background, output, 1,
        is_intro=kind == "intro", is_cta=kind == "cta", hide_text=True,
        circle_video=overlay, overlay_shape="Split Screen" if overlay else "Circle", export_fps=10,
    )
    assert result == output
    renderer.sd_manager.generate.assert_called_once()
    assert get_video_duration(output) > 0
    assert not list(tmp_path.glob("grad*.png"))


def test_generation_failure_never_renders_gradient(renderer, tmp_path):
    renderer.sd_manager.generate.return_value = None
    output = tmp_path / "output.mp4"
    result = renderer._create_slide_with_ffmpeg("Forest", audio(tmp_path), None, output, 1, hide_text=True)
    assert result is None and not output.exists()
    assert not list(tmp_path.glob("grad*.png"))


def test_complete_render_is_required():
    tree = ast.parse(Path("core/video/ffmpeg_generator.py").read_text())
    boundaries = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "len(valid_slide_paths) != len(slides_data)"]
    assert len(boundaries) == 1
    assert isinstance(boundaries[0].body[0], ast.Raise)


def test_real_video_does_not_generate_image(renderer, tmp_path):
    video = tmp_path / "stock.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x240:r=10:d=0.4", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
    result = renderer._create_slide_with_ffmpeg("Ocean", audio(tmp_path), video, tmp_path / "out.mp4", 1, hide_text=True)
    assert result is not None
    renderer.sd_manager.generate.assert_not_called()


def test_sd_cache_is_resolution_specific_and_writes_atomically(tmp_path):
    # Exercise real generator code without importing/downloading torch models.
    import contextlib
    import hashlib
    import logging
    import threading
    import time
    tree = ast.parse(Path("core/ai/stable_diffusion.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SDTurboGenerator")
    ns = dict(Optional=__import__('typing').Optional, Tuple=__import__('typing').Tuple,
              Path=Path, Config=object, threading=threading, hashlib=hashlib,
              logging=logging, logger=logging.getLogger("sd-test"), time=time, uuid=uuid,
              SD_TURBO_AVAILABLE=True, Image=Image, ImageOps=__import__('PIL.ImageOps', fromlist=['fit']),
              torch=SimpleNamespace(no_grad=contextlib.nullcontext))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "sd", "exec"), ns)
    config = SimpleNamespace(IMAGE_GENERATION_CACHE_DIR=tmp_path, ROOT_DIR=tmp_path)
    generator = ns["SDTurboGenerator"](config=config)
    generator._load_model = lambda: None
    generator._pipeline = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (32, 32), "red")]))
    first = generator.generate("forest", target_size=(160, 240))
    second = generator.generate("forest", target_size=(240, 160))
    assert first != second
    assert Image.open(first).size == (160, 240)
    assert Image.open(second).size == (240, 160)
    assert generator.generate("forest", target_size=(160, 240)) == first
    assert generator._pipeline.call_count == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_video_only_lookup_ignores_cached_photo(tmp_path):
    import threading
    tree = ast.parse(Path("core/media/manager.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MediaManager")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_random_media"]
    ns = dict(List=__import__('typing').List, Optional=__import__('typing').Optional, Path=Path, random=__import__('random'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "media", "exec"), ns)
    manager = ns["MediaManager"]()
    photo = tmp_path / "cached.jpg"
    photo.write_bytes(b"photo")
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    manager.search_cache = {("forest", None): str(photo)}
    manager._selection_lock = threading.RLock()
    manager._used_asset_ids = set()
    manager.config = SimpleNamespace(VIDEOS_DIR=tmp_path)
    manager._search_and_download = MagicMock(return_value=video)
    result = manager.get_random_media(["forest"], videos_only=True)
    assert result == video
    assert manager._search_and_download.call_args.kwargs["videos_only"] is True


def test_video_only_search_skips_photo_providers_and_candidates(tmp_path):
    import concurrent.futures
    import threading
    import time
    from collections import defaultdict
    tree = ast.parse(Path("core/media/manager.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MediaManager")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_search_and_download"]
    video_url = "https://example.invalid/clip.mp4"
    candidate = {"url": video_url, "ext": "mp4", "media_type": "video", "_source": "Video"}
    def rerank(**kwargs):
        assert kwargs["candidates_by_source"] == {"Video": [candidate]}
        return [candidate]
    ns = dict(List=__import__('typing').List, Dict=dict, Optional=__import__('typing').Optional,
              Path=Path, random=__import__('random'), concurrent=concurrent, time=time,
              rerank_pooled_candidates=rerank, candidate_identity=lambda *a: video_url,
              path_identity=lambda p: str(p))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "media", "exec"), ns)
    manager = ns["MediaManager"]()
    image_only = SimpleNamespace(search_images=MagicMock(return_value=[]))
    video_provider = SimpleNamespace(search_videos=lambda *a, **k: [candidate, {"url": "photo", "ext": "jpg", "media_type": "image"}],
                                    download_video=MagicMock(return_value=True))
    manager.apis = {"Photo": image_only, "Video": video_provider}
    manager._get_ordered_sources = lambda source: ["Photo", "Video"]
    manager.config = SimpleNamespace(VIDEOS_DIR=tmp_path)
    manager.entity_handler = None
    manager._used_media_urls = set()
    manager._source_usage_count = defaultdict(int)
    manager._source_last_used = {}
    manager._selected_media = []
    manager._selection_lock = threading.RLock()
    manager._reserved_candidate_ids = set()
    manager._used_asset_ids = set()
    manager.source_success_counts = defaultdict(int)
    manager.successful_keywords = defaultdict(int)
    manager.search_cache = {}
    result = manager._search_and_download("forest", None, videos_only=True)
    assert result.suffix == ".mp4"
    image_only.search_images.assert_not_called()
    video_provider.download_video.assert_called_once()


def test_renderer_has_no_gradient_background_branch():
    tree = ast.parse(Path("core/video/ffmpeg_generator.py").read_text())
    renderer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_create_slide_with_ffmpeg")
    calls = [n for n in ast.walk(renderer) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Name) and n.func.id == "create_gradient_image" for n in calls)
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == "_generate_background_image" for n in calls)
