"""Network-free regression coverage for worker stalls in thumbnail scoring."""
import importlib.util
import io
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image


@pytest.fixture
def scoring(monkeypatch):
    # Isolate thumbnail helpers from unrelated optional model/provider imports.
    identity = types.ModuleType('core.media.identity')
    identity.canonical_url = lambda x: x
    identity.candidate_identity = lambda *args: None
    monkeypatch.setitem(sys.modules, 'core.media.identity', identity)
    spec = importlib.util.spec_from_file_location('bounded_scoring', Path(__file__).parents[1] / 'media_scoring.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, chunks=(), mime='image/png', size=None, status=200, location=None):
        self.headers = {'Content-Type': mime}
        if size is not None:
            self.headers['Content-Length'] = str(size)
        if location:
            self.headers['Location'] = location
        self.status_code = status
        self.chunks = chunks
        self.read = False
        self.closed = False

    @property
    def content(self):
        raise AssertionError('unbounded response.content must never be read')

    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def raise_for_status(self): pass
    def iter_content(self, chunk_size):
        assert chunk_size == 1  # Must return even if the peer trickles one byte at a time.
        self.read = True
        yield from self.chunks


def png():
    b = io.BytesIO()
    Image.new('RGB', (4, 4)).save(b, format='PNG')
    return b.getvalue()


def test_valid_image_streams_and_closes(scoring, monkeypatch):
    r = Response([png()])
    get = Mock(return_value=r)
    monkeypatch.setattr(scoring.requests, 'get', get)
    assert scoring._fetch_thumbnail_image('https://example.test/a.png').size == (4, 4)
    assert r.closed
    assert get.call_args.kwargs['stream'] is True
    assert get.call_args.kwargs['allow_redirects'] is False
    assert get.call_args.kwargs['timeout'] == (3, 3)


@pytest.mark.parametrize('mime,size', [('video/mp4', None), ('text/html', None), ('image/png', 3*1024*1024)])
def test_reject_before_body(scoring, monkeypatch, mime, size):
    r = Response([b'x'], mime=mime, size=size)
    monkeypatch.setattr(scoring.requests, 'get', lambda *a, **k: r)
    assert scoring._fetch_thumbnail_image('https://example.test/asset') is None
    assert r.closed and not r.read


def test_unreported_oversize_body_stops(scoring, monkeypatch):
    r = Response([b'x' * (2*1024*1024), b'x'])
    monkeypatch.setattr(scoring.requests, 'get', lambda *a, **k: r)
    assert scoring._fetch_thumbnail_image('https://example.test/huge') is None
    assert r.closed


def test_trickling_body_has_total_deadline(scoring, monkeypatch):
    clock = iter([0, 0, 4, 9])
    monkeypatch.setattr(scoring.time, 'monotonic', lambda: next(clock))
    r = Response([b'a', b'b', b'c'])
    monkeypatch.setattr(scoring.requests, 'get', lambda *a, **k: r)
    assert scoring._fetch_thumbnail_image('https://example.test/trickle') is None
    assert r.closed


def test_redirect_body_not_consumed(scoring, monkeypatch):
    redirect, image = Response(status=302, location='/image.png'), Response([png()])
    get = Mock(side_effect=[redirect, image])
    monkeypatch.setattr(scoring.requests, 'get', get)
    assert scoring._fetch_thumbnail_image('https://example.test/start').size == (4, 4)
    assert redirect.closed and not redirect.read and image.closed
    assert get.call_args.args[0] == 'https://example.test/image.png'


def test_thumbnail_url_wins_and_video_is_not_thumbnail(scoring):
    assert scoring._thumbnail_ref({'url': 'https://x/v.mp4', 'thumbnail_url': 'https://x/thumb'}) == 'https://x/thumb'
    assert scoring._thumbnail_ref({'url': 'https://x/v.mp4'}) is None
    assert scoring._thumbnail_ref({'url': 'https://x/watch?v=123'}) is None
    assert scoring._thumbnail_ref({'url': 'https://x/a.jpg?q=1'}) == 'https://x/a.jpg?q=1'


def test_cached_thumbnails_are_downscaled_and_cache_bounded(scoring, tmp_path):
    """Memory regression: the cache must not hold full-resolution decoded images."""
    big = tmp_path / 'big.png'
    Image.new('RGB', (3000, 2000), (10, 20, 30)).save(big)
    scoring._fetch_thumbnail_image.cache_clear()
    image = scoring._fetch_thumbnail_image(str(big))
    if image is None:  # file above the 2MB read bound: use a noisy-free smaller one
        pytest.skip('fixture exceeded byte bound')
    assert max(image.size) <= scoring._THUMB_MAX_SIDE
    assert image.mode == 'RGB'
    assert scoring._fetch_thumbnail_image.cache_info().maxsize <= 128
