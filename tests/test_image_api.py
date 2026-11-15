"""Tests for the bounded authenticated still-image endpoint (core.api.image_api)."""

import io
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from core.api.image_api import register_image_api


def _png_bytes(size=(64, 64), color=(10, 200, 90)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class FakeSD:
    """Stands in for SDTurboGenerator; records the call and serves a real PNG."""

    def __init__(self, tmp_path, delay=0.0, block_event=None):
        self.calls = []
        self._tmp = tmp_path
        self._delay = delay
        self._block = block_event

    def generate(self, prompt, width=None, height=None, target_size=None):
        self.calls.append(
            dict(prompt=prompt, width=width, height=height, target_size=target_size)
        )
        if self._block is not None:
            self._block.wait(timeout=10)
        elif self._delay:
            time.sleep(self._delay)
        path = self._tmp / f"img_{len(self.calls)}.png"
        path.write_bytes(_png_bytes((width or 64, height or 64)))
        return path


def _client(sd, monkeypatch, key="test-key"):
    monkeypatch.setenv("IMAGE_API_KEY", key)
    app = FastAPI()
    register_image_api(app, lambda: sd)
    return TestClient(app, raise_server_exceptions=False)


def test_requires_api_key(tmp_path, monkeypatch):
    sd = FakeSD(tmp_path)
    c = _client(sd, monkeypatch)
    assert c.post("/api/image", json={"prompt": "a cat"}).status_code == 401
    assert (
        c.post(
            "/api/image", json={"prompt": "a cat"}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert sd.calls == []


def test_disabled_without_env_key(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGE_API_KEY", raising=False)
    app = FastAPI()
    register_image_api(app, lambda: FakeSD(tmp_path))
    c = TestClient(app)
    r = c.post(
        "/api/image", json={"prompt": "a cat"}, headers={"X-API-Key": "x"}
    )
    assert r.status_code == 503


def test_success_png_and_default_dims(tmp_path, monkeypatch):
    sd = FakeSD(tmp_path)
    c = _client(sd, monkeypatch)
    r = c.post(
        "/api/image",
        json={"prompt": "a cat on a sofa"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.format == "PNG"
    assert sd.calls[0]["width"] == 512 and sd.calls[0]["height"] == 512
    assert sd.calls[0]["target_size"] is None


def test_dimensions_clamped_and_rounded(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_API_MAX_DIM", "1024")
    sd = FakeSD(tmp_path)
    c = _client(sd, monkeypatch)
    r = c.post(
        "/api/image",
        json={"prompt": "big", "width": 5000, "height": 10},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert sd.calls[0]["width"] == 1024
    assert sd.calls[0]["height"] == 64  # min clamp, multiple of 8


def test_jpeg_format(tmp_path, monkeypatch):
    sd = FakeSD(tmp_path)
    c = _client(sd, monkeypatch)
    r = c.post(
        "/api/image",
        json={"prompt": "a cat", "format": "jpeg", "width": 128, "height": 128},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).format == "JPEG"


def test_bad_format_rejected(tmp_path, monkeypatch):
    sd = FakeSD(tmp_path)
    c = _client(sd, monkeypatch)
    r = c.post(
        "/api/image",
        json={"prompt": "a cat", "format": "tiff"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 422
    assert sd.calls == []


def test_prompt_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_API_MAX_PROMPT", "20")
    sd = FakeSD(tmp_path)
    c = _client(sd, monkeypatch)
    r = c.post(
        "/api/image",
        json={"prompt": "x" * 5000},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert len(sd.calls[0]["prompt"]) == 20


def test_single_flight_returns_429(tmp_path, monkeypatch):
    block = threading.Event()
    sd = FakeSD(tmp_path, block_event=block)
    c = _client(sd, monkeypatch)
    result = {}

    def first_call():
        result["first"] = c.post(
            "/api/image", json={"prompt": "slow"}, headers={"X-API-Key": "test-key"}
        )

    t = threading.Thread(target=first_call)
    t.start()
    time.sleep(0.3)  # let the first request take the flight slot
    r2 = c.post(
        "/api/image", json={"prompt": "second"}, headers={"X-API-Key": "test-key"}
    )
    block.set()
    t.join(timeout=10)
    assert r2.status_code == 429
    assert result["first"].status_code == 200


def test_timeout_returns_504(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_API_TIMEOUT", "1")
    sd = FakeSD(tmp_path, delay=5)
    c = _client(sd, monkeypatch)
    r = c.post(
        "/api/image", json={"prompt": "slow"}, headers={"X-API-Key": "test-key"}
    )
    assert r.status_code == 504
