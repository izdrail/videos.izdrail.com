"""The mounted app must honor X-Forwarded-Proto so Gradio emits https asset
URLs behind the TLS-terminating proxy; otherwise browsers block theme.css as
mixed content and the UI renders unstyled (prod regression 2026-11-15).

The pinned behavior is the proxy-header trust itself (Gradio's page handler
builds absolute asset URLs from request.url, whose scheme this middleware
rewrites), plus that any absolute links the page does emit follow the
forwarded proto.
"""

import re

import gradio as gr
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.routing import Route

from core.api.app_factory import create_app


async def _scheme_probe(request):
    return JSONResponse({"scheme": request.url.scheme})


def _app(monkeypatch):
    monkeypatch.setenv("IMAGE_API_KEY", "k")
    with gr.Blocks() as demo:
        gr.Textbox(label="probe")
    app = create_app(demo, lambda: None)
    # Insert before the catch-all Gradio mount so the probe is reachable.
    app.routes.insert(0, Route("/__scheme_probe", _scheme_probe))
    return app


def test_forwarded_https_rewrites_request_scheme(monkeypatch):
    c = TestClient(_app(monkeypatch), base_url="http://internal:1603")
    r = c.get("/__scheme_probe", headers={"X-Forwarded-Proto": "https"})
    assert r.json()["scheme"] == "https"


def test_plain_request_keeps_http_scheme(monkeypatch):
    c = TestClient(_app(monkeypatch), base_url="http://internal:1603")
    r = c.get("/__scheme_probe")
    assert r.json()["scheme"] == "http"


def test_no_mixed_content_absolute_links_when_forwarded_https(monkeypatch):
    c = TestClient(_app(monkeypatch), base_url="http://internal:1603")
    r = c.get("/", headers={"X-Forwarded-Proto": "https"})
    assert r.status_code == 200
    absolute = re.findall(r'(?:href|src)="(https?://[^"]+)"', r.text)
    for link in absolute:
        assert link.startswith("https://"), f"mixed-content link: {link}"


def test_image_api_route_still_mounted(monkeypatch):
    c = TestClient(_app(monkeypatch), base_url="http://internal:1603")
    assert c.post("/api/image", json={"prompt": "x"}).status_code == 401
