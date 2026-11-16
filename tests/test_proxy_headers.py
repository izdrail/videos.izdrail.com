"""Gradio emits absolute asset URLs (theme.css) from the inbound
x-forwarded-proto header. Behind Cloudflare (Flexible) -> Traefik's web
entrypoint that header arrives as http (or not at all), so the https page
referenced an http stylesheet, browsers blocked it as mixed content and the
UI rendered unstyled (prod regression 2026-11-15/16).

Fix: FORCE_HTTPS=1 (set in Coolify env) rewrites x-forwarded-proto to https
at the ASGI layer. These tests pin both the forced scheme and that a page
served under prod-like headers (X-Forwarded-Host only, no proto) emits no
http absolute links.
"""

import re

import gradio as gr
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.routing import Route

from core.api.app_factory import create_app


async def _scheme_probe(request):
    return JSONResponse({"scheme": request.url.scheme})


def _app(monkeypatch, force_https: bool):
    monkeypatch.setenv("IMAGE_API_KEY", "k")
    if force_https:
        monkeypatch.setenv("FORCE_HTTPS", "1")
    else:
        monkeypatch.delenv("FORCE_HTTPS", raising=False)
    with gr.Blocks() as demo:
        gr.Textbox(label="probe")
    app = create_app(demo, lambda: None)
    # Insert before the catch-all Gradio mount so the probe is reachable.
    app.routes.insert(0, Route("/__scheme_probe", _scheme_probe))
    return app


def test_force_https_rewrites_scheme_without_proto_header(monkeypatch):
    c = TestClient(_app(monkeypatch, force_https=True), base_url="http://internal:1603")
    r = c.get("/__scheme_probe", headers={"X-Forwarded-Host": "videos.izdrail.com"})
    assert r.json()["scheme"] == "https"


def test_force_https_page_has_no_http_absolute_links(monkeypatch):
    # Reproduce the prod condition: only X-Forwarded-Host arrives, no proto.
    c = TestClient(_app(monkeypatch, force_https=True), base_url="http://internal:1603")
    r = c.get("/", headers={"X-Forwarded-Host": "videos.izdrail.com"})
    assert r.status_code == 200
    absolute = re.findall(r'(?:href|src)="(https?://[^"]+)"', r.text)
    for link in absolute:
        assert link.startswith("https://"), f"mixed-content link: {link}"


def test_gradio_root_url_resolves_https_under_prod_headers(monkeypatch):
    # Pin the actual consumer: gradio.route_utils.get_root_url builds the
    # absolute theme.css URL from the request headers.
    from fastapi.responses import JSONResponse as _J
    from gradio.route_utils import get_root_url

    async def root_probe(request):
        return _J({"root_url": get_root_url(request, "/", None)})

    app = _app(monkeypatch, force_https=True)
    app.routes.insert(0, Route("/__root_probe", root_probe))
    c = TestClient(app, base_url="http://internal:1603")
    r = c.get("/__root_probe", headers={"X-Forwarded-Host": "videos.izdrail.com"})
    assert r.json()["root_url"] == "https://videos.izdrail.com"


def test_without_force_env_plain_request_keeps_http(monkeypatch):
    c = TestClient(_app(monkeypatch, force_https=False), base_url="http://internal:1603")
    r = c.get("/__scheme_probe")
    assert r.json()["scheme"] == "http"


def test_forwarded_proto_header_still_honored_without_env(monkeypatch):
    c = TestClient(_app(monkeypatch, force_https=False), base_url="http://internal:1603")
    r = c.get("/__scheme_probe", headers={"X-Forwarded-Proto": "https"})
    assert r.json()["scheme"] == "https"


def test_image_api_route_still_mounted(monkeypatch):
    c = TestClient(_app(monkeypatch, force_https=True), base_url="http://internal:1603")
    assert c.post("/api/image", json={"prompt": "x"}).status_code == 401
