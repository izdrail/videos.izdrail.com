"""Build the FastAPI app that serves both the Gradio UI and /api/image.

Kept separate from main.py so tests can construct the exact same app.
"""

import os

import gradio as gr
from fastapi import FastAPI
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from core.api.image_api import register_image_api


class ForceForwardedProtoMiddleware:
    """Rewrite the inbound x-forwarded-proto header to https.

    The public HTTPS app sits behind proxies that can supply an absent or
    http X-Forwarded-Proto. Gradio builds absolute asset URLs (theme.css)
    from that header,
    so the https page references an http stylesheet and browsers block it as
    mixed content, leaving the UI unstyled. The public site is only ever
    served over https, so forcing the proto is safe. Enabled via FORCE_HTTPS=1
    so local http development is unaffected.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = [
                (k, v) for k, v in scope.get("headers", []) if k != b"x-forwarded-proto"
            ]
            headers.append((b"x-forwarded-proto", b"https"))
            scope["headers"] = headers
        await self.app(scope, receive, send)


def create_app(demo, sd_manager_getter) -> FastAPI:
    app = FastAPI()
    # Honors the (possibly rewritten) forwarded proto for request.url in our
    # own routes; Gradio reads the header directly (see above).
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    # Added last = outermost: rewrites the raw header before anything reads
    # it, so both Gradio's header check and ProxyHeadersMiddleware see https.
    if os.environ.get("FORCE_HTTPS") == "1":
        app.add_middleware(ForceForwardedProtoMiddleware)
    register_image_api(app, sd_manager_getter)
    gr.mount_gradio_app(app, demo, path="/")
    return app
