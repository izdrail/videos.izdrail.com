"""Build the FastAPI app that serves both the Gradio UI and /api/image.

Kept separate from main.py so tests can construct the exact same app.
"""

import gradio as gr
from fastapi import FastAPI
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from core.api.image_api import register_image_api


def create_app(demo, sd_manager_getter) -> FastAPI:
    app = FastAPI()
    # Behind TLS-terminating Traefik the app sees plain http and would emit
    # http:// asset URLs (theme.css), which browsers block as mixed content,
    # leaving the UI unstyled. Trust X-Forwarded-Proto from the proxy so
    # Gradio generates https URLs. Only reachable via the proxy in prod.
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
    register_image_api(app, sd_manager_getter)
    gr.mount_gradio_app(app, demo, path="/")
    return app
