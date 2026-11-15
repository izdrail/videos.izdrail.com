"""Bounded, authenticated still-image HTTP endpoint around the SD-Turbo pipeline.

Exposes POST /api/image on the Gradio FastAPI app: prompt in, PNG/JPEG out.
The endpoint is a side door for sibling services (e.g. the Social app); the
Gradio UI and video job flow are untouched.

Guards:
- API key auth via the IMAGE_API_KEY env var (X-API-Key header). When the env
  var is unset the endpoint answers 503 and does no work.
- Fixed bounds: dimensions are clamped to [IMAGE_API_MIN_DIM, IMAGE_API_MAX_DIM]
  and rounded down to a multiple of 8; prompt length is capped.
- Single-flight: only one API generation at a time; concurrent callers get
  429 immediately instead of queueing GPU work ahead of video jobs. The
  underlying SDTurboGenerator._generation_lock still serializes against any
  in-flight video job generation.
- Timeout: a request that exceeds IMAGE_API_TIMEOUT seconds answers 504.
"""

import concurrent.futures
import hmac
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MIN_DIM_DEFAULT = 64
MAX_DIM_DEFAULT = 1024
TIMEOUT_DEFAULT = 120
MAX_PROMPT_DEFAULT = 1000

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="image-api"
)
_flight_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _clamp_dim(value: Optional[int], default: int) -> int:
    lo = _env_int("IMAGE_API_MIN_DIM", MIN_DIM_DEFAULT)
    hi = _env_int("IMAGE_API_MAX_DIM", MAX_DIM_DEFAULT)
    v = value if value else default
    v = max(lo, min(hi, v))
    return (v // 8) * 8


def register_image_api(app, sd_manager_getter: Callable[[], object]) -> None:
    """Attach POST /api/image to a FastAPI app (Gradio's demo.app)."""
    from fastapi import Header, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    class ImageRequest(BaseModel):
        prompt: str = Field(..., min_length=1)
        width: Optional[int] = None
        height: Optional[int] = None
        format: str = "png"

    @app.post("/api/image")
    def generate_image_api(
        req: ImageRequest, x_api_key: Optional[str] = Header(default=None)
    ):
        configured_key = os.environ.get("IMAGE_API_KEY", "")
        if not configured_key:
            raise HTTPException(status_code=503, detail="image API disabled")
        if not x_api_key or not hmac.compare_digest(x_api_key, configured_key):
            raise HTTPException(status_code=401, detail="invalid API key")

        max_prompt = _env_int("IMAGE_API_MAX_PROMPT", MAX_PROMPT_DEFAULT)
        prompt = req.prompt.strip()[:max_prompt]
        if not prompt:
            raise HTTPException(status_code=422, detail="empty prompt")

        width = _clamp_dim(req.width, 512)
        height = _clamp_dim(req.height, 512)
        fmt = (req.format or "png").lower()
        if fmt not in ("png", "jpeg", "jpg"):
            raise HTTPException(status_code=422, detail="format must be png or jpeg")

        if not _flight_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="image generation busy; retry later"
            )
        try:
            sd_manager = sd_manager_getter()
            if sd_manager is None:
                raise HTTPException(
                    status_code=503, detail="image generator unavailable"
                )
            timeout = _env_int("IMAGE_API_TIMEOUT", TIMEOUT_DEFAULT)
            future = _executor.submit(
                sd_manager.generate,
                prompt=prompt,
                width=width,
                height=height,
                target_size=None,
            )
            try:
                image_path = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise HTTPException(
                    status_code=504, detail="generation timed out"
                )
            if not image_path or not Path(image_path).exists():
                raise HTTPException(status_code=500, detail="generation failed")

            if fmt == "png":
                return Response(
                    content=Path(image_path).read_bytes(), media_type="image/png"
                )
            from PIL import Image

            with Image.open(image_path) as im:
                import io

                buf = io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=92)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
        finally:
            _flight_lock.release()
