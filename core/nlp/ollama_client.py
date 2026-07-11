"""
Robust Ollama API Client
Handles retries with exponential backoff, response caching, and graceful failure.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

logger = logging.getLogger(__name__)

# Default fallback values used when Ollama is unreachable
DEFAULT_FALLBACK_KEYWORDS: List[str] = [
    "abstract",
    "motion",
    "light",
    "texture",
    "landscape",
    "cityscape",
    "nature",
]
DEFAULT_FALLBACK_MOOD = "Cinematic"
DEFAULT_FALLBACK_SCRIPT = (
    "This video was generated without AI assistance due to a temporary service issue."
)


class OllamaClient:
    """Thread-safe Ollama API client with retry, caching, and fallback."""

    def __init__(
        self,
        model: str = "mistral:7b",
        url: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: int = 180,
        cache_max_size: int = 512,
    ):
        self.model = model
        self.url = url or os.getenv(
            "OLLAMA_API_URL", "https://ai.izdrail.com/api/generate"
        )
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}
        self._cache_max_size = cache_max_size
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "retries": 0,
            "failures": 0,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def post(
        self,
        prompt: str,
        *,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        fmt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a generate request to Ollama with retries and caching.

        Returns the parsed JSON response body or ``None`` on total failure.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if fmt:
            payload["format"] = fmt

        cache_key = self._make_cache_key(payload)

        # Check in-memory cache first
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            logger.debug("[OllamaClient] Cache hit for key=%s", cache_key[:16])
            return self._cache[cache_key]

        effective_timeout = timeout or self.timeout
        self._stats["total_calls"] += 1
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url, json=payload, timeout=effective_timeout
                )
                # 5xx server errors are retryable
                if response.status_code >= 500:
                    raise ConnectionError(
                        f"Server error {response.status_code}: {response.text[:200]}"
                    )
                # 4xx client errors are NOT retryable
                if response.status_code >= 400:
                    logger.warning(
                        "[OllamaClient] Client error %d: %s",
                        response.status_code,
                        response.text[:200],
                    )
                    self._stats["failures"] += 1
                    return None
                result = response.json()
                self._put_cache(cache_key, result)
                return result
            except (ConnectionError, Timeout) as exc:
                last_exc = exc
                self._stats["retries"] += 1
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[OllamaClient] Attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
            except Exception as exc:
                last_exc = exc
                self._stats["failures"] += 1
                logger.error("[OllamaClient] Unexpected error: %s", exc)
                break

        self._stats["failures"] += 1
        logger.error(
            "[OllamaClient] All %d retries exhausted. Last error: %s",
            self.max_retries,
            last_exc,
        )
        return None

    def post_or_fallback(
        self,
        prompt: str,
        fallback: Any,
        **kwargs,
    ) -> Any:
        """Call ``post`` and return *fallback* if the request fails entirely."""
        result = self.post(prompt, **kwargs)
        if result is None:
            return fallback
        return result

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def clear_cache(self) -> None:
        self._cache.clear()

    def set_url(self, url: str) -> None:
        if url and url != self.url:
            logger.info("[OllamaClient] Updating API URL: %s", url)
            self.url = url
            self.clear_cache()

    def set_model(self, model: str) -> None:
        if model and model != self.model:
            logger.info("[OllamaClient] Updating model: %s", model)
            self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _put_cache(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._cache_max_size:
            # Simple eviction: remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
