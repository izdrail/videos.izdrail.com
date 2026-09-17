"""Stable identities and defensive uniqueness for selected media."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


def canonical_url(url: Any) -> str | None:
    """Return a stable URL identity while retaining meaningful query parameters."""
    if not isinstance(url, str) or not url.strip():
        return None
    value = url.strip()
    parsed = urlsplit(value)
    if not parsed.netloc:
        return value.casefold()
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
               if k.casefold() not in _TRACKING_KEYS),
        doseq=True,
    )
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(),
                       parsed.path.rstrip("/"), query, ""))


def candidate_identity(candidate: Mapping[str, Any], source: str | None = None) -> Hashable:
    """Prefer a provider asset id, then a canonical source URL."""
    provider = str(candidate.get("source") or source or candidate.get("_source") or "").casefold()
    asset_id = candidate.get("asset_id", candidate.get("video_id", candidate.get("id")))
    if asset_id not in (None, ""):
        return ("provider", provider, str(asset_id))
    url = canonical_url(candidate.get("url"))
    if url:
        return ("url", url)
    return ("object", provider, id(candidate))


def path_identity(path: Any) -> Hashable:
    """Recognise path aliases and symlinks without reading a full video into memory."""
    p = Path(path).expanduser()
    resolved = Path(os.path.realpath(p))
    try:
        stat = resolved.stat()
        return ("file", stat.st_dev, stat.st_ino)
    except OSError:
        return ("path", os.path.normcase(str(resolved)))


def unique_slide_assets(slide_assets: Mapping[int, Any]) -> Tuple[Dict[int, Any], int]:
    """Keep the first use of each source asset, preserving gradients and order."""
    unique: Dict[int, Any] = {}
    seen = set()
    removed = 0
    for slide_num, asset in slide_assets.items():
        if asset is None or asset == "__gradient__":
            unique[slide_num] = asset
            continue
        key = path_identity(asset)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique[slide_num] = asset
    return unique, removed
