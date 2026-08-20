"""
SearXNG Image Search API Client
Searches for images via a SearXNG instance and uses them as static video backgrounds
"""

from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI


class SearXNGAPI(BaseMediaAPI):
    """SearXNG-based image search for background generation"""

    def __init__(
        self,
        base_url: str = "https://search.izdrail.com",
        api_key: Optional[str] = None,
    ):
        super().__init__(api_key)
        self.base_url = base_url.rstrip("/")

    def search_videos(
        self, query: str, orientation: str = "portrait", per_page: int = 10
    ) -> List[Dict]:
        cache_key = f"{query}_{orientation}_{per_page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        params = {
            "q": query,
            "format": "json",
            "categories": "images",
            "pageno": 1,
            "safesearch": 0,
        }
        if orientation == "portrait":
            params["aspect_ratio"] = "tall"

        raw_results = []
        try:
            resp = self.session.get(
                f"{self.base_url}/search", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results", [])

            results = []
            for item in raw_results:
                img_url = item.get("img_src", "") or item.get("url", "")
                # thumbnail_src is frequently empty in SearXNG responses; fall
                # back to thumbnail / img_src so the preview always has an image.
                thumbnail = (
                    item.get("thumbnail_src") or item.get("thumbnail") or img_url
                )
                if not img_url:
                    continue

                # SearXNG may return relative URLs for proxied images.
                if thumbnail and thumbnail.startswith("/"):
                    thumbnail = self.base_url + thumbnail
                if img_url.startswith("/"):
                    img_url = self.base_url + img_url

                ext = ".jpg"
                if img_url.lower().endswith(".png"):
                    ext = ".png"
                elif img_url.lower().endswith(".webp"):
                    ext = ".webp"
                elif img_url.lower().endswith(".gif"):
                    ext = ".gif"

                results.append(
                    {
                        "url": img_url,
                        "thumbnail": thumbnail,
                        "title": item.get("title", query),
                        "source": item.get("source", ""),
                        "resolution": item.get("resolution", ""),
                        "ext": ext,
                        "content": item.get("content", ""),
                    }
                )

            self.search_cache[cache_key] = results
            return results
        except Exception as e:
            print(f"[SearXNG] Search error: {e}")
            return []
        finally:
            if raw_results:
                print(
                    f"[SearXNG] {len(raw_results)} results; sample keys: "
                    f"{list(raw_results[0].keys())}"
                )

    def download_video(self, img_url: str, output_path: Path) -> bool:
        return self._download_file(img_url, output_path)

    def search_images(self, query: str, per_page: int = 10) -> List[Dict]:
        """Alias for search_videos that makes the intent clearer for image sources"""
        return self.search_videos(query, "portrait", per_page)
