"""
Media Manager
Coordinates multiple media source APIs to find the best background media
"""

import random
import time
import concurrent.futures
from collections import defaultdict
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .pexels import PexelsAPI
from .giphy import GiphyAPI
from .youtube import YouTubeAPI
from .pixabay import PixabayAPI
from .unsplash import UnsplashAPI
from .searxng import SearXNGAPI
from .openverse import OpenverseProvider
from .wikimedia import WikimediaProvider
from .internet_archive import InternetArchiveProvider
from .base import MediaType, Media
from core.nlp.neuron_extractor import NeuronExtractor
from core.nlp.entity import EntityHandler
from media_scoring import (
    clip_relevance_score,
    compute_quality_score,
    rerank_pooled_candidates,
)


class MediaManager:
    """Coordinates search and download across multiple media APIs"""

    def __init__(self, config=None):
        self.config = config
        self.apis = {
            "Pexels": PexelsAPI(),
            "Pixabay": PixabayAPI(),
            "Giphy": GiphyAPI(),
            "YouTube": YouTubeAPI(),
            "Unsplash": UnsplashAPI(config.UNSPLASH_ACCESS_KEY if config else None),
            "SearXNG": SearXNGAPI(),
            "Openverse": OpenverseProvider(),
            "Wikimedia": WikimediaProvider(),
            "InternetArchive": InternetArchiveProvider(),
        }
        self.neuron_extractor = NeuronExtractor(
            model=config.AI_MODEL if config else "gemma4:e2b"
        )
        self.entity_handler = EntityHandler()
        # Open, key-less providers are tried first so the system prefers freely
        # licensed media and only falls back to commercial APIs when needed.
        self.preferred_order = [
            "Openverse",
            "Wikimedia",
            "SearXNG",
            "InternetArchive",
            "YouTube",
            "Pexels",
            "Pixabay",
            "Unsplash",
            "Giphy",
        ]
        self.search_cache = {}
        # Track successful fetches per source for dynamic prioritization
        self.source_success_counts = defaultdict(int)
        # Track successful keywords for learning
        self.successful_keywords = defaultdict(int)
        # Maximum attempts across sources to improve hit rate
        self.MAX_ATTEMPTS = 3
        self._used_media_urls = set()
        # Bandit tracking
        self._source_usage_count = defaultdict(int)
        self._source_last_used = {}
        # Last source actually chosen, used to force source rotation so a single
        # high-scoring source does not win every query.
        self._last_used_source = None

    def get_random_media(
        self,
        queries: List[str],
        preferred_source: Optional[str] = None,
        context: Optional[str] = None,
        use_snn: bool = False,
        return_keyword: bool = False,
        theme: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        """Fetch a background video for a list of *queries*.
        Tries each query in the list across all sources.
        When return_keyword=True, returns (path, keyword_that_worked).
        If ``entity`` is supplied, searches are enriched with that entity.
        """
        if isinstance(queries, str):
            queries = [queries]

        # Append theme-based broad queries to the search list as an extra
        # chance of finding relevant footage before falling back to local files.
        if theme and isinstance(theme, str) and theme.strip():
            queries = list(queries) + [theme.strip()]

        unique_queries = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                unique_queries.append(q)
                seen.add(q)
        queries = unique_queries[:4]

        def _find_cached(q):
            ck = (q, preferred_source)
            if ck in self.search_cache:
                cp = self.search_cache[ck]
                if cp and Path(cp).exists():
                    return Path(cp)
            return None

        # 1. Cache hit + direct search (3 queries max)
        for query in queries:
            cached = _find_cached(query)
            if cached:
                print(f"🔁 [MediaManager] Using cached video for '{query}'")
                return (cached, query) if return_keyword else cached

            print(f"🔍 [MediaManager] Searching for '{query}'...")
            result = self._search_and_download(
                query,
                preferred_source,
                context=context,
                use_snn=use_snn,
                theme=theme,
                entity=entity,
            )
            if result:
                return (result, query) if return_keyword else result

        # 2. Word simplification fallback
        for query in queries:
            simplified = self._simplify_query(query)
            if simplified and simplified != query:
                cached = _find_cached(simplified)
                if cached:
                    return (cached, simplified) if return_keyword else cached
                result = self._search_and_download(
                    simplified,
                    preferred_source,
                    context=context,
                    theme=theme,
                    entity=entity,
                )
                if result:
                    return (result, simplified) if return_keyword else result

        # 3. Local file fallback
        for ext in ["*.mp4", "*.mov", "*.avi", "*.jpg", "*.png"]:
            if self.config and self.config.VIDEOS_DIR.exists():
                files = list(self.config.VIDEOS_DIR.glob(ext))
                if files:
                    selected = random.choice(files)
                    print(
                        f"📁 [Local] Using local background video (Last Resort): {selected.name}"
                    )
                    return (selected, None) if return_keyword else selected
        return (None, None) if return_keyword else None

    def _search_and_download(
        self,
        query: str,
        preferred_source: Optional[str],
        context: Optional[str] = None,
        use_snn: bool = False,
        source_timeout: int = 60,
        theme: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> Optional[Path]:
        """Semantic Multi-Armed Bandit: search all sources in parallel,
        score each by semantic_match, quality, freshness & diversity,
        then download from the highest-scoring source.
        """
        ordered_sources = self._get_ordered_sources(preferred_source)

        # Enrich the search query with any supplied entity context so results
        # are more targeted. Done before safe_q/cache_key so entity searches
        # are cached independently.
        entity_dict = self.entity_handler.parse_entity(entity) if entity else None
        if entity_dict:
            query = self.entity_handler.enrich_query(query, entity_dict)

        safe_q = "".join([c if c.isalnum() else "_" for c in query.lower()])

        if self.config:
            keyword_folder = self.config.VIDEOS_DIR / safe_q
            keyword_folder.mkdir(parents=True, exist_ok=True)
        else:
            keyword_folder = None

        # ── Stage 1: check for existing local files (skip bandit, instant reuse) ──
        if keyword_folder and self.config:
            existing_videos = list(keyword_folder.glob("*.mp4")) + list(
                keyword_folder.glob("*.mov")
            )
            if existing_videos:
                selected_existing = random.choice(existing_videos)
                print(
                    f"🔁 [MediaManager] Reusing existing local video for '{query}': {selected_existing.name}"
                )
                cache_key = (query, preferred_source)
                self.search_cache[cache_key] = str(selected_existing)
                return selected_existing

            existing_images = list(keyword_folder.glob("*.jpg")) + list(
                keyword_folder.glob("*.png")
            )
            if existing_images:
                selected_existing = random.choice(existing_images)
                print(
                    f"🔁 [MediaManager] Reusing existing local image for '{query}': {selected_existing.name}"
                )
                cache_key = (query, preferred_source)
                self.search_cache[cache_key] = str(selected_existing)
                return selected_existing

        # ── Stage 2: search EVERY eligible source in parallel ──
        def _search_source(source_name: str):
            api = self.apis.get(source_name)
            if not api or not hasattr(api, "search_videos"):
                return None
            # Skip a source only if it genuinely requires a key that is missing.
            # Key-less open providers (Openverse, Wikimedia, Internet Archive,
            # YouTube, SearXNG) must never be filtered out here.
            try:
                requires_key = api.capabilities().get(
                    "requires_key", bool(getattr(api, "api_key", None))
                )
            except Exception:
                requires_key = bool(getattr(api, "api_key", None))
            if requires_key and not getattr(api, "api_key", None):
                return None
            try:
                raw = api.search_videos(query, per_page=10)
                if not raw:
                    return None
                filtered = [r for r in raw if r.get("url") not in self._used_media_urls]
                return filtered if filtered else raw
            except Exception:
                return None

        source_results: Dict[str, list] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(ordered_sources) or 1
        ) as pool:
            fut_map = {pool.submit(_search_source, s): s for s in ordered_sources}
            for fut in concurrent.futures.as_completed(fut_map):
                src = fut_map[fut]
                try:
                    res = fut.result(timeout=source_timeout)
                    if res:
                        source_results[src] = res
                except concurrent.futures.TimeoutError:
                    print(
                        f"⏱️ [Bandit] {src} timed out after {source_timeout}s for '{query}'. Skipping."
                    )
                except Exception:
                    pass

        if entity_dict:
            print(
                f"🔎 [Search] query='{query}' entity='{entity_dict['raw']}' "
                f"({entity_dict['type']}) results/source="
                f"{ {s: len(v) for s, v in source_results.items()} }"
            )

        if not source_results:
            print(f"💡 [Bandit] No results found for '{query}' from any source.")
            # Theme-based fallback: broaden to the script's theme and generic
            # terms so we still get on-theme (or at least neutral) footage.
            if theme and theme.strip() and theme.strip().lower() not in query.lower():
                print(f"🎯 [Bandit] Trying theme-based query: '{theme.strip()}'")
                try:
                    theme_result = self._search_and_download(
                        theme.strip(),
                        preferred_source,
                        context=context,
                        use_snn=use_snn,
                        source_timeout=source_timeout,
                    )
                    if theme_result:
                        return theme_result
                except Exception:
                    pass
            return None

        # ── Stage 3: Pool all candidates and rerank globally with media_scoring ──
        now = time.time()
        narration_text = context if (context and context.strip()) else query

        best_clips = rerank_pooled_candidates(
            narration_text=narration_text,
            candidates_by_source=source_results,
            used_urls=self._used_media_urls,
            source_usage_count=self._source_usage_count,
            source_last_used=self._source_last_used,
            now=now,
            preferred_source=preferred_source,
            top_k=1,
            target_width=1080,
            target_height=1920,
        )

        if not best_clips:
            print(f"💡 [Rerank] No eligible candidate clips after reranking for '{query}'.")
            return None

        best_media = best_clips[0]
        best_src = best_media.get("_source", "Unknown")
        best_score = best_media.get("_score", 0.0)

        self._last_used_source = best_src
        print(
            f"🎰 [MediaScoring] {best_src} wins (score={best_score:.3f}) "
            f"for '{query}'"
        )

        # ── Stage 4: download from the winning source ──
        raw_ext = best_media.get("ext")
        if not raw_ext or not isinstance(raw_ext, str) or raw_ext.strip().lower() in ("none", "null", ""):
            ext = ".mp4"
        else:
            ext = raw_ext.strip().lower()
            if not ext.startswith("."):
                ext = f".{ext}"

        if ext == ".mp4" and (best_src in ("Unsplash", "SearXNG")):
            url = best_media.get("url", "")
            if ".jpg" in url:
                ext = ".jpg"
            elif ".png" in url:
                ext = ".png"
            else:
                ext = ".jpg"

        output_path = (
            keyword_folder
            / f"{safe_q}_{best_src.lower()}_{random.randint(1000, 9999)}{ext}"
        )

        api = self.apis.get(best_src)
        if api and api.download_video(best_media.get("url"), output_path):
            print(f"✅ [Bandit] Downloaded from {best_src} for query: '{query}'")
            self.source_success_counts[best_src] += 1
            self.successful_keywords[query] += 1
            self._source_usage_count[best_src] += 1
            self._source_last_used[best_src] = now
            cache_key = (query, preferred_source)
            self.search_cache[cache_key] = str(output_path)
            self._used_media_urls.add(best_media.get("url"))
            return output_path

        return None

    @staticmethod
    def _compute_quality_score(media: dict) -> float:
        """Score a media item's visual quality (resolution + aspect ratio fit)."""
        return compute_quality_score(media)

    def _simplify_query(self, query: str) -> str:
        """Reduce query complexity to improve search hit rate."""
        words = query.split()
        if len(words) > 1:
            simplified = " ".join(words[:2])
            return simplified
        return query

    def is_keyword_available(
        self, keyword: str, preferred_source: Optional[str] = None
    ) -> bool:
        """Quick availability dry-run for a single keyword.

        Returns True if at least one eligible source returns ≥1 result for a
        minimal (``per_page=1``) search. Used by the availability fallback so we
        only commit to keyword candidates that can actually yield footage.
        """
        if not keyword:
            return False
        ordered = self._get_ordered_sources(preferred_source)
        for src in ordered:
            api = self.apis.get(src)
            if not api or not hasattr(api, "search_videos"):
                continue
            try:
                requires_key = api.capabilities().get(
                    "requires_key", bool(getattr(api, "api_key", None))
                )
            except Exception:
                requires_key = bool(getattr(api, "api_key", None))
            if requires_key and not getattr(api, "api_key", None):
                continue
            try:
                res = api.search_videos(keyword, per_page=1)
                if res:
                    return True
            except Exception:
                continue
        return False

    def _get_ordered_sources(self, preferred: Optional[str] = None) -> List[str]:
        sources = list(self.apis.keys())
        if preferred and preferred in sources:
            sources.remove(preferred)
            ordered = [preferred]
        elif self.preferred_order:
            # When no specific source selected, use preferred_order as the base
            available = (
                self.preferred_order if len(self.preferred_order) > 1 else sources
            )
            ordered = [s for s in available if s in sources]
            sources = [s for s in sources if s not in ordered]
        else:
            ordered = []
        # Sort remaining by success count
        remaining = sorted(
            sources, key=lambda s: self.source_success_counts.get(s, 0), reverse=True
        )
        ordered.extend(remaining)
        return ordered

    # ------------------------------------------------------------------
    # Unified, de-duplicated discovery across all providers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_key(media: "Media") -> str:
        """Stable de-duplication key: normalized URL, else title+source."""
        url = (media.url or "").strip()
        if url:
            from urllib.parse import urlparse

            p = urlparse(url)
            path = p.path.split("?")[0].rstrip("/").lower()
            return f"{p.netloc}{path}"
        return f"{(media.title or '').lower()}|{media.source or ''}"

    def search(
        self,
        query: str,
        media_type: MediaType = MediaType.ANY,
        limit: int = 50,
        min_results: int = 50,
    ) -> List["Media"]:
        """Aggregate results from every provider (open sources first).

        Walks the priority list querying each provider (skipping those that don't
        support the requested ``media_type`` or require a missing key) until at
        least ``min_results`` items are collected or all providers are exhausted.
        Results are de-duplicated by canonical URL so the same file surfacing on
        multiple providers appears only once.

        Returns:
            A list of unified :class:`Media` objects.
        """
        collected: List[Media] = []
        seen = set()
        discovery_order = [s for s in self.preferred_order if s in self.apis]

        for src in discovery_order:
            api = self.apis.get(src)
            if not api:
                continue
            try:
                caps = api.capabilities()
            except Exception:
                caps = {}
            supports = caps.get("supports_media_types", [MediaType.ANY])
            if media_type != MediaType.ANY and media_type not in supports:
                continue
            if caps.get("requires_key") and not getattr(api, "api_key", None):
                continue

            try:
                if hasattr(api, "search"):
                    items = api.search(query, media_type=media_type, limit=limit)
                else:
                    items = [
                        Media.from_dict(d, src)
                        for d in api.search_videos(query, "portrait", limit)
                    ]
            except Exception as e:
                print(f"⚠️ [Search] {src} failed: {e}")
                items = []

            for m in items:
                key = self._canonical_key(m)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(m)
                if len(collected) >= min_results:
                    break
            if len(collected) >= min_results:
                break

        return collected[:limit]
