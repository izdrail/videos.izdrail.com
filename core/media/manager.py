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
from core.nlp.neuron_extractor import NeuronExtractor


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
        }
        self.neuron_extractor = NeuronExtractor(
            model=config.AI_MODEL if config else "mistral:7b"
        )
        self.preferred_order = [
            "YouTube",
            "Pexels",
            "Unsplash",
            "SearXNG",
            "Pixabay",
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

    def get_random_media(
        self,
        queries: List[str],
        preferred_source: Optional[str] = None,
        context: Optional[str] = None,
        use_snn: bool = False,
        return_keyword: bool = False,
    ):
        """Fetch a background video for a list of *queries*.
        Tries each query in the list across all sources.
        When return_keyword=True, returns (path, keyword_that_worked).
        """
        if isinstance(queries, str):
            queries = [queries]

        unique_queries = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                unique_queries.append(q)
                seen.add(q)
        queries = unique_queries[:3]

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
                query, preferred_source, context=context, use_snn=use_snn
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
                    simplified, preferred_source, context=context
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
    ) -> Optional[Path]:
        """Semantic Multi-Armed Bandit: search all sources in parallel,
        score each by semantic_match, quality, freshness & diversity,
        then download from the highest-scoring source.
        """
        ordered_sources = self._get_ordered_sources(preferred_source)
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
            if (
                hasattr(api, "api_key")
                and source_name not in ("YouTube", "SearXNG")
                and not api.api_key
            ):
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

        if not source_results:
            print(f"💡 [Bandit] No results found for '{query}' from any source.")
            return None

        # ── Stage 3: score every source with the bandit formula ──
        now = time.time()
        scores: Dict[str, float] = {}
        best_media_per_source: Dict[str, dict] = {}

        for src, results in source_results.items():
            evaluated = None
            if context and len(results) > 0:
                try:
                    evaluated = self.neuron_extractor.evaluate_media(
                        context, results, use_snn=use_snn
                    )
                except Exception:
                    pass

            if evaluated:
                top = evaluated[0]
                media = top.get("media") or top
                semantic_score = max(0.0, top.get("decision_score", 0.5))
            else:
                media = results[0]
                semantic_score = 0.5

            best_media_per_source[src] = media
            quality = self._compute_quality_score(media)
            last_used = self._source_last_used.get(src, 0)
            freshness = 1.0 / (now - last_used + 1)
            usage = self._source_usage_count.get(src, 0)
            diversity = 1.0 / (usage + 1)

            boost = 0.3 if (preferred_source and src == preferred_source) else 0.0
            scores[src] = (
                semantic_score * 0.6
                + quality * 0.2
                + freshness * 0.1
                + diversity * 0.1
                + boost
            )

        best_src = max(scores, key=scores.get)
        best_media = best_media_per_source[best_src]
        print(
            f"🎰 [Bandit] {best_src} wins (score={scores[best_src]:.3f}) "
            f"for '{query}'  |  scores: { {s: f'{v:.2f}' for s, v in sorted(scores.items())} }"
        )

        # ── Stage 4: download from the winning source ──
        ext = best_media.get("ext", ".mp4")
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
        """Score a media item's visual quality (resolution-based)."""
        w = media.get("width") or 0
        h = media.get("height") or 0
        if w > 0 and h > 0:
            pixels = w * h
            ref = 1920 * 1080
            return min(1.0, pixels / ref)
        return 0.5

    def _simplify_query(self, query: str) -> str:
        """Reduce query complexity to improve search hit rate."""
        words = query.split()
        if len(words) > 1:
            simplified = " ".join(words[:2])
            return simplified
        return query

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
