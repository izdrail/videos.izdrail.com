"""
Media Manager
Coordinates multiple media source APIs to find the best background media
"""
import random
from collections import defaultdict
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .pexels import PexelsAPI
from .giphy import GiphyAPI
from .youtube import YouTubeAPI
from .pixabay import PixabayAPI

class MediaManager:
    """Coordinates search and download across multiple media APIs"""
    
    def __init__(self, config=None):
        self.config = config
        self.apis = {
            "Pexels": PexelsAPI(),
            "Pixabay": PixabayAPI(),
            "Giphy": GiphyAPI(),
            "YouTube": YouTubeAPI()
        }
        self.preferred_order = ["Pexels", "YouTube", "Giphy"]
        self.search_cache = {}
        # Track successful fetches per source for dynamic prioritization
        self.source_success_counts = defaultdict(int)
        # Maximum attempts across sources to improve hit rate
        self.MAX_ATTEMPTS = 3
        self._used_media_urls = set()

    def get_random_media(self, queries: List[str], preferred_source: Optional[str] = None) -> Optional[Path]:
        """Fetch a background video for a list of *queries*.
        Tries each query in the list across all sources.
        """
        # Ensure queries is a list
        if isinstance(queries, str):
            queries = [queries]
            
        # Deduplicate and filter empty
        unique_queries = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                unique_queries.append(q)
                seen.add(q)
        queries = unique_queries

        # 1. First pass: Try exact matches for all queries
        for query in queries:
            # check cache first
            cache_key = (query, preferred_source)
            if cache_key in self.search_cache:
                cp = self.search_cache[cache_key]
                if cp and Path(cp).exists():
                    print(f"🔁 [MediaManager] Using cached video for '{query}'")
                    return Path(cp)

            result = self._search_and_download(query, preferred_source)
            if result:
                return result

        # 2. Second pass: aggressive simplification
        # If we are here, no video was found for any of the specific keywords.
        # Try simplifying the *first* (most relevant) keyword, or iterate all if needed.
        # Let's try simplifying all of them in order.
        print(f"⚠️ [MediaManager] No videos found for queries: {queries}. Trying simplified queries...")
        
        for query in queries:
            simplified = self._simplify_query(query)
            if simplified != query:
                print(f"🔍 [MediaManager] Trying simplified: '{simplified}' (derived from '{query}')")
                # check cache
                cache_key = (simplified, preferred_source)
                if cache_key in self.search_cache:
                    cp = self.search_cache[cache_key]
                    if cp and Path(cp).exists():
                        print(f"🔁 [MediaManager] Using cached video for '{simplified}'")
                        return Path(cp)
                        
                result = self._search_and_download(simplified, preferred_source)
                if result:
                    return result

        # Final fallback to local
        for ext in ['*.mp4', '*.mov', '*.avi']:
            if self.config and self.config.VIDEOS_DIR.exists():
                files = list(self.config.VIDEOS_DIR.glob(ext))
                if files:
                    selected = random.choice(files)
                    print(f"📁 [Local] Using local background video (Last Resort): {selected.name}")
                    return selected
        return None

    def _search_and_download(self, query: str, preferred_source: Optional[str]) -> Optional[Path]:
        """Internal helper to search single query across all sources"""
        ordered_sources = self._get_ordered_sources(preferred_source)
        
        for source_name in ordered_sources:
            try:
                # Basic validation
                api = self.apis[source_name]
                if hasattr(api, 'api_key') and source_name != "YouTube" and not api.api_key:
                    continue
                
                # Build output path
                # Use query in filename
                safe_q = "".join([c if c.isalnum() else "_" for c in query.lower()])
                
                if self.config:
                    keyword_folder = self.config.VIDEOS_DIR / safe_q
                    keyword_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Caching: Check if we already have a video in this keyword folder
                    existing_videos = list(keyword_folder.glob("*.mp4"))
                    if existing_videos:
                        selected_existing = random.choice(existing_videos)
                        print(f"🔁 [MediaManager] Reusing existing local video for '{query}': {selected_existing.name}")
                        self.source_success_counts[source_name] += 1
                        cache_key = (query, preferred_source)
                        self.search_cache[cache_key] = str(selected_existing)
                        return selected_existing
                
                # If no config or no existing video, proceed to search/download
                results = api.search_videos(query)
                if not results:
                    continue
                    
                # Filter results that were already downloaded in this session
                filtered = [r for r in results if r.get('url') not in self._used_media_urls]
                
                if not filtered and results:
                    filtered = results 
                
                if not filtered: 
                    continue
                
                selected = random.choice(filtered)
                
                if self.config:
                    output_path = keyword_folder / f"{safe_q}_{source_name.lower()}_{random.randint(1000, 9999)}.mp4"
                else:
                    output_path = Path(f"{safe_q}_{source_name.lower()}_{random.randint(1000, 9999)}.mp4")
                    
                # Download
                print(f"⬇️ [MediaManager] Downloading video from {source_name} for '{query}'...")
                if api.download_video(selected.get('url'), output_path):
                    print(f"✅ [MediaManager] Selected {source_name} for query: '{query}'")
                    self.source_success_counts[source_name] += 1
                    cache_key = (query, preferred_source)
                    self.search_cache[cache_key] = str(output_path)
                    self._used_media_urls.add(selected.get('url'))
                    return output_path
                    
            except Exception as e:
                print(f"[MediaManager] Error with {source_name}: {e}")
        return None

    def _simplify_query(self, query: str) -> str:
        """Reduce query complexity to improve search hit rate."""
        words = query.split()
        if len(words) > 1:
            # Try taking just the first word
            return words[0]
        return query

    def _get_ordered_sources(self, preferred: Optional[str] = None) -> List[str]:
        sources = list(self.apis.keys())
        if preferred and preferred in sources:
            sources.remove(preferred)
            ordered = [preferred]
        else:
            ordered = []
        # Sort remaining by success count
        remaining = sorted(sources, key=lambda s: self.source_success_counts.get(s, 0), reverse=True)
        ordered.extend(remaining)
        return ordered
