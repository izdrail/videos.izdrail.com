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

class MediaManager:
    """Coordinates search and download across multiple media APIs"""
    
    def __init__(self, config=None):
        self.config = config
        self.apis = {
            "Pexels": PexelsAPI(),
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

    def get_random_media(self, query: str, preferred_source: Optional[str] = None) -> Optional[Path]:
        """Fetch a background video for *query*.
        Attempts multiple times across sources, simplifying the query on retries to always get results.
        """
        # Use cached source if available
        cache_key = (query, preferred_source)
        if cache_key in self.search_cache:
            cp = self.search_cache[cache_key]
            if cp and Path(cp).exists():
                print(f"🔁 [MediaManager] Using cached video for '{query}'")
                return Path(cp)

        current_query = query
        attempts = 0
        
        while attempts < self.MAX_ATTEMPTS:
            attempts += 1
            ordered_sources = self._get_ordered_sources(preferred_source)
            
            for source_name in ordered_sources:
                try:
                    api = self.apis[source_name]
                    if hasattr(api, 'api_key') and source_name != "YouTube" and not api.api_key:
                        continue
                    
                    results = api.search_videos(current_query)
                    if not results:
                        continue
                        
                    # Filter results that were already downloaded in this session if possible
                    filtered = [r for r in results if r.get('url') not in self._used_media_urls]
                    if not filtered: filtered = results # Fallback if all used
                    
                    selected = random.choice(filtered)
                    
                    # Build output path
                    if self.config:
                        safe_q = "".join([c if c.isalnum() else "_" for c in current_query.lower()])
                        keyword_folder = self.config.VIDEOS_DIR / safe_q
                        keyword_folder.mkdir(parents=True, exist_ok=True)
                        output_path = keyword_folder / f"{safe_q}_{source_name.lower()}_{random.randint(1000, 9999)}.mp4"
                    else:
                        safe_q = "".join([c if c.isalnum() else "_" for c in current_query.lower()])
                        output_path = Path(f"{safe_q}_{source_name.lower()}_{random.randint(1000, 9999)}.mp4")
                        
                    if api.download_video(selected.get('url'), output_path):
                        print(f"🎯 [MediaManager] Selected {source_name} for query: '{current_query}' (attempt {attempts})")
                        self.source_success_counts[source_name] += 1
                        self.search_cache[cache_key] = str(output_path)
                        self._used_media_urls.add(selected.get('url'))
                        return output_path
                        
                except Exception as e:
                    print(f"[MediaManager] Error with {source_name}: {e}")
            
            # If we failed, simplify the query for the next attempt
            old_q = current_query
            current_query = self._simplify_query(current_query)
            if current_query == old_q: # No more simplification possible
                break
                
        # Final fallback to local
        for ext in ['*.mp4', '*.mov', '*.avi']:
            if self.config and self.config.VIDEOS_DIR.exists():
                files = list(self.config.VIDEOS_DIR.glob(ext))
                if files:
                    selected = random.choice(files)
                    print(f"📁 [Local] Using local background video: {selected.name}")
                    return selected
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
