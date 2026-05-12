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
        self.neuron_extractor = NeuronExtractor(model=config.AI_MODEL if config else "mistral:7b")
        self.preferred_order = ["YouTube", "Pexels", "Unsplash", "SearXNG", "Pixabay", "Giphy"]
        self.search_cache = {}
        # Track successful fetches per source for dynamic prioritization
        self.source_success_counts = defaultdict(int)
        # Track successful keywords for learning
        self.successful_keywords = defaultdict(int)
        # Maximum attempts across sources to improve hit rate
        self.MAX_ATTEMPTS = 3
        self._used_media_urls = set()

    def get_random_media(self, queries: List[str], preferred_source: Optional[str] = None, context: Optional[str] = None, use_snn: bool = False) -> Optional[Path]:
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
        # Limit to top 5 unique queries to reduce API requests
        queries = unique_queries[:5]

        # 1. First pass: Try exact matches for all queries
        for query in queries:
            # check cache first
            cache_key = (query, preferred_source)
            if cache_key in self.search_cache:
                cp = self.search_cache[cache_key]
                if cp and Path(cp).exists():
                    print(f"🔁 [MediaManager] Using cached video for '{query}'")
                    return Path(cp)

            print(f"🔍 [MediaManager] Searching across all sources for query: '{query}'...")
            result = self._search_and_download(query, preferred_source, context=context, use_snn=use_snn)
            if result:
                return result

        # 2. Second pass: Try fallback keywords (broader/related terms)
        print(f"⚠️ [MediaManager] No videos found for queries: {queries}. Trying fallback keywords...")
        
        # Import keyword extractor for fallback generation
        from core.nlp.keyword_extractor import KeywordExtractor
        ke = KeywordExtractor()
        
        for query in queries[:2]:  # Try fallbacks for top 2 keywords only to save requests
            fallback_keywords = ke.generate_fallback_keywords(query)
            if fallback_keywords:
                print(f"🔄 [MediaManager] Trying fallbacks for '{query}': {fallback_keywords}")
                for fallback in fallback_keywords:
                    # check cache
                    cache_key = (fallback, preferred_source)
                    if cache_key in self.search_cache:
                        cp = self.search_cache[cache_key]
                        if cp and Path(cp).exists():
                            print(f"🔁 [MediaManager] Using cached video for fallback '{fallback}'")
                            return Path(cp)
                    
                    result = self._search_and_download(fallback, preferred_source, context=context)
                    if result:
                        return result
        
        # 3. Third pass: Simple word simplification as last resort
        print(f"⚠️ [MediaManager] Fallback keywords failed. Trying word simplification...")
        for query in queries:
            simplified = self._simplify_query(query)
            if simplified != query:
                print(f"🔍 [MediaManager] Trying simplified: '{simplified}' (derived from '{query}')")
                cache_key = (simplified, preferred_source)
                if cache_key in self.search_cache:
                    cp = self.search_cache[cache_key]
                    if cp and Path(cp).exists():
                        print(f"🔁 [MediaManager] Using cached video for '{simplified}'")
                        return Path(cp)
                        
                result = self._search_and_download(simplified, preferred_source, context=context)
                if result:
                    return result

        # Final fallback to local
        for ext in ['*.mp4', '*.mov', '*.avi', '*.jpg', '*.png']:
            if self.config and self.config.VIDEOS_DIR.exists():
                files = list(self.config.VIDEOS_DIR.glob(ext))
                if files:
                    selected = random.choice(files)
                    print(f"📁 [Local] Using local background video (Last Resort): {selected.name}")
                    return selected
        return None

    def _search_and_download(self, query: str, preferred_source: Optional[str], context: Optional[str] = None, use_snn: bool = False) -> Optional[Path]:
        """Internal helper to search single query across all sources"""
        ordered_sources = self._get_ordered_sources(preferred_source)
        
        for source_name in ordered_sources:
            try:
                # Basic validation
                api = self.apis[source_name]
                if hasattr(api, 'api_key') and source_name not in ("YouTube", "SearXNG") and not api.api_key:
                    continue
                
                # Build output path
                # Use query in filename
                safe_q = "".join([c if c.isalnum() else "_" for c in query.lower()])
                
                if self.config:
                    keyword_folder = self.config.VIDEOS_DIR / safe_q
                    keyword_folder.mkdir(parents=True, exist_ok=True)

                    skip_cache = preferred_source and preferred_source in self.apis and preferred_source == source_name

                    if not skip_cache:
                        existing_videos = list(keyword_folder.glob("*.mp4")) + list(keyword_folder.glob("*.mov"))
                        if existing_videos:
                            selected_existing = random.choice(existing_videos)
                            print(f"🔁 [MediaManager] Reusing existing local video for '{query}': {selected_existing.name}")
                            cache_key = (query, preferred_source)
                            self.search_cache[cache_key] = str(selected_existing)
                            return selected_existing

                        if source_name == "Unsplash":
                            existing_images = list(keyword_folder.glob("*.jpg")) + list(keyword_folder.glob("*.png"))
                            if existing_images:
                                selected_existing = random.choice(existing_images)
                                print(f"🔁 [MediaManager] Reusing existing local image for '{query}': {selected_existing.name}")
                                cache_key = (query, preferred_source)
                                self.search_cache[cache_key] = str(selected_existing)
                                return selected_existing
                
                # Optimization: Evaluation is now LOCAL and INSTANT (spaCy vectors).
                # We can fetch multiple candidates and filter them without LLM latency.
                num_candidates = 10
                
                # Check if API supports video search (some might only be images)
                if not hasattr(api, 'search_videos'):
                    print(f"[MediaManager] {source_name} does not support search_videos. Skipping.")
                    continue
                
                results = api.search_videos(query, per_page=num_candidates)
                if not results:
                    continue
                    
                # Filter results that were already downloaded in this session
                # This filtering should happen before selection, regardless of SNN
                filtered_results = [r for r in results if r.get('url') not in self._used_media_urls]
                
                # If all results were already used, allow re-using them for this search attempt
                if not filtered_results and results:
                    filtered_results = results 
                
                if not filtered_results: 
                    continue
                
                # Selection Logic
                if len(filtered_results) > 1:
                     print(f"[MediaManager] Using Neuron AI to evaluate {len(filtered_results)} candidates for: '{context[:50]}...'")
                     evaluated = self.neuron_extractor.evaluate_media(context, filtered_results, use_snn=use_snn)
                     
                     # Safe extraction: evaluated list might be empty or missing 'media' key if Ollama had an error
                     selected = None
                     if evaluated:
                        try:
                            # Try to get 'media' key, fallback to direct object or first filtered result
                            selected = evaluated[0].get('media') or evaluated[0]
                        except Exception as eSelection:
                            print(f"⚠️ [MediaManager] Selection error: {eSelection}")
                            selected = filtered_results[0]
                     else:
                        selected = filtered_results[0]
                        
                     if selected and isinstance(selected, dict):
                        print(f"🧠 [MediaManager] Neuron AI chose: {selected.get('title', 'Untitled')}")
                else:
                    selected = filtered_results[0]
                
                if self.config:
                    ext = selected.get('ext', ".mp4")
                    if ext == ".mp4" and (source_name == "Unsplash" or source_name == "SearXNG"):
                        if selected.get('url') and '.jpg' in selected.get('url'):
                            ext = ".jpg"
                        elif selected.get('url') and '.png' in selected.get('url'):
                            ext = ".png"
                        else:
                            ext = ".jpg"
                    output_path = keyword_folder / f"{safe_q}_{source_name.lower()}_{random.randint(1000, 9999)}{ext}"
                else:
                    ext = selected.get('ext', ".mp4")
                    if ext == ".mp4" and (source_name == "Unsplash" or source_name == "SearXNG"):
                        if selected.get('url') and '.jpg' in selected.get('url'):
                            ext = ".jpg"
                        elif selected.get('url') and '.png' in selected.get('url'):
                            ext = ".png"
                        else:
                            ext = ".jpg"
                    output_path = Path(f"{safe_q}_{source_name.lower()}_{random.randint(1000, 9999)}{ext}")
                    
                # Download
                print(f"⬇️ [MediaManager] Downloading video from {source_name} for '{query}'...")
                if api.download_video(selected.get('url'), output_path):
                    print(f"✅ [MediaManager] Selected {source_name} for query: '{query}'")
                    self.source_success_counts[source_name] += 1
                    self.successful_keywords[query] += 1  # Track successful keyword
                    cache_key = (query, preferred_source)
                    self.search_cache[cache_key] = str(output_path)
                    self._used_media_urls.add(selected.get('url'))
                    print(f"📊 [MediaManager] Keyword '{query}' success count: {self.successful_keywords[query]}")
                    return output_path
                    
            except Exception as e:
                print(f"[MediaManager] Error with {source_name}: {e}")
        return None

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
            available = self.preferred_order if len(self.preferred_order) > 1 else sources
            ordered = [s for s in available if s in sources]
            sources = [s for s in sources if s not in ordered]
        else:
            ordered = []
        # Sort remaining by success count
        remaining = sorted(sources, key=lambda s: self.source_success_counts.get(s, 0), reverse=True)
        ordered.extend(remaining)
        return ordered
