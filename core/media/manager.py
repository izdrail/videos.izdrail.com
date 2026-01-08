"""
Media Manager
Coordinates multiple media source APIs to find the best background media
"""
import random
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

    def get_random_media(self, query: str, preferred_source: Optional[str] = None) -> Optional[Path]:
        """
        Get a random video/media matching the query from available sources
        
        Args:
            query: Search query
            preferred_source: Optional source to prioritize (e.g., "Pexels")
            
        Returns:
            Path to downloaded media or None
        """
        all_sources = list(self.apis.keys())
        
        # Determine source order
        if preferred_source == "Random" or not preferred_source:
            random.shuffle(all_sources)
        else:
            priority_list = []
            if preferred_source in self.apis:
                priority_list.append(preferred_source)
            for s in self.preferred_order:
                if s not in priority_list and s in self.apis:
                    priority_list.append(s)
            for s in all_sources:
                if s not in priority_list:
                    priority_list.append(s)
            all_sources = priority_list

        for source_name in all_sources:
            try:
                api = self.apis[source_name]
                # Check if API is functional (has key if needed)
                if hasattr(api, 'api_key') and source_name != "YouTube" and not api.api_key:
                    continue
                
                # Search and get a random result
                results = api.search_videos(query)
                if results:
                    selected = random.choice(results)
                    # We need a download path. If config is available, use it.
                    if self.config:
                        # Sanitize query for folder name
                        safe_query = "".join([c if c.isalnum() else "_" for c in query.lower()])
                        keyword_folder = self.config.VIDEOS_DIR / safe_query
                        keyword_folder.mkdir(parents=True, exist_ok=True)
                        output_path = keyword_folder / f"{source_name.lower()}_{random.randint(1000, 9999)}.mp4"
                    else:
                        # Fallback to current dir if no config
                        output_path = Path(f"{source_name.lower()}_{random.randint(1000, 9999)}.mp4")
                        
                    if api.download_video(selected.get('url'), output_path):
                        print(f"🎯 [MediaManager] Selected {source_name} for query: '{query}'")
                        return output_path
            except Exception as e:
                print(f"[MediaManager] Error with {source_name}: {e}")
            
        return None
