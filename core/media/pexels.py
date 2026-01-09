"""
Pexels API Client
Provides access to Pexels video library
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI


class PexelsAPI(BaseMediaAPI):
    """Pexels video search and download API"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv('PEXELS_API_KEY', ''))
        if self.api_key:
            self.session.headers.update({'Authorization': self.api_key})
    
    def search_videos(self, query: str, orientation: str = "portrait", 
                     per_page: int = 15) -> List[Dict]:
        """
        Search Pexels for videos
        
        Args:
            query: Search query
            orientation: Video orientation (portrait, landscape, square)
            per_page: Number of results per page
            
        Returns:
            List of video metadata dictionaries with 'url' and 'id' keys
        """
        if not self.api_key:
            print("[PexelsAPI] No API key configured")
            return []
        
        # Check cache
        cache_key = f"{query}_{orientation}_{per_page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        try:
            url = "https://api.pexels.com/videos/search"
            params = {
                'query': query,
                'orientation': orientation,
                'per_page': per_page,
                'page': 1
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for video in data.get('videos', []):
                # Get the best quality video file
                video_files = video.get('video_files', [])
                if video_files:
                    # Sort by resolution (width) ascending to get the smallest version
                    best_file = min(video_files, key=lambda x: x.get('width', 99999))
                    
                    results.append({
                        'url': best_file.get('link'),
                        'id': video.get('id'),
                        'duration': video.get('duration', 0),
                        'width': best_file.get('width'),
                        'height': best_file.get('height')
                    })
            
            # Cache results
            self.search_cache[cache_key] = results
            return results
            
        except Exception as e:
            print(f"[PexelsAPI] Search error: {e}")
            return []
    
    def download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download video from Pexels
        
        Args:
            video_url: URL of the video
            output_path: Path to save the video
            
        Returns:
            True if successful, False otherwise
        """
        return self._download_file(video_url, output_path)
