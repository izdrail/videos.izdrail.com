"""
Base Media API Interface
Abstract base class for media source APIs (Pexels, Giphy, YouTube, etc.)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import requests


class BaseMediaAPI(ABC):
    """Abstract base class for media source APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.search_cache: Dict[str, List[Dict]] = {}
    
    @abstractmethod
    def search_videos(self, query: str, orientation: str = "portrait", 
                     per_page: int = 15) -> List[Dict]:
        """
        Search for videos matching the query
        
        Args:
            query: Search query
            orientation: Video orientation (portrait, landscape, square)
            per_page: Number of results per page
            
        Returns:
            List of video metadata dictionaries
        """
        pass
    
    @abstractmethod
    def download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download a video from URL to output path
        
        Args:
            video_url: URL of the video to download
            output_path: Path where video should be saved
            
        Returns:
            True if download successful, False otherwise
        """
        pass
    
    def _download_file(self, url: str, output_path: Path, timeout: int = 30) -> bool:
        """
        Generic file download helper
        
        Args:
            url: URL to download from
            output_path: Path to save file
            timeout: Request timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Write to temporary file first
            temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Move to final location
            temp_path.replace(output_path)
            return output_path.exists()
            
        except Exception as e:
            print(f"[{self.__class__.__name__}] Download failed: {e}")
            # Clean up temp file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            return False
