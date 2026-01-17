"""
YouTube API Client
Provides access to YouTube video downloads via yt-dlp
"""
import os
import subprocess
from typing import List, Dict, Optional
from pathlib import Path
from .base import BaseMediaAPI


class YouTubeAPI(BaseMediaAPI):
    """YouTube video search and download via yt-dlp"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        # YouTube doesn't require API key for yt-dlp downloads
    
    def search_videos(self, query: str, orientation: str = "portrait", 
                     per_page: int = 15) -> List[Dict]:
        """
        Search YouTube for videos using yt-dlp
        
        Args:
            query: Search query
            orientation: Not used for YouTube
            per_page: Number of results
            
        Returns:
            List of video metadata dictionaries
        """
        # Check cache
        cache_key = f"{query}_{per_page}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        try:
            # Add negative search terms to proactively filter out music/interviews
            clean_query = f"{query} -lyrics -\"official video\" -\"music video\" -interview -commentary"
            cmd = [
                'yt-dlp',
                f'ytsearch{per_page}:{clean_query}',
                '--get-id',
                '--get-title',
                '--get-duration',
                '--no-warnings',
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                results = []
                
                # Parse output (alternating title and ID)
                bad_keywords = ["lyrics", "official video", "music video", "interview", "commentary", "podcast", "vlog"]
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        title = lines[i]
                        # Final filter: skip if any bad keyword is still in title
                        if any(bk in title.lower() for bk in bad_keywords):
                            continue
                            
                        results.append({
                            'url': f'https://www.youtube.com/watch?v={lines[i+1]}',
                            'id': lines[i+1],
                            'title': title
                        })
                
                # Cache results
                self.search_cache[cache_key] = results
                return results
            
        except Exception as e:
            print(f"[YouTubeAPI] Search error: {e}")
        
        return []
    
    def download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download video from YouTube using yt-dlp
        
        Args:
            video_url: YouTube video URL
            output_path: Path to save the video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                'yt-dlp',
                '-f', 'worst[ext=mp4]/worst',  # Pick the smallest available (prefer MP4)
                '-o', str(output_path),
                '--no-warnings',
                '--quiet',
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            return result.returncode == 0 and output_path.exists()
            
        except Exception as e:
            print(f"[YouTubeAPI] Download error: {e}")
            return False
