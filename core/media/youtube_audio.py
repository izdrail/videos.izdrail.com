"""
YouTube Free Audio Library API Service
Provides access to royalty-free music from the YouTube Audio Library
"""
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import time

class YouTubeAudioLibraryAPI:
    def __init__(self, config=None):
        self.config = config
        self.api_url = getattr(config, 'YOUTUBE_AUDIO_API_URL', 'https://thibaultjanbeyer.github.io/YouTube-Free-Audio-Library-API/api.json')
        self.cache_file = Path(config.ROOT_DIR if config else ".") / "yt_audio_library_cache.json"
        self.tracks: List[Dict] = []
        self._load_tracks()

    def _load_tracks(self):
        """Load tracks from cache or fetch from API"""
        if self.cache_file.exists():
            # Check age (expire after 7 days)
            age = time.time() - self.cache_file.stat().st_mtime
            if age < 7 * 24 * 3600:
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)
                        self.tracks = data.get('all', [])
                        if self.tracks:
                            print(f"[YouTubeAudio] Loaded {len(self.tracks)} tracks from cache.")
                            return
                except Exception as e:
                    print(f"[YouTubeAudio] Cache load error: {e}")

        self.fetch_tracks()

    def fetch_tracks(self) -> bool:
        """Fetch all tracks from the remote API"""
        try:
            print(f"[YouTubeAudio] Fetching library from {self.api_url}...")
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.tracks = data.get('all', [])
            
            # Save to cache
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
            
            print(f"[YouTubeAudio] Successfully fetched {len(self.tracks)} tracks.")
            return True
        except Exception as e:
            print(f"[YouTubeAudio] Fetch failed: {e}")
            return False

    def search(self, query: str) -> List[Dict]:
        """Search tracks by name (case-insensitive)"""
        if not query:
            return []
        
        print(f"🔍 [YouTubeAudio] Searching for keywords: '{query}' in library ({len(self.tracks)} tracks)...")
        query_words = query.lower().split()
        results = []
        for track in self.tracks:
            track_name = track.get('name', '').lower()
            # Match if any word in query matches full or partial track name
            if any(word in track_name for word in query_words):
                results.append(track)
        
        print(f"📊 [YouTubeAudio] Found {len(results)} potential matches for '{query}'.")
        return results

    def download_track(self, track_id: str, output_path: Path) -> bool:
        """
        Download a track from Google Drive using its ID
        Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
        """
        download_url = f"https://drive.google.com/uc?export=download&id={track_id}"
        
        try:
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Handle large file confirmation page from Google Drive
            token = self._get_confirm_token(response)
            if token:
                params = {'id': track_id, 'confirm': token}
                response = session.get(download_url, params=params, stream=True)

            response.raise_for_status()
            
            # Write to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            
            return output_path.exists()
        except Exception as e:
            print(f"❌ [YouTubeAudio] Download error for {track_id} ({output_path.name}): {e}")
            return False

    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
