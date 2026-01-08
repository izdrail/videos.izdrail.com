"""
Video utilities for processing, metadata extraction and formatting
"""
import subprocess
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[VideoUtils] Error getting duration for {video_path}: {e}")
        return 0.0

def has_audio_stream(video_path: Path) -> bool:
    """Check if video has an audio stream using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ], capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except Exception:
        return False

def is_video_file(path: Path) -> bool:
    """Detect if file is a video based on extension and magic bytes"""
    if not path or not path.exists():
        return False
        
    ext = path.suffix.lower()
    if ext in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
        return True
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
        return False

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "v:0", str(path)],
            capture_output=True, text=True, timeout=8
        )
        return bool(probe.stdout.strip())
    except Exception:
        return False

def sanitize_url_filename(url: str) -> str:
    """Create a safe filename for a URL"""
    parsed = urlparse(url)
    base = Path(parsed.path).name or hashlib.sha256(url.encode()).hexdigest()[:12]
    ext = Path(base).suffix
    if not ext:
        mime, _ = mimetypes.guess_type(url)
        ext = {
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif"
        }.get(mime, ".bin")
    
    # Use hash of URL to avoid collisions with same-name files from different domains
    name_hash = hashlib.sha256(url.encode()).hexdigest()[:20]
    return f"{name_hash}{ext}"
