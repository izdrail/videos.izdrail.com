"""
Video utilities for processing, metadata extraction and formatting
"""
import random
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

# New helper: extract a smart thumbnail frame (1s from end, or middle if short)
def get_smart_thumbnail_frame(video_path: Path, output_path: Path) -> Path:
    """Extract a representative thumbnail frame from the video.

    Formula:
    - If video > 1.0s: Extract at (Duration - 1.0s)
    - If video <= 1.0s: Extract at 50% (middle)
    
    This avoids black frames at the very end (fade-outs) and ensures
    we capture content on short clips.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Unable to determine duration for video {video_path}")

    # Simplified, robust formula
    if duration >= 1.0:
        timestamp = duration - 1.0
    else:
        timestamp = duration * 0.5 # Middle of short clip
    
    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
        "-y",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to extract smart thumbnail frame: {e}")
    return output_path

def get_last_minus_one_second_frame(video_path: Path, output_path: Path) -> Path:
    """Extract a frame from the video at (duration - 1 second).

    If the video is shorter than 1 second, extracts the first frame.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Unable to determine duration for video {video_path}")
    timestamp = max(duration - 1.0, 0.0)
    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
        "-y",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to extract last-minus-1s frame: {e}")
    return output_path

def get_random_middle_frame(video_path: Path, output_path: Path, middle_start: float = 0.4, middle_end: float = 0.6) -> Path:
    if not (0 <= middle_start < middle_end <= 1):
        raise ValueError("middle_start must be < middle_end and both within [0,1]")
    duration = get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Unable to determine duration for video {video_path}")
    start_sec = duration * middle_start
    end_sec = duration * middle_end
    timestamp = random.uniform(start_sec, end_sec)
    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
        "-y",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to extract frame: {e}")
    return output_path


def validate_background_asset(path: Path) -> tuple[bool, str]:
    """Validate that background asset file exists, is regular, has valid extension and is FFmpeg-readable."""
    if not path:
        return False, "path is None"

    path_str = str(path)
    if path_str.endswith("None") or "/app/background_videos/" in path_str and path_str.endswith("None"):
        return False, f"malformed path ending in None: {path_str}"

    path_obj = Path(path)
    if not path_obj.exists():
        return False, f"file does not exist: {path_obj}"
    if not path_obj.is_file():
        return False, f"not a regular file: {path_obj}"
    if path_obj.stat().st_size == 0:
        return False, f"file is empty (0 bytes): {path_obj}"

    valid_exts = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
    if path_obj.suffix.lower() not in valid_exts:
        return False, f"unsupported extension: {path_obj.suffix}"

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path_obj)],
            capture_output=True, text=True, timeout=10
        )
        if probe.returncode != 0:
            return False, f"ffprobe failed: {probe.stderr.strip()}"
    except Exception as e:
        return False, f"ffprobe error: {e}"

    return True, "OK"


def validate_slide(slide_path: Path) -> tuple[bool, str]:
    """Validate generated slide video before concatenation."""
    if not slide_path or not Path(slide_path).exists():
        return False, f"slide file missing: {slide_path}"

    p = Path(slide_path)
    if not p.is_file():
        return False, f"slide is not a regular file: {p}"
    if p.stat().st_size == 0:
        return False, f"slide file is empty: {p}"

    duration = get_video_duration(p)
    if duration <= 0:
        return False, f"slide duration is invalid or 0s: {p}"

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=10
        )
        if not probe.stdout.strip():
            return False, f"slide has no video stream: {p}"
    except Exception as e:
        return False, f"slide video stream probe failed: {e}"

    return True, "OK"
