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

import random

def get_random_middle_frame(video_path: Path, output_path: Path, middle_start: float = 0.4, middle_end: float = 0.6) -> Path:
    """Extract a random frame from the middle portion of a video.

    Args:
        video_path (Path): Path to the source video file.
        output_path (Path): Destination path for the extracted frame image.
        middle_start (float): Fraction of duration where the middle section starts (default 0.4).
        middle_end (float): Fraction of duration where the middle section ends (default 0.6).

    Returns:
        Path: The path to the extracted frame image.
    """
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

# New helper: extract frame at (duration - 1 second) (or last frame if video shorter)
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

import random

def get_random_middle_frame(video_path: Path, output_path: Path, middle_start: float = 0.4, middle_end: float = 0.6) -> Path:
    """Extract a random frame from the middle portion of a video.

    Args:
        video_path (Path): Path to the source video file.
        output_path (Path): Destination path for the extracted frame image.
        middle_start (float): Fraction of duration where the middle section starts (default 0.4).
        middle_end (float): Fraction of duration where the middle section ends (default 0.6).

    Returns:
        Path: The path to the extracted frame image.
    """
    # Ensure fractions are valid
    if not (0 <= middle_start < middle_end <= 1):
        raise ValueError("middle_start must be < middle_end and both within [0,1]")

    duration = get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Unable to determine duration for video {video_path}")

    # Compute the time window for the middle section
    start_sec = duration * middle_start
    end_sec = duration * middle_end
    # Choose a random timestamp within this window
    timestamp = random.uniform(start_sec, end_sec)

    # Build ffmpeg command to extract a single frame at the timestamp
    # -ss before -i for fast seeking
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
