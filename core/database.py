"""
Database Management
Centralized database operations for caching TTS and video generation
"""
import sqlite3
import hashlib
import threading
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class GenerationDB:
    """Database for caching TTS audio and video generation results"""
    
    def __init__(self, db_path: Path = Path("generation_cache.db")):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()
    
    def init_db(self):
        """Initialize database schema"""
        # Ensure directory exists for db
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tts_cache table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tts_cache'")
            table_exists = cursor.fetchone() is not None
            
            # Check for schema compatibility
            if table_exists:
                cursor.execute("PRAGMA table_info(tts_cache)")
                columns = [info[1] for info in cursor.fetchall()]
                
                # Handle migration from old schema
                # If any required column is missing, drop and recreate
                if 'speaker_id' not in columns or 'voice_id' not in columns:
                    print("[DB] Missing required columns. Recreating tts_cache table...")
                    cursor.execute("DROP TABLE tts_cache")
                    conn.commit()
            
            # Create tables with flexible schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tts_cache (
                    text_hash TEXT PRIMARY KEY,
                    speaker_id TEXT,
                    voice_id TEXT,
                    language TEXT,
                    audio_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT UNIQUE,
                    video_path TEXT,
                    audio_path TEXT,
                    output_dir TEXT,
                    sentence_count INTEGER,
                    created_at TEXT,
                    processing_time REAL,
                    system_stats TEXT
                )
            """)
            
            conn.commit()
    
    def get_cached_tts(self, text: str, identifier: str, language: str = 'en', speed: float = 1.0) -> Optional[Path]:
        """
        Get cached TTS audio
        identifier can be speaker_id or voice_id depending on the TTS system
        """
        cache_key = f"{text}_{identifier}_{language}_{speed}"
        text_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        
        with self.lock, sqlite3.connect(self.db_path) as conn:
            # Try both speaker_id and voice_id for compatibility
            row = conn.execute(
                """SELECT audio_path FROM tts_cache 
                   WHERE text_hash = ? AND (speaker_id = ? OR voice_id = ?)""",
                (text_hash, identifier, identifier)
            ).fetchone()
            
            if row and Path(row[0]).exists():
                return Path(row[0])
        return None
    
    def save_tts(self, text: str, identifier: str, language: str, audio_path: Path, speed: float = 1.0):
        """
        Save TTS audio to cache
        identifier can be speaker_id or voice_id depending on the TTS system
        """
        cache_key = f"{text}_{identifier}_{language}_{speed}"
        text_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tts_cache 
                   (text_hash, speaker_id, voice_id, language, audio_path, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (text_hash, identifier, identifier, language, str(audio_path), datetime.now().isoformat())
            )
    
    def get_cached_video(self, input_params: Dict) -> Optional[Dict]:
        """Get cached video generation result"""
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()) 
                            if k not in {'progress_callback'})
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        
        with self.lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT video_path, audio_path, output_dir, sentence_count 
                   FROM video_logs WHERE input_hash = ?""",
                (input_hash,)
            ).fetchone()
            
            if row and all(Path(p).exists() for p in row[:2] if p):
                return {
                    "video_path": row[0],
                    "audio_path": row[1],
                    "output_directory": row[2],
                    "sentence_count": row[3],
                    "success": True
                }
        return None
    
    def save_video(self, input_params: Dict, result: Dict, processing_time: float = None):
        """Save video generation result"""
        param_str = "|".join(str(v) for k, v in sorted(input_params.items()) 
                            if k not in {'progress_callback'})
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()
        
        with self.lock, sqlite3.connect(self.db_path) as conn:
            system_stats = str({
                "timestamp": datetime.now().isoformat()
            })
            
            conn.execute(
                """INSERT OR REPLACE INTO video_logs
                   (input_hash, video_path, audio_path, output_dir, sentence_count, 
                    created_at, processing_time, system_stats)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    input_hash,
                    result.get("video_path"),
                    result.get("audio_path"),
                    result.get("output_directory"),
                    result.get("sentence_count", 0),
                    datetime.now().isoformat(),
                    processing_time,
                    system_stats
                )
            )

# Default instance for shared use
DB = GenerationDB()
