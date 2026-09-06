"""
Database Management
Centralized database operations for caching TTS and video generation
"""

import logging
import sqlite3
import hashlib
import json
import threading
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class GenerationDB:
    """Database for caching TTS audio and video generation results"""

    def __init__(self, db_path: Path = Path("generation_cache.db")):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()

    def init_db(self):
        """Initialize database schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # WAL mode allows concurrent readers (the selection audit can be
            # written from worker threads without blocking generation readers).
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass

            cursor = conn.cursor()

            # Check if tts_cache table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tts_cache'"
            )
            table_exists = cursor.fetchone() is not None

            # Check for schema compatibility
            if table_exists:
                cursor.execute("PRAGMA table_info(tts_cache)")
                columns = [info[1] for info in cursor.fetchall()]

                # Handle migration from old schema
                # If any required column is missing, drop and recreate
                if "speaker_id" not in columns or "voice_id" not in columns:
                    print(
                        "[DB] Missing required columns. Recreating tts_cache table..."
                    )
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

            # Audit trail for keyword-selection calibration (see keyword_extractor
            # selection_history). Captures every finalized selection so weights can
            # later be tuned with a regression/classifier instead of hand-picked.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keyword_selection_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    script_id TEXT,
                    sentence_idx INTEGER,
                    keyword TEXT,
                    context_preview TEXT,
                    signals_json TEXT,
                    decision_score REAL,
                    was_used INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS clip_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_url TEXT UNIQUE,
                    keyword TEXT,
                    source TEXT,
                    selected_count INTEGER DEFAULT 0,
                    replaced_count INTEGER DEFAULT 0,
                    watch_duration REAL DEFAULT 0.0,
                    completion_rate REAL DEFAULT 0.0,
                    feedback_score REAL DEFAULT 0.0,
                    last_updated TEXT NOT NULL
                )
            """)

            conn.commit()

    def get_cached_tts(
        self, text: str, identifier: str, language: str = "en", speed: float = 1.0
    ) -> Optional[Path]:
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
                (text_hash, identifier, identifier),
            ).fetchone()

            if row and Path(row[0]).exists():
                return Path(row[0])
        return None

    def save_tts(
        self,
        text: str,
        identifier: str,
        language: str,
        audio_path: Path,
        speed: float = 1.0,
    ):
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
                (
                    text_hash,
                    identifier,
                    identifier,
                    language,
                    str(audio_path),
                    datetime.now().isoformat(),
                ),
            )

    def get_cached_video(self, input_params: Dict) -> Optional[Dict]:
        """Get cached video generation result"""
        param_str = "|".join(
            str(v)
            for k, v in sorted(input_params.items())
            if k not in {"progress_callback"}
        )
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()

        with self.lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT video_path, audio_path, output_dir, sentence_count 
                   FROM video_logs WHERE input_hash = ?""",
                (input_hash,),
            ).fetchone()

            if row and all(Path(p).exists() for p in row[:2] if p):
                return {
                    "video_path": row[0],
                    "audio_path": row[1],
                    "output_directory": row[2],
                    "sentence_count": row[3],
                    "success": True,
                }
        return None

    def save_video(
        self, input_params: Dict, result: Dict, processing_time: float = None
    ):
        """Save video generation result"""
        param_str = "|".join(
            str(v)
            for k, v in sorted(input_params.items())
            if k not in {"progress_callback"}
        )
        input_hash = hashlib.sha256(param_str.encode()).hexdigest()

        with self.lock, sqlite3.connect(self.db_path) as conn:
            system_stats = str({"timestamp": datetime.now().isoformat()})

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
                    system_stats,
                ),
            )

    def log_keyword_selection(
        self,
        script_id: Optional[str],
        sentence_idx: Optional[int],
        keyword: Optional[str],
        context_preview: str = "",
        signals: Optional[Any] = None,
        decision_score: Optional[float] = None,
        was_used: bool = False,
    ) -> None:
        """Record a finalized keyword selection for later calibration.

        Best-effort and non-blocking: failures are logged and swallowed so the
        generation pipeline is never stalled by the audit write. The
        ``signals_json`` / ``decision_score`` columns are left nullable so the
        raw neuron signals can be piped in later without a schema change.
        """
        try:
            with self.lock, sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO keyword_selection_audit
                       (timestamp, script_id, sentence_idx, keyword,
                        context_preview, signals_json, decision_score, was_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        script_id,
                        sentence_idx,
                        keyword,
                        (context_preview or "")[:200],
                        json.dumps(signals) if signals is not None else None,
                        decision_score,
                        int(bool(was_used)),
                    ),
                )
        except Exception as e:  # pragma: no cover - best-effort audit
            logger.warning("[DB] keyword selection audit write failed: %s", e)

    def log_clip_performance(
        self,
        media_url: str,
        keyword: str = "",
        source: str = "",
        event_type: str = "select",
        watch_duration: float = 0.0,
        completion_rate: float = 0.0,
        score_delta: float = 0.0,
    ) -> None:
        """Log user engagement / feedback for a clip to update bandit performance scores."""
        if not media_url:
            return
        try:
            with self.lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT selected_count, replaced_count, watch_duration, completion_rate, feedback_score FROM clip_performance WHERE media_url=?",
                    (media_url,),
                )
                row = cursor.fetchone()
                now_str = datetime.now().isoformat()

                if row:
                    sel_cnt, rep_cnt, prev_dur, prev_comp, prev_score = row
                    if event_type == "select":
                        sel_cnt += 1
                        prev_score += 0.1
                    elif event_type == "replace":
                        rep_cnt += 1
                        prev_score -= 0.2
                    elif event_type == "watch":
                        prev_dur += watch_duration
                        prev_comp = max(prev_comp, completion_rate)
                        prev_score += completion_rate * 0.2
                    elif event_type == "feedback":
                        prev_score += score_delta

                    cursor.execute(
                        """UPDATE clip_performance SET
                           selected_count=?, replaced_count=?, watch_duration=?,
                           completion_rate=?, feedback_score=?, last_updated=?
                           WHERE media_url=?""",
                        (sel_cnt, rep_cnt, prev_dur, prev_comp, prev_score, now_str, media_url),
                    )
                else:
                    sel_cnt = 1 if event_type == "select" else 0
                    rep_cnt = 1 if event_type == "replace" else 0
                    fb_score = (
                        0.1 if event_type == "select" else (-0.2 if event_type == "replace" else score_delta)
                    )
                    cursor.execute(
                        """INSERT INTO clip_performance
                           (media_url, keyword, source, selected_count, replaced_count,
                            watch_duration, completion_rate, feedback_score, last_updated)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            media_url,
                            keyword,
                            source,
                            sel_cnt,
                            rep_cnt,
                            watch_duration,
                            completion_rate,
                            fb_score,
                            now_str,
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.warning("[DB] clip_performance log write failed: %s", e)

    def get_clip_performance_score(
        self, media_url: str = None, keyword: str = None, source: str = None
    ) -> float:
        """Returns performance multiplier/offset based on past engagement history."""
        try:
            with self.lock, sqlite3.connect(self.db_path) as conn:
                score = 0.0
                cursor = conn.cursor()
                if media_url:
                    row = cursor.execute(
                        "SELECT feedback_score, completion_rate, replaced_count FROM clip_performance WHERE media_url=?",
                        (media_url,),
                    ).fetchone()
                    if row:
                        fb_score, comp_rate, rep_cnt = row
                        score += fb_score * 0.2 + comp_rate * 0.1 - rep_cnt * 0.05

                if source:
                    row = cursor.execute(
                        "SELECT AVG(feedback_score) FROM clip_performance WHERE source=?",
                        (source,),
                    ).fetchone()
                    if row and row[0] is not None:
                        score += float(row[0]) * 0.1

                return max(-0.5, min(0.5, score))
        except Exception as e:
            logger.warning("[DB] get_clip_performance_score failed: %s", e)
            return 0.0


# Default instance for shared use
DB = GenerationDB()
