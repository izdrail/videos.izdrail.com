import json
import queue
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.database import DB


class JobManager:
    def __init__(self, config=None, max_workers: Optional[int] = None):
        from core.config import Config

        self.config = config or Config()
        self.max_workers = max_workers or getattr(self.config, "MAX_CONCURRENT_JOBS", 2)
        self.queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._generator_factory: Optional[Callable[[], Any]] = None
        self._gpu_sem = threading.Semaphore(self.max_workers)
        self._start_workers()

    def set_generator_factory(self, factory: Callable[[], Any]):
        self._generator_factory = factory

    def _start_workers(self):
        for i in range(self.max_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"job-worker-{i}", daemon=True
            )
            t.start()
            self._workers.append(t)

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                job_id = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._process_job(job_id)
            except Exception as e:
                traceback.print_exc()
                try:
                    self.update_job(job_id, status="failed", error_message=str(e))
                except Exception:
                    pass
            finally:
                self.queue.task_done()

    def _process_job(self, job_id: str):
        job = self.get_job(job_id)
        if not job or job["status"] == "canceled":
            return
        self.update_job(job_id, status="processing", progress=5)
        params = (
            json.loads(job["params"])
            if isinstance(job["params"], str)
            else job["params"]
        )
        job_temp = self.config.TEMP_DIR / f"job_{job_id[:8]}"
        job_temp.mkdir(parents=True, exist_ok=True)

        def progress_cb(current, total, msg=""):
            pct = 0
            if total:
                pct = int(current / total * 100)
            pct = max(5, min(95, pct))
            self.update_job(job_id, progress=pct)

        if self._generator_factory is None:
            raise RuntimeError("Generator factory not set")
        gen = self._generator_factory()
        try:
            result = gen.generate_video(**params, progress_callback=progress_cb)
            if not result.get("success"):
                raise RuntimeError(result.get("error", "Unknown generation error"))
            self.update_job(
                job_id,
                status="completed",
                progress=100,
                output_dir=result.get("output_directory")
                or result.get("video_path", ""),
                video_path=result.get("video_path"),
                audio_path=result.get("audio_path"),
                thumbnail_path=result.get("thumbnail_path") or "",
            )
        except Exception as e:
            traceback.print_exc()
            self.update_job(job_id, status="failed", error_message=str(e)[:2000])
            raise

    def submit_job(self, params: Dict[str, Any]) -> str:
        if not params.get("text") or not str(params["text"]).strip():
            raise ValueError("Text cannot be empty")
        sanitized = {}
        for k, v in params.items():
            if k == "progress_callback":
                continue
            if isinstance(v, Path):
                sanitized[k] = str(v)
            else:
                sanitized[k] = v
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        with DB.lock, sqlite3.connect(DB.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id,status,progress,created_at,updated_at,params) VALUES (?,?,?,?,?,?)",
                (job_id, "queued", 0, now, now, json.dumps(sanitized, default=str)),
            )
        self.queue.put(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        with DB.lock, sqlite3.connect(DB.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_jobs(self, limit: int = 100) -> List[Dict]:
        with DB.lock, sqlite3.connect(DB.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_job(self, job_id: str, **kwargs):
        if not kwargs:
            return
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [job_id]
        with DB.lock, sqlite3.connect(DB.db_path) as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE job_id=?", vals)

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job or job["status"] not in ("queued", "processing"):
            return False
        self.update_job(job_id, status="canceled")
        return True

    def retry_job(self, job_id: str) -> Optional[str]:
        job = self.get_job(job_id)
        if not job:
            return None
        params = (
            json.loads(job["params"])
            if isinstance(job["params"], str)
            else job["params"]
        )
        return self.submit_job(params)

    def recover_interrupted_jobs(self) -> int:
        """Mark jobs stuck in 'queued'/'processing' from a previous run as failed.

        Worker threads are in-memory only: after an app restart any job left in
        those states will never move again. Reclassifying them as failed (with a
        clear message) is what makes them restartable via the UI retry button.
        Returns the number of jobs recovered.
        """
        now = datetime.now().isoformat()
        with DB.lock, sqlite3.connect(DB.db_path) as conn:
            cur = conn.execute(
                "UPDATE jobs SET status='failed', error_message=?, updated_at=? "
                "WHERE status IN ('queued','processing')",
                ("Interrupted: app restarted before this job finished. Safe to retry.", now),
            )
            return cur.rowcount

    def shutdown(self):
        self._stop.set()
