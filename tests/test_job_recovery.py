"""Restart recovery: orphaned queued jobs re-enqueue, interrupted jobs restart.

A container restart kills the in-memory queue but the SQLite jobs table
survives. On startup the database layer marks 'processing' jobs 'interrupted'
(bumping interrupt_count), and JobManager puts 'queued' jobs back on the
queue and resumes interrupted ones under their original id. Legacy rows
already covered by active/completed work are not replayed.
"""
import json
import sqlite3
from pathlib import Path

import pytest

import core.database as db_mod
from core.config import Config
from core.database import DB, GenerationDB
from core.job_manager import JobManager


def _insert_job(db_path, job_id, status, created, interrupt_count=0):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id,status,progress,created_at,updated_at,params,interrupt_count)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                job_id,
                status,
                40 if status == "processing" else 0,
                created,
                created,
                json.dumps({"text": "hello world"}),
                interrupt_count,
            ),
        )


def _setup_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    GenerationDB(db_path=db_path)  # creates schema
    DB.db_path = db_path  # JobManager reads through the shared handle
    return db_path


def _make_manager(monkeypatch, **kwargs):
    # No worker threads: recovery must be observable before any job runs.
    monkeypatch.setattr(JobManager, "_start_workers", lambda self: None)
    return JobManager(config=Config(), **kwargs)


def test_processing_jobs_become_interrupted_on_startup(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "job_running", "processing", "2026-11-15T08:00:00")
    GenerationDB(db_path=db_path)  # simulates the next server start
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, interrupt_count, error_message FROM jobs WHERE job_id='job_running'"
        ).fetchone()
    assert row[0] == "interrupted"
    assert row[1] == 1
    assert "restart" in row[2].lower()


def test_queued_jobs_are_requeued_and_interrupted_autorestarted(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "job_queued", "queued", "2026-11-15T08:00:00")
    _insert_job(db_path, "job_running", "processing", "2026-11-15T08:01:00")
    GenerationDB(db_path=db_path)  # startup marks job_running interrupted

    jm = _make_manager(monkeypatch)

    # Existing queued work with identical params covers the legacy interruption.
    queued_ids = {jm.queue.get_nowait() for _ in range(jm.queue.qsize())}
    assert queued_ids == {"job_queued"}
    assert jm.get_job("job_running")["status"] == "interrupted"
    assert len(jm.get_all_jobs()) == 2


def test_interrupted_reuses_id_once_and_keeps_attempt_count(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "same_id", "interrupted", "2026-11-15T08:00:00", interrupt_count=2)
    jm = _make_manager(monkeypatch)
    assert jm.queue.get_nowait() == "same_id"
    assert jm.get_job("same_id")["status"] == "queued"
    assert jm.get_job("same_id")["interrupt_count"] == 2
    jm.recover_interrupted_jobs()
    assert jm.queue.empty()
    assert len(jm.get_all_jobs()) == 1


@pytest.mark.parametrize("cover_status", ["completed", "processing", "queued"])
def test_legacy_interrupted_covered_by_existing_work_not_replayed(tmp_path, monkeypatch, cover_status):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "old_interruption", "interrupted", "2026-11-15T08:00:00", 1)
    _insert_job(db_path, "successor", cover_status, "2026-11-15T08:01:00")
    jm = _make_manager(monkeypatch)
    assert jm.get_job("old_interruption")["status"] == "interrupted"
    assert len(jm.get_all_jobs()) == 2
    assert list(jm.queue.queue) == (["successor"] if cover_status == "queued" else [])


def test_different_params_do_not_suppress_recovery(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "resume", "interrupted", "2026-11-15T08:00:00", 1)
    _insert_job(db_path, "other", "completed", "2026-11-15T08:01:00")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE jobs SET params=? WHERE job_id='other'", (json.dumps({"text":"different"}),))
    jm = _make_manager(monkeypatch)
    assert jm.queue.get_nowait() == "resume"
    assert jm.get_job("resume")["status"] == "queued"


def test_auto_restart_can_be_disabled(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "job_running", "processing", "2026-11-15T08:00:00")
    GenerationDB(db_path=db_path)
    monkeypatch.setenv("VIDEOS_AUTO_RESTART_INTERRUPTED", "0")

    jm = _make_manager(monkeypatch)
    assert jm.queue.qsize() == 0
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT status FROM jobs WHERE job_id='job_running'"
            ).fetchone()[0]
            == "interrupted"
        )


def test_crash_loop_guard_stops_restarting_after_three_interrupts(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(
        db_path, "job_poison", "processing", "2026-11-15T08:00:00", interrupt_count=2
    )
    GenerationDB(db_path=db_path)  # interrupt_count becomes 3

    jm = _make_manager(monkeypatch)
    assert jm.queue.qsize() == 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, interrupt_count FROM jobs WHERE job_id='job_poison'"
        ).fetchone()
    assert row == ("interrupted", 3)


def test_retry_job_creates_new_id_with_same_params(tmp_path, monkeypatch):
    db_path = _setup_db(tmp_path)
    _insert_job(db_path, "job_failed", "failed", "2026-11-15T08:00:00")
    jm = _make_manager(monkeypatch)
    new_id = jm.retry_job("job_failed")
    assert new_id and new_id != "job_failed"
    old = jm.get_job("job_failed")
    new = jm.get_job(new_id)
    assert old["status"] == "failed"
    assert new["status"] == "queued"
    assert json.loads(new["params"])["text"] == "hello world"

def test_recovery_waits_for_factory_before_consuming_queue(tmp_path, monkeypatch):
    import threading
    import time
    db_path = _setup_db(tmp_path)
    _insert_job(db_path,'waiting_factory','queued','2026-11-16T10:00:00')
    config=Config()
    config.TEMP_DIR=tmp_path
    jm=JobManager(config=config,max_workers=1)
    try:
        time.sleep(0.1)
        assert jm.get_job('waiting_factory')['status']=='queued'
        ran=threading.Event()
        class Generator:
            def generate_video(self,**params):
                ran.set()
                return {'success':True}
        jm.set_generator_factory(Generator)
        assert ran.wait(2)
        jm.queue.join()
        assert jm.get_job('waiting_factory')['status']=='completed'
    finally:
        jm.shutdown()
