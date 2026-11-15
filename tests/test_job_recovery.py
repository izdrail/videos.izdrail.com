"""Restart recovery: orphaned queued jobs re-enqueue, interrupted jobs restart.

A container restart kills the in-memory queue but the SQLite jobs table
survives. On startup the database layer marks 'processing' jobs 'interrupted'
(bumping interrupt_count), and JobManager puts 'queued' jobs back on the
queue and restarts interrupted ones as fresh jobs (new ids - no duplicate
outputs for the same job id).
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

    # The never-started job is back on the in-memory queue, and the
    # interrupted job was restarted as a NEW queued job.
    queued_ids = {jm.queue.get_nowait() for _ in range(jm.queue.qsize())}
    assert "job_queued" in queued_ids
    assert "job_running" not in queued_ids  # old id never re-runs
    new_ids = queued_ids - {"job_queued"}
    assert len(new_ids) == 1
    with sqlite3.connect(db_path) as conn:
        old = conn.execute(
            "SELECT status FROM jobs WHERE job_id='job_running'"
        ).fetchone()
        new = conn.execute(
            "SELECT status FROM jobs WHERE job_id=?", (new_ids.pop(),)
        ).fetchone()
    assert old[0] == "interrupted"  # kept for audit, no duplicate outputs
    assert new[0] == "queued"


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
