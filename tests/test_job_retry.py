"""
Tests for failed-job restart: JobManager.retry_job + recover_interrupted_jobs.

The UI fix (Jobs dropdown listing failed/canceled jobs) lives in main.py; these
tests pin the manager-level semantics that make restarting a failed render work.
"""

import json
import sqlite3

import pytest

from core.database import DB
from core.job_manager import JobManager


@pytest.fixture()
def jm(tmp_path, monkeypatch):
    """A JobManager with an isolated DB and no worker threads (no factory set)."""
    monkeypatch.setattr(DB, "db_path", tmp_path / "jobs_test.db")
    DB.init_db()
    monkeypatch.setattr(JobManager, "_start_workers", lambda self: None)
    return JobManager(config=None)


def _set_status(jm, job_id, status):
    with DB.lock, sqlite3.connect(DB.db_path) as conn:
        conn.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, job_id))


def test_retry_failed_job_creates_new_queued_job(jm):
    job_id = jm.submit_job({"text": "hello world", "language": "en"})
    _set_status(jm, job_id, "failed")

    new_id = jm.retry_job(job_id)

    assert new_id is not None and new_id != job_id
    new_job = jm.get_job(new_id)
    assert new_job["status"] == "queued"
    params = json.loads(new_job["params"])
    assert params["text"] == "hello world"
    assert params["language"] == "en"


def test_retry_canceled_job(jm):
    job_id = jm.submit_job({"text": "retry me"})
    _set_status(jm, job_id, "canceled")
    assert jm.retry_job(job_id) is not None


def test_retry_unknown_job_returns_none(jm):
    assert jm.retry_job("doesnotexist") is None


def test_recover_interrupted_jobs_marks_stale_jobs_failed(jm):
    stuck_processing = jm.submit_job({"text": "was processing"})
    stuck_queued = jm.submit_job({"text": "was queued"})
    done = jm.submit_job({"text": "finished fine"})
    _set_status(jm, stuck_processing, "processing")
    _set_status(jm, done, "completed")

    recovered = jm.recover_interrupted_jobs()

    assert recovered == 2
    for jid in (stuck_processing, stuck_queued):
        job = jm.get_job(jid)
        assert job["status"] == "failed"
        assert "retry" in job["error_message"].lower()
    assert jm.get_job(done)["status"] == "completed"

    # And the recovered jobs are immediately restartable.
    assert jm.retry_job(stuck_processing) is not None
