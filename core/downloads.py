"""Choose the completed job artifact without hiding missing-file failures."""
from pathlib import Path


def job_download(job):
    if not job:
        return None, 'Select a completed job first.'
    if job.get('status') != 'completed':
        return None, 'This job is not completed yet.'
    for key in ('video_path', 'audio_path'):
        path = job.get(key)
        if path and Path(path).is_file():
            return str(Path(path)), 'Ready to download.'
    return None, 'The output file is missing. Regenerate this job to create it again.'
