import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('downloads', Path(__file__).parents[1] / 'core/downloads.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_audio_only(tmp_path):
    p = tmp_path / 'voice.mp3'; p.write_bytes(b'audio')
    assert module.job_download({'status':'completed','video_path':None,'audio_path':str(p)})[0] == str(p)

def test_video_first(tmp_path):
    p=tmp_path/'video.mp4';p.write_bytes(b'video')
    assert module.job_download({'status':'completed','video_path':str(p)})[0] == str(p)

def test_missing_and_unselected():
    assert module.job_download(None)[0] is None
    assert 'missing' in module.job_download({'status':'completed','video_path':'/missing/file'})[1]
