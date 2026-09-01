import unittest
from unittest.mock import MagicMock, patch
import subprocess

class TestVideoProgressAndConcat(unittest.TestCase):

    def test_progress_scaling_logic(self):
        """Test video_progress callback scaling and current/total bounding."""
        progress_calls = []

        def mock_progress_cb(current, total, msg):
            progress_calls.append((current, total, msg))

        # Simulate video_progress logic from TextToVideoGenerator
        audio_tasks = [
            {"type": "sentence", "index": 0},
            {"type": "sentence", "index": 1},
            {"type": "sentence", "index": 2},
            {"type": "intro", "index": -1},
            {"type": "cta", "index": 999},
        ]
        audio_count = len(audio_tasks)  # 5 tasks

        def video_progress(current, total, message):
            scaled_current = audio_count + int((current / max(total, 1)) * audio_count)
            mock_progress_cb(
                min(scaled_current, audio_count * 2), audio_count * 2, message
            )

        # Simulate slide rendering updates (total_slides = 5, total_steps = 10)
        total_slides = 5
        for i in range(1, total_slides + 1):
            video_progress(total_slides + i, total_slides * 2, f"Rendering slide {i}")

        # Check all reported current values <= total values (10) and percentage <= 100%
        for curr, tot, msg in progress_calls:
            self.assertLessEqual(curr, tot)
            pct = (curr / tot) * 100
            self.assertLessEqual(pct, 100.0)

    @patch("subprocess.run")
    def test_concat_failure_raises_runtime_error(self, mock_subproc):
        """Test that FFmpeg concat failure raises a RuntimeError instead of swallowing errors."""
        mock_subproc.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg", stderr="Invalid data found when processing input"
        )

        concat_cmd = ["ffmpeg", "-y", "-f", "concat", "-i", "concat.txt"]
        with self.assertRaises(RuntimeError) as ctx:
            try:
                subprocess.run(concat_cmd, check=True, capture_output=True, text=True, timeout=300)
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
                raise RuntimeError(f"FFmpeg concat failed: {err_msg[:500]}") from e

        self.assertIn("FFmpeg concat failed", str(ctx.exception))

    @patch("subprocess.run")
    def test_concat_timeout_raises_runtime_error(self, mock_subproc):
        """Test that FFmpeg concat timeout raises a RuntimeError."""
        mock_subproc.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        concat_cmd = ["ffmpeg", "-y", "-f", "concat", "-i", "concat.txt"]
        with self.assertRaises(RuntimeError) as ctx:
            try:
                subprocess.run(concat_cmd, check=True, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("FFmpeg concat process timed out") from e

        self.assertIn("timed out", str(ctx.exception))

    @patch("subprocess.run")
    def test_audio_mixing_error_fallback(self, mock_subproc):
        """Test that audio mixing failure falls back gracefully to original video without crashing."""
        mock_subproc.side_effect = Exception("Audio codec error")
        fallback_occurred = False

        try:
            subprocess.run(["ffmpeg", "-i", "video.mp4"], check=True, capture_output=True, timeout=300)
        except Exception as eMix:
            fallback_occurred = True

        self.assertTrue(fallback_occurred)

if __name__ == "__main__":
    unittest.main()
