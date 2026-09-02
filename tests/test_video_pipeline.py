import unittest
import subprocess
import shutil
import tempfile
from pathlib import Path
import torch

from core.utils.video import validate_background_asset, validate_slide, get_video_duration, has_audio_stream
from core.config import Config


class TestVideoPipeline(unittest.TestCase):

    def test_device_selection_logic(self):
        """Verify GPU/CPU device selection logic."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.assertIn(device, ("cuda", "cpu"))
        config = Config()
        self.assertIn(config.DEVICE, ("cuda", "cpu"))

    def test_background_asset_validation(self):
        """Test background asset validation for valid, missing, None in path, bad extension, empty file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Missing file
            missing_file = tmp_path / "non_existent.mp4"
            valid, reason = validate_background_asset(missing_file)
            self.assertFalse(valid)
            self.assertIn("does not exist", reason)

            # 2. Path ending in None
            none_path = tmp_path / "video_wikimedia_123None"
            valid, reason = validate_background_asset(none_path)
            self.assertFalse(valid)
            self.assertTrue("malformed" in reason or "does not exist" in reason)

            # 3. Empty file
            empty_file = tmp_path / "empty.mp4"
            empty_file.touch()
            valid, reason = validate_background_asset(empty_file)
            self.assertFalse(valid)
            self.assertIn("empty", reason)

            # 4. Bad extension
            bad_ext = tmp_path / "file.exe"
            bad_ext.write_bytes(b"12345")
            valid, reason = validate_background_asset(bad_ext)
            self.assertFalse(valid)
            self.assertIn("unsupported extension", reason)

    def test_synthetic_media_validation_and_crossfade(self):
        """Create synthetic video slides with FFmpeg and test validation & crossfade concat."""
        if not shutil.which("ffmpeg"):
            self.skipTest("FFmpeg not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            slide1 = tmp_path / "slide_1.mp4"
            slide2 = tmp_path / "slide_2.mp4"

            # Generate slide 1 (2 sec 1080x1920 video with audio)
            cmd1 = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=1080x1920:r=30:d=2",
                "-f", "lavfi", "-i", "sine=f=440:d=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                str(slide1)
            ]
            subprocess.run(cmd1, check=True, capture_output=True)

            # Generate slide 2 (2 sec 1080x1920 video with audio)
            cmd2 = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=30:d=2",
                "-f", "lavfi", "-i", "sine=f=880:d=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                str(slide2)
            ]
            subprocess.run(cmd2, check=True, capture_output=True)

            # Validate synthetic slides
            valid1, reason1 = validate_slide(slide1)
            self.assertTrue(valid1, f"Slide 1 invalid: {reason1}")

            valid2, reason2 = validate_slide(slide2)
            self.assertTrue(valid2, f"Slide 2 invalid: {reason2}")

            # Test crossfade concat filter graph execution
            out_video = tmp_path / "crossfade_out.mp4"
            crossfade_cmd = [
                "ffmpeg", "-y",
                "-i", str(slide1),
                "-i", str(slide2),
                "-filter_complex",
                "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,settb=1/AVTB,setpts=PTS-STARTPTS[v0];"
                "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,aresample=async=1,asetpts=PTS-STARTPTS[a0];"
                "[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,settb=1/AVTB,setpts=PTS-STARTPTS[v1];"
                "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,aresample=async=1,asetpts=PTS-STARTPTS[a1];"
                "[v0][v1]xfade=transition=fade:duration=0.3:offset=1.7[v_final];"
                "[a0][a1]acrossfade=d=0.3[a_final]",
                "-map", "[v_final]",
                "-map", "[a_final]",
                "-c:v", "libx264",
                "-c:a", "aac",
                str(out_video)
            ]
            subprocess.run(crossfade_cmd, check=True, capture_output=True)

            self.assertTrue(out_video.exists())
            self.assertGreater(out_video.stat().st_size, 0)
            duration = get_video_duration(out_video)
            self.assertGreater(duration, 3.0)  # ~3.7s duration



    def test_validate_slide_edge_cases(self):
        """Test validate_slide with missing, invalid, empty, corrupt, and non-file paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. None / missing path
            valid, reason = validate_slide(None)
            self.assertFalse(valid)
            self.assertIn("missing", reason)

            missing_file = tmp_path / "missing_slide.mp4"
            valid, reason = validate_slide(missing_file)
            self.assertFalse(valid)
            self.assertIn("missing", reason)

            # 2. Directory instead of regular file
            dir_path = tmp_path / "slide_dir"
            dir_path.mkdir()
            valid, reason = validate_slide(dir_path)
            self.assertFalse(valid)
            self.assertIn("not a regular file", reason)

            # 3. Empty file (0 bytes)
            empty_file = tmp_path / "empty_slide.mp4"
            empty_file.touch()
            valid, reason = validate_slide(empty_file)
            self.assertFalse(valid)
            self.assertIn("empty", reason)

            # 4. Corrupt / non-video file
            corrupt_file = tmp_path / "corrupt_slide.mp4"
            corrupt_file.write_bytes(b"not a real video file content")
            valid, reason = validate_slide(corrupt_file)
            self.assertFalse(valid)

    def test_main_imports_and_validates_slide(self):
        """Verify that main module imports validate_slide and can invoke slide validation."""
        import main
        self.assertTrue(hasattr(main, 'validate_slide'))
        self.assertEqual(main.validate_slide, validate_slide)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_slide = tmp_path / "slide_1.mp4"
            # Non-existent file validation via main's validate_slide reference
            valid, reason = main.validate_slide(fake_slide)
            self.assertFalse(valid)
            self.assertIn("missing", reason)


if __name__ == "__main__":
    unittest.main()
