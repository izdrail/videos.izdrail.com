"""
Unit tests for SD-Turbo generator, device detection, prompt generator, and caching.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from core.ai.prompt_generator import PromptGenerator
from core.ai.stable_diffusion import SDTurboGenerator, StableDiffusionManager
from core.config import Config
from core.utils.gpu import get_optimal_device


class TestGPUUtils:
    def test_get_optimal_device_cpu(self):
        assert get_optimal_device("cpu") == "cpu"

    def test_get_optimal_device_auto(self):
        dev = get_optimal_device("auto")
        assert dev in ["cuda", "cpu"]

    def test_get_optimal_device_cuda_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                get_optimal_device("cuda")


class TestPromptGenerator:
    def test_generate_with_keyword_and_entities(self):
        pg = PromptGenerator()
        prompt = pg.generate(
            sentence="A CEO delivering a keynote speech",
            keyword="CEO",
            entities=["Silicon Valley", "Conference"],
        )
        assert "CEO" in prompt
        assert "Silicon Valley" in prompt
        assert "Cinematic editorial photograph" in prompt

    def test_generate_with_sentence_only(self):
        pg = PromptGenerator()
        prompt = pg.generate(sentence="The sun rises over snowy mountains")
        assert "Cinematic editorial photograph" in prompt
        assert "rises" in prompt or "mountain" in prompt or "snowy" in prompt


class TestSDTurboGenerator:
    @pytest.fixture
    def mock_config(self, tmp_path):
        cfg = Config()
        cfg.IMAGE_GENERATION_CACHE_DIR = tmp_path / "cache"
        cfg.IMAGE_GENERATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cfg

    def test_init(self, mock_config):
        generator = SDTurboGenerator(config=mock_config)
        assert generator._loaded is False
        assert generator._pipeline is None

    def test_cache_key_generation(self, mock_config):
        generator = SDTurboGenerator(config=mock_config)
        key1 = generator._generate_cache_key("test prompt", 512, 512, 1, 0.0)
        key2 = generator._generate_cache_key("test prompt", 512, 512, 1, 0.0)
        key3 = generator._generate_cache_key("different prompt", 512, 512, 1, 0.0)

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64

    def test_cache_hit(self, mock_config, tmp_path):
        generator = SDTurboGenerator(config=mock_config)
        prompt = "test cache hit"
        cache_key = generator._generate_cache_key(prompt, 512, 512, 1, 0.0)
        cache_file = mock_config.IMAGE_GENERATION_CACHE_DIR / f"{cache_key}.png"

        # Create dummy image in cache
        img = Image.new("RGB", (1080, 1920), color="blue")
        img.save(cache_file)

        # Call generate without loading pipeline
        res = generator.generate(prompt, width=512, height=512, steps=1, guidance_scale=0.0)
        assert res == cache_file
        assert generator._loaded is False

    def test_generation_mock_pipeline(self, mock_config):
        generator = SDTurboGenerator(config=mock_config)

        mock_pil = Image.new("RGB", (512, 512), color="red")
        mock_output = MagicMock()
        mock_output.images = [mock_pil]

        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_output

        with patch.object(generator, "_load_model") as mock_load:
            generator._pipeline = mock_pipe
            generator._loaded = True

            result_path = generator.generate(
                prompt="a red square",
                width=512,
                height=512,
                steps=1,
                guidance_scale=0.0,
                target_size=(1080, 1920),
            )

            assert result_path is not None
            assert result_path.exists()

            saved_img = Image.open(result_path)
            assert saved_img.size == (1080, 1920)

    def test_stable_diffusion_manager_wrapper(self, mock_config):
        mgr = StableDiffusionManager(config=mock_config)
        assert mgr.generator is not None
