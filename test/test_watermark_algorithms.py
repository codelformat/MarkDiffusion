"""
Parameterized pytest tests for all watermark algorithms in MarkDiffusion.

Usage:
    # Test all image watermark algorithms
    pytest test/test_watermark_algorithms.py -v

    # Test specific algorithm
    pytest test/test_watermark_algorithms.py -v -k "test_image_watermark[TR]"

    # Test specific algorithms using markers
    pytest test/test_watermark_algorithms.py -v -m "image"
    pytest test/test_watermark_algorithms.py -v -m "video"

    # Test with custom parameters
    pytest test/test_watermark_algorithms.py -v --algorithm TR --image-model-path /path/to/model
"""

import pytest
from PIL import Image
from typing import Dict, Any

from watermark.auto_watermark import AutoWatermark, PIPELINE_SUPPORTED_WATERMARKS
from utils.pipeline_utils import (
    get_pipeline_type,
    PIPELINE_TYPE_IMAGE,
    PIPELINE_TYPE_TEXT_TO_VIDEO,
)

# Import test constants from conftest
from .conftest import (
    TEST_PROMPT_IMAGE,
    TEST_PROMPT_VIDEO,
    IMAGE_SIZE,
    NUM_FRAMES,
)


# ============================================================================
# Test Cases - Image Watermarks
# ============================================================================

@pytest.mark.image
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_initialization(algorithm_name, image_diffusion_config):
    """Test that image watermark algorithms can be initialized correctly."""
    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )
        assert watermark is not None
        assert watermark.config is not None
        assert get_pipeline_type(watermark.config.pipe) == PIPELINE_TYPE_IMAGE
        print(f"✓ {algorithm_name} initialized successfully")
    except Exception as e:
        pytest.fail(f"Failed to initialize {algorithm_name}: {e}")


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_generation(algorithm_name, image_diffusion_config, skip_generation):
    """Test watermarked image generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )

        # Generate watermarked image
        watermarked_image = watermark.generate_watermarked_media(TEST_PROMPT_IMAGE)

        # Validate output
        assert watermarked_image is not None
        assert isinstance(watermarked_image, Image.Image)
        assert watermarked_image.size == (IMAGE_SIZE[1], IMAGE_SIZE[0])

        print(f"✓ {algorithm_name} generated watermarked image successfully")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermarked image generation")
    except Exception as e:
        pytest.fail(f"Failed to generate watermarked image with {algorithm_name}: {e}")


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_unwatermarked_generation(algorithm_name, image_diffusion_config, skip_generation):
    """Test unwatermarked image generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )

        # Generate unwatermarked image
        unwatermarked_image = watermark.generate_unwatermarked_media(TEST_PROMPT_IMAGE)

        # Validate output
        assert unwatermarked_image is not None
        assert isinstance(unwatermarked_image, Image.Image)
        assert unwatermarked_image.size == (IMAGE_SIZE[1], IMAGE_SIZE[0])

        print(f"✓ {algorithm_name} generated unwatermarked image successfully")

    except Exception as e:
        pytest.fail(f"Failed to generate unwatermarked image with {algorithm_name}: {e}")


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_detection(algorithm_name, image_diffusion_config, skip_detection):
    """Test watermark detection in images for each algorithm."""
    if skip_detection:
        pytest.skip("Detection tests skipped by --skip-detection flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )

        # Generate watermarked and unwatermarked images
        watermarked_image = watermark.generate_watermarked_media(TEST_PROMPT_IMAGE)
        unwatermarked_image = watermark.generate_unwatermarked_media(TEST_PROMPT_IMAGE)

        # Detect watermark in watermarked image
        detection_result_wm = watermark.detect_watermark_in_media(watermarked_image)
        assert detection_result_wm is not None
        assert isinstance(detection_result_wm, dict)
        assert detection_result_wm['is_watermarked'] is True

        # Detect watermark in unwatermarked image
        detection_result_unwm = watermark.detect_watermark_in_media(unwatermarked_image)
        assert detection_result_unwm is not None
        assert isinstance(detection_result_unwm, dict)
        assert detection_result_unwm['is_watermarked'] is False

        print(f"✓ {algorithm_name} detection results:")
        print(f"  Watermarked: {detection_result_wm}")
        print(f"  Unwatermarked: {detection_result_unwm}")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermark detection")
    except Exception as e:
        pytest.fail(f"Failed to detect watermark with {algorithm_name}: {e}")


# ============================================================================
# Test Cases - Video Watermarks
# ============================================================================

@pytest.mark.video
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_initialization(algorithm_name, video_diffusion_config):
    """Test that video watermark algorithms can be initialized correctly."""
    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )
        assert watermark is not None
        assert watermark.config is not None
        assert get_pipeline_type(watermark.config.pipe) == PIPELINE_TYPE_TEXT_TO_VIDEO
        print(f"✓ {algorithm_name} initialized successfully")
    except Exception as e:
        pytest.fail(f"Failed to initialize {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_generation(algorithm_name, video_diffusion_config, skip_generation):
    """Test watermarked video generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate watermarked video
        watermarked_frames = watermark.generate_watermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Validate output
        assert watermarked_frames is not None
        assert isinstance(watermarked_frames, list)
        assert len(watermarked_frames) > 0
        assert all(isinstance(frame, Image.Image) for frame in watermarked_frames)

        print(f"✓ {algorithm_name} generated {len(watermarked_frames)} watermarked frames")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermarked video generation")
    except Exception as e:
        pytest.fail(f"Failed to generate watermarked video with {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_unwatermarked_generation(algorithm_name, video_diffusion_config, skip_generation):
    """Test unwatermarked video generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate unwatermarked video
        unwatermarked_frames = watermark.generate_unwatermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Validate output
        assert unwatermarked_frames is not None
        assert isinstance(unwatermarked_frames, list)
        assert len(unwatermarked_frames) > 0
        assert all(isinstance(frame, Image.Image) for frame in unwatermarked_frames)

        print(f"✓ {algorithm_name} generated {len(unwatermarked_frames)} unwatermarked frames")

    except Exception as e:
        pytest.fail(f"Failed to generate unwatermarked video with {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_detection(algorithm_name, video_diffusion_config, skip_detection):
    """Test watermark detection in videos for each algorithm."""
    if skip_detection:
        pytest.skip("Detection tests skipped by --skip-detection flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate watermarked and unwatermarked videos
        watermarked_frames = watermark.generate_watermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )
        unwatermarked_frames = watermark.generate_unwatermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Detect watermark in watermarked video
        detection_result_wm = watermark.detect_watermark_in_media(
            watermarked_frames,
            prompt=TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )
        assert detection_result_wm is not None
        assert isinstance(detection_result_wm, dict)
        assert detection_result_wm['is_watermarked'] is True

        # Detect watermark in unwatermarked video
        detection_result_unwm = watermark.detect_watermark_in_media(
            unwatermarked_frames,
            prompt=TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )
        assert detection_result_unwm is not None
        assert isinstance(detection_result_unwm, dict)
        assert detection_result_unwm['is_watermarked'] is False

        print(f"✓ {algorithm_name} detection results:")
        print(f"  Watermarked: {detection_result_wm}")
        print(f"  Unwatermarked: {detection_result_unwm}")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermark detection")
    except Exception as e:
        pytest.fail(f"Failed to detect watermark with {algorithm_name}: {e}")


# ============================================================================
# Test Cases - Algorithm Compatibility
# ============================================================================

def test_algorithm_list():
    """Test that all algorithms are properly registered."""
    image_algorithms = AutoWatermark.list_supported_algorithms(PIPELINE_TYPE_IMAGE)
    video_algorithms = AutoWatermark.list_supported_algorithms(PIPELINE_TYPE_TEXT_TO_VIDEO)

    assert len(image_algorithms) > 0, "No image algorithms found"
    assert len(video_algorithms) > 0, "No video algorithms found"

    print(f"Image algorithms: {image_algorithms}")
    print(f"Video algorithms: {video_algorithms}")


def test_invalid_algorithm():
    """Test that invalid algorithm names raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid algorithm name"):
        AutoWatermark.load("InvalidAlgorithm", diffusion_config=None)
