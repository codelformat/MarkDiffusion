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


# ============================================================================
# Test Cases - Inversion Modules
# ============================================================================

@pytest.mark.inversion
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_inversion_4d_image_input(inversion_type, device, image_pipeline):
    """Test inversion modules with 4D image input (batch_size, channels, height, width)."""
    import torch
    from inversions import DDIMInversion, ExactInversion

    pipe, scheduler = image_pipeline

    # Create inversion instance
    if inversion_type == "ddim":
        inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)
    else:  # exact
        inversion = ExactInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create 4D test input: (batch_size, channels, height, width)
    batch_size = 1
    channels = 4  # latent space channels
    height = 64   # latent space height (512 / 8)
    width = 64    # latent space width (512 / 8)

    latents_input = torch.randn(batch_size, channels, height, width).to(device)

    # Create text embeddings
    text_embeddings = torch.randn(1, 77, 768).to(device)  # Standard CLIP embedding size

    try:
        # Test forward diffusion (image to noise)
        intermediate_latents = inversion.forward_diffusion(
            text_embeddings=text_embeddings,
            latents=latents_input,
            num_inference_steps=10,  # Use fewer steps for testing
            guidance_scale=1.0
        )

        # Validate output
        assert intermediate_latents is not None
        assert isinstance(intermediate_latents, list)
        assert len(intermediate_latents) > 0

        # Get final inverted latent (Z_T)
        z_t = intermediate_latents[-1]
        assert z_t.shape == latents_input.shape

        print(f"✓ {inversion_type} inversion for 4D image input successful")
        print(f"  Input shape: {latents_input.shape}")
        print(f"  Output Z_T shape: {z_t.shape}")
        print(f"  Number of intermediate steps: {len(intermediate_latents)}")

    except Exception as e:
        pytest.fail(f"Failed to invert 4D image with {inversion_type}: {e}")


@pytest.mark.inversion
@pytest.mark.slow
@pytest.mark.parametrize("inversion_type", ["ddim"])
def test_inversion_5d_video_input(inversion_type, device, video_pipeline):
    """Test inversion modules with 5D video input (batch_size, num_frames, channels, height, width)."""
    import torch
    from inversions import DDIMInversion

    pipe, scheduler = video_pipeline

    # Create inversion instance
    inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create 5D test input: (batch_size, num_frames, channels, height, width)
    batch_size = 1
    num_frames = 8   # number of video frames
    channels = 4     # latent space channels
    height = 64      # latent space height
    width = 64       # latent space width

    # Reshape to 5D for video: (batch_size, num_frames, channels, height, width)
    latents_input = torch.randn(batch_size, num_frames, channels, height, width).to(device)

    # For video, we need to process frame by frame
    # Flatten the frame dimension into batch dimension
    latents_flat = latents_input.reshape(batch_size * num_frames, channels, height, width)

    # Create text embeddings
    text_embeddings = torch.randn(1, 77, 768).to(device)
    # Expand for all frames
    text_embeddings_expanded = text_embeddings.repeat(num_frames, 1, 1)

    try:
        # Test forward diffusion (video frames to noise)
        intermediate_latents = inversion.forward_diffusion(
            text_embeddings=text_embeddings_expanded,
            latents=latents_flat,
            num_inference_steps=10,  # Use fewer steps for testing
            guidance_scale=1.0
        )

        # Validate output
        assert intermediate_latents is not None
        assert isinstance(intermediate_latents, list)
        assert len(intermediate_latents) > 0

        # Get final inverted latent (Z_T)
        z_t_flat = intermediate_latents[-1]
        assert z_t_flat.shape == latents_flat.shape

        # Reshape back to 5D
        z_t = z_t_flat.reshape(batch_size, num_frames, channels, height, width)
        assert z_t.shape == latents_input.shape

        print(f"✓ {inversion_type} inversion for 5D video input successful")
        print(f"  Input shape: {latents_input.shape}")
        print(f"  Output Z_T shape: {z_t.shape}")
        print(f"  Number of intermediate steps: {len(intermediate_latents)}")

    except Exception as e:
        pytest.fail(f"Failed to invert 5D video with {inversion_type}: {e}")


@pytest.mark.inversion
def test_inversion_reconstruction_accuracy(device, image_pipeline):
    """Test that inversion can accurately reconstruct the latent vector."""
    import torch
    from inversions import DDIMInversion

    pipe, scheduler = image_pipeline
    inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create test input
    latents_input = torch.randn(1, 4, 64, 64).to(device)
    text_embeddings = torch.randn(1, 77, 768).to(device)

    try:
        # Forward diffusion: x_0 -> x_T
        forward_result = inversion.forward_diffusion(
            text_embeddings=text_embeddings,
            latents=latents_input,
            num_inference_steps=10,
            guidance_scale=1.0
        )

        z_t = forward_result[-1]

        # Backward diffusion: x_T -> x_0
        backward_result = inversion.backward_diffusion(
            text_embeddings=text_embeddings,
            latents=z_t,
            num_inference_steps=10,
            guidance_scale=1.0,
            reverse_process=False
        )

        reconstructed = backward_result[-1]

        # Calculate reconstruction error
        mse = torch.nn.functional.mse_loss(reconstructed, latents_input)

        print(f"✓ Inversion reconstruction test completed")
        print(f"  MSE between original and reconstructed: {mse.item():.6f}")
        print(f"  Original shape: {latents_input.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")

        # The reconstruction should be reasonably close
        # Note: DDIM is not perfectly reversible, so we expect some error
        assert mse.item() < 1.0, f"Reconstruction error too high: {mse.item()}"

    except Exception as e:
        pytest.fail(f"Failed reconstruction accuracy test: {e}")
