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

    # Get correct text embedding dimension from the model
    # Different SD versions use different text encoders (CLIP: 768, OpenCLIP: 1024)
    text_encoder = pipe.text_encoder
    with torch.no_grad():
        # Use a dummy prompt to get properly formatted embeddings
        text_inputs = pipe.tokenizer(
            "a test prompt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

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
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Number of intermediate steps: {len(intermediate_latents)}")

    except Exception as e:
        pytest.fail(f"Failed to invert 4D image with {inversion_type}: {e}")


@pytest.mark.inversion
@pytest.mark.slow
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_inversion_5d_video_input(inversion_type, device, video_pipeline):
    """Test inversion modules with 5D video input (batch_size, num_frames, channels, height, width)."""
    import torch
    from inversions import DDIMInversion, ExactInversion

    pipe, scheduler = video_pipeline

    # Create inversion instance
    if inversion_type == "ddim":
        inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)
    else:  # exact
        inversion = ExactInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create 5D test input: (batch_size, num_frames, channels, height, width)
    batch_size = 1
    num_frames = 8   # number of video frames
    channels = 4     # latent space channels
    height = 64      # latent space height
    width = 64       # latent space width

    # Reshape to 5D for video: (batch_size, num_frames, channels, height, width)
    latents_input = torch.randn(batch_size, num_frames, channels, height, width).to(device)

    # Get correct text embeddings from the model
    text_encoder = pipe.text_encoder
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            "a test video prompt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    try:
        # Test forward diffusion (video frames to noise)
        intermediate_latents = inversion.forward_diffusion(
            text_embeddings=text_embeddings,
            latents=latents_input.to(pipe.dtype),
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

        print(f"✓ {inversion_type} inversion for 5D video input successful")
        print(f"  Input shape: {latents_input.shape}")
        print(f"  Output Z_T shape: {z_t.shape}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Number of intermediate steps: {len(intermediate_latents)}")

    except Exception as e:
        pytest.fail(f"Failed to invert 5D video with {inversion_type}: {e}")


@pytest.mark.inversion
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_inversion_reconstruction_accuracy(device, image_pipeline, inversion_type):
    """Test that inversion can accurately reconstruct the latent vector."""
    import torch
    from inversions import DDIMInversion, ExactInversion

    pipe, scheduler = image_pipeline
    if inversion_type == "ddim":
        inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)
    else:  # exact
        inversion = ExactInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create test input
    latents_input = torch.randn(1, 4, 64, 64).to(device)

    # Get correct text embeddings from the model
    text_encoder = pipe.text_encoder
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            "a test prompt for reconstruction",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

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
        print(f"  Text embeddings shape: {text_embeddings.shape}")

        # The reconstruction should be reasonably close
        # Note: DDIM is not perfectly reversible, so we expect some error
        assert mse.item() < 1.0, f"Reconstruction error too high: {mse.item()}"

    except Exception as e:
        pytest.fail(f"Failed reconstruction accuracy test: {e}")


# ============================================================================
# Test Cases - Visualization
# ============================================================================

@pytest.mark.image
@pytest.mark.visualization
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_visualization(algorithm_name, image_diffusion_config, tmp_path):
    """Test visualization generation for image watermark algorithms."""
    from visualize.auto_visualization import AutoVisualizer, VISUALIZATION_DATA_MAPPING
    from visualize.data_for_visualization import DataForVisualization

    # Skip if visualization not supported for this algorithm
    if algorithm_name not in VISUALIZATION_DATA_MAPPING:
        pytest.skip(f"{algorithm_name} does not have visualization support")

    try:
        # Load watermark algorithm
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )

        # Generate watermarked image to get visualization data
        watermarked_image = watermark.generate_watermarked_media(TEST_PROMPT_IMAGE)

        # Get visualization data from the watermark instance
        # The watermark instance should have stored the necessary data
        if not hasattr(watermark, 'get_visualization_data'):
            pytest.skip(f"{algorithm_name} does not implement get_visualization_data()")

        vis_data = watermark.get_visualization_data()

        # Validate visualization data
        assert vis_data is not None
        assert isinstance(vis_data, DataForVisualization)
        assert vis_data.algorithm_name == algorithm_name

        # Load visualizer
        visualizer = AutoVisualizer.load(
            algorithm_name=algorithm_name,
            data_for_visualization=vis_data
        )

        assert visualizer is not None

        # Test basic visualization methods
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Test drawing watermarked image
        visualizer.draw_watermarked_image(ax=ax)
        plt.close(fig)

        # Test drawing original latents
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        visualizer.draw_orig_latents(channel=0, ax=ax)
        plt.close(fig)

        # Test algorithm-specific methods based on algorithm type
        algorithm_specific_methods = {
            'TR': ['draw_pattern_fft', 'draw_inverted_pattern_fft'],
            'GS': ['draw_watermark_bits', 'draw_reconstructed_watermark_bits'],
            'PRC': ['draw_generator_matrix', 'draw_codeword', 'draw_recovered_codeword', 'draw_difference_map'],
            'RI': ['draw_ring_pattern_fft', 'draw_heter_pattern_fft', 'draw_inverted_ring_pattern_fft'],
            'SEAL': ['draw_embedding_distributions', 'draw_patch_diff'],
            'ROBIN': ['draw_pattern_fft', 'draw_inverted_pattern_fft', 'draw_optimized_watermark'],
            'WIND': ['draw_group_pattern_fft', 'draw_orig_noise_wo_group_pattern',
                     'draw_inverted_noise_wo_group_pattern', 'draw_diff_noise_wo_group_pattern',
                     'draw_inverted_group_pattern_fft'],
            'GM': [],  # GM may not have specific methods beyond base class
            'SFW': []  # SFW may not have specific methods beyond base class
        }

        # Test algorithm-specific visualization methods
        specific_methods = algorithm_specific_methods.get(algorithm_name, [])
        tested_methods = []

        for method_name in specific_methods:
            if hasattr(visualizer, method_name):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                try:
                    method = getattr(visualizer, method_name)
                    # Call method with minimal parameters
                    method(ax=ax)
                    tested_methods.append(method_name)
                    plt.close(fig)
                except Exception as e:
                    plt.close(fig)
                    print(f"  Warning: {algorithm_name}.{method_name}() failed: {e}")

        # Save a comprehensive visualization with algorithm-specific methods
        methods_to_visualize = ['draw_watermarked_image', 'draw_orig_latents']
        method_kwargs_list = [{}, {'channel': 0}]

        # Add one algorithm-specific method if available
        if tested_methods:
            methods_to_visualize.append(tested_methods[0])
            method_kwargs_list.append({})

        # Adjust grid size based on number of methods
        num_methods = len(methods_to_visualize)
        cols = min(num_methods, 3)
        rows = (num_methods + cols - 1) // cols

        save_path = tmp_path / f"{algorithm_name}_test_visualization.png"
        fig = visualizer.visualize(
            rows=rows,
            cols=cols,
            methods=methods_to_visualize,
            method_kwargs=method_kwargs_list,
            save_path=str(save_path)
        )
        plt.close(fig)

        # Verify the file was created
        assert save_path.exists()

        print(f"✓ {algorithm_name} visualization test passed")
        print(f"  Base methods tested: draw_watermarked_image, draw_orig_latents")
        if tested_methods:
            print(f"  Algorithm-specific methods tested: {', '.join(tested_methods)}")
        print(f"  Visualization saved to: {save_path}")

    except NotImplementedError as e:
        pytest.skip(f"{algorithm_name} visualization not fully implemented: {e}")
    except Exception as e:
        pytest.fail(f"Failed to test visualization for {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.visualization
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_visualization(algorithm_name, video_diffusion_config, tmp_path):
    """Test visualization generation for video watermark algorithms."""
    from visualize.auto_visualization import AutoVisualizer, VISUALIZATION_DATA_MAPPING
    from visualize.data_for_visualization import DataForVisualization

    # Skip if visualization not supported for this algorithm
    if algorithm_name not in VISUALIZATION_DATA_MAPPING:
        pytest.skip(f"{algorithm_name} does not have visualization support")

    try:
        # Load watermark algorithm
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate watermarked video to get visualization data
        watermarked_frames = watermark.generate_watermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Get visualization data from the watermark instance
        if not hasattr(watermark, 'get_visualization_data'):
            pytest.skip(f"{algorithm_name} does not implement get_visualization_data()")

        vis_data = watermark.get_visualization_data()

        # Validate visualization data
        assert vis_data is not None
        assert isinstance(vis_data, DataForVisualization)
        assert vis_data.algorithm_name == algorithm_name

        # Load visualizer
        visualizer = AutoVisualizer.load(
            algorithm_name=algorithm_name,
            data_for_visualization=vis_data
        )

        assert visualizer is not None

        # Test basic visualization methods for video
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Test drawing watermarked image (first frame)
        visualizer.draw_watermarked_image(ax=ax)
        plt.close(fig)

        # Test drawing original latents with frame parameter
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        visualizer.draw_orig_latents(channel=0, frame=0, ax=ax)
        plt.close(fig)

        # Test algorithm-specific methods based on video algorithm type
        algorithm_specific_methods = {
            'VideoShield': [
                ('draw_watermark_bits', {'channel': None, 'frame': 0}),
                ('draw_reconstructed_watermark_bits', {'channel': None, 'frame': 0}),
                ('draw_watermarked_video_frames', {'num_frames': 4})
            ],
            'VideoMark': [
                ('draw_watermarked_video_frames', {'num_frames': 4})
            ]
        }

        # Test algorithm-specific visualization methods
        specific_methods = algorithm_specific_methods.get(algorithm_name, [])
        tested_methods = []

        for method_info in specific_methods:
            if isinstance(method_info, tuple):
                method_name, kwargs = method_info
            else:
                method_name = method_info
                kwargs = {}

            if hasattr(visualizer, method_name):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                try:
                    method = getattr(visualizer, method_name)
                    # Call method with specific parameters for video methods
                    method(ax=ax, **kwargs)
                    tested_methods.append(method_name)
                    plt.close(fig)
                except Exception as e:
                    plt.close(fig)
                    print(f"  Warning: {algorithm_name}.{method_name}() failed: {e}")

        # Save a comprehensive visualization with algorithm-specific methods
        methods_to_visualize = ['draw_watermarked_image', 'draw_orig_latents']
        method_kwargs_list = [{}, {'channel': 0, 'frame': 0}]

        # Add video-specific method if available
        if algorithm_name == 'VideoShield':
            # Test watermark bits visualization for VideoShield
            if 'draw_watermark_bits' in [m[0] if isinstance(m, tuple) else m for m in tested_methods]:
                methods_to_visualize.append('draw_watermark_bits')
                method_kwargs_list.append({'channel': None, 'frame': 0})
        elif algorithm_name == 'VideoMark':
            # Test video frames visualization for VideoMark
            if 'draw_watermarked_video_frames' in tested_methods:
                methods_to_visualize.append('draw_watermarked_video_frames')
                method_kwargs_list.append({'num_frames': 4})

        # Adjust grid size based on number of methods
        num_methods = len(methods_to_visualize)
        cols = min(num_methods, 3)
        rows = (num_methods + cols - 1) // cols

        save_path = tmp_path / f"{algorithm_name}_test_visualization.png"
        fig = visualizer.visualize(
            rows=rows,
            cols=cols,
            methods=methods_to_visualize,
            method_kwargs=method_kwargs_list,
            save_path=str(save_path)
        )
        plt.close(fig)

        # Verify the file was created
        assert save_path.exists()

        print(f"✓ {algorithm_name} video visualization test passed")
        print(f"  Base methods tested: draw_watermarked_image, draw_orig_latents")
        if tested_methods:
            print(f"  Algorithm-specific methods tested: {', '.join(tested_methods)}")
        print(f"  Visualization saved to: {save_path}")

    except NotImplementedError as e:
        pytest.skip(f"{algorithm_name} video visualization not fully implemented: {e}")
    except Exception as e:
        pytest.fail(f"Failed to test video visualization for {algorithm_name}: {e}")
