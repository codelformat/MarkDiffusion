# MarkDiffusion Watermark Algorithm Unit Tests

This directory contains all parameterized unit tests for the watermark algorithms and inversion modules in the MarkDiffusion project.

## 📋 Directory Structure

```text
test/
├── test_watermark_algorithms.py  # Main test file (parameterized tests)
├── conftest.py                   # Pytest configuration and fixtures
├── pytest.ini                    # Pytest config file
├── requirements-test.txt         # Test dependencies
├── run_tests.sh                  # Convenience test script
├── README.md                     # This document
└── test_method.py                # Legacy test file (kept for reference)
```

## 🎯 What Is Covered by the Tests

### Watermark Algorithms

#### Image watermark algorithms (9)

- **TR** (Tree-Ring)
- **GS** (Gaussian Shading)
- **PRC** (Perceptual Robust Coding)
- **RI** (Robust Invisible)
- **SEAL** (Secure Embedding Algorithm)
- **ROBIN** (Robust Invisible Noise)
- **WIND** (Watermark in Noise Domain)
- **GM** (Generative Model / GaussMarker)
- **SFW** (Stable Feature Watermark)

#### Video watermark algorithms (2)

- **VideoShield**
- **VideoMark**

### Inversion Modules

- **DDIM Inversion** – supports 4D image input and 5D video input  
- **Exact Inversion** – supports 4D image input

### Visualization Modules

- Visualization support for all image and video watermark algorithms  
- Visualization content includes: watermarked images, original latent vectors, inverted latent vectors, frequency-domain analysis, etc.  
- Each algorithm has its own dedicated visualizer

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install -r test/requirements-test.txt
```

Test dependencies include:

- pytest
- pytest-timeout
- pytest-html (optional, to generate HTML reports)
- pytest-cov (optional, for coverage reports)
- pytest-xdist (optional, for parallel testing)

### 2. Run Tests

#### Run directly with pytest

```bash
# Test all algorithms and modules
pytest test/test_watermark_algorithms.py -v

# Test a specific algorithm
pytest test/test_watermark_algorithms.py -v --algorithm TR

# Quick tests (initialization only)
pytest test/test_watermark_algorithms.py -v -k initialization
```

#### Use the convenience script

```bash
# Test all algorithms
./test/run_tests.sh

# Test image algorithms
./test/run_tests.sh --type image

# Test a specific algorithm
./test/run_tests.sh --algorithm TR

# Quick tests (initialization only)
./test/run_tests.sh --type quick
```

## 📋 Test Types and Coverage

### Watermark Algorithm Tests

#### 1. Initialization tests (11 tests)

Verify that watermark algorithms can be initialized correctly:

- Load configuration files
- Create watermark instances
- Validate pipeline type

```bash
pytest test/test_watermark_algorithms.py -v -k initialization
```

#### 2. Generation tests (22 tests)

Verify the generation functionality of watermark algorithms:

- Generate watermarked media (image/video)
- Generate non-watermarked media
- Validate output format and dimensions

```bash
# Test all generation functionality
pytest test/test_watermark_algorithms.py -v -k generation

# Skip generation tests
pytest test/test_watermark_algorithms.py -v --skip-generation
```

#### 3. Detection tests (11 tests)

Verify the detection functionality of watermark algorithms:

- Detect watermarks in watermarked media
- Detect on non-watermarked media (negative samples)
- Validate the result format of detection

```bash
# Test all detection functionality
pytest test/test_watermark_algorithms.py -v -k detection

# Skip detection tests
pytest test/test_watermark_algorithms.py -v --skip-detection
```

### Inversion Tests

#### 4. 4D image inversion tests (2 tests: DDIM + Exact)

Test the ability of inversion modules to handle 4D image input:

- Input shape: `(batch_size, channels, height, width)`
- Test both DDIM and Exact inversion methods
- Verify accurate recovery of latent vector Z_T

```bash
# Test 4D image inversion
pytest test/test_watermark_algorithms.py -v -k "test_inversion_4d"

# Test DDIM inversion
pytest test/test_watermark_algorithms.py -v -k "test_inversion_4d[ddim]"

# Test Exact inversion
pytest test/test_watermark_algorithms.py -v -k "test_inversion_4d[exact]"
```

#### 5. 5D video inversion tests (1 test: DDIM)

Test the ability of inversion modules to handle 5D video frame input:

- Input shape: `(batch_size, num_frames, channels, height, width)`
- Test DDIM inversion method
- Verify accurate recovery of latent vector Z_T for video frames

```bash
# Test 5D video inversion
pytest test/test_watermark_algorithms.py -v -k "test_inversion_5d"
```

#### 6. Inversion reconstruction accuracy tests (1 test)

Test the reconstruction accuracy of inversion modules:

- Forward diffusion: x₀ → x_T
- Reverse diffusion: x_T → x₀
- Validate that reconstruction error is within an acceptable range

```bash
# Test reconstruction accuracy
pytest test/test_watermark_algorithms.py -v -k "test_inversion_reconstruction"
```

#### Inversion test summary

```bash
# Test all inversion modules
pytest test/test_watermark_algorithms.py -v -m inversion

# Test inversion modules (excluding slow video tests)
pytest test/test_watermark_algorithms.py -v -m "inversion and not slow"
```

### Visualization Tests

#### 7. Image watermark visualization tests (9 tests)

Test visualization for image watermark algorithms:

- Load visualization data
- Create visualizer instances
- Test basic plotting methods (watermarked images, latent vectors, etc.)
- Generate and save visualization images

```bash
# Test visualization of all image algorithms
pytest test/test_watermark_algorithms.py -v -k "test_image_watermark_visualization"

# Test visualization for a specific algorithm
pytest test/test_watermark_algorithms.py -v -k "test_image_watermark_visualization[TR]"
```

#### 8. Video watermark visualization tests (2 tests)

Test visualization for video watermark algorithms:

- Load video visualization data
- Create visualizer instances
- Test visualization of video frames
- Generate and save visualization images

```bash
# Test visualization of all video algorithms
pytest test/test_watermark_algorithms.py -v -k "test_video_watermark_visualization"

# Test visualization for a specific video algorithm
pytest test/test_watermark_algorithms.py -v -k "test_video_watermark_visualization[VideoShield]"
```

#### Visualization test summary

```bash
# Test all visualization functionality
pytest test/test_watermark_algorithms.py -v -m visualization

# Test image visualization only (exclude video)
pytest test/test_watermark_algorithms.py -v -m "visualization and image"

# Test video visualization
pytest test/test_watermark_algorithms.py -v -m "visualization and video"
```

**Total**: 58+ parameterized test cases (44 watermark algorithm tests + 4 inversion tests + 11 visualization tests)

## 📖 Quick Command Reference

| Purpose | Command |
|--------|---------|
| Test all algorithms | `pytest test/test_watermark_algorithms.py -v` |
| Test image algorithms | `pytest test/test_watermark_algorithms.py -v -m image` |
| Test video algorithms | `pytest test/test_watermark_algorithms.py -v -m video` |
| Test inversion modules | `pytest test/test_watermark_algorithms.py -v -m inversion` |
| Test visualization functionality | `pytest test/test_watermark_algorithms.py -v -m visualization` |
| Test TR algorithm | `pytest test/test_watermark_algorithms.py -v -k TR` |
| Quick test (initialization) | `pytest test/test_watermark_algorithms.py -v -k initialization` |
| Skip generation tests | `pytest test/test_watermark_algorithms.py -v --skip-generation` |
| Run in parallel | `pytest test/test_watermark_algorithms.py -v -n auto` |
| Generate HTML report | `pytest test/test_watermark_algorithms.py -v --html=report.html` |
| Test 4D image inversion | `pytest test/test_watermark_algorithms.py -v -k test_inversion_4d` |
| Test 5D video inversion | `pytest test/test_watermark_algorithms.py -v -k test_inversion_5d` |
| Test image visualization | `pytest test/test_watermark_algorithms.py -v -k test_image_watermark_visualization` |
| Test video visualization | `pytest test/test_watermark_algorithms.py -v -k test_video_watermark_visualization` |

## ⚙️ Command-line Arguments

| Argument | Description | Default |
|---------|-------------|---------|
| `--algorithm` | Name of algorithm to test | None (test all) |
| `--image-model-path` | Image generation model path | `stabilityai/stable-diffusion-2-1-base` |
| `--video-model-path` | Video generation model path | `damo-vilab/text-to-video-ms-1.7b` |
| `--skip-generation` | Skip generation tests | False |
| `--skip-detection` | Skip detection tests | False |

## 🏷️ Test Markers

| Marker | Description | How to use |
|--------|-------------|------------|
| `@pytest.mark.image` | Image watermark tests | `-m image` |
| `@pytest.mark.video` | Video watermark tests | `-m video` |
| `@pytest.mark.inversion` | Inversion module tests | `-m inversion` |
| `@pytest.mark.visualization` | Visualization tests | `-m visualization` |
| `@pytest.mark.slow` | Slow tests (generation and detection) | `-m "not slow"` |

Use markers to filter tests:

```bash
# Run only image tests
pytest test/test_watermark_algorithms.py -v -m image

# Run only video tests
pytest test/test_watermark_algorithms.py -v -m video

# Run only inversion tests
pytest test/test_watermark_algorithms.py -v -m inversion

# Run only visualization tests
pytest test/test_watermark_algorithms.py -v -m visualization

# Exclude slow tests
pytest test/test_watermark_algorithms.py -v -m "not slow"

# Combined markers: initialization tests for image algorithms
pytest test/test_watermark_algorithms.py -v -m image -k initialization

# Combined markers: visualization tests for image algorithms
pytest test/test_watermark_algorithms.py -v -m "image and visualization"
```

## 💡 Practical Examples

### Example 1: Quickly verify all algorithms can initialize

```bash
pytest test/test_watermark_algorithms.py -v -k "initialization"
```

**Expected result**: 11 algorithm tests pass, taking about 10–30 seconds.

### Example 2: Fully test a single algorithm

```bash
pytest test/test_watermark_algorithms.py -v --algorithm TR
```

**Expected result**: 3 tests pass (initialization, generation, detection).

### Example 3: Test generation functionality of all image algorithms

```bash
pytest test/test_watermark_algorithms.py -v -m image -k "generation"
```

**Expected result**: 18 tests (9 algorithms × 2 generation types).

### Example 4: Test all inversion modules

```bash
pytest test/test_watermark_algorithms.py -v -m inversion
```

**Expected result**: 4 tests pass (2×4D tests + 1×5D test + 1 reconstruction test).

### Example 4.5: Test all visualization functionality

```bash
pytest test/test_watermark_algorithms.py -v -m visualization
```

**Expected result**: 11 tests pass (9 image visualizations + 2 video visualizations).

### Example 5: Run in CI/CD (skip slow tests)

```bash
pytest test/test_watermark_algorithms.py -v \
    -m "not slow" \
    --tb=short \
    --maxfail=3
```

### Example 6: Generate a full test report

```bash
pytest test/test_watermark_algorithms.py -v \
    --html=report.html \
    --cov=watermark \
    --cov=inversions \
    --cov-report=html
```

**Output**:

- `report.html` – test report
- `htmlcov/` – coverage report

### Example 7: Debug failures of a specific algorithm

```bash
pytest test/test_watermark_algorithms.py -v \
    --algorithm TR \
    -s \
    --tb=long \
    --pdb
```

### Example 8: Parallel testing for speed

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest test/test_watermark_algorithms.py -v -n auto
```

### Example 9: Test visualization of a specific algorithm

```bash
# Test visualization for TR algorithm
pytest test/test_watermark_algorithms.py -v -k "test_image_watermark_visualization[TR]"
```

**Expected result**: 1 test passes, visualization image generated in a temporary directory.

### Example 10: Test visualization for all image algorithms (skip video)

```bash
pytest test/test_watermark_algorithms.py -v -m "visualization and image"
```

**Expected result**: 9 image visualization tests pass.

## 📊 Test Reports

### View detailed output

```bash
# Show detailed test output (including print statements)
pytest test/test_watermark_algorithms.py -v -s

# Show test coverage
pytest test/test_watermark_algorithms.py -v --cov=watermark --cov=inversions
```

### Generate HTML report

```bash
# Install pytest-html
pip install pytest-html

# Generate HTML report
pytest test/test_watermark_algorithms.py -v --html=report.html --self-contained-html
```

## 🔧 Troubleshooting

### Issue 1: Model failed to load

**Error message**: `Failed to load image/video model`

**Solutions**:

1. Check that the model path is correct.
2. Ensure there is enough disk space and memory.
3. Use `--image-model-path` or `--video-model-path` to specify a local model path:

```bash
pytest test/test_watermark_algorithms.py -v \
    --image-model-path /local/path/to/model
```

### Issue 2: CUDA out of memory

**Error message**: `CUDA out of memory`

**Solutions**:

1. Reduce batch size.
2. Run tests on CPU (auto-detected).
3. Test a single algorithm at a time:

```bash
pytest test/test_watermark_algorithms.py -v --algorithm TR
```

### Issue 3: Test timeout

**Error message**: `Test timeout`

**Solutions**:

1. Increase timeout by changing the `timeout` value in `pytest.ini`.
2. Skip slow tests:

```bash
pytest test/test_watermark_algorithms.py -v --skip-generation --skip-detection
```

3. Run only initialization tests:

```bash
pytest test/test_watermark_algorithms.py -v -k "initialization"
```

### Issue 4: Config file not found

**Error message**: `Config file not found`

**Solutions**:

1. Make sure tests are run from the project root.
2. Check that the corresponding JSON config file exists under the `config/` directory.
3. Verify the exact filename and its case.

### Issue 5: Inversion tests failed

**Error message**: `Failed to invert 4D/5D input`

**Solutions**:

1. Check that the device has enough GPU memory.
2. Verify that scheduler and UNet models are loaded correctly.
3. Inspect detailed error output:

```bash
pytest test/test_watermark_algorithms.py -v -s -k inversion
```

### Issue 6: Visualization tests failed

**Error message**: `Algorithm does not implement get_visualization_data()`

**Solutions**:

1. Ensure the watermark algorithm implements `get_visualization_data()`.
2. Check that the algorithm is registered in `VISUALIZATION_DATA_MAPPING`.
3. Inspect detailed error output:

```bash
pytest test/test_watermark_algorithms.py -v -s -k visualization
```

### Issue 7: matplotlib-related errors

**Error message**: `No module named 'matplotlib'`

**Solutions**:

1. Install matplotlib:

```bash
pip install matplotlib
```

2. If running in a headless environment (e.g., server), set a backend:

```bash
export MPLBACKEND=Agg
pytest test/test_watermark_algorithms.py -v -m visualization
```

## 📈 Performance Optimization

### Test durations

- **Quick tests** (initialization only): ~10–30 seconds  
- **Full tests** (including generation and detection): ~10–30 minutes (depending on hardware)  
- **Inversion tests**: ~1–3 minutes (4D), ~5–10 minutes (5D)  
- **Visualization tests**: ~5–15 minutes (requires generating watermarked images first)

### Optimization tips

1. **Use `-k initialization` for quick verification**

   ```bash
   pytest test/test_watermark_algorithms.py -v -k initialization
   ```

2. **Use `--skip-generation` to skip expensive generation tests**

   ```bash
   pytest test/test_watermark_algorithms.py -v --skip-generation
   ```

3. **Use `-n auto` for parallel execution**

   ```bash
   pip install pytest-xdist
   pytest test/test_watermark_algorithms.py -v -n auto
   ```

4. **Use `--algorithm` to test a single algorithm**

   ```bash
   pytest test/test_watermark_algorithms.py -v --algorithm TR
   ```

5. **Use session-scoped fixtures to cache models**

   - Models are loaded only once and shared across tests.
   - Handled automatically in `conftest.py`.

6. **Use GPU acceleration**

   - Tests automatically detect and use available CUDA devices.
   - Significantly speeds up testing.

7. **Skip visualization tests to save time**

   ```bash
   pytest test/test_watermark_algorithms.py -v -m "not visualization"
   ```

## 📝 Adding New Tests

### Add tests for a new watermark algorithm

To add tests for a new watermark algorithm:

1. Register the new algorithm in `watermark/auto_watermark.py`.
2. Add its config file under the `config/` directory.
3. The test framework will automatically discover and test the new algorithm.

**No test code changes required!**

### Add new tests for inversion modules

Add a new test function in `test_watermark_algorithms.py`:

```python
@pytest.mark.inversion
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_new_inversion_feature(inversion_type, device, image_pipeline):
    # Test code
    pass
```

### Modify test parameters

Edit constants in `conftest.py`:

```python
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
NUM_FRAMES = 16
```

Or override the default values via command-line arguments.

## ✨ Key Features

1. ✅ **Zero redundancy** – one test file covers all 11 algorithms + inversion modules + visualization modules  
2. ✅ **Parameterized tests** – test cases are automatically generated per algorithm  
3. ✅ **Flexible filtering** – filter by algorithm, type, or functionality  
4. ✅ **Command-line options** – customize model paths, skip tests, etc.  
5. ✅ **Session-scoped fixtures** – models loaded once for all tests, improving efficiency  
6. ✅ **Comprehensive docs** – full usage instructions and examples  
7. ✅ **Convenience scripts** – friendly CLI tooling  
8. ✅ **CI/CD ready** – includes example GitHub Actions configuration  
9. ✅ **Extensible** – add new algorithms without modifying test code  
10. ✅ **Graceful error handling** – handles unimplemented functionality cleanly  
11. ✅ **Inversion tests** – complete 4D/5D input tests and reconstruction validation  
12. ✅ **Visualization tests** – automated testing of visualization for all algorithms  

## 🎯 Test Coverage Summary

### Algorithm Test Matrix

| Test type | Image algos | Video algos | Inversion | Visualization | Total |
|-----------|-------------|-------------|-----------|---------------|-------|
| Initialization | 9 | 2 | - | - | 11 |
| Generation (with watermark) | 9 | 2 | - | - | 11 |
| Generation (without watermark) | 9 | 2 | - | - | 11 |
| Detection | 9 | 2 | - | - | 11 |
| 4D inversion tests | - | - | 2 | - | 2 |
| 5D inversion tests | - | - | 1 | - | 1 |
| Reconstruction accuracy tests | - | - | 1 | - | 1 |
| Visualization tests | 9 | 2 | - | - | 11 |
| **Total** | **45** | **10** | **4** | **11** | **59** |

### Inversion Test Details

| Test name | Input dimension | Inversion method | What is tested |
|-----------|-----------------|------------------|----------------|
| `test_inversion_4d_image_input[ddim]` | 4D (B,C,H,W) | DDIM | Image latent inversion |
| `test_inversion_4d_image_input[exact]` | 4D (B,C,H,W) | Exact | Image latent inversion |
| `test_inversion_5d_video_input[ddim]` | 5D (B,F,C,H,W) | DDIM | Video frame latent inversion |
| `test_inversion_reconstruction_accuracy` | 4D (B,C,H,W) | DDIM | Forward + reverse reconstruction accuracy |

**Notation**:

- B: batch_size  
- C: channels (latent-space channels, usually 4)  
- H: height (latent-space height)  
- W: width (latent-space width)  
- F: num_frames (number of video frames)

### Visualization Test Details

| Test name | Algorithm type | What is tested |
|-----------|----------------|----------------|
| `test_image_watermark_visualization[TR]` | Image | TR visualization (watermarked image, latent vectors, frequency analysis) |
| `test_image_watermark_visualization[GS]` | Image | GS visualization (watermark bits, reconstructed bits) |
| `test_image_watermark_visualization[PRC]` | Image | PRC visualization |
| `test_image_watermark_visualization[RI]` | Image | RI visualization |
| `test_image_watermark_visualization[SEAL]` | Image | SEAL visualization |
| `test_image_watermark_visualization[ROBIN]` | Image | ROBIN visualization |
| `test_image_watermark_visualization[WIND]` | Image | WIND visualization |
| `test_image_watermark_visualization[GM]` | Image | GM visualization |
| `test_image_watermark_visualization[SFW]` | Image | SFW visualization |
| `test_video_watermark_visualization[VideoShield]` | Video | VideoShield visualization (video frames) |
| `test_video_watermark_visualization[VideoMark]` | Video | VideoMark visualization (video frames) |

## 🤝 Contributing

### Contribute test improvements

If you find issues in the tests or want to improve the test framework:

1. Open an issue describing the problem or suggestion.  
2. Fork the project and create a new branch.  
3. Submit a pull request with test results.  
4. Make sure all existing tests still pass.

### Add tests for new features

1. Add new test functions in `test_watermark_algorithms.py`.  
2. Use `@pytest.mark.parametrize` decorators.  
3. Use fixtures from `conftest.py`.  
4. Add appropriate test markers.  
5. Update this document.

## 🎓 Learning Resources

### pytest

- [Official pytest docs](https://docs.pytest.org/)  
- [pytest fixtures docs](https://docs.pytest.org/en/stable/fixture.html)  
- [pytest parametrize docs](https://docs.pytest.org/en/stable/parametrize.html)  
- [pytest marks docs](https://docs.pytest.org/en/stable/mark.html)

### Project

- MarkDiffusion project documentation  
- Implementations under the `watermark/` directory  
- Inversion modules under the `inversions/` directory  
- Config files under the `config/` directory  

## 💻 CI/CD Integration

### GitHub Actions Example

See the `.github_workflows_example.yml` file, which includes:

1. **Quick tests**: initialization only (good for each commit)  
2. **Full tests**: includes generation and detection (good for PRs and releases)  
3. **Matrix tests**: parallel testing across multiple algorithms  

### Local CI-style Testing

Simulate CI environment locally:

```bash
# Quick CI-style tests
pytest test/test_watermark_algorithms.py -v \
    -k initialization \
    --tb=short \
    --maxfail=3

# Full CI tests
pytest test/test_watermark_algorithms.py -v \
    --html=report.html \
    --cov=watermark \
    --cov=inversions \
    --cov-report=html
```

## 📄 License

These test codes follow the MarkDiffusion project’s Apache 2.0 license.
